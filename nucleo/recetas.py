"""
Recetas de entrada de cada modelo.

Un ONNX no dice qué espera en cada canal. Eso va en una receta JSON junto al
.onnx, con el mismo nombre de base:

    modelos/deeplabv3plus.onnx
    modelos/deeplabv3plus.json

La receta fija los canales y su orden, la normalización con sus constantes, el
tipo de salida y el umbral. La función `armar` construye el tensor a partir de
canales ya expresados en unidades de Landslide4Sense, y la usan por igual el
backend (con imágenes de cubo) y la verificación (con los h5 de L4S). Así lo que
se mide contra Landslide4Sense es exactamente lo que corre en producción.
"""
import json
import os

import numpy as np

from . import config

# Posición de cada canal en los h5 de Landslide4Sense (128 x 128 x 14).
IDX_L4S = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6,
           'B08': 7, 'B09': 8, 'B10': 9, 'B11': 10, 'B12': 11,
           'pendiente': 12, 'elevacion': 13}
INDICES = {'NDVI': ('B08', 'B04'), 'NDWI': ('B03', 'B08'), 'GRVI': ('B03', 'B04')}
# Nombres de Landslide4Sense para los canales de relieve: B13 = pendiente y
# B14 = DEM. Una receta puede usar cualquiera de los dos nombres.
ALIAS = {'B13': 'pendiente', 'B14': 'elevacion', 'B1': 'B01', 'B2': 'B02', 'B3': 'B03',
         'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', 'B8': 'B08', 'B9': 'B09'}
TERRENO = ('pendiente', 'elevacion')
DERIVADOS_TERRENO = ('CURVATURA', 'TPI')
SALIDAS = ('sigmoide', 'softmax')


class RecetaInvalida(ValueError):
    pass


def normalizar_nombres(r):
    """B13 -> pendiente, B14 -> elevacion, B1 -> B01... en canales y constantes."""
    al = lambda c: ALIAS.get(c, c)
    r['canales'] = [al(c) for c in r['canales']]
    n = r.get('normalizacion', {})
    if 'canales' in n:
        n['canales'] = [al(c) for c in n['canales']]
    if 'por_canal' in n:
        n['por_canal'] = {al(c): v for c, v in n['por_canal'].items()}
    if 'escala_terreno' in r:
        r['escala_terreno'] = {al(c): v for c, v in r['escala_terreno'].items()}
    if 'rango_entrenamiento' in r:
        r['rango_entrenamiento'] = {al(c): v for c, v in r['rango_entrenamiento'].items()}
    return r


def leer(ruta_json):
    with open(ruta_json, encoding='utf-8') as f:
        r = json.load(f)
    normalizar_nombres(r)
    validar(r, os.path.basename(ruta_json))
    return r


def validar(r, origen='receta'):
    for campo in ('nombre', 'onnx', 'canales', 'lado_px', 'normalizacion', 'salida',
                  'umbral'):
        if campo not in r:
            raise RecetaInvalida(f'{origen}: falta "{campo}"')
    for c in r['canales']:
        if c not in IDX_L4S and c not in INDICES and c not in DERIVADOS_TERRENO:
            raise RecetaInvalida(f'{origen}: canal desconocido "{c}"')
    if r['salida'] not in SALIDAS:
        raise RecetaInvalida(f'{origen}: salida "{r["salida"]}" no es {SALIDAS}')
    n = r['normalizacion']
    if n.get('metodo') not in ('percentil', 'media_desv', 'ninguna'):
        raise RecetaInvalida(f'{origen}: normalización "{n.get("metodo")}" desconocida')
    if n['metodo'] != 'ninguna':
        for c in n.get('canales', []):
            if c not in n.get('por_canal', {}):
                raise RecetaInvalida(f'{origen}: faltan las constantes de "{c}"')
    for c, k in r.get('escala_terreno', {}).items():
        if c not in TERRENO or not float(k) > 0:
            raise RecetaInvalida(f'{origen}: escala_terreno["{c}"] no es válida')
    if not 0 < float(r['umbral']) < 1:
        raise RecetaInvalida(f'{origen}: el umbral tiene que estar entre 0 y 1')


def bandas_s2(r):
    """Bandas de Sentinel-2 que hay que pedir para esta receta."""
    s = set()
    for c in r['canales']:
        if c in INDICES:
            s.update(INDICES[c])
        elif c.startswith('B'):
            s.add(c)
    return sorted(s)


def necesita_dem(r):
    return any(c in TERRENO or c in DERIVADOS_TERRENO for c in r['canales'])


def _indice(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    den = a + b
    out = np.zeros_like(a)
    np.divide(a - b, den, out=out, where=den != 0)
    return np.clip(out, -1.0, 1.0)


def _curvatura_dem(elev):
    """Curvatura laplaciana discreta del DEM normalizada en [-1, 1].
    Positivo = ladera cóncava/convergente (acumula agua); negativo = cresta convexa."""
    z = elev.astype(np.float32)
    zp = np.pad(z, 1, mode='edge')
    lap = (zp[:-2, 1:-1] + zp[2:, 1:-1] + zp[1:-1, :-2] + zp[1:-1, 2:]) - 4.0 * z
    return np.tanh(lap * 15.0).astype(np.float32)


def _tpi_dem(elev):
    """Topographic Position Index (TPI) en ventana 5×5, acotado en [-1, 1].
    Positivo = crestas/zonas altas locales; negativo = fondos de valle/quebradas."""
    z = elev.astype(np.float32)
    zp = np.pad(z, 2, mode='edge')
    acc = np.zeros_like(z)
    for dy in range(5):
        for dx in range(5):
            acc += zp[dy:dy + z.shape[0], dx:dx + z.shape[1]]
    media_local = acc / 25.0
    return np.tanh((z - media_local) * 15.0).astype(np.float32)


def armar(fuente, valido, r):
    """fuente: canal -> (H, W) en unidades de Landslide4Sense.
    valido: (H, W) bool. Devuelve (C, H, W) float32 listo para el ONNX."""
    n = r['normalizacion']
    a_normalizar = set(n.get('canales', []))
    capas = []
    for c in r['canales']:
        if c in INDICES:
            x = _indice(fuente[INDICES[c][0]], fuente[INDICES[c][1]])
        elif c == 'CURVATURA':
            x = _curvatura_dem(fuente['elevacion'])
        elif c == 'TPI':
            x = _tpi_dem(fuente['elevacion'])
        else:
            x = fuente[c].astype(np.float32)
        if c in a_normalizar:
            k = n['por_canal'][c]
            if n['metodo'] == 'percentil':
                lo, hi = float(k['lo']), float(k['hi'])
                x = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            elif n['metodo'] == 'media_desv':
                x = (x - float(k['media'])) / max(float(k['desv']), 1e-9)
        capas.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    t = np.stack(capas).astype(np.float32)
    return t * valido.astype(np.float32)[None]


# ------------------------------------------------------------ dos adaptadores
def desde_l4s(h5, r):
    """Un parche de Landslide4Sense, (128, 128, 14) o (14, 128, 128), ya viene en
    sus unidades. Solo hay que tomar los canales."""
    a = np.asarray(h5, np.float32)
    if a.shape[-1] == 14:
        a = a.transpose(2, 0, 1)
    valido = np.isfinite(a).all(axis=0)
    a = np.nan_to_num(a, nan=0.0)
    necesarios = set()
    for c in r['canales']:
        if c in DERIVADOS_TERRENO:
            necesarios.add('elevacion')
        else:
            necesarios.update(INDICES.get(c, (c,)))
    fuente = {c: a[IDX_L4S[c]] for c in necesarios}
    return armar(fuente, valido, r), valido


def desde_l2a(bandas_dn, terreno, valido, r, escala=None):
    """Bandas de Sentinel-2 L2A en niveles digitales (ya sin el desplazamiento de
    la baseline 04.00), y pendiente en grados y elevación en metros. Se llevan a
    unidades de Landslide4Sense dividiendo cada banda por su media de escena,
    que es como se construyó Landslide4Sense."""
    escala = escala or r.get('escala_bandas') or config.ESCALA_L4S
    if not escala:
        raise RecetaInvalida('no hay escala de bandas: falta escala_laconvencion.json')
    fuente = {}
    for b, x in bandas_dn.items():
        if b not in escala:
            raise RecetaInvalida(f'no hay constante de escala para {b}')
        fuente[b] = x.astype(np.float32) / float(escala[b])
    # pendiente en grados y elevación en metros, a unidades de Landslide4Sense;
    # si la receta no trae sus constantes valen las de Landslide4Sense
    k_terreno = dict(config.ESCALA_TERRENO_L4S, **r.get('escala_terreno', {}))
    for c, x in (terreno or {}).items():
        fuente[c] = x.astype(np.float32) / float(k_terreno[c])
    return armar(fuente, valido, r)


def probabilidad(salida, tipo):
    """(N, C, H, W) -> (N, H, W) en [0, 1] para la clase deslizamiento."""
    z = np.asarray(salida, np.float32)
    if tipo == 'sigmoide':
        if z.shape[1] != 1:
            raise RecetaInvalida(f'la receta dice sigmoide y el modelo da {z.shape[1]} canales')
        return 1.0 / (1.0 + np.exp(-np.clip(z[:, 0], -60, 60)))
    if z.shape[1] != 2:
        raise RecetaInvalida(f'la receta dice softmax y el modelo da {z.shape[1]} canales')
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)
