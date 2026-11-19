"""
Ajuste fino de un modelo ya entrenado (U-Net v4) con los parches marcados de La Convención.

Parte del punto de control de entrenar.py, sigue entrenando con tasa baja mezclando en cada
lote parches locales y parches de Landslide4Sense (para que no olvide lo que ya sabe) y mide
el antes y el después sobre las MISMAS zonas de validación local, que el modelo no ve al
entrenar, y sobre la prueba oficial de Landslide4Sense.

Qué hace, en orden:

  1. Lee el punto de control base y sus estadísticas de normalización. NO se recalculan:
     el modelo ajustado usa la misma receta de entrada, así el servidor no cambia.
  2. Arma los parches locales con `recetas.desde_l2a` y escala_laconvencion.json, la MISMA
     función y la misma escala que usa el servidor al detectar.
  3. Usa solo parches con máscara de nivel 1. En la máscara, 255 se ignora.
     La partición sale de la columna «conjunto» de indice_parches.csv (por zona).
     Con --debiles se suman, solo para entrenar y con peso bajo, parches con máscara de
     nivel 2 (propuestas de INGEMMET) que tengan alguna pieza aceptada.
  4. Mide el modelo base (época 0) y después cada época, en validación local y en la
     validación de Landslide4Sense.
     BatchNorm queda fijo (medias y varianzas del modelo base) salvo --bn-libre.
  5. Elige la época con mejor IoU en validación local entre las que no pierden más de
     --tolerancia-l4s de IoU en la validación de Landslide4Sense. La época 0 compite: si
     nada la supera, se queda el modelo base y no se escribe uno nuevo.
  6. Elige el umbral en validación (local si hay cicatrices suficientes; si no, el de L4S)
     y mide una sola vez la prueba oficial de Landslide4Sense con el base y con el ajustado.
  7. Da el intervalo de la mejora local remuestreando ZONAS (no píxeles ni parches).

Ojo al leer el resultado: la validación local se usa para elegir época y umbral, así que su
IoU es optimista. Con pocas cicatrices el intervalo es ancho y el resultado es exploratorio;
el script lo dice. Para una cifra final hace falta un tercer grupo de zonas que no se use
para decidir nada.

Uso (Colab con GPU):
  python3 ajuste_fino.py --l4s /content/l4s \
      --desde  .../mejor_unet_b1b4b13b14_b2809be440d6.pt \
      --stats  .../stats_norm_b2809be440d6.json \
      --parches /content/local/parches_128 --mascaras /content/local/mascaras_nivel1/parches_128 \
      --salida /content/drive/MyDrive/ajuste_fino

Deja, con la misma forma que entrenar.py (exportar_onnx.py los lee sin cambios):
  mejor_<arq>_b1b4b13b14_<firma>.pt   stats_norm_<firma>.json   resultados_<arq>_b1b4b13b14_<firma>.json
  historial_ajuste_<firma>.csv
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..', 'backend'))
sys.path.insert(0, os.path.join(AQUI, '..'))

from entrenar import (ETIQUETA_CANALES, Conteo, cargar, factor_lr, modelo_con_indices, objetos,  # noqa: E402
                      pares, perdida_mixta, predecir, receta_de)

UMBRALES = np.round(np.arange(0.05, 0.96, 0.05), 2)
BANDAS = ['B01', 'B02', 'B03', 'B04']


# ------------------------------------------------------------------- datos locales
def cargar_locales(carpeta_parches, carpeta_mascaras, receta, escala, conjunto_permitido, solo_con_positivos=False,
                   tiers=None):
    """Parches locales con máscara. Devuelve X, Y, V y la lista de filas del índice.
    V = 1 donde el píxel cuenta: con dato y máscara distinta de 255.
    tiers: si se da (conjunto de nombres, p. ej. {'oro', 'plata'}), con catálogo de Tiers solo pasan esos Tiers."""
    import rasterio
    from nucleo import recetas
    filas = []
    indice_amp = os.path.join(carpeta_parches, 'indice_parches_ampliado.csv')
    indice = indice_amp if os.path.exists(indice_amp) else os.path.join(carpeta_parches, 'indice_parches.csv')
    if not os.path.exists(indice):
        raise SystemExit(f'falta {indice}: corre cortar_parches_128.py')
    with open(indice, encoding='utf-8') as f_ind:
        filas_ind = list(csv.DictReader(f_ind))

    # Si existe catalogo_parches_tiers.csv, leer pesos estratificados por parche (Oro=1.0, HardNeg=0.8, Plata=0.5, Bronce=0.3/0.2)
    pesos_tiers, nombres_tier = None, {}
    posibles_tiers = [
        os.path.join(os.path.dirname(carpeta_mascaras.rstrip('/\\')), 'catalogo_parches_tiers.csv'),
        os.path.join(carpeta_mascaras, '..', 'catalogo_parches_tiers.csv'),
    ]
    for rt in posibles_tiers:
        if os.path.exists(rt):
            with open(rt, encoding='utf-8') as ft:
                pesos_tiers, nombres_tier = {}, {}
                for row in csv.DictReader(ft):
                    if float(row.get('peso_perdida') or 0) > 0:
                        pesos_tiers[(row['proyecto'], row['parche'])] = float(row['peso_perdida'])
                        nombres_tier[(row['proyecto'], row['parche'])] = (row.get('tier') or '').strip().lower()
            break

    # Si existe mascaras_nivel2_completo.csv en el padre de carpeta_mascaras, respetar incluir_debil
    permitidos_n2 = None
    posibles_csv = [
        os.path.join(os.path.dirname(carpeta_mascaras.rstrip('/\\')), 'mascaras_nivel2_completo.csv'),
        os.path.join(carpeta_mascaras, '..', 'mascaras_nivel2_completo.csv'),
    ]
    for rc in posibles_csv:
        if os.path.exists(rc):
            with open(rc, encoding='utf-8') as fc:
                permitidos_n2 = {
                    (row['proyecto'], row['tesela'])
                    for row in csv.DictReader(fc)
                    if row.get('incluir_debil') == 'si'
                }
            break

    for r in filas_ind:
        if r['conjunto'] not in conjunto_permitido:
            continue
        m = os.path.join(carpeta_mascaras, r['proyecto'], r['parche'] + '.tif')
        es_raiz_n2 = False
        if not os.path.exists(m):
            m = os.path.join(carpeta_mascaras, r['parche'] + '.tif')          # nivel 2 clásico: sin subcarpeta
            es_raiz_n2 = True
            if not os.path.exists(m) or r['proyecto'] != 'datos_ingemmet':
                continue
        peso_p = None
        if solo_con_positivos and not es_raiz_n2:
            if pesos_tiers is not None:
                if (r['proyecto'], r['parche']) not in pesos_tiers:
                    continue
                if tiers is not None and nombres_tier.get((r['proyecto'], r['parche'])) not in tiers:
                    continue
                peso_p = pesos_tiers[(r['proyecto'], r['parche'])]
            elif permitidos_n2 is not None:
                if (r['proyecto'], r['tesela']) not in permitidos_n2:
                    continue
        filas.append(dict(r, _mascara=m, _imagen=os.path.join(carpeta_parches, r['archivo']),
                          _es_raiz_n2=es_raiz_n2, _peso_tier=peso_p,
                          _tier=nombres_tier.get((r['proyecto'], r['parche']))))
    X, Y, V, usadas = [], [], [], []
    for r in filas:
        with rasterio.open(r['_mascara']) as d:
            mk = d.read(1)
        if solo_con_positivos and (pesos_tiers is None and permitidos_n2 is None or r.get('_es_raiz_n2')) and not (mk == 1).any():
            continue
        with rasterio.open(r['_imagen']) as d:
            x = d.read().astype(np.float32)
        if x.shape != (6, 128, 128) or mk.shape != (128, 128):
            raise SystemExit(f"{r['parche']}: tamaño inesperado {x.shape} / {mk.shape}")
        valido = (x[1] > 0) & (mk != 255)
        t = recetas.desde_l2a({b: x[k] for k, b in enumerate(BANDAS)},
                              {'pendiente': x[4], 'elevacion': x[5]}, valido, receta, escala=escala)
        X.append(t); Y.append((mk == 1).astype(np.float32)); V.append(valido.astype(np.float32)); usadas.append(r)
    if not X:
        z = np.zeros((0, len(receta['canales']), 128, 128), np.float32)
        return z, np.zeros((0, 128, 128), np.float32), np.zeros((0, 128, 128), np.float32), []
    return np.stack(X), np.stack(Y), np.stack(V), usadas


def cargar_l4spe(ruta, receta, conjuntos):
    """Parches del profesor (L4SPe) con máscara manual, de empaquetar_l4spe.py.
    Las bandas vienen en niveles digitales y se llevan a unidades de Landslide4Sense con la escala
    guardada en el archivo (media del entrenamiento), igual que se hace con La Convención.
    La pendiente viene en radianes y la elevación en metros. Devuelve X, Y, V y las filas."""
    from nucleo import config as cfg, recetas
    with np.load(ruta, allow_pickle=False) as z:
        # se lee cada arreglo UNA vez: d['x'][i] dentro del bucle descomprime el archivo entero en cada vuelta
        d = {k: z[k] for k in ('x', 'y', 'punto', 'zona', 'conjunto', 'escala_json')}
    escala = json.loads(str(d['escala_json']))
    k_terreno = dict(cfg.ESCALA_TERRENO_L4S, **receta.get('escala_terreno', {}))
    sel = [i for i, c in enumerate(d['conjunto']) if str(c) in conjuntos]
    X, Y, V, filas = [], [], [], []
    for i in sel:
        x = d['x'][i]
        valido = np.isfinite(x).all(axis=0)
        x = np.nan_to_num(x, nan=0.0)
        bandas = {b: x[k] for k, b in enumerate(BANDAS)}
        terreno = {'pendiente': np.degrees(x[4]).astype(np.float32), 'elevacion': x[5]}
        X.append(recetas.desde_l2a(bandas, terreno, valido, receta, escala=escala))
        Y.append(d['y'][i].astype(np.float32)); V.append(valido.astype(np.float32))
        filas.append({'proyecto': 'l4spe', 'parche': str(d['punto'][i]), 'tesela': str(d['punto'][i]),
                      'zona': str(d['zona'][i]), 'conjunto': str(d['conjunto'][i])})
    if not X:
        z = np.zeros((0, len(receta['canales']), 128, 128), np.float32)
        return z, np.zeros((0, 128, 128), np.float32), np.zeros((0, 128, 128), np.float32), []
    return np.stack(X), np.stack(Y), np.stack(V), filas


# ------------------------------------------------------------------- medición
def barrido(prob, Y, V):
    c = Conteo(UMBRALES)
    for i in range(0, len(prob), 64):
        c.sumar(prob[i:i + 64], Y[i:i + 64], V[i:i + 64])
    return [c.metricas(i) for i in range(len(UMBRALES))]


def en_umbral(lista, u):
    return min(lista, key=lambda r: abs(r['umbral'] - u))


def resumen_local(prob, Y, V, filas, u):
    """Métricas locales a un umbral: píxeles, cicatrices tocadas y falsas alarmas en negativos."""
    m = en_umbral(barrido(prob, Y, V), u)
    objs = objetos(Y * V)
    tocadas = sum(bool(((prob[i][sl] >= u) & (V[i][sl] > 0) & mm).any()) for i, sl, mm, _ in objs)
    teselas = {}
    for i, r in enumerate(filas):
        t = teselas.setdefault((r['proyecto'], r['tesela']), [0.0, 0.0, 0.0])
        t[0] += float((Y[i] * V[i]).sum()); t[1] += float(((prob[i] >= u) & (V[i] > 0)).sum()); t[2] += float(V[i].sum())
    neg = [t for t in teselas.values() if t[0] == 0]
    px_neg, fa_neg = sum(t[2] for t in neg), sum(t[1] for t in neg)
    return dict(m, cicatrices=len(objs), cicatrices_tocadas=tocadas, teselas=len(teselas),
                teselas_negativas=len(neg),
                falsas_alarmas_ha_por_km2_en_negativos=round(fa_neg / max(px_neg, 1) * 100, 3))


def regla_suelo_pendiente(X, receta, pend_min=10.0, b4_min_l4s=1.29):
    """Comparador sin aprendizaje (regla espectral y de pendiente, análoga a Devara 2024):
    marca 1,0 donde hay suelo desnudo (GRVI <= 0 y B4 >= 1,29 en unidades L4S, ~600 DN)
    sobre ladera (pendiente >= pend_min grados)."""
    if len(X) == 0:
        return np.zeros((0, 128, 128), np.float32)
    can = receta['canales']
    pc = receta['normalizacion']['por_canal']
    crudo = lambda c: X[:, can.index(c)] * float(pc[c]['desv']) + float(pc[c]['media'])
    g, r, p = crudo('B03'), crudo('B04'), crudo('pendiente') * 20.292
    grvi = (g - r) / (np.abs(g + r) + 1e-3)
    return ((grvi <= 0.0) & (r >= b4_min_l4s) & (p >= pend_min)).astype(np.float32)


def bootstrap_zonas(prob_a, u_a, prob_b, u_b, Y, V, filas, n=2000, semilla=7):
    """Intervalo del 95 % de IoU(b) − IoU(a) remuestreando zonas completas."""
    zonas = sorted({r['zona'] for r in filas})
    if len(zonas) < 2:
        return None
    cuenta = {z: np.zeros((2, 3)) for z in zonas}
    for i, r in enumerate(filas):
        ok, t = V[i] > 0, Y[i] > 0
        for k, (p, u) in enumerate(((prob_a, u_a), (prob_b, u_b))):
            mm = p[i] >= u
            cuenta[r['zona']][k] += [np.sum(mm & t & ok), np.sum(mm & ~t & ok), np.sum(~mm & t & ok)]
    C = np.stack([cuenta[z] for z in zonas])
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(n):
        s = C[rng.integers(0, len(zonas), len(zonas))].sum(0)
        iou = s[:, 0] / np.maximum(s.sum(1), 1)
        difs.append(iou[1] - iou[0])
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {'zonas': len(zonas), 'ic95': [round(float(lo), 4), round(float(hi), 4)],
            'gana_pct': round(float(np.mean(np.array(difs) > 0)) * 100, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--l4s', help='carpeta de Landslide4Sense (no hace falta con --sin-l4s)')
    ap.add_argument('--desde', required=True, help='punto de control .pt del modelo base (entrenar.py)')
    ap.add_argument('--stats', required=True, help='stats_norm_<firma>.json del modelo base')
    ap.add_argument('--parches', required=True, help='carpeta parches_128 con indice_parches.csv')
    ap.add_argument('--mascaras', required=True, help='mascaras_nivel1/parches_128')
    ap.add_argument('--debiles', help='(opcional) mascaras_provisionales/nivel2/parches_128, solo para entrenar')
    ap.add_argument('--peso-debil', dest='peso_debil', type=float, default=0.3)
    ap.add_argument('--escala', help='escala_laconvencion.json; por defecto la del servidor (nucleo.config)')
    ap.add_argument('--salida', required=True)
    ap.add_argument('--epocas', type=int, default=40)
    ap.add_argument('--lote', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5, help='diez veces menor que en el entrenamiento original')
    ap.add_argument('--weight-decay', dest='wd', type=float, default=1e-4)
    ap.add_argument('--frac-local', dest='frac_local', type=float, default=0.5, help='parte de cada lote que es local')
    ap.add_argument('--solo-local', dest='solo_local', action='store_true',
                    help='entrena SOLO con parches de La Convención (lote 100 %% local) y no exige conservar el IoU en '
                         'Landslide4Sense; L4S se sigue midiendo y se informa. Más riesgo de olvidar lo aprendido')
    ap.add_argument('--pasos', type=int, default=0, help='lotes por época (0 = según la cantidad de parches locales)')
    ap.add_argument('--ohem', type=float, default=None, help='por defecto, el del modelo base')
    ap.add_argument('--congelar-encoder', dest='congelar', action='store_true',
                    help='entrena solo el decodificador (más seguro con muy pocos datos)')
    ap.add_argument('--bn-libre', dest='bn_libre', action='store_true',
                    help='deja que BatchNorm recalcule sus medias (solo con cientos de parches locales)')
    ap.add_argument('--tolerancia-l4s', dest='tol', type=float, default=0.03,
                    help='IoU que se acepta perder en la validación de Landslide4Sense')
    ap.add_argument('--min-cicatrices', dest='min_cic', type=int, default=10,
                    help='cicatrices mínimas en validación local para elegir ahí el umbral')
    ap.add_argument('--umbral-base', dest='umbral_base', type=float, default=0.45,
                    help='umbral de la receta del modelo base, para comparar a umbral fijo')
    ap.add_argument('--geomorfologia', action='store_true',
                    help='añade CURVATURA y TPI calculados desde el DEM como canales extra (Vía 3)')
    ap.add_argument('--tiers', help='Tiers de nivel 2 que entran a entrenar, separados por coma (p. ej. oro,plata,hard_negative). '
                    'Por defecto todos menos evaluacion_ciega, que se reserva para medir')
    ap.add_argument('--val-provisional', dest='val_provisional', action='store_true',
                    help='si no hay validación de nivel 1, valida con parches de nivel 2 de zonas de validación '
                         '(--tiers-val). Es una medida provisional: las máscaras son propuestas, no verificadas')
    ap.add_argument('--tiers-val', dest='tiers_val', default='oro,plata,bronce_negativo',
                    help='Tiers de nivel 2 usados en la validación provisional')
    ap.add_argument('--paciencia', type=int, default=0,
                    help='corta si pasan tantas épocas seguidas pegando por debajo del piso de L4S (0 = no cortar)')
    ap.add_argument('--perdida-por-grupo', dest='perdida_grupo', action=argparse.BooleanOptionalAction, default=True,
                    help='calcula la pérdida (y el OHEM) por separado para la parte local y la de L4S, y las promedia '
                         'según la fracción del lote. Evita que los píxeles locales más difíciles dominen todo el lote')
    ap.add_argument('--l4spe', help='l4spe_profesor.npz (empaquetar_l4spe.py): parches con máscara manual del profesor, '
                    'de otras zonas del Perú. Entran a entrenar, a validar (elegir época y umbral) y a la prueba final')
    ap.add_argument('--frac-l4spe', dest='frac_l4spe', type=float, default=0.3,
                    help='parte de cada lote que sale de L4SPe (solo con --l4spe)')
    ap.add_argument('--peso-l4spe', dest='peso_l4spe', type=float, default=1.0,
                    help='peso en la pérdida de los píxeles de L4SPe (1,0 = como Landslide4Sense)')
    ap.add_argument('--max-l4s-entrenamiento', dest='max_l4s', type=int, default=0,
                    help='usa solo N parches de entrenamiento de L4S como ancla, al azar (0 = todos). Ahorra memoria; '
                         'la validación y la prueba de L4S quedan completas')
    ap.add_argument('--sin-local', dest='sin_local', action='store_true',
                    help='no entrena con parches de La Convención (solo L4SPe y L4S); la validación local se sigue midiendo')
    ap.add_argument('--interpolar', action='store_true',
                    help='al final prueba mezclas de pesos base + alfa·(ajustado − base) (WiSE-FT, Wortsman et al. 2022) '
                         'con alfa 0,25 / 0,5 / 0,75 y se queda con la mejor que respete el piso de L4S. Suele '
                         'conservar casi todo lo ganado perdiendo mucho menos en Landslide4Sense')
    ap.add_argument('--sin-l4s', dest='sin_l4s', action='store_true',
                    help='no usa Landslide4Sense en nada: ni en el lote, ni como piso, ni en la medición. Solo datos del '
                         'Perú (La Convención y L4SPe). El modelo de partida sí viene de L4S (es el que ya se entrenó)')
    ap.add_argument('--semilla', type=int, default=42)
    ap.add_argument('--limite', type=int, help='solo para probar: parches de L4S por conjunto')
    a = ap.parse_args()
    if not a.sin_l4s and not a.l4s:
        ap.error('falta --l4s (o usa --sin-l4s)')

    import torch
    import segmentation_models_pytorch as smp
    from nucleo import config as cfg_srv

    torch.manual_seed(a.semilla); np.random.seed(a.semilla)
    rng = np.random.default_rng(a.semilla)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = disp == 'cuda'
    os.makedirs(a.salida, exist_ok=True)
    t0 = time.time()

    ck0 = torch.load(a.desde, map_location='cpu', weights_only=False)
    cfg0 = ck0.get('config') or {}
    arq = ck0.get('arquitectura', 'unet')
    canales_base = list(cfg0.get('canales') or ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'])
    if canales_base != ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion']:
        raise SystemExit(f'el modelo base usa los canales {canales_base}; los parches locales traen B1 a B4, pendiente y DEM')
    canales = list(canales_base)
    stats = json.load(open(a.stats, encoding='utf-8'))
    if a.geomorfologia:
        canales.extend(['CURVATURA', 'TPI'])
        stats['CURVATURA'] = {'media': 0.0, 'desv': 1.0}
        stats['TPI'] = {'media': 0.0, 'desv': 1.0}
    receta = receta_de(canales, stats)
    escala = json.load(open(a.escala, encoding='utf-8'))['media_dn'] if a.escala else cfg_srv.ESCALA_L4S
    if not escala:
        raise SystemExit('no hay escala de bandas: pasa --escala escala_laconvencion.json')
    ohem = a.ohem if a.ohem is not None else float((cfg0.get('v4') or {}).get('ohem', 0.0))
    indices = bool((cfg0.get('v4') or {}).get('indices_verdor')) or any(k.startswith('base.') for k in ck0['modelo'])
    if a.solo_local:
        print('MODO SOLO LOCAL: lotes 100 % de La Convención, sin ancla de Landslide4Sense ni piso de IoU en L4S.')
    print(f"base: {os.path.basename(a.desde)} | {arq} | época {ck0.get('epoca')} | IoU val {ck0.get('iou_val')} | "
          f"índices de verdor {'sí' if indices else 'no'} | geomorfología {'sí' if a.geomorfologia else 'no'} | ohem {ohem} | {disp}")

    # ---------------------------------------------------------------- datos
    tiers_ent = {t.strip().lower() for t in a.tiers.split(',')} if a.tiers else None
    tiers_val = {t.strip().lower() for t in a.tiers_val.split(',')}
    if tiers_ent is None:
        tiers_ent = {'oro', 'plata', 'bronce_positivo', 'bronce_negativo', 'hard_negative'}   # todos menos evaluacion_ciega
    if tiers_ent & {'evaluacion_ciega'} and a.val_provisional and 'evaluacion_ciega' in tiers_val:
        raise SystemExit('evaluacion_ciega no puede estar a la vez en entrenamiento y en validación')
    if a.sin_l4s:
        tr = va = te = []
    else:
        tr, va, te = (pares(a.l4s, s, a.limite) for s in ('train', 'validation', 'test'))
    if a.max_l4s and len(tr) > a.max_l4s:
        tr = [tr[i] for i in sorted(np.random.default_rng(a.semilla).choice(len(tr), a.max_l4s, replace=False))]
    Xtr, Ytr, Vtr = cargar(tr, receta)
    Xva, Yva, Vva = cargar(va, receta)
    Xte, Yte, Vte = cargar(te, receta)
    Xl, Yl, Vl, Fl = cargar_locales(a.parches, a.mascaras, receta, escala, {'entrenamiento'})
    Xv, Yv, Vv, Fv = cargar_locales(a.parches, a.mascaras, receta, escala, {'validacion'})
    origen_val = 'nivel 1' if len(Xv) else 'ninguna'
    if not len(Xv) and a.debiles and a.val_provisional:
        Xv, Yv, Vv, Fv = cargar_locales(a.parches, a.debiles, receta, escala, {'validacion'}, solo_con_positivos=True,
                                        tiers=tiers_val)
        origen_val = 'nivel 2 provisional (' + ', '.join(sorted(tiers_val)) + ')' if len(Xv) else 'ninguna'
    Wl = np.ones(len(Xl), np.float32)
    n_debiles = 0
    if a.debiles and not a.sin_local:
        con_n1 = {r['parche'] for r in Fl}
        Xd, Yd, Vd, Fd = cargar_locales(a.parches, a.debiles, receta, escala, {'entrenamiento'}, solo_con_positivos=True,
                                        tiers=tiers_ent)
        quedan = [i for i, r in enumerate(Fd) if r['parche'] not in con_n1]
        if quedan:
            Xl = np.concatenate([Xl, Xd[quedan]]); Yl = np.concatenate([Yl, Yd[quedan]]); Vl = np.concatenate([Vl, Vd[quedan]])
            pesos_q = np.array([Fd[i].get('_peso_tier') or a.peso_debil for i in quedan], dtype=np.float32)
            Wl = np.concatenate([Wl, pesos_q])
            Fl = Fl + [Fd[i] for i in quedan]
            n_debiles = len(quedan)
    Xp = Yp = Vp = Xpv = Ypv = Vpv = Xpt = Ypt = Vpt = None
    Fp = Fpv = Fpt = []
    if a.l4spe:
        Xp, Yp, Vp, Fp = cargar_l4spe(a.l4spe, receta, {'entrenamiento'})
        Xpv, Ypv, Vpv, Fpv = cargar_l4spe(a.l4spe, receta, {'validacion'})
        Xpt, Ypt, Vpt, Fpt = cargar_l4spe(a.l4spe, receta, {'prueba'})
        if len(Xp) < 4 or len(Xpv) < 1:
            raise SystemExit(f'L4SPe: {len(Xp)} parches de entrenamiento y {len(Xpv)} de validación; revisa {a.l4spe}')
        print(f'L4SPe (profesor): entrenamiento {len(Xp)} parches / {len({r["zona"] for r in Fp})} zonas, '
              f'validación {len(Xpv)} / {len({r["zona"] for r in Fpv})}, prueba {len(Xpt)} / {len({r["zona"] for r in Fpt})}')
    zonas_tr, zonas_va = {r['zona'] for r in Fl}, {r['zona'] for r in Fv}
    if zonas_tr & zonas_va:
        raise SystemExit(f'zonas en entrenamiento y validación a la vez: {sorted(zonas_tr & zonas_va)}')
    if len(Xl) < 4 and not a.sin_local:
        raise SystemExit(f'solo hay {len(Xl)} parches locales de entrenamiento con máscara: faltan teselas marcadas')
    cic_tr, cic_va = len(objetos(Yl * Vl)), len(objetos(Yv * Vv))
    print(f'L4S: entrenamiento {len(Xtr)}, validación {len(Xva)}, prueba {len(Xte)}')
    print(f'local entrenamiento: {len(Xl)} parches ({n_debiles} de nivel 2, con el peso de su Tier; {a.peso_debil} si no hay catálogo), {len(zonas_tr)} zonas, '
          f'{cic_tr} cicatrices, {float((Yl * Vl).sum()) * 0.01:.1f} ha')
    print(f'local validación  : {len(Xv)} parches, {len(zonas_va)} zonas, {cic_va} cicatrices, '
          f'{float((Yv * Vv).sum()) * 0.01:.1f} ha')
    frac = lambda Y_, V_: float((Y_ * V_).sum()) / max(float(V_.sum()), 1.0) * 100
    print(f'píxeles de deslizamiento: L4S entrenamiento {frac(Ytr, Vtr):.2f} %, local entrenamiento {frac(Yl, Vl):.2f} %'
          f'. Validación local: {origen_val}')
    hay_pe = Xp is not None
    cic_pe_va = len(objetos(Ypv * Vpv)) if hay_pe else 0
    hay_pe_val = hay_pe and cic_pe_va > 0
    if hay_pe:
        print(f'L4SPe validación: {cic_pe_va} cicatrices. Píxeles de deslizamiento: L4SPe entrenamiento {frac(Yp, Vp):.2f} %')
    hay_val = len(Xv) > 0
    hay_pos_val = cic_va > 0
    if not hay_val:
        print('AVISO: no hay validación local marcada. Se entrena, pero la época se elige solo con Landslide4Sense '
              'y NO se puede afirmar ninguna mejora en La Convención.')
    elif origen_val.startswith('nivel 2'):
        print('AVISO: la validación local usa máscaras de nivel 2 (propuestas sin verificar). Sirve para elegir época y '
              'detectar si el ajuste ayuda o daña, no como cifra final de tesis.')
    if hay_val and cic_va < a.min_cic:
        print(f'AVISO: la validación local tiene {cic_va} cicatrices (menos de {a.min_cic}): resultado exploratorio.')

    # ---------------------------------------------------------------- modelo
    clase = {'unet': smp.Unet, 'deeplabv3plus': smp.DeepLabV3Plus}[arq]
    modelo = clase(encoder_name=cfg0.get('encoder', 'resnet34'), encoder_weights=None,
                   in_channels=len(canales) + (2 if indices else 0), classes=1)
    if indices:
        modelo = modelo_con_indices(modelo, canales)
    sd = dict(ck0['modelo'])
    if a.geomorfologia:
        clave_conv = 'base.encoder.conv1.weight' if indices else 'encoder.conv1.weight'
        if clave_conv in sd:
            w_old = sd[clave_conv]
            w_new = torch.zeros((w_old.shape[0], w_old.shape[1] + 2, w_old.shape[2], w_old.shape[3]), dtype=w_old.dtype)
            w_new[:, :6] = w_old[:, :6]
            if w_old.shape[1] > 6:
                w_new[:, 8:] = w_old[:, 6:]
            sd[clave_conv] = w_new
    modelo.load_state_dict(sd, strict=True)
    modelo = modelo.to(disp)
    nucleo_red = modelo.base if indices else modelo
    if a.congelar:
        for q in nucleo_red.encoder.parameters():
            q.requires_grad_(False)

    def medir(m):
        out = {'l4s': None, 'l4s_barrido': None}
        if not a.sin_l4s:
            b_l4s = barrido(predecir(m, Xva, disp), Yva, Vva)
            out = {'l4s': max(b_l4s, key=lambda r: r['iou']), 'l4s_barrido': b_l4s}
        if hay_val:
            pl = predecir(m, Xv, disp)
            b_loc = barrido(pl, Yv, Vv)
            out['local_barrido'] = b_loc
            out['local'] = max(b_loc, key=lambda r: r['iou']) if hay_pos_val else min(b_loc, key=lambda r: (r['fp'], r['umbral']))
        if hay_pe_val:
            b_pe = barrido(predecir(m, Xpv, disp), Ypv, Vpv)
            out['pe_barrido'] = b_pe
            out['pe'] = max(b_pe, key=lambda r: r['iou'])
        return out

    def puntaje(m):
        # promedio de los IoU de validación que existen: L4SPe (manual, otras zonas del Perú) y La Convención
        comp = ([m['pe']['iou']] if hay_pe_val else []) + ([m['local']['iou']] if hay_val and hay_pos_val else [])
        if comp:
            return float(np.mean(comp))
        if not hay_val:
            if a.sin_l4s:
                raise SystemExit('--sin-l4s necesita validación de L4SPe o local para elegir la época')
            return m['l4s']['iou']
        fp05 = en_umbral(m['local_barrido'], 0.5)
        return -fp05['fp'] / max(fp05['fp'] + fp05['tn'], 1)

    m0 = medir(modelo)
    # con --solo-local el IoU de Landslide4Sense deja de ser condición para aceptar una época
    piso_l4s = float('-inf') if a.solo_local or a.sin_l4s else m0['l4s']['iou'] - a.tol
    iou4 = lambda m: m['l4s']['iou'] if m['l4s'] else None
    print(f"época  0 (modelo base)" + (f" | IoU val L4S {m0['l4s']['iou']:.4f} (umbral {m0['l4s']['umbral']})" if m0['l4s'] else '')
          + (f" | IoU val local {m0['local']['iou']:.4f} (umbral {m0['local']['umbral']})" if hay_val and hay_pos_val else '')
          + (f" | IoU val L4SPe {m0['pe']['iou']:.4f} (umbral {m0['pe']['umbral']})" if hay_pe_val else ''))
    config = dict(cfg0, ajuste_fino={'desde': os.path.basename(a.desde), 'firma_base': ck0.get('firma'),
                                     'lr': a.lr, 'lote': a.lote, 'sin_local': a.sin_local, 'sin_l4s': a.sin_l4s, 'interpolar': a.interpolar, 'frac_local': 1.0 if a.solo_local else a.frac_local,
                                     'solo_local': a.solo_local, 'epocas_max': a.epocas,
                                     'ohem': ohem, 'pasos': a.pasos, 'l4spe': os.path.basename(a.l4spe) if a.l4spe else None,
                                     'frac_l4spe': a.frac_l4spe if a.l4spe else None, 'peso_l4spe': a.peso_l4spe if a.l4spe else None,
                                     'tiers_entrenamiento': sorted(tiers_ent) if a.debiles else None,
                                     'validacion_local': origen_val, 'perdida_por_grupo': a.perdida_grupo,
                                     'congelar_encoder': a.congelar, 'bn_libre': a.bn_libre, 'tolerancia_l4s': a.tol,
                                     'semilla': a.semilla, 'parches_locales': len(Xl), 'parches_debiles': n_debiles,
                                     'peso_debil': a.peso_debil if n_debiles else None,
                                     'zonas_entrenamiento': sorted(zonas_tr), 'zonas_validacion': sorted(zonas_va)})
    firma = hashlib.sha1(json.dumps(dict(config, arquitectura=arq), sort_keys=True).encode()).hexdigest()[:12]
    base = f'{arq}_{ETIQUETA_CANALES}_{firma}'
    ruta_ckpt = os.path.join(a.salida, f'mejor_{base}.pt')

    params = [q for q in modelo.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: factor_lr(e, a.epocas, 2, 0.05))
    escalador = torch.amp.GradScaler('cuda', enabled=amp)
    n_pe = max(1, int(round(a.lote * a.frac_l4spe))) if hay_pe else 0
    if a.sin_l4s and not a.sin_local:
        n_loc = a.lote - n_pe
    elif a.sin_local:
        if not hay_pe:
            raise SystemExit('--sin-local necesita --l4spe')
        n_loc = 0
    elif a.solo_local:
        n_loc = a.lote - n_pe
    else:
        n_loc = max(1, min(a.lote - n_pe - 1, int(round(a.lote * a.frac_local))))
    n_l4s = 0 if a.sin_l4s else a.lote - n_loc - n_pe
    if a.sin_l4s and a.sin_local:
        n_pe = a.lote
    print(f'lote de {a.lote}: {n_loc} de La Convención, {n_pe} de L4SPe, {n_l4s} de Landslide4Sense')
    pasos = a.pasos or max(20, math.ceil(len(Xl) / n_loc if n_loc else len(Xp) / max(n_pe, 1)))
    historial = [{'epoca': 0, 'perdida': None, 'iou_l4s': iou4(m0),
                  'iou_local': m0['local']['iou'] if hay_val else None,
                  'iou_l4spe': m0['pe']['iou'] if hay_pe_val else None, 'puntaje': puntaje(m0), 'admisible': True}]
    mejor = {'epoca': 0, 'puntaje': puntaje(m0), 'medida': m0, 'estado': None, 'alfa': 1.0}
    mejor_libre = {'epoca': 0, 'puntaje': puntaje(m0), 'estado': None}
    cola_l4s = rng.permutation(max(len(Xtr), 1)); pos = 0
    seguidas_malas, ep_hechas = 0, 0
    for ep in range(1, a.epocas + 1):
        ep_hechas = ep
        modelo.train()
        if not a.bn_libre:
            # las medias y varianzas de BatchNorm se quedan como en el modelo base: con pocos
            # parches locales por lote se desajustan y el modelo empeora aunque no aprenda nada
            for capa in modelo.modules():
                if isinstance(capa, torch.nn.modules.batchnorm._BatchNorm):
                    capa.eval()
        perdidas = []; te0 = time.time()
        for _ in range(pasos):
            xs, ys, vs = [], [], []
            if n_loc:
                jl = rng.integers(0, len(Xl), n_loc)
                xs.append(Xl[jl]); ys.append(Yl[jl]); vs.append(Vl[jl] * Wl[jl][:, None, None])
            if n_pe:
                jp = rng.integers(0, len(Xp), n_pe)
                xs.append(Xp[jp]); ys.append(Yp[jp]); vs.append(Vp[jp] * a.peso_l4spe)
            if n_l4s:
                if pos + n_l4s > len(cola_l4s):
                    cola_l4s = rng.permutation(len(Xtr)); pos = 0
                jg = cola_l4s[pos:pos + n_l4s]; pos += n_l4s
                xs.append(Xtr[jg]); ys.append(Ytr[jg]); vs.append(Vtr[jg])
            x, y, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
            k = int(rng.integers(0, 4)); espejo = bool(rng.integers(0, 2))
            x, y, v = (np.rot90(mm, k, axes=(-2, -1)) for mm in (x, y, v))
            if espejo:
                x, y, v = x[..., ::-1], y[..., ::-1], v[..., ::-1]
            x, y, v = (torch.from_numpy(np.ascontiguousarray(mm)).to(disp) for mm in (x, y, v))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=amp):
                logits = modelo(x)[:, 0]
            if a.perdida_grupo and (n_pe or n_l4s):
                # cada grupo con su propia pérdida: si no, el OHEM elige los píxeles más difíciles de TODO el lote
                # y los locales (máscaras provisionales) acaparan el gradiente aunque el lote sea mitad L4S
                lg = logits.float()
                loss, ini = 0.0, 0
                for n_g in (n_loc, n_pe, n_l4s):
                    if n_g:
                        loss = loss + n_g * perdida_mixta(lg[ini:ini + n_g], y[ini:ini + n_g], v[ini:ini + n_g], 0.5, ohem)
                        ini += n_g
                loss = loss / a.lote
            else:
                loss = perdida_mixta(logits.float(), y, v, 0.5, ohem)
            escalador.scale(loss).backward()
            escalador.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            escalador.step(opt); escalador.update()
            perdidas.append(float(loss.detach()))
        sched.step()
        m = medir(modelo)
        admisible = m['l4s'] is None or m['l4s']['iou'] >= piso_l4s
        p = puntaje(m)
        marca = ''
        if p > mejor_libre['puntaje'] + 1e-6:
            mejor_libre = {'epoca': ep, 'puntaje': p,
                           'estado': {kk: vv.detach().cpu().clone() for kk, vv in modelo.state_dict().items()}}
        if admisible and p > mejor['puntaje'] + 1e-6:
            mejor = {'epoca': ep, 'puntaje': p, 'medida': m,
                     'estado': {kk: vv.detach().cpu().clone() for kk, vv in modelo.state_dict().items()}}
            marca = '  <- mejor'
        historial.append({'epoca': ep, 'perdida': round(float(np.mean(perdidas)), 5), 'iou_l4s': iou4(m),
                          'iou_local': m['local']['iou'] if hay_val else None,
                          'iou_l4spe': m['pe']['iou'] if hay_pe_val else None, 'puntaje': round(float(p), 5),
                          'admisible': admisible})
        print(f"época {ep:2d} | pérdida {np.mean(perdidas):.4f}" + (f" | IoU val L4S {m['l4s']['iou']:.4f}" if m['l4s'] else '')
              + ('' if admisible else ' (pierde demasiado)')
              + (f" | IoU val local {m['local']['iou']:.4f}" if hay_val and hay_pos_val else '')
              + (f" | IoU val L4SPe {m['pe']['iou']:.4f}" if hay_pe_val else '')
              + f" | {time.time() - te0:.0f} s{marca}", flush=True)
        seguidas_malas = 0 if admisible else seguidas_malas + 1
        if a.paciencia and seguidas_malas >= a.paciencia:
            print(f'se corta: {seguidas_malas} épocas seguidas por debajo del piso de L4S ({piso_l4s:.4f}). '
                  'Baja --lr, congela el encoder o revisa las máscaras.')
            break
    interpolaciones = []
    if a.interpolar and mejor_libre['estado'] is not None:
        # WiSE-FT: mezcla lineal de pesos entre el modelo base y la mejor época (sin mirar el piso de L4S)
        print(f"interpolación de pesos con la época {mejor_libre['epoca']} (la de mejor puntaje aunque pierda en L4S):")
        for alfa in (0.25, 0.5, 0.75):
            mezcla = {kk: (sd[kk].float() + alfa * (vv.float() - sd[kk].float())).to(vv.dtype) if vv.is_floating_point() else vv
                      for kk, vv in mejor_libre['estado'].items()}
            modelo.load_state_dict(mezcla)
            m = medir(modelo)
            p = puntaje(m)
            admisible = m['l4s'] is None or m['l4s']['iou'] >= piso_l4s
            marca = ''
            if admisible and p > mejor['puntaje'] + 1e-6:
                mejor = {'epoca': mejor_libre['epoca'], 'puntaje': p, 'medida': m, 'alfa': alfa,
                         'estado': {kk: vv.detach().cpu().clone() for kk, vv in mezcla.items()}}
                marca = '  <- mejor'
            interpolaciones.append({'alfa': alfa, 'iou_l4s': iou4(m), 'iou_local': m['local']['iou'] if hay_val else None,
                                    'iou_l4spe': m['pe']['iou'] if hay_pe_val else None, 'puntaje': round(float(p), 5),
                                    'admisible': admisible})
            print(f"  alfa {alfa:.2f}" + (f" | IoU val L4S {m['l4s']['iou']:.4f}" if m['l4s'] else '') + ('' if admisible else ' (pierde demasiado)')
                  + (f" | IoU val local {m['local']['iou']:.4f}" if hay_val and hay_pos_val else '')
                  + (f" | IoU val L4SPe {m['pe']['iou']:.4f}" if hay_pe_val else '') + marca, flush=True)
    with open(os.path.join(a.salida, f'historial_ajuste_{firma}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0])); w.writeheader(); w.writerows(historial)

    # ---------------------------------------------------------------- medición final
    base_modelo = clase(encoder_name=cfg0.get('encoder', 'resnet34'), encoder_weights=None,
                        in_channels=len(canales) + (2 if indices else 0), classes=1)
    if indices:
        base_modelo = modelo_con_indices(base_modelo, canales)
    base_modelo.load_state_dict(sd, strict=True)
    base_modelo = base_modelo.to(disp)
    mejoro = mejor['epoca'] > 0
    if mejoro:
        modelo.load_state_dict(mejor['estado'])
    else:
        modelo.load_state_dict(sd, strict=True)
    mF = mejor['medida']
    usar_local = hay_val and hay_pos_val and cic_va >= a.min_cic
    usar_pe = hay_pe_val and cic_pe_va >= a.min_cic          # máscara manual: manda sobre la provisional

    def umbral_de(m):
        if usar_pe:
            return m['pe']['umbral']
        if usar_local or m['l4s'] is None:
            return m['local']['umbral'] if hay_val and hay_pos_val else (m['pe']['umbral'] if hay_pe_val else 0.5)
        return m['l4s']['umbral']

    u0, uF = umbral_de(m0), umbral_de(mF)
    res = {'firma': firma, 'arquitectura': arq, 'mejor_epoca': mejor['epoca'], 'mejoro': mejoro,
           'alfa_interpolacion': mejor.get('alfa', 1.0), 'interpolaciones': interpolaciones,
           'umbral_operacion': uF, 'umbral_elegido_en': ('validación de L4SPe (máscara manual del profesor)' if usar_pe else
                                 'validación local' if usar_local else 'validación de Landslide4Sense'),
           'config': config, 'epocas_corridas': ep_hechas, 'epocas_max': a.epocas,
           'validacion_local_origen': origen_val, 'base': {'firma': ck0.get('firma'), 'umbral': u0}}
    # prueba oficial de Landslide4Sense, una vez, con el umbral elegido en SU validación
    res['comparador_sin_aprendizaje'] = {'regla': 'GRVI <= 0, B4 >= 1.29 L4S (~600 DN) y pendiente >= 10°'}
    res['usa_l4s'] = not a.sin_l4s
    if not a.sin_l4s:
        pt0, ptF = predecir(base_modelo, Xte, disp), predecir(modelo, Xte, disp)
        ptR = regla_suelo_pendiente(Xte, receta)
        res['base']['l4s_validacion'] = m0['l4s']
        res['base']['l4s_prueba'] = en_umbral(barrido(pt0, Yte, Vte), m0['l4s']['umbral'])
        res['validacion'] = mF['l4s']
        res['prueba'] = en_umbral(barrido(ptF, Yte, Vte), mF['l4s']['umbral'])
        res['comparador_sin_aprendizaje']['l4s_prueba'] = en_umbral(barrido(ptR, Yte, Vte), 0.5)
        res['barrido_validacion'] = mF['l4s_barrido']
    if hay_val:
        pl0, plF = predecir(base_modelo, Xv, disp), predecir(modelo, Xv, disp)
        plR = regla_suelo_pendiente(Xv, receta)
        res['base']['local_validacion'] = resumen_local(pl0, Yv, Vv, Fv, u0)
        res['local_validacion'] = resumen_local(plF, Yv, Vv, Fv, uF)
        res['comparador_sin_aprendizaje']['local_validacion'] = resumen_local(plR, Yv, Vv, Fv, 0.5)
        res['local_validacion_umbral_receta_base'] = {
            'umbral': a.umbral_base, 'base': resumen_local(pl0, Yv, Vv, Fv, a.umbral_base),
            'ajustado': resumen_local(plF, Yv, Vv, Fv, a.umbral_base)}
        res['local_mejora_iou'] = bootstrap_zonas(pl0, u0, plF, uF, Yv, Vv, Fv) if hay_pos_val else None
        res['local_advertencias'] = [s for s in (
            'la validación local se usó para elegir época y umbral: su IoU es optimista',
            f'solo {cic_va} cicatrices en validación local: resultado exploratorio' if cic_va < a.min_cic else '',
            f'solo {len(zonas_va)} zonas de validación' if len(zonas_va) < 5 else '',
            'validación hecha con máscaras de nivel 2 (propuestas sin verificar): no sirve como cifra final'
            if origen_val.startswith('nivel 2') else '') if s]
    if hay_pe:
        pv0, pvF = predecir(base_modelo, Xpv, disp), predecir(modelo, Xpv, disp)
        pt0_, ptF_ = predecir(base_modelo, Xpt, disp), predecir(modelo, Xpt, disp)
        res['l4spe'] = {
            'nota': 'máscaras manuales del profesor, otras zonas del Perú (no La Convención); entrada de Sentinel-2 L1C',
            'validacion': {'base': resumen_local(pv0, Ypv, Vpv, Fpv, u0), 'ajustado': resumen_local(pvF, Ypv, Vpv, Fpv, uF)},
            'prueba': {'base': resumen_local(pt0_, Ypt, Vpt, Fpt, u0), 'ajustado': resumen_local(ptF_, Ypt, Vpt, Fpt, uF),
                       'regla': resumen_local(regla_suelo_pendiente(Xpt, receta), Ypt, Vpt, Fpt, 0.5)},
            'mejora_iou_prueba': bootstrap_zonas(pt0_, u0, ptF_, uF, Ypt, Vpt, Fpt)}
        res['local_advertencias'] = res.get('local_advertencias', []) + [
            'L4SPe es de otras zonas del Perú y de imágenes L1C: que mejore ahí no garantiza mejora en La Convención (L2A)']
    res['minutos'] = round((time.time() - t0) / 60, 1)

    print('\n' + '=' * 78)
    if not mejoro:
        print('Ninguna época superó al modelo base' + ('' if a.solo_local or a.sin_l4s else ' sin perder en Landslide4Sense') + ': SE QUEDA EL MODELO BASE.')
        print('No se escribe un modelo nuevo. Suele querer decir que faltan teselas marcadas.')
    else:
        torch.save({'modelo': modelo.state_dict(), 'arquitectura': arq, 'epoca': mejor['epoca'],
                    'iou_val': iou4(mF) if mF['l4s'] else mejor['puntaje'], 'firma': firma, 'config': config}, ruta_ckpt)
        json.dump(stats, open(os.path.join(a.salida, f'stats_norm_{firma}.json'), 'w', encoding='utf-8'), indent=1)
        print(f"mejor época {mejor['epoca']}" + (f" mezclada con alfa {mejor['alfa']}" if mejor.get('alfa', 1.0) < 1 else '')
              + f"; umbral {uF} (elegido en {res['umbral_elegido_en']})")
    if hay_pe:
        pr = res['l4spe']['prueba']
        print(f"L4SPe (profesor), prueba de {len(Xpt)} parches de otras zonas del Perú:")
        for nombre in ('regla', 'base', 'ajustado'):
            m_ = pr[nombre]
            print(f"  {nombre:8s}: IoU {m_['iou']:.4f}  F1 {m_['f1']:.4f}  precisión {m_['precision']:.4f}  "
                  f"exhaustividad {m_['recall']:.4f} | cicatrices {m_['cicatrices_tocadas']}/{m_['cicatrices']}")
        bz = res['l4spe']['mejora_iou_prueba']
        if bz:
            print(f"mejora de IoU en L4SPe (prueba), IC 95 % por zonas ({bz['zonas']}): [{bz['ic95'][0]:+.4f}, {bz['ic95'][1]:+.4f}]")
    f4 = lambda r: f"IoU {r['iou']:.4f}  F1 {r['f1']:.4f}  precisión {r['precision']:.4f}  exhaustividad {r['recall']:.4f}"
    if not a.sin_l4s:
        print(f"Landslide4Sense, prueba oficial   regla   : {f4(res['comparador_sin_aprendizaje']['l4s_prueba'])}")
        print(f"                                  base    : {f4(res['base']['l4s_prueba'])}")
        print(f"                                  ajustado: {f4(res['prueba'])}")
    if hay_val:
        rg = res['comparador_sin_aprendizaje']['local_validacion']
        b, j = res['base']['local_validacion'], res['local_validacion']
        print(f"La Convención, validación local   regla   : {f4(rg)}  | cicatrices {rg['cicatrices_tocadas']}/{rg['cicatrices']} "
              f"| falsas alarmas {rg['falsas_alarmas_ha_por_km2_en_negativos']} ha/km² en negativos")
        print(f"                                  base    : {f4(b)}  | cicatrices {b['cicatrices_tocadas']}/{b['cicatrices']} "
              f"| falsas alarmas {b['falsas_alarmas_ha_por_km2_en_negativos']} ha/km² en negativos")
        print(f"                                  ajustado: {f4(j)}  | cicatrices {j['cicatrices_tocadas']}/{j['cicatrices']} "
              f"| falsas alarmas {j['falsas_alarmas_ha_por_km2_en_negativos']} ha/km² en negativos")
        if mejoro and res.get('local_mejora_iou'):
            bz = res['local_mejora_iou']
            print(f"mejora de IoU local, IC 95 % por zonas ({bz['zonas']}): [{bz['ic95'][0]:+.4f}, {bz['ic95'][1]:+.4f}]; "
                  f"gana en el {bz['gana_pct']} % de los remuestreos")
        for s in res['local_advertencias']:
            print('  ojo:', s)
    json.dump(res, open(os.path.join(a.salida, f'resultados_{base}.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('archivos en', a.salida)


if __name__ == '__main__':
    main()
