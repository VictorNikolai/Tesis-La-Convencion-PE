"""
Exportador de parches de 128×128 px para IRIS (ESA-PhiLab / fork gt-csse) y adaptador de salida
hacia `mascaras_nivel1/`.

Permite marcar directamente sobre parches de 128×128 px en IRIS (sin redimensionar, a 10 m por
píxel) a partir de las teselas ya descargadas en `datos_laconvencion`, `datos_indeci`,
`datos_ingemmet` y `datos_azar`, y convertir las máscaras resultantes (`masks/<parche>.png`) al
mismo formato exacto que produce `mascaras_desde_marcas.py` (`mascaras_nivel1/parches_128/`),
listo para `empaquetar_para_colab.py` y `ajuste_fino.py`.

Uso:
  1. Armar el proyecto IRIS de 128×128 (por defecto solo las teselas de ronda y validación):
       python herramientas/iris_128.py exportar --salida iris_128

  2. Abrir en IRIS:
       iris label iris_128/deslizamientos_128.json

  3. Convertir las máscaras marcadas en IRIS a `mascaras_nivel1/`:
       python herramientas/iris_128.py importar --iris iris_128 --salida mascaras_nivel1
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from cortar_parches_128 import LADO, cuadrantes  # noqa: E402

BASES = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']
NOMBRES_8 = [
    'B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)',
    'probabilidad del modelo', 'cajas del modelo (contorno)'
]


def configuracion_iris_128(nombre='deslizamientos-laconvencion-128', lado=128, puerto=5000):
    return {
        'name': nombre,
        'port': puerto,
        'images': {
            'path': {'S2': 'images/{id}/image.tif'},
            'shape': [lado, lado],
            'metadata': 'images/{id}/metadata.json',
        },
        'segmentation': {
            'path': 'masks/{id}.png',
            'mask_encoding': 'integer',
            'mask_area': [0, 0, lado, lado],
            'score': 'f1',
            'unverified_threshold': 1,
            'test_images': None,
        },
        'classes': [
            {
                'name': 'Sin deslizamiento',
                'colour': [255, 255, 255, 0],
                'user_colour': [0, 255, 255, 70],
                'description': 'Fondo confirmado: bosque, pasto, río, camino, cultivo o techo.',
            },
            {
                'name': 'Deslizamiento',
                'colour': [255, 40, 60, 110],
                'description': 'Cicatriz de deslizamiento visible en ladera.',
            },
            {
                'name': 'Dudoso',
                'colour': [255, 200, 0, 90],
                'description': 'No se puede decidir (nube, sombra, cauce sin arranque visible). Se ignora (255).',
            },
        ],
        'views': {
            'Color natural con cajas': {
                'description': 'Cajas amarillas del modelo como guía visual.',
                'type': 'image', 'vmin': 0, 'vmax': 1500,
                'data': ['$S2.B4*(1-$S2.B8)+5000*$S2.B8', '$S2.B3*(1-$S2.B8)+5000*$S2.B8', '$S2.B2*(1-$S2.B8)'],
            },
            'Color natural': {
                'description': 'B4, B3 y B2 de Sentinel-2.',
                'type': 'image', 'vmin': 0, 'vmax': 1500,
                'data': ['$S2.B4', '$S2.B3', '$S2.B2'],
            },
            'Probabilidad del modelo': {
                'description': 'Solo de guía.',
                'type': 'image', 'data': '$S2.B7', 'cmap': 'magma', 'vmin': 0, 'vmax': 1,
            },
            'Pendiente': {
                'description': 'B13 en grados.',
                'type': 'image', 'data': '$S2.B5', 'cmap': 'viridis', 'vmin': 0, 'vmax': 60,
            },
            'Relieve': {
                'description': 'B14 en metros.',
                'type': 'image', 'data': '$S2.B6', 'cmap': 'terrain', 'vmin': 500, 'vmax': 3500,
            },
        },
        'view_groups': {
            'default': ['Color natural con cajas', 'Color natural', 'Pendiente'],
            'probabilidad': ['Color natural', 'Probabilidad del modelo', 'Pendiente'],
            'relieve': ['Color natural', 'Relieve', 'Pendiente'],
        },
    }


def exportar_128(raiz, salida, solo_prioritarios=True, limite=None, puerto=5000):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform as tr_coords
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    part = {}
    ruta_part = os.path.join(raiz, 'particion.csv')
    if os.path.exists(ruta_part):
        with open(ruta_part, encoding='utf-8') as f:
            part = {(r['proyecto'], r['tesela']): r for r in csv.DictReader(f)}

    os.makedirs(os.path.join(salida, 'images'), exist_ok=True)
    os.makedirs(os.path.join(salida, 'masks'), exist_ok=True)
    indice = []
    n_tes = 0
    for proy in BASES:
        for ruta in sorted(glob.glob(os.path.join(raiz, proy, 'images', '*', 'image.tif'))):
            tes = os.path.basename(os.path.dirname(ruta))
            info = part.get((proy, tes), {'zona': '', 'conjunto': 'entrenamiento', 'uso': 'reserva'})
            if info['conjunto'] in ('excluida', 'pendiente'):
                continue
            if solo_prioritarios and not (
                info['uso'].startswith('ronda') or info['uso'] == 'validacion' or info['conjunto'] == 'validacion'
            ):
                continue
            with rasterio.open(ruta) as ds:
                x = ds.read().astype('float32')
                perfil = ds.profile
            meta_orig = {}
            ruta_meta = os.path.join(os.path.dirname(ruta), 'metadata.json')
            if os.path.exists(ruta_meta):
                with open(ruta_meta, encoding='utf-8') as fm:
                    meta_orig = json.load(fm)
            for q, f0, c0 in cuadrantes(x.shape[1], x.shape[2], LADO):
                pid = f'{proy}__{tes}_{q}'
                carpeta_p = os.path.join(salida, 'images', pid)
                os.makedirs(carpeta_p, exist_ok=True)
                tr = win_transform(Window(c0, f0, LADO, LADO), perfil['transform'])
                p = dict(
                    driver='GTiff', height=LADO, width=LADO, count=x.shape[0],
                    dtype='float32', crs=perfil['crs'], transform=tr, compress='deflate'
                )
                trozo = x[:, f0:f0 + LADO, c0:c0 + LADO]
                with MemoryFile() as mem:
                    with mem.open(**p) as d:
                        d.write(trozo)
                        d.descriptions = tuple(NOMBRES_8[:x.shape[0]])
                    with open(os.path.join(carpeta_p, 'image.tif'), 'wb') as fo:
                        fo.write(mem.read())
                cx, cy = tr * (LADO / 2, LADO / 2)
                lon, lat = tr_coords(perfil['crs'], 'EPSG:4326', [cx], [cy])
                meta_p = dict(
                    meta_orig, location=[round(lon[0], 5), round(lat[0], 5)],
                    proyecto=proy, tesela=tes, cuadrante=q, parche=f'{tes}_{q}',
                    zona=info['zona'], conjunto=info['conjunto'], uso=info['uso']
                )
                with open(os.path.join(carpeta_p, 'metadata.json'), 'w', encoding='utf-8') as fm:
                    json.dump(meta_p, fm, indent=1, ensure_ascii=False)
                indice.append({
                    'id_iris': pid, 'parche': f'{tes}_{q}', 'proyecto': proy, 'tesela': tes,
                    'cuadrante': q, 'zona': info['zona'], 'conjunto': info['conjunto'], 'uso': info['uso']
                })
            n_tes += 1
            if limite and n_tes >= limite:
                break
        if limite and n_tes >= limite:
            break

    with open(os.path.join(salida, 'indice_iris_128.csv'), 'w', newline='', encoding='utf-8') as f:
        if indice:
            w = csv.DictWriter(f, fieldnames=list(indice[0].keys()))
            w.writeheader()
            w.writerows(indice)
    cfg = configuracion_iris_128(lado=LADO, puerto=puerto)
    with open(os.path.join(salida, 'deslizamientos_128.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    print(f'{len(indice)} parches de 128×128 ({n_tes} teselas) en {os.path.abspath(salida)}')
    return len(indice)


def importar_128(carpeta_iris, salida_nivel1, quien='IRIS'):
    """Convierte las máscaras PNG de IRIS (128×128) al formato de `mascaras_nivel1/`.
    Codificación de IRIS:
      - Si los valores están en {1, 2, 3} (clases 1-based de IRIS): 1 -> 0 (fondo), 2 -> 1 (deslizamiento), 3 -> 255 (dudoso), 0 -> 255 (sin pintar).
      - Si los valores están en {0, 1, 2} (0-based): 0 -> 0 (fondo), 1 -> 1 (deslizamiento), 2 -> 255 (dudoso).
    Respeta la regla de nube blanca (B2 > 1000 -> 255 salvo donde la persona marcó deslizamiento = 1)."""
    from PIL import Image
    import rasterio
    from rasterio.io import MemoryFile
    from scipy import ndimage

    ruta_ind = os.path.join(carpeta_iris, 'indice_iris_128.csv')
    if not os.path.exists(ruta_ind):
        raise SystemExit(f'Falta {ruta_ind}: corre primero iris_128.py exportar')
    with open(ruta_ind, encoding='utf-8') as f:
        indice = list(csv.DictReader(f))

    por_tesela = collections.defaultdict(list)
    n_parches = 0
    for r in indice:
        pid = r['id_iris']
        ruta_png = os.path.join(carpeta_iris, 'masks', f'{pid}.png')
        if not os.path.exists(ruta_png):
            continue
        raw = np.asarray(Image.open(ruta_png))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        ruta_img = os.path.join(carpeta_iris, 'images', pid, 'image.tif')
        with rasterio.open(ruta_img) as ds:
            b2 = ds.read(2)
            tr, crs, forma = ds.transform, ds.crs, ds.shape
        # Determinar codificación (1-based de IRIS cuando se pintó con clases 1..3 vs 0-based 0..2)
        vals = set(np.unique(raw).tolist())
        if 3 in vals or (vals <= {1, 2, 3} and 0 not in vals):
            des = raw == 2
            dud = (raw == 3) | (raw == 0)
        else:
            des = raw == 1
            dud = (raw == 2) | (raw == 255)
        sin_dato = (b2 <= 0) | ((b2 > 1000) & ~des)
        mask = np.zeros(forma, dtype='uint8')
        mask[des] = 1
        mask[dud | sin_dato] = 255

        proy, parche = r['proyecto'], r['parche']
        dest_dir = os.path.join(salida_nivel1, 'parches_128', proy)
        os.makedirs(dest_dir, exist_ok=True)
        p8 = dict(driver='GTiff', height=forma[0], width=forma[1], count=1, dtype='uint8',
                  crs=crs, transform=tr, compress='deflate')
        with MemoryFile() as mem:
            with mem.open(**p8) as d:
                d.write(mask, 1)
                d.update_tags(nivel='1', valores='0 fondo, 1 deslizamiento, 255 ignorar', origen='marcado humano en IRIS 128x128')
            with open(os.path.join(dest_dir, f'{parche}.tif'), 'wb') as fo:
                fo.write(mem.read())
        por_tesela[(proy, r['tesela'], r['zona'], r['conjunto'])].append(mask)
        n_parches += 1

    if not n_parches:
        raise SystemExit(f'No hay máscaras PNG en {os.path.join(carpeta_iris, "masks")}')

    filas = []
    for (proy, tes, zona, conj), ms in sorted(por_tesela.items()):
        cics = sum(int(ndimage.label(m == 1)[1]) for m in ms)
        ha_des = round(sum(float((m == 1).sum()) for m in ms) * 0.01, 2)
        ha_ign = round(sum(float((m == 255).sum()) for m in ms) * 0.01, 2)
        filas.append({
            'proyecto': proy, 'tesela': tes, 'quien': quien,
            'poligonos_deslizamiento': cics, 'poligonos_dudoso': 0,
            'cicatrices': cics, 'ha_deslizamiento': ha_des, 'ha_ignorar': ha_ign,
            'zona': zona, 'conjunto': conj, 'nota': f'{len(ms)} parches importados de IRIS 128'
        })
    os.makedirs(salida_nivel1, exist_ok=True)
    with open(os.path.join(salida_nivel1, 'mascaras_nivel1.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f'{n_parches} parches de nivel 1 importados ({len(filas)} teselas) en {os.path.abspath(salida_nivel1)}')
    return n_parches


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_exp = sub.add_parser('exportar', help='arma el proyecto de IRIS con parches de 128×128')
    p_exp.add_argument('--raiz', default='.')
    p_exp.add_argument('--salida', default='iris_128')
    p_exp.add_argument('--todas', action='store_true', help='incluir también las teselas de reserva')
    p_exp.add_argument('--limite', type=int, help='máximo de teselas a exportar (para probar)')
    p_exp.add_argument('--puerto', type=int, default=5000)

    p_imp = sub.add_parser('importar', help='convierte las máscaras PNG de IRIS a mascaras_nivel1/')
    p_imp.add_argument('--iris', default='iris_128')
    p_imp.add_argument('--salida', default='mascaras_nivel1')
    p_imp.add_argument('--quien', default='IRIS')

    a = ap.parse_args()
    if a.cmd == 'exportar':
        exportar_128(a.raiz, a.salida, solo_prioritarios=not a.todas, limite=a.limite, puerto=a.puerto)
    else:
        importar_128(a.iris, a.salida, quien=a.quien)


if __name__ == '__main__':
    main()
