"""
Exportador de parches de 128×128 px para IRIS (ESA-PhiLab / fork gt-csse), lanzador local y
adaptador de salida hacia `mascaras_nivel1/`.

Permite marcar directamente sobre parches de 128×128 px en IRIS (sin redimensionar, a 10 m por
píxel) a partir de las teselas ya descargadas en `datos_laconvencion`, `datos_indeci`,
`datos_ingemmet` y `datos_azar`, y convertir las máscaras resultantes (`masks/<parche>.png`) al
mismo formato exacto que produce `mascaras_desde_marcas.py` (`mascaras_nivel1/parches_128/` y
`mascaras_nivel1/mascaras_256/`), listo para `empaquetar_para_colab.py` y `ajuste_fino.py`.

Uso:
  1. Armar el proyecto IRIS de 128×128 (ordenado por prioridad de tesis):
       python herramientas/iris_128.py exportar --salida iris_128

  2. Abrir el servidor web de IRIS (crea usuario admin/laconvencion automáticamente):
       python herramientas/iris_128.py abrir --iris iris_128

  3. Convertir las máscaras marcadas en IRIS a `mascaras_nivel1/`:
       python herramientas/iris_128.py importar --iris iris_128 --salida mascaras_nivel1

  4. Generar un ejemplo demostrativo aislado (sin tocar datos reales de evaluación):
       python herramientas/iris_128.py ejemplo --salida iris_ejemplo_demo
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
RAIZ_PROY = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from cortar_parches_128 import LADO, cuadrantes  # noqa: E402

BASES = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']
NOMBRES_8 = [
    'B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)',
    'probabilidad del modelo', 'cajas del modelo (contorno)'
]

TESELAS_ACUERDO = ('sinpad_15525_001', 'sinpad_48878_001', 'sinpad_557_001')
TESELAS_CICATRICES_CLARAS = (
    'ingemmet_002_001', 'ingemmet_003_001', 'ingemmet_036_001', 'ingemmet_037_001', 'ingemmet_046_001',
    'indeci_005_001', 'indeci_009_001', 'indeci_040_001', 'indeci_046_001',
    'sinpad_005_001', 'sinpad_009_001', 'sinpad_040_001', 'sinpad_046_001',
)
ZONAS_VAL_CICATRICES = ('Z04_SANTA_ANA', 'Z08_QUELLOUNO', 'Z09_INKAWASI')
TESELAS_VAL_NEGATIVOS = (
    'sinpad_13640_001', 'sinpad_7448_001', 'sinpad_56208_001', 'sinpad_9709_001', 'sinpad_61214_001',
    'sinpad_96_001', 'sinpad_36869_001', 'sinpad_62691_001', 'sinpad_86528_001', 'sinpad_5559_001',
)


def prioridad_tesela(proy, tes, info):
    """Orden determinista según el plan de trabajo de la tesis."""
    if tes in TESELAS_ACUERDO:
        return (0, TESELAS_ACUERDO.index(tes), proy, tes)
    if tes in TESELAS_CICATRICES_CLARAS:
        return (1, TESELAS_CICATRICES_CLARAS.index(tes), proy, tes)
    if info.get('zona') in ZONAS_VAL_CICATRICES:
        return (2, ZONAS_VAL_CICATRICES.index(info['zona']), proy, tes)
    if info.get('uso') == 'ronda1':
        return (3, 0, proy, tes)
    if tes in TESELAS_VAL_NEGATIVOS:
        return (3, 1 + TESELAS_VAL_NEGATIVOS.index(tes), proy, tes)
    if info.get('conjunto') == 'validacion' or info.get('uso') == 'validacion':
        return (4, 0, proy, tes)
    if info.get('uso', '').startswith('ronda'):
        return (4, 1, proy, tes)
    return (5, 0, proy, tes)


def configuracion_iris_128(nombre='deslizamientos-laconvencion-128', lado=128, puerto=5000):
    return {
        'name': nombre,
        'host': '127.0.0.1',
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
            'prioritise_unmarked_images': False,
            'shuffle_images': False,
            'unverified_threshold': 1,
            'test_images': None,
            'ai_model': {
                'bands': ['$S2.B1', '$S2.B2', '$S2.B3', '$S2.B4', '$S2.B5', '$S2.B6'],
                'train_ratio': 0.8,
                'max_train_pixels': 20000,
                'n_estimators': 25,
                'max_depth': 10,
                'n_leaves': 12,
                'suppression_threshold': 0,
                'suppression_filter_size': 5,
                'suppression_default_class': 0,
                'use_edge_filter': True,
                'use_superpixels': False,
                'use_meshgrid': False,
                'meshgrid_cells': '3x3',
            },
        },
        'classes': [
            {
                'name': 'Sin deslizamiento',
                'colour': [255, 255, 255, 0],
                'user_colour': [0, 255, 255, 70],
                'description': 'Fondo confirmado (tecla 1): bosque, pasto, río, camino, cultivo o techo.',
            },
            {
                'name': 'Deslizamiento',
                'colour': [255, 40, 60, 110],
                'description': 'Cicatriz de deslizamiento visible en ladera (tecla 2): corona arriba y material arrastrado.',
            },
            {
                'name': 'Dudoso',
                'colour': [255, 200, 0, 90],
                'description': 'No se puede decidir (tecla 3: nube, sombra, cauce sin arranque visible). Se ignora (255).',
            },
        ],
        'views': {
            'Color natural con cajas': {
                'description': 'Cajas amarillas del modelo como guía visual. Pinta la forma real con «Deslizamiento» (2).',
                'type': 'image', 'vmin': 0, 'vmax': 1500,
                'data': ['$S2.B4*(1-$S2.B8)+5000*$S2.B8', '$S2.B3*(1-$S2.B8)+5000*$S2.B8', '$S2.B2*(1-$S2.B8)'],
            },
            'Color natural': {
                'description': 'B4 (rojo), B3 (verde) y B2 (azul) de Sentinel-2 L2A.',
                'type': 'image', 'vmin': 0, 'vmax': 1500,
                'data': ['$S2.B4', '$S2.B3', '$S2.B2'],
            },
            'Probabilidad del modelo': {
                'description': 'Mapa de calor U-Net v4 (solo de guía: lo que vale es lo que se ve en la imagen).',
                'type': 'image', 'data': '$S2.B7', 'cmap': 'magma', 'vmin': 0, 'vmax': 1,
            },
            'Pendiente': {
                'description': 'B13 pendiente del terreno en grados (las cicatrices arrancan en ladera >= 10-15°).',
                'type': 'image', 'data': '$S2.B5', 'cmap': 'viridis', 'vmin': 0, 'vmax': 60,
            },
            'Relieve': {
                'description': 'B14 elevación DEM en metros sobre el nivel del mar.',
                'type': 'image', 'data': '$S2.B6', 'cmap': 'terrain', 'vmin': 500, 'vmax': 3500,
            },
            'Bordes': {
                'description': 'Gradiente Sobel en las bandas visibles: resalta escarpas y coronas.',
                'type': 'image', 'data': 'edges($S2.B2+$S2.B3+$S2.B4)', 'cmap': 'gray',
            },
            'Suelo vs Vegetacion': {
                'description': 'Contraste Rojo/Verde (B4-B3)/(B4+B3+1): tonos cálidos indican suelo desnudo o roca.',
                'type': 'image', 'data': '($S2.B4-$S2.B3)/($S2.B4+$S2.B3+1)', 'cmap': 'RdYlGn_r', 'vmin': -0.3, 'vmax': 0.3,
            },
            'Mapa': {
                'description': 'Mapa aéreo Bing de referencia contextual.',
                'type': 'bingmap',
            },
        },
        'view_groups': {
            'default': ['Color natural con cajas', 'Color natural', 'Pendiente'],
            'probabilidad': ['Color natural', 'Probabilidad del modelo', 'Pendiente'],
            'espectral': ['Color natural', 'Suelo vs Vegetacion', 'Pendiente'],
            'relieve': ['Color natural', 'Relieve', 'Bordes'],
            'mapa': ['Color natural', 'Mapa'],
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

    candidatas = []
    for proy in BASES:
        for ruta in sorted(glob.glob(os.path.join(raiz, proy, 'images', '*', 'image.tif'))):
            tes = os.path.basename(os.path.dirname(ruta))
            info = part.get((proy, tes), {'zona': '', 'conjunto': 'entrenamiento', 'uso': 'reserva'})
            if info['conjunto'] in ('excluida', 'pendiente'):
                continue
            if solo_prioritarios and not (
                info['uso'].startswith('ronda')
                or info['uso'] == 'validacion'
                or info['conjunto'] == 'validacion'
                or tes in TESELAS_ACUERDO
                or tes in TESELAS_CICATRICES_CLARAS
            ):
                continue
            candidatas.append((prioridad_tesela(proy, tes, info), proy, tes, ruta, info))

    candidatas.sort(key=lambda t: t[0])
    if limite:
        candidatas = candidatas[:limite]

    os.makedirs(os.path.join(salida, 'images'), exist_ok=True)
    os.makedirs(os.path.join(salida, 'masks'), exist_ok=True)
    indice = []
    n_tes = 0
    for prio, proy, tes, ruta, info in candidatas:
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
                zona=info['zona'], conjunto=info['conjunto'], uso=info['uso'],
                bloque_prioridad=prio[0]
            )
            with open(os.path.join(carpeta_p, 'metadata.json'), 'w', encoding='utf-8') as fm:
                json.dump(meta_p, fm, indent=1, ensure_ascii=False)
            indice.append({
                'orden': len(indice) + 1,
                'bloque_prioridad': prio[0],
                'id_iris': pid, 'parche': f'{tes}_{q}', 'proyecto': proy, 'tesela': tes,
                'cuadrante': q, 'zona': info['zona'], 'conjunto': info['conjunto'], 'uso': info['uso']
            })
        n_tes += 1

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


def _cargar_propuestas_hashes(carpeta_iris):
    ruta = os.path.join(carpeta_iris, 'propuestas_iniciales_hashes.json')
    if not os.path.exists(ruta):
        return None, {}
    with open(ruta, encoding='utf-8') as f:
        meta = json.load(f)
    return meta.get('fecha_respaldo_unix'), meta.get('archivos', {})


def _acciones_iris_db(carpeta_iris):
    """Devuelve el conjunto de image_ids guardados por usuarios en iris.db (si existe)."""
    import sqlite3
    dbs = glob.glob(os.path.join(carpeta_iris, '**', '*.db'), recursive=True)
    guardados = set()
    for db in dbs:
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            tablas = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'action' in tablas:
                cols = {r[1] for r in cur.execute("PRAGMA table_info(action)").fetchall()}
                if 'image_id' in cols:
                    for (img_id,) in cur.execute("SELECT DISTINCT image_id FROM action WHERE image_id IS NOT NULL").fetchall():
                        guardados.add(str(img_id))
            con.close()
        except Exception:
            pass
    return guardados


def _cargar_decisiones_agente(ruta_decisiones):
    if not ruta_decisiones or not os.path.exists(ruta_decisiones):
        return set()
    revisados = set()
    with open(ruta_decisiones, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('parche'):
                revisados.add(r['parche'].strip())
            if r.get('id_iris'):
                revisados.add(r['id_iris'].strip())
    return revisados


def importar_128(carpeta_iris, salida_nivel1, quien='IRIS', raiz='.', decisiones=None, fecha_respaldo=None):
    """Convierte las máscaras PNG revisadas de IRIS (128×128) al formato de `mascaras_nivel1/`.
    Solo importa parches revisados:
      - Si `quien == 'AGENTE'`: exige que el parche esté registrado en `informes/decisiones_agente.csv`
        (o en `--decisiones`), y respeta las máscaras humanas previas sin sobrescribirlas.
      - Si `quien != 'AGENTE'` y existe `propuestas_iniciales_hashes.json` (o `fecha_respaldo`):
        solo importa PNGs guardados después de `fecha_respaldo` y distintos de las propuestas automáticas
        sin editar (o registrados como acción de usuario en `iris.db`). Nunca importa un PNG sin revisar.
    Codificación de IRIS:
      - 0-based estándar de IRIS (`mask_encoding: integer`): 0 -> 0 (fondo), 1 -> 1 (deslizamiento), 2 -> 255 (dudoso).
      - 1-based alternativo (cuando contiene clase 3): 1 -> 0 (fondo), 2 -> 1 (deslizamiento), 3 -> 255 (dudoso).
    Respeta la regla física de nube blanca: B2 > 1000 se marca como 255 salvo donde es suelo desnudo claro
    (B2 <= 2000, B4 > B3, B4 > 1.2*B2) o donde se marcó deslizamiento (1)."""
    import hashlib
    from PIL import Image
    import rasterio
    from rasterio.features import shapes
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_geom
    from scipy import ndimage

    ruta_ind = os.path.join(carpeta_iris, 'indice_iris_128.csv')
    if not os.path.exists(ruta_ind):
        raise SystemExit(f'Falta {ruta_ind}: corre primero iris_128.py exportar')
    with open(ruta_ind, encoding='utf-8') as f:
        indice = list(csv.DictReader(f))

    es_agente = str(quien).strip().upper() == 'AGENTE'
    origen_tif = 'agente en IRIS 128x128' if es_agente else 'marcado humano en IRIS 128x128'
    origen_gj = 'IRIS_128_AGENTE' if es_agente else 'IRIS_128'

    ruta_decisiones = decisiones or os.path.join(raiz, 'informes', 'decisiones_agente.csv')
    decisiones_agente = _cargar_decisiones_agente(ruta_decisiones) if es_agente else set()
    fecha_resp_auto, hashes_propuestas = _cargar_propuestas_hashes(carpeta_iris)
    corte_fecha = fecha_respaldo if fecha_respaldo is not None else fecha_resp_auto
    acciones_db = _acciones_iris_db(carpeta_iris) if (corte_fecha is not None and not es_agente) else set()

    ruta_csv = os.path.join(salida_nivel1, 'mascaras_nivel1.csv')
    existentes = {}
    if os.path.exists(ruta_csv):
        with open(ruta_csv, encoding='utf-8') as f_ex:
            for rex in csv.DictReader(f_ex):
                existentes[(rex['proyecto'], rex['tesela'])] = rex

    por_tesela = collections.defaultdict(dict)
    n_parches = 0
    omitidos_sin_revision = 0
    omitidos_humano_protegido = 0
    feats_geojson = []

    for r in indice:
        pid = r['id_iris']
        proy, tes, q, parche = r['proyecto'], r['tesela'], r['cuadrante'], r['parche']
        ruta_png = os.path.join(carpeta_iris, 'masks', f'{pid}.png')
        if not os.path.exists(ruta_png):
            continue

        if es_agente:
            if parche not in decisiones_agente and pid not in decisiones_agente:
                omitidos_sin_revision += 1
                continue
            prev = existentes.get((proy, tes))
            if prev and prev.get('quien', '').strip() not in ('', 'AGENTE'):
                omitidos_humano_protegido += 1
                continue
        elif corte_fecha is not None or hashes_propuestas:
            mtime_png = os.path.getmtime(ruta_png)
            with open(ruta_png, 'rb') as fp:
                sha_actual = hashlib.sha256(fp.read()).hexdigest()
            info_prop = hashes_propuestas.get(f'{pid}.png')
            es_propuesta_intacta = info_prop is not None and sha_actual == info_prop.get('sha256')
            en_db = pid in acciones_db
            if (corte_fecha is not None and mtime_png <= corte_fecha) or (es_propuesta_intacta and not en_db):
                omitidos_sin_revision += 1
                continue

        raw = np.asarray(Image.open(ruta_png))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        ruta_img = os.path.join(carpeta_iris, 'images', pid, 'image.tif')
        with rasterio.open(ruta_img) as ds:
            b2 = ds.read(2)
            b3 = ds.read(3) if ds.count >= 3 else b2
            b4 = ds.read(4) if ds.count >= 4 else b2
            tr, crs, forma = ds.transform, ds.crs, ds.shape

        vals = set(np.unique(raw).tolist())
        if 3 in vals:
            des = raw == 2
            dud = (raw == 3) | (raw == 0)
        else:
            des = raw == 1
            dud = (raw == 2) | (raw == 255)

        suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
        sin_dato = (b2 <= 0) | ((b2 > 1000) & ~suelo_claro & ~des)
        mask = np.zeros(forma, dtype='uint8')
        mask[des] = 1
        mask[dud | sin_dato] = 255

        dest_dir = os.path.join(salida_nivel1, 'parches_128', proy)
        os.makedirs(dest_dir, exist_ok=True)
        p8 = dict(driver='GTiff', height=forma[0], width=forma[1], count=1, dtype='uint8',
                  crs=crs, transform=tr, compress='deflate')
        with MemoryFile() as mem:
            with mem.open(**p8) as d:
                d.write(mask, 1)
                d.update_tags(nivel='1', valores='0 fondo, 1 deslizamiento, 255 ignorar', origen=origen_tif, quien=quien)
            with open(os.path.join(dest_dir, f'{parche}.tif'), 'wb') as fo:
                fo.write(mem.read())

        for val_c, nom_c in ((1, 'deslizamiento'), (255, 'dudoso')):
            bin_m = (mask == 1) if val_c == 1 else dud
            if bin_m.any():
                for geom, _ in shapes(bin_m.astype('uint8'), mask=bin_m, transform=tr):
                    g4326 = transform_geom(crs, 'EPSG:4326', geom)
                    feats_geojson.append({
                        'type': 'Feature',
                        'properties': {'clase': nom_c, 'proyecto': proy, 'tesela': tes, 'parche': parche, 'origen': origen_gj, 'quien': quien},
                        'geometry': g4326,
                    })

        por_tesela[(proy, tes, r['zona'], r['conjunto'])][q] = (mask, crs, tr)
        n_parches += 1

    if omitidos_sin_revision:
        print(f'Aviso: se omitieron {omitidos_sin_revision} PNG(s) sin revisión confirmada.')
    if omitidos_humano_protegido:
        print(f'Aviso: se protegieron {omitidos_humano_protegido} parche(s) con máscara humana previa.')

    if not n_parches:
        raise SystemExit(
            f'No hay máscaras PNG revisadas para importar en {os.path.join(carpeta_iris, "masks")} '
            f'(omitidas sin revisión: {omitidos_sin_revision})'
        )

    # Reconstruir máscaras 256×256 por tesela y actualizar tabla resumen
    filas_nuevas = {}
    for (proy, tes, zona, conj), q_dict in sorted(por_tesela.items()):
        ms = [v[0] for v in q_dict.values()]
        crs0, tr_q1 = next(iter(q_dict.values()))[1], next(iter(q_dict.values()))[2]
        ruta_tes_orig = os.path.join(raiz, proy, 'images', tes, 'image.tif')
        if os.path.exists(ruta_tes_orig):
            with rasterio.open(ruta_tes_orig) as ds_t:
                tr_t, crs_t, forma_t = ds_t.transform, ds_t.crs, ds_t.shape
        else:
            tr_t, crs_t, forma_t = tr_q1, crs0, (256, 256)

        m256 = np.full(forma_t, 255, dtype='uint8')
        for q, f0, c0 in cuadrantes(forma_t[0], forma_t[1], LADO):
            if q in q_dict:
                m256[f0:f0 + LADO, c0:c0 + LADO] = q_dict[q][0]
        dir_256 = os.path.join(salida_nivel1, 'mascaras_256', proy)
        os.makedirs(dir_256, exist_ok=True)
        p256 = dict(driver='GTiff', height=forma_t[0], width=forma_t[1], count=1, dtype='uint8',
                    crs=crs_t, transform=tr_t, compress='deflate')
        with MemoryFile() as mem:
            with mem.open(**p256) as d:
                d.write(m256, 1)
                d.update_tags(nivel='1', valores='0 fondo, 1 deslizamiento, 255 ignorar', origen=origen_tif, quien=quien)
            with open(os.path.join(dir_256, f'{tes}.tif'), 'wb') as fo:
                fo.write(mem.read())

        cics = int(ndimage.label(m256 == 1)[1])
        ha_des = round(float((m256 == 1).sum()) * 0.01, 2)
        ha_ign = round(float((m256 == 255).sum()) * 0.01, 2)
        filas_nuevas[(proy, tes)] = {
            'proyecto': proy, 'tesela': tes, 'quien': quien,
            'poligonos_deslizamiento': cics, 'poligonos_dudoso': 0,
            'cicatrices': cics, 'ha_deslizamiento': ha_des, 'ha_ignorar': ha_ign,
            'zona': zona, 'conjunto': conj, 'nota': f'{len(ms)}/4 parches importados de IRIS 128'
        }

    os.makedirs(salida_nivel1, exist_ok=True)
    existentes.update(filas_nuevas)
    filas = [existentes[k] for k in sorted(existentes.keys())]

    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    ruta_gj = os.path.join(salida_nivel1, 'marcas_desde_iris.geojson')
    with open(ruta_gj, 'w', encoding='utf-8') as fg:
        json.dump({'type': 'FeatureCollection', 'features': feats_geojson}, fg, ensure_ascii=False)

    print(f'{n_parches} parches de nivel 1 importados ({len(filas_nuevas)} teselas) en {os.path.abspath(salida_nivel1)}')
    return n_parches


def crear_ejemplo_demo(raiz='.', salida='iris_ejemplo_demo', tesela_ejemplo='ingemmet_003_001'):
    """Crea un ejemplo demostrativo completo y aislado (1 tesela = 4 parches de Sillapata ingemmet_003_001)
    usando como referencia didáctica las piezas geológicas de Nivel 2 (INGEMMET ∩ suelo desnudo),
    sin tocar `datos_laconvencion/marcas.geojson` ni `revisadas.csv`, verificando todo el flujo
    Exportar -> Pintar en IRIS -> Importar a Nivel 1 -> Generar lámina visual."""
    import shutil
    from PIL import Image, ImageDraw
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform as tr_coords
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    if os.path.exists(salida):
        shutil.rmtree(salida)
    dir_iris = os.path.join(salida, 'proyecto_iris')
    dir_n1 = os.path.join(salida, 'mascaras_nivel1_demo')
    os.makedirs(os.path.join(dir_iris, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_iris, 'masks'), exist_ok=True)

    proy = 'datos_ingemmet' if tesela_ejemplo.startswith('ingemmet') else 'datos_laconvencion'
    ruta_tes = os.path.join(raiz, proy, 'images', tesela_ejemplo, 'image.tif')
    with rasterio.open(ruta_tes) as ds:
        x_full = ds.read().astype('float32')
        perfil = ds.profile

    indice = []
    paneles = []
    for q, f0, c0 in cuadrantes(x_full.shape[1], x_full.shape[2], LADO):
        pid = f'{proy}__{tesela_ejemplo}_{q}'
        carpeta_p = os.path.join(dir_iris, 'images', pid)
        os.makedirs(carpeta_p, exist_ok=True)
        tr = win_transform(Window(c0, f0, LADO, LADO), perfil['transform'])
        p = dict(
            driver='GTiff', height=LADO, width=LADO, count=x_full.shape[0],
            dtype='float32', crs=perfil['crs'], transform=tr, compress='deflate'
        )
        trozo = x_full[:, f0:f0 + LADO, c0:c0 + LADO]
        with MemoryFile() as mem:
            with mem.open(**p) as d:
                d.write(trozo)
                d.descriptions = tuple(NOMBRES_8[:x_full.shape[0]])
            with open(os.path.join(carpeta_p, 'image.tif'), 'wb') as fo:
                fo.write(mem.read())
        cx, cy = tr * (LADO / 2, LADO / 2)
        lon, lat = tr_coords(perfil['crs'], 'EPSG:4326', [cx], [cy])
        with open(os.path.join(carpeta_p, 'metadata.json'), 'w', encoding='utf-8') as fm:
            json.dump({
                'location': [round(lon[0], 5), round(lat[0], 5)],
                'proyecto': proy, 'tesela': tesela_ejemplo, 'cuadrante': q,
                'parche': f'{tesela_ejemplo}_{q}', 'zona': 'Z04_SANTA_ANA',
                'conjunto': 'validacion', 'uso': 'ronda1'
            }, fm, indent=1, ensure_ascii=False)
        indice.append({
            'orden': len(indice) + 1, 'bloque_prioridad': 1,
            'id_iris': pid, 'parche': f'{tesela_ejemplo}_{q}', 'proyecto': proy,
            'tesela': tesela_ejemplo, 'cuadrante': q, 'zona': 'Z04_SANTA_ANA',
            'conjunto': 'validacion', 'uso': 'ronda1'
        })

        # Usar la propuesta geológica INGEMMET de Nivel 2 como ejemplo didáctico de lo que se pinta en IRIS
        ruta_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2', 'parches_128', f'{tesela_ejemplo}_{q}.tif')
        m_demo = np.zeros((LADO, LADO), dtype=np.uint8)
        if os.path.exists(ruta_n2):
            with rasterio.open(ruta_n2) as dn2:
                mn2 = dn2.read(1)
            m_demo[mn2 == 255] = 2   # Clase 2 en IRIS (tecla 3): Dudoso
            m_demo[mn2 == 1] = 1     # Clase 1 en IRIS (tecla 2): Deslizamiento
        else:
            b3, b4, pend, prob = trozo[2], trozo[3], trozo[4], trozo[6]
            m_demo[(prob >= 0.55) & (pend >= 18.0) & (b4 > b3)] = 1
        Image.fromarray(m_demo).save(os.path.join(dir_iris, 'masks', f'{pid}.png'))

        b2, b3, b4, pend, cajas = trozo[1], trozo[2], trozo[3], trozo[4], trozo[7]
        rgb = np.clip(np.stack([b4, b3, b2], axis=-1) / 1500.0 * 255.0, 0, 255).astype(np.uint8)
        rgb_cajas = rgb.copy()
        rgb_cajas[cajas > 0] = [255, 230, 0]

        iris_vis = rgb.copy().astype(np.float32)
        iris_vis[m_demo == 1] = 0.35 * iris_vis[m_demo == 1] + 0.65 * np.array([255, 40, 60])
        iris_vis[m_demo == 2] = 0.50 * iris_vis[m_demo == 2] + 0.50 * np.array([255, 200, 0])
        iris_vis = np.clip(iris_vis, 0, 255).astype(np.uint8)
        paneles.append((f'{tesela_ejemplo}_{q}', rgb_cajas, rgb, iris_vis))

    with open(os.path.join(dir_iris, 'indice_iris_128.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(indice[0].keys()))
        w.writeheader()
        w.writerows(indice)
    cfg = configuracion_iris_128(nombre='demo-iris-128-sillapata', lado=LADO, puerto=5000)
    with open(os.path.join(dir_iris, 'deslizamientos_128.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)

    importar_128(dir_iris, dir_n1, quien='DEMO_EJEMPLO', raiz=raiz)

    # Completar la lámina visual de 4 columnas
    ESC = 2
    W_P = LADO * ESC
    H_FILA = W_P + 28
    lienzo = Image.new('RGB', (W_P * 4 + 40, H_FILA * len(paneles) + 46), (18, 22, 28))
    dib = ImageDraw.Draw(lienzo)
    dib.text(
        (10, 8),
        f'EJEMPLO PRACTICO IRIS 128x128 ({tesela_ejemplo} - Sillapata) | NO modifica datos reales de evaluacion',
        fill=(255, 235, 140)
    )
    dib.text(
        (10, 24),
        'Col 1: Vista IRIS "Color natural con cajas" | Col 2: "Color natural" | Col 3: Trazo en IRIS (Rojo=Deslizamiento [2], Amarillo=Dudoso [3]) | Col 4: Mascara Nivel 1 (.tif)',
        fill=(220, 225, 235)
    )
    for idx, (nom_parche, rgb_cajas, rgb_limpio, iris_vis) in enumerate(paneles):
        ruta_m1 = os.path.join(dir_n1, 'parches_128', proy, f'{nom_parche}.tif')
        with rasterio.open(ruta_m1) as dm:
            m1 = dm.read(1)
        m1_vis = np.zeros((LADO, LADO, 3), dtype=np.uint8)
        m1_vis[m1 == 0] = [28, 45, 36]        # 0 = fondo confirmado
        m1_vis[m1 == 1] = [255, 45, 55]       # 1 = deslizamiento confirmado
        m1_vis[m1 == 255] = [255, 200, 0]     # 255 = ignorar (dudoso / nube)

        y0 = 46 + idx * H_FILA
        dib.text(
            (10, y0),
            f'Parche {nom_parche} (128x128 px): Deslizamiento (valor 1) = {(m1==1).sum()} px ({(m1==1).sum()*0.01:.2f} ha) | Dudoso/Ignorar (255) = {(m1==255).sum()} px | Fondo (0) = {(m1==0).sum()} px',
            fill=(160, 220, 255)
        )
        for j, arr_img in enumerate((rgb_cajas, rgb_limpio, iris_vis, m1_vis)):
            im_big = Image.fromarray(arr_img).resize((W_P, W_P), Image.NEAREST)
            lienzo.paste(im_big, (8 + j * (W_P + 8), y0 + 18))

    ruta_lamina = os.path.join(salida, 'lamina_ejemplo_iris_128.png')
    lienzo.save(ruta_lamina)
    print(f'Lámina de ejemplo guardada en {os.path.abspath(ruta_lamina)}')
    return ruta_lamina


def abrir_servidor(carpeta_iris='iris_128', usuario='admin', clave='laconvencion', debug=False):
    from iris import start_server
    ruta_cfg = os.path.join(carpeta_iris, 'deslizamientos_128.json')
    if not os.path.exists(ruta_cfg):
        raise SystemExit(f'No existe {ruta_cfg}: corre primero python herramientas/iris_128.py exportar')
    print(f'Iniciando IRIS en http://127.0.0.1:5000/segmentation/ (usuario: {usuario} | clave: {clave})')
    start_server(
        project_file=os.path.abspath(ruta_cfg),
        debug=debug,
        production=False,
        admin_user=usuario,
        admin_password=clave,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_exp = sub.add_parser('exportar', help='arma el proyecto de IRIS con parches de 128×128 ordenados por prioridad')
    p_exp.add_argument('--raiz', default='.')
    p_exp.add_argument('--salida', default='iris_128')
    p_exp.add_argument('--todas', action='store_true', help='incluir también las teselas de reserva')
    p_exp.add_argument('--limite', type=int, help='máximo de teselas a exportar (para probar)')
    p_exp.add_argument('--puerto', type=int, default=5000)

    p_imp = sub.add_parser('importar', help='convierte las máscaras PNG revisadas de IRIS a mascaras_nivel1/')
    p_imp.add_argument('--iris', default='iris_128')
    p_imp.add_argument('--salida', default='mascaras_nivel1')
    p_imp.add_argument('--quien', default='IRIS')
    p_imp.add_argument('--raiz', default='.')
    p_imp.add_argument('--decisiones', default=None, help='ruta a decisiones_agente.csv (cuando --quien AGENTE)')
    p_imp.add_argument('--fecha-respaldo', type=float, default=None, help='timestamp unix de corte para exigir PNGs guardados después')

    p_abr = sub.add_parser('abrir', help='inicia el servidor web local de IRIS en http://127.0.0.1:5000')
    p_abr.add_argument('--iris', default='iris_128')
    p_abr.add_argument('--usuario', default='admin')
    p_abr.add_argument('--clave', default='laconvencion')
    p_abr.add_argument('--debug', action='store_true')

    p_eje = sub.add_parser('ejemplo', help='genera un ejemplo demostrativo completo y aislado en iris_ejemplo_demo/')
    p_eje.add_argument('--raiz', default='.')
    p_eje.add_argument('--salida', default='iris_ejemplo_demo')

    a = ap.parse_args()
    if a.cmd == 'exportar':
        exportar_128(a.raiz, a.salida, solo_prioritarios=not a.todas, limite=a.limite, puerto=a.puerto)
    elif a.cmd == 'importar':
        importar_128(a.iris, a.salida, quien=a.quien, raiz=a.raiz, decisiones=a.decisiones, fecha_respaldo=a.fecha_respaldo)
    elif a.cmd == 'abrir':
        abrir_servidor(a.iris, usuario=a.usuario, clave=a.clave, debug=a.debug)
    elif a.cmd == 'ejemplo':
        crear_ejemplo_demo(a.raiz, a.salida)


if __name__ == '__main__':
    main()
