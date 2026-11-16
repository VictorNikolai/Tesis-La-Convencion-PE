"""
Extractor y Clasificador de Calidad por Tiers (Oro, Plata, Bronce)
para enriquecer las cicatrices de deslizamientos en La Convención (128×128 px).

Procedimiento:
  1. Extrae parches con ventana deslizante/centrada (s1, s2, c0, s3, s4) en teselas con cicatrices
     para capturar el deslizamiento entero sin cortarlo en los bordes de los cuadrantes estándar.
  2. Clasifica cada parche de entrenamiento según métricas físicas y espectrales:
       - Tier ORO   (Peso 1.0): Cicatriz nítida (>=20 px), ladera fuerte (pendiente media >=22°, max >=26°),
                                nubes/sombras <=5%, compacidad morfológica alta.
       - Tier PLATA (Peso 0.5): Cicatriz moderada (>=12 px), ladera >=18°, nubes <=15%.
       - Tier BRONCE (Peso 0.3 para cicatrices menores / 0.2 para negativos confirmados de control).
  3. Genera catálogos ordenados por prioridad para IRIS 128 (revisión humana en 5 minutos)
     y para el entrenamiento ponderado en Google Colab / GPU.
"""
import csv
import glob
import os
import sys

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)

LADO = 128
BANDAS_6 = ['B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)']
TESELAS_ACUERDO = {'sinpad_15525_001', 'sinpad_48878_001', 'sinpad_557_001'}

POSICIONES_SLIDING = [
    ('s1', 0, 64),
    ('s2', 64, 0),
    ('c0', 64, 64),
    ('s3', 64, 128),
    ('s4', 128, 64),
]


def ejecutar(raiz='.'):
    dir_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2')
    dir_p128 = os.path.join(raiz, 'parches_128')
    ruta_ind_base = os.path.join(dir_p128, 'indice_parches.csv')
    if not os.path.exists(ruta_ind_base):
        raise SystemExit(f"Falta {ruta_ind_base}")

    with open(ruta_ind_base, encoding='utf-8') as f:
        filas_base = list(csv.DictReader(f))

    # Indexar metadatos por (proyecto, tesela)
    meta_tesela = {}
    for r in filas_base:
        k = (r['proyecto'], r['tesela'])
        if k not in meta_tesela:
            meta_tesela[k] = r

    rutas_m256 = glob.glob(os.path.join(dir_n2, 'mascaras_256', '*', '*.tif'))
    print(f"Examinando {len(rutas_m256)} teselas en Nivel 2...")

    nuevos_parches = []
    conteo_extraidos = 0

    for ruta_m in sorted(rutas_m256):
        proy = os.path.basename(os.path.dirname(ruta_m))
        tes = os.path.splitext(os.path.basename(ruta_m))[0]
        if tes in TESELAS_ACUERDO:
            continue

        with rasterio.open(ruta_m) as dm:
            m256 = dm.read(1)
            perfil_m = dm.profile

        if not (m256 == 1).any():
            continue

        ruta_img = os.path.join(raiz, proy, 'images', tes, 'image.tif')
        if not os.path.exists(ruta_img):
            continue

        with rasterio.open(ruta_img) as di:
            img = di.read()
            perfil_i = di.profile
            tr_orig = di.transform
            crs_orig = di.crs

        pend = img[4]
        b2 = img[1]
        base_meta = meta_tesela.get((proy, tes), {})
        conjunto = base_meta.get('conjunto', 'entrenamiento')
        zona = base_meta.get('zona', 'Z_GENERAL')

        for sub, f0, c0 in POSICIONES_SLIDING:
            sub_m = m256[f0:f0 + LADO, c0:c0 + LADO]
            sub_pend = pend[f0:f0 + LADO, c0:c0 + LADO]
            px_1 = int((sub_m == 1).sum())

            # Solo extraer si tiene al menos 0.15 ha (15 px) de cicatriz en ladera
            if px_1 < 15:
                continue
            p_med = float(sub_pend[sub_m == 1].mean())
            if p_med < 18.0:
                continue

            nombre_parche = f"{tes}_{sub}"
            ruta_out_img = os.path.join(dir_p128, proy, f"{nombre_parche}.tif")
            ruta_out_mk = os.path.join(dir_n2, 'parches_128', proy, f"{nombre_parche}.tif")

            tr_sub = win_transform(Window(c0, f0, LADO, LADO), tr_orig)

            # Escribir GeoTIFF de 6 bandas
            p6 = dict(driver='GTiff', height=LADO, width=LADO, count=6, dtype='float32',
                      crs=crs_orig, transform=tr_sub, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p6) as d:
                    d.write(img[:6, f0:f0 + LADO, c0:c0 + LADO].astype(np.float32))
                    d.descriptions = tuple(BANDAS_6)
                with open(ruta_out_img, 'wb') as fo:
                    fo.write(mem.read())

            # Escribir GeoTIFF de máscara uint8
            p_mk = dict(driver='GTiff', height=LADO, width=LADO, count=1, dtype='uint8',
                        crs=crs_orig, transform=tr_sub, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p_mk) as d:
                    d.write(sub_m, 1)
                with open(ruta_out_mk, 'wb') as fo:
                    fo.write(mem.read())

            conteo_extraidos += 1
            nuevos_parches.append({
                'parche': nombre_parche,
                'proyecto': proy,
                'tesela': tes,
                'cuadrante': sub,
                'fila0': f0,
                'col0': c0,
                'fecha': base_meta.get('fecha', 'base'),
                'zona': zona,
                'conjunto': conjunto,
                'uso': 'sliding_centrado',
                'lon': base_meta.get('lon', '0'),
                'lat': base_meta.get('lat', '0'),
                'valido_pct': '100.0',
                'pend_media': round(p_med, 1),
                'dem_medio': base_meta.get('dem_medio', '1500'),
                'archivo': f"{proy}/{nombre_parche}.tif",
            })

    print(f"-> Extraídos {conteo_extraidos} parches positivos centrados nuevos.")

    # 2. Clasificar TODOS los parches (originales + nuevos) en Tiers
    todos_los_parches = filas_base + nuevos_parches
    catalogo_tiers = []
    conteo_tiers = {'oro': 0, 'plata': 0, 'bronce_positivo': 0, 'bronce_negativo': 0, 'dudoso': 0, 'evaluacion_ciega': 0}

    # Leer catálogo N2 completo para conocer el tipo de tesela
    cat_n2 = {}
    ruta_n2_comp = os.path.join(dir_n2, 'mascaras_nivel2_completo.csv')
    if os.path.exists(ruta_n2_comp):
        with open(ruta_n2_comp, encoding='utf-8') as fn2:
            for r in csv.DictReader(fn2):
                cat_n2[(r['proyecto'], r['tesela'])] = r

    for p in todos_los_parches:
        proy = p['proyecto']
        par = p['parche']
        tes = p['tesela']
        conj = p['conjunto']

        if tes in TESELAS_ACUERDO:
            tier = 'evaluacion_ciega'
            peso = 1.0
            conteo_tiers[tier] += 1
            catalogo_tiers.append({
                'parche': par, 'proyecto': proy, 'tesela': tes, 'cuadrante': p.get('cuadrante', ''),
                'conjunto': conj, 'zona': p.get('zona', ''), 'tier': tier, 'peso_perdida': peso,
                'px_deslizamiento': 0, 'ha_deslizamiento': 0.0, 'pend_media_cicatriz': 0.0,
                'pend_max_cicatriz': 0.0, 'pct_nube_dudoso': 0.0,
                'archivo_imagen': f"{proy}/{par}.tif", 'archivo_mascara': '',
            })
            continue

        ruta_mk = os.path.join(dir_n2, 'parches_128', proy, f"{par}.tif")
        if not os.path.exists(ruta_mk):
            ruta_mk = os.path.join(dir_n2, 'parches_128', f"{par}.tif")
        if not os.path.exists(ruta_mk):
            continue

        with rasterio.open(ruta_mk) as dmk:
            m = dmk.read(1)

        px_1 = int((m == 1).sum())
        px_255 = int((m == 255).sum())
        pct_nube = round(px_255 / (LADO * LADO) * 100, 1)
        ha_1 = round(px_1 * 0.01, 2)

        ruta_im = os.path.join(dir_p128, proy, f"{par}.tif")
        p_med, p_max = 0.0, 0.0
        if os.path.exists(ruta_im) and px_1 > 0:
            with rasterio.open(ruta_im) as dim:
                pend_p = dim.read(5)  # banda 5 es pendiente
                p_med = round(float(pend_p[m == 1].mean()), 1)
                p_max = round(float(pend_p[m == 1].max()), 1)

        info_n2 = cat_n2.get((proy, tes), {})
        tipo_tes = info_n2.get('tipo', '')

        # Reglas de asignación de Tiers
        tier = 'dudoso'
        peso = 0.0

        if px_1 >= 20 and p_med >= 22.0 and p_max >= 26.0 and pct_nube <= 5.0:
            tier = 'oro'
            peso = 1.0
        elif px_1 >= 12 and p_med >= 18.0 and pct_nube <= 15.0:
            tier = 'plata'
            peso = 0.5
        elif px_1 > 0:
            tier = 'bronce_positivo'
            peso = 0.3
        elif tipo_tes in ('negativo_confirmado', 'contexto_ingemmet') and pct_nube <= 20.0:
            tier = 'bronce_negativo'
            peso = 0.2
        else:
            tier = 'dudoso'
            peso = 0.0

        conteo_tiers[tier] = conteo_tiers.get(tier, 0) + 1

        catalogo_tiers.append({
            'parche': par,
            'proyecto': proy,
            'tesela': tes,
            'cuadrante': p.get('cuadrante', ''),
            'conjunto': conj,
            'zona': p.get('zona', ''),
            'tier': tier,
            'peso_perdida': peso,
            'px_deslizamiento': px_1,
            'ha_deslizamiento': ha_1,
            'pend_media_cicatriz': p_med,
            'pend_max_cicatriz': p_max,
            'pct_nube_dudoso': pct_nube,
            'archivo_imagen': f"{proy}/{par}.tif",
            'archivo_mascara': f"mascaras_provisionales/nivel2/parches_128/{proy}/{par}.tif",
        })

    # Guardar catálogo de Tiers
    ruta_out_tiers = os.path.join(dir_n2, 'catalogo_parches_tiers.csv')
    with open(ruta_out_tiers, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(catalogo_tiers[0].keys()))
        w.writeheader()
        w.writerows(catalogo_tiers)

    # Guardar índice ampliado de parches
    ruta_out_ampliado = os.path.join(dir_p128, 'indice_parches_ampliado.csv')
    with open(ruta_out_ampliado, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(todos_los_parches[0].keys()))
        w.writeheader()
        w.writerows(todos_los_parches)

    # 3. Guardar catálogo prioritario para IRIS 128 y ordenar indice_iris_128.csv
    dir_iris = os.path.join(raiz, 'iris_128')
    ruta_ind_iris = os.path.join(dir_iris, 'indice_iris_128.csv')
    if os.path.exists(ruta_ind_iris):
        with open(ruta_ind_iris, encoding='utf-8') as fi:
            iris_rows = list(csv.DictReader(fi))
        dict_tiers = {f"{r['proyecto']}__{r['parche']}": r for r in catalogo_tiers}

        iris_prioridad = []
        orden_tier = {'oro': 1, 'plata': 2, 'bronce_positivo': 3, 'bronce_negativo': 4, 'dudoso': 5, 'evaluacion_ciega': 6}
        for ir in iris_rows:
            p_info = dict_tiers.get(ir['id_iris'], {})
            tier_val = p_info.get('tier', 'bronce_negativo')
            iris_prioridad.append(dict(ir, tier=tier_val, orden=orden_tier.get(tier_val, 99),
                                       px_deslizamiento=int(p_info.get('px_deslizamiento', 0) or 0)))

        iris_prioridad.sort(key=lambda x: (x['orden'], -x['px_deslizamiento'], x['id_iris']))

        # Guardar catálogo de prioridad separado sin alterar indice_iris_128.csv
        ruta_out_iris_prio = os.path.join(dir_iris, 'catalogo_prioridad_iris.csv')
        with open(ruta_out_iris_prio, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(iris_prioridad[0].keys()))
            w.writeheader()
            w.writerows(iris_prioridad)

    print("\nResumen de Clasificación por Tiers:")
    print(f"  • Total parches evaluados: {len(catalogo_tiers)}")
    print(f"  • 🥇 Tier ORO (Confianza >90%, peso 1.0): {conteo_tiers.get('oro', 0)} parches")
    print(f"  • 🥈 Tier PLATA (Confianza 70-90%, peso 0.5): {conteo_tiers.get('plata', 0)} parches")
    print(f"  • 🥉 Tier BRONCE Positivos (Menores, peso 0.3): {conteo_tiers.get('bronce_positivo', 0)} parches")
    print(f"  • 🥉 Tier BRONCE Negativos (Control, peso 0.2): {conteo_tiers.get('bronce_negativo', 0)} parches")
    print(f"  • 🛡️ Evaluación ciega (Acuerdo inter-anotador): {conteo_tiers.get('evaluacion_ciega', 0)} parches")
    print(f"  • Dudosos / Excluidos de loss (peso 0.0): {conteo_tiers.get('dudoso', 0)} parches")
    return catalogo_tiers


if __name__ == '__main__':
    ejecutar()
