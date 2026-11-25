"""
Pruebas automáticas de las herramientas del conjunto local de La Convención.

Comprueba sin tocar los datos reales (usa carpetas temporales para todo lo que escribe):
  1. Partición espacial por zonas (`particion_por_zona.py`) y ausencia de fuga entre conjuntos.
  2. Corte de parches de 128×128 y georreferencia por cuadrante (`cortar_parches_128.py`).
  3. Generación de máscaras de nivel 1 (`mascaras_desde_marcas.py`), incluyendo:
     - negativos confirmados (tesela con «si» y sin polígonos -> máscara en 0),
     - regla de nube blanca (B2 > 1000) que NO pisa los polígonos de deslizamiento,
     - polígonos «dudoso» -> 255,
     - polígonos con nota «2025» -> 255 en imagen base y 1 en datos_laconvencion_2025.
  4. Regla de nube blanca frente a suelo desnudo muy claro (`candidatos_para_qgis.py`,
     `descargar_desde_puntos.py`, `cambios_para_qgis.py`).
  5. Acuerdo entre anotadores: IoU y kappa de Cohen (`acuerdo_anotadores.py`).
  6. Carga de parches locales, normalización L2A -> L4S, pérdida con máscara 255,
     resumen por cicatrices y remuestreo bootstrap por zonas (`ajuste_fino.py`).

Ejecución:
  python herramientas/test_herramientas.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, RAIZ)


def _crear_tesela_sintetica(ruta_tif, x0=750000.0, y0=8600000.0, lado=256, b2_val=300.0, b3_val=450.0, b4_val=300.0):
    os.makedirs(os.path.dirname(ruta_tif), exist_ok=True)
    tr = from_origin(x0, y0, 10.0, 10.0)
    arr = np.zeros((8, lado, lado), dtype=np.float32)
    arr[0] = 250.0          # B1
    arr[1] = b2_val         # B2
    arr[2] = b3_val         # B3
    arr[3] = b4_val         # B4
    arr[4] = 25.0           # B13 pendiente (grados)
    arr[5] = 1500.0         # B14 DEM (m)
    arr[6] = 0.05           # probabilidad
    arr[7] = 0.0            # cajas
    with rasterio.open(
        ruta_tif, 'w', driver='GTiff', height=lado, width=lado, count=8,
        dtype='float32', crs='EPSG:32718', transform=tr
    ) as ds:
        ds.write(arr)
    return tr


class TestParticionPorZona(unittest.TestCase):
    def test_particion_real_sin_fuga_espacial(self):
        """Comprueba que particion.csv y particion_nuevas_zonas.csv no comparten zonas ni tienen fuga (<3560m) entre entrenamiento, validación y prueba."""
        ruta = os.path.join(RAIZ, 'particion.csv')
        self.assertTrue(os.path.exists(ruta), 'Falta particion.csv en la raíz')
        with open(ruta, encoding='utf-8') as f_p:
            filas = list(csv.DictReader(f_p))
        self.assertGreaterEqual(len(filas), 242)
        conj_activos = ('entrenamiento', 'validacion', 'prueba')
        z_por_c = {c: {r['zona'] for r in filas if r['conjunto'] == c} for c in conj_activos}
        self.assertEqual(z_por_c['entrenamiento'] & z_por_c['validacion'], set())
        self.assertEqual(z_por_c['entrenamiento'] & z_por_c['prueba'], set())
        self.assertEqual(z_por_c['validacion'] & z_por_c['prueba'], set())
        # Verificar que las 3 anclas con cicatrices están en validación
        por_tes = {r['tesela']: r for r in filas}
        for ancla in ('ingemmet_002_001', 'sinpad_21603_001', 'sinpad_37418_001'):
            self.assertEqual(por_tes[ancla]['conjunto'], 'validacion')
        # Verificar que teselas y puntos nuevos en conjuntos distintos están a >= 3560 m en x o y
        activas = [r for r in filas if r['conjunto'] in conj_activos]
        ruta_nz = os.path.join(RAIZ, 'particion_nuevas_zonas.csv')
        if os.path.exists(ruta_nz):
            with open(ruta_nz, encoding='utf-8') as f_nz:
                for r in csv.DictReader(f_nz):
                    if r['conjunto'] in conj_activos:
                        activas.append({'tesela': r['id_origen'], 'zona': r['zona'], 'conjunto': r['conjunto'], 'x': r['x'], 'y': r['y']})
        for i, a in enumerate(activas):
            xa, ya = float(a['x']), float(a['y'])
            for b in activas[i + 1:]:
                if a['conjunto'] != b['conjunto']:
                    dx = abs(xa - float(b['x']))
                    dy = abs(ya - float(b['y']))
                    self.assertFalse(
                        dx < 3560 and dy < 3560,
                        f"Elementos {a['tesela']} ({a['conjunto']}) y {b['tesela']} ({b['conjunto']}) "
                        f"están a dx={dx:.0f}m, dy={dy:.0f}m (<3560m) en conjuntos distintos"
                    )
        import particion_por_zona as ppz
        errs_prov = ppz.comprobar_teselas_en_provincia(filas, raiz=RAIZ)
        self.assertEqual(errs_prov, [], 'Teselas fuera de La Convención (<50%): ' + '; '.join(errs_prov))

    def test_comprobar_teselas_en_provincia_sintetico(self):
        import particion_por_zona as ppz
        from shapely.geometry import box
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer
        lim_gj = json.load(open(os.path.join(RAIZ, 'herramientas', 'limite_laconvencion.geojson'), encoding='utf-8'))
        # Centro aproximado dentro de La Convencion (Quillabamba: -12.86, -72.69) y punto externo (-13.16, -72.54)
        tr_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32718', always_xy=True).transform
        pt_in = shp_transform(tr_utm, box(-72.70, -12.87, -72.68, -12.85)).centroid
        pt_out = shp_transform(tr_utm, box(-72.20, -13.40, -72.18, -13.38)).centroid
        filas = [
            {'tesela': 'IN_001', 'zona': 'Z1', 'conjunto': 'prueba', 'x': pt_in.x, 'y': pt_in.y},
            {'tesela': 'OUT_001', 'zona': 'Z2', 'conjunto': 'prueba', 'x': pt_out.x, 'y': pt_out.y},
            {'tesela': 'EXT_001', 'zona': 'Z3', 'conjunto': 'prueba_externa', 'x': pt_out.x, 'y': pt_out.y},
        ]
        errs = ppz.comprobar_teselas_en_provincia(filas, raiz=RAIZ)
        self.assertEqual(len(errs), 1)
        self.assertIn('OUT_001', errs[0])

    def test_encadenamiento_y_reglas_sinteticas(self):
        """Tres teselas encadenadas (A-B-C a 2 km) van a una sola zona; descartar y rebajar se respetan."""
        with tempfile.TemporaryDirectory() as tmp:
            # A en x=750000, B en x=752000 (<3560m de A), C en x=754000 (<3560m de B pero >3560m de A)
            # D lejos en x=770000
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_1_001', 'image.tif'), 750000, 8600000)
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_2_001', 'image.tif'), 752000, 8600000)
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_3_001', 'image.tif'), 754000, 8600000)
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_37418_001', 'image.tif'), 770000, 8600000)
            with open(os.path.join(tmp, 'datos_laconvencion', 'revision_teselas.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'distrito', 'uso_propuesto'])
                w.writeheader()
                w.writerows([
                    {'tesela': 'sinpad_1_001', 'distrito': 'SANTA TERESA', 'uso_propuesto': 'ronda1'},
                    {'tesela': 'sinpad_2_001', 'distrito': 'SANTA TERESA', 'uso_propuesto': 'validacion'},
                    {'tesela': 'sinpad_3_001', 'distrito': 'SANTA TERESA', 'uso_propuesto': 'descartar'},
                    {'tesela': 'sinpad_37418_001', 'distrito': 'SANTA ANA', 'uso_propuesto': 'ronda1'},
                ])
            r = subprocess.run(
                [sys.executable, os.path.join(AQUI, 'particion_por_zona.py'), '--raiz', tmp],
                capture_output=True, text=True, check=True
            )
            with open(os.path.join(tmp, 'particion.csv'), encoding='utf-8') as f_p:
                filas = {row['tesela']: row for row in csv.DictReader(f_p)}
            # A, B, C comparten zona por encadenamiento
            self.assertEqual(filas['sinpad_1_001']['zona'], filas['sinpad_2_001']['zona'])
            self.assertEqual(filas['sinpad_2_001']['zona'], filas['sinpad_3_001']['zona'])
            # La zona tiene ronda1 -> entrena (incluso sinpad_2_001 que era validacion en la guía), y sinpad_3_001 queda excluida
            self.assertEqual(filas['sinpad_1_001']['conjunto'], 'entrenamiento')
            self.assertEqual(filas['sinpad_2_001']['conjunto'], 'entrenamiento')
            self.assertIn('era de validación', filas['sinpad_2_001']['nota'])
            self.assertEqual(filas['sinpad_3_001']['conjunto'], 'excluida')
            # El ancla sinpad_37418_001 va a validación aunque sea ronda1
            self.assertEqual(filas['sinpad_37418_001']['conjunto'], 'validacion')


class TestMascarasDesdeMarcas(unittest.TestCase):
    def test_mascaras_nivel1_y_regla_nubes(self):
        """Comprueba nivel 1: suelo claro dentro de cicatriz no se borra por nube, dudoso->255, negativo->0, nota 2025."""
        from rasterio.warp import transform
        with tempfile.TemporaryDirectory() as tmp:
            # Tesela 1: con cicatriz cuyo centro tiene suelo muy claro (B2=1300 > 1000) y una nube fuera (B2=1500)
            t1_path = os.path.join(tmp, 'datos_laconvencion', 'images', 'tes_pos', 'image.tif')
            tr = _crear_tesela_sintetica(t1_path, 750000, 8600000)
            with rasterio.open(t1_path, 'r+') as ds:
                b2 = ds.read(2)
                b2[20:40, 20:40] = 1300.0   # suelo brillante DENTRO del deslizamiento (filas 10..50, cols 10..50)
                b2[200:220, 200:220] = 1500.0  # nube blanca FUERA del deslizamiento
                ds.write(b2, 2)

            # Tesela 1 en 2025 (para probar nota="2025")
            t1_2025 = os.path.join(tmp, 'datos_laconvencion_2025', 'images', 'tes_pos', 'image.tif')
            _crear_tesela_sintetica(t1_2025, 750000, 8600000)

            # Tesela 2: negativo confirmado (revisada=si, sin polígonos)
            t2_path = os.path.join(tmp, 'datos_laconvencion', 'images', 'tes_neg', 'image.tif')
            _crear_tesela_sintetica(t2_path, 760000, 8600000)

            # Tesela 3: no revisada (revisada=no) -> no debe generar máscara
            t3_path = os.path.join(tmp, 'datos_laconvencion', 'images', 'tes_norev', 'image.tif')
            _crear_tesela_sintetica(t3_path, 770000, 8600000)

            with open(os.path.join(tmp, 'datos_laconvencion', 'revisadas.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'revisada', 'quien', 'nota'])
                w.writeheader()
                w.writerows([
                    {'tesela': 'tes_pos', 'revisada': 'si', 'quien': 'VN', 'nota': ''},
                    {'tesela': 'tes_neg', 'revisada': 'si', 'quien': 'VN', 'nota': 'negativo limpio'},
                    {'tesela': 'tes_norev', 'revisada': '', 'quien': '', 'nota': ''},
                ])

            with open(os.path.join(tmp, 'particion.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'proyecto', 'zona', 'conjunto', 'uso'])
                w.writeheader()
                w.writerows([
                    {'tesela': 'tes_pos', 'proyecto': 'datos_laconvencion', 'zona': 'Z01', 'conjunto': 'entrenamiento', 'uso': 'ronda1'},
                    {'tesela': 'tes_neg', 'proyecto': 'datos_laconvencion', 'zona': 'Z02', 'conjunto': 'validacion', 'uso': 'validacion'},
                ])

            def caja_wgs84(f0, f1, c0, c1):
                xs = [750000 + c0 * 10, 750000 + c1 * 10, 750000 + c1 * 10, 750000 + c0 * 10, 750000 + c0 * 10]
                ys = [8600000 - f0 * 10, 8600000 - f0 * 10, 8600000 - f1 * 10, 8600000 - f1 * 10, 8600000 - f0 * 10]
                lons, lats = transform('EPSG:32718', 'EPSG:4326', xs, ys)
                return {'type': 'Polygon', 'coordinates': [list(zip(lons, lats))]}

            gj = {
                'type': 'FeatureCollection',
                'features': [
                    # Cicatriz principal en filas 10..50, cols 10..50
                    {'type': 'Feature', 'properties': {'clase': 'deslizamiento', 'nota': ''}, 'geometry': caja_wgs84(10, 50, 10, 50)},
                    # Zona dudosa en filas 70..90, cols 70..90
                    {'type': 'Feature', 'properties': {'clase': 'dudoso', 'nota': ''}, 'geometry': caja_wgs84(70, 90, 70, 90)},
                    # Cicatriz exclusiva de 2025 en filas 140..160, cols 140..160
                    {'type': 'Feature', 'properties': {'clase': 'deslizamiento', 'nota': 'solo en 2025'}, 'geometry': caja_wgs84(140, 160, 140, 160)},
                ]
            }
            ruta_marcas = os.path.join(tmp, 'datos_laconvencion', 'marcas.geojson')
            with open(ruta_marcas, 'w', encoding='utf-8') as f_m:
                json.dump(gj, f_m)

            subprocess.run(
                [sys.executable, os.path.join(AQUI, 'mascaras_desde_marcas.py'), '--raiz', tmp],
                capture_output=True, text=True, check=True
            )

            # Comprobar tes_pos base
            with rasterio.open(os.path.join(tmp, 'mascaras_nivel1', 'mascaras_256', 'datos_laconvencion', 'tes_pos.tif')) as d_pos:
                m_pos = d_pos.read(1)
            # El centro brillante (25, 25) con B2=1300 dentro del deslizamiento DEBE valer 1, no 255
            self.assertEqual(m_pos[25, 25], 1, 'La regla de nubes borró el suelo desnudo brillante dentro de la cicatriz')
            # El polígono dudoso (80, 80) DEBE valer 255
            self.assertEqual(m_pos[80, 80], 255)
            # La nube fuera del deslizamiento (210, 210) DEBE valer 255
            self.assertEqual(m_pos[210, 210], 255)
            # El polígono de 2025 (150, 150) DEBE valer 255 en la imagen base
            self.assertEqual(m_pos[150, 150], 255)
            # Y en la máscara de 2025 DEBE valer 1, mientras la cicatriz base vale 255
            with rasterio.open(os.path.join(tmp, 'mascaras_nivel1', 'mascaras_256', 'datos_laconvencion_2025', 'tes_pos.tif')) as d_25:
                m_25 = d_25.read(1)
            self.assertEqual(m_25[150, 150], 1)
            self.assertEqual(m_25[25, 25], 255)

            # Comprobar negativo confirmado: todo 0
            with rasterio.open(os.path.join(tmp, 'mascaras_nivel1', 'mascaras_256', 'datos_laconvencion', 'tes_neg.tif')) as d_neg:
                m_neg = d_neg.read(1)
            self.assertTrue((m_neg == 0).all())

            # Comprobar que tes_norev NO existe
            self.assertFalse(os.path.exists(os.path.join(tmp, 'mascaras_nivel1', 'mascaras_256', 'datos_laconvencion', 'tes_norev.tif')))

            # Comprobar que los 4 cuadrantes de 128x128 se crearon
            for q in ('q1', 'q2', 'q3', 'q4'):
                pq = os.path.join(tmp, 'mascaras_nivel1', 'parches_128', 'datos_laconvencion', f'tes_pos_{q}.tif')
                self.assertTrue(os.path.exists(pq))
                with rasterio.open(pq) as dq:
                    self.assertEqual(dq.shape, (128, 128))


class TestReglaNubesYAnomalias(unittest.TestCase):
    def test_es_nube_blanca_distingue_suelo_claro(self):
        from candidatos_para_qgis import es_nube_blanca
        # Nube blanca (B2=1400, B3=1450, B4=1400 -> B4/B2 = 1.0)
        b2 = np.array([1400.0, 1200.0, 400.0, 2500.0], dtype=np.float32)
        b3 = np.array([1450.0, 1600.0, 500.0, 2600.0], dtype=np.float32)
        b4 = np.array([1400.0, 1900.0, 600.0, 3100.0], dtype=np.float32)
        res = es_nube_blanca(b2, b3, b4)
        # Pixel 0: nube blanca -> True
        self.assertTrue(res[0])
        # Pixel 1: suelo desnudo claro tipo Sillapata (B2=1200, B3=1600, B4=1900) -> False
        self.assertFalse(res[1])
        # Pixel 2: suelo normal (B2=400) -> False
        self.assertFalse(res[2])
        # Pixel 3: saturación extrema / nieve / nube gruesa (B2=2500 > 2000) -> True
        self.assertTrue(res[3])


class TestAjusteFinoComponentes(unittest.TestCase):
    def test_cargar_locales_y_bootstrap_zonas(self):
        from ajuste_fino import bootstrap_zonas, cargar_locales, regla_suelo_pendiente, resumen_local
        from cortar_parches_128 import escribir_parches
        from entrenar import receta_de
        from nucleo import config

        stats = {
            'B01': {'media': 0.5, 'desv': 0.2},
            'B02': {'media': 0.5, 'desv': 0.2},
            'B03': {'media': 0.5, 'desv': 0.2},
            'B04': {'media': 0.5, 'desv': 0.2},
            'pendiente': {'media': 1.0, 'desv': 0.5},
            'elevacion': {'media': 2.0, 'desv': 1.0},
        }
        receta = receta_de(['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'], stats)
        with tempfile.TemporaryDirectory() as tmp:
            t_path = os.path.join(tmp, 'img.tif')
            tr = _crear_tesela_sintetica(t_path)
            with rasterio.open(t_path) as ds:
                x = ds.read()[:6]
                perfil = ds.profile
            escribir_parches(x, perfil, os.path.join(tmp, 'parches_128', 'datos_laconvencion'), 'tes1')
            mask = np.zeros((1, 256, 256), dtype=np.uint8)
            mask[0, 10:30, 10:30] = 1     # cicatriz en q1
            mask[0, 50:70, 50:70] = 255   # zona ignorada en q1
            escribir_parches(mask, perfil, os.path.join(tmp, 'mascaras', 'datos_laconvencion'), 'tes1')
            with open(os.path.join(tmp, 'parches_128', 'indice_parches.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['parche', 'proyecto', 'tesela', 'cuadrante', 'fecha', 'zona', 'conjunto', 'archivo'])
                w.writeheader()
                for q in ('q1', 'q2', 'q3', 'q4'):
                    w.writerow({
                        'parche': f'tes1_{q}', 'proyecto': 'datos_laconvencion', 'tesela': 'tes1',
                        'cuadrante': q, 'fecha': 'base', 'zona': 'Z01' if q in ('q1', 'q2') else 'Z02',
                        'conjunto': 'validacion', 'archivo': f'datos_laconvencion/tes1_{q}.tif'
                    })
            X, Y, V, filas = cargar_locales(
                os.path.join(tmp, 'parches_128'), os.path.join(tmp, 'mascaras'),
                receta, config.ESCALA_L4S, {'validacion'}
            )
            self.assertEqual(X.shape, (4, 6, 128, 128))
            # En q1 (índice 0), los píxeles 10:30, 10:30 son positivos y 50:70, 50:70 tienen V=0
            self.assertEqual(int(Y[0, 10:30, 10:30].sum()), 400)
            self.assertEqual(int(V[0, 50:70, 50:70].sum()), 0)

            # Probar resumen_local y bootstrap_zonas
            prob_base = np.full_like(Y, 0.1)
            prob_ajust = np.where(Y > 0, 0.8, 0.1)
            res_b = resumen_local(prob_base, Y, V, filas, 0.45)
            res_a = resumen_local(prob_ajust, Y, V, filas, 0.45)
            self.assertEqual(res_b['cicatrices'], 1)
            self.assertEqual(res_b['cicatrices_tocadas'], 0)
            self.assertEqual(res_a['cicatrices_tocadas'], 1)
            self.assertGreater(res_a['iou'], res_b['iou'])

            bz = bootstrap_zonas(prob_base, 0.45, prob_ajust, 0.45, Y, V, filas, n=200)
            self.assertEqual(bz['zonas'], 2)
            self.assertGreaterEqual(bz['ic95'][1], 0.0)

            # Probar comparador sin aprendizaje (GRVI <= 0, B4 >= 1.29 L4S, pendiente >= 10°)
            pred_regla = regla_suelo_pendiente(X, receta)
            self.assertEqual(pred_regla.shape, (4, 128, 128))
            # Como el sintético tiene B3=450, B4=300 (vegetación verde), la regla debe dar todo 0
            self.assertEqual(int(pred_regla.sum()), 0)

    def test_iris_128_exportar_e_importar(self):
        from PIL import Image
        from iris_128 import exportar_128, importar_128
        with tempfile.TemporaryDirectory() as tmp:
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_1_001', 'image.tif'))
            with open(os.path.join(tmp, 'particion.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'proyecto', 'zona', 'conjunto', 'uso'])
                w.writeheader()
                w.writerow({'tesela': 'sinpad_1_001', 'proyecto': 'datos_laconvencion', 'zona': 'Z01', 'conjunto': 'entrenamiento', 'uso': 'ronda1'})
            dir_iris = os.path.join(tmp, 'iris_128')
            n_exp = exportar_128(tmp, dir_iris, limite=1)
            self.assertEqual(n_exp, 4)
            # Simular marcado en IRIS 0-based (0=fondo, 1=deslizamiento, 2=dudoso) en q1 y 1-based en q2
            m_q1 = np.zeros((128, 128), dtype=np.uint8)
            m_q1[10:20, 10:20] = 1
            m_q1[30:40, 30:40] = 2
            Image.fromarray(m_q1).save(os.path.join(dir_iris, 'masks', 'datos_laconvencion__sinpad_1_001_q1.png'))

            m_q2 = np.ones((128, 128), dtype=np.uint8)
            m_q2[10:20, 10:20] = 2
            m_q2[30:40, 30:40] = 3
            Image.fromarray(m_q2).save(os.path.join(dir_iris, 'masks', 'datos_laconvencion__sinpad_1_001_q2.png'))

            dir_n1 = os.path.join(tmp, 'mascaras_nivel1')
            n_imp = importar_128(dir_iris, dir_n1, raiz=tmp)
            self.assertEqual(n_imp, 2)
            for q in ('q1', 'q2'):
                with rasterio.open(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', f'sinpad_1_001_{q}.tif')) as dm:
                    mk = dm.read(1)
                self.assertEqual(mk[15, 15], 1)
                self.assertEqual(mk[35, 35], 255)
                self.assertEqual(mk[80, 80], 0)
            # Verificar reconstrucción de máscara 256×256 y exportación de GeoJSON
            self.assertTrue(os.path.exists(os.path.join(dir_n1, 'mascaras_256', 'datos_laconvencion', 'sinpad_1_001.tif')))
            self.assertTrue(os.path.exists(os.path.join(dir_n1, 'marcas_desde_iris.geojson')))
            with rasterio.open(os.path.join(dir_n1, 'mascaras_256', 'datos_laconvencion', 'sinpad_1_001.tif')) as d256:
                m256 = d256.read(1)
            self.assertEqual(m256.shape, (256, 256))
            self.assertEqual(m256[15, 15], 1)      # q1 pintado
            self.assertEqual(m256[15, 128 + 15], 1)  # q2 pintado
            self.assertEqual(m256[200, 200], 255)  # q4 sin pintar -> 255 (ignorar)

    def test_iris_128_solo_importa_revisados_y_origen_por_quien(self):
        import hashlib, json as _json, time
        from PIL import Image
        from iris_128 import exportar_128, importar_128
        with tempfile.TemporaryDirectory() as tmp:
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_1_001', 'image.tif'))
            with open(os.path.join(tmp, 'particion.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'proyecto', 'zona', 'conjunto', 'uso'])
                w.writeheader()
                w.writerow({'tesela': 'sinpad_1_001', 'proyecto': 'datos_laconvencion', 'zona': 'Z01', 'conjunto': 'entrenamiento', 'uso': 'ronda1'})
            dir_iris = os.path.join(tmp, 'iris_128')
            exportar_128(tmp, dir_iris, limite=1)

            # q1 y q2 en masks/: solo q1 está en decisiones_agente.csv, q2 NO fue revisado
            m_q1 = np.zeros((128, 128), dtype=np.uint8)
            m_q1[10:20, 10:20] = 1
            p_q1 = os.path.join(dir_iris, 'masks', 'datos_laconvencion__sinpad_1_001_q1.png')
            Image.fromarray(m_q1).save(p_q1)

            m_q2 = np.zeros((128, 128), dtype=np.uint8)
            m_q2[20:30, 20:30] = 1
            p_q2 = os.path.join(dir_iris, 'masks', 'datos_laconvencion__sinpad_1_001_q2.png')
            Image.fromarray(m_q2).save(p_q2)

            # Simular propuestas_iniciales_hashes.json con fecha corte en el futuro para q2
            t_corte = time.time() - 10.0
            os.utime(p_q2, (t_corte - 50.0, t_corte - 50.0))
            with open(p_q2, 'rb') as fq2:
                sha_q2 = hashlib.sha256(fq2.read()).hexdigest()
            with open(os.path.join(dir_iris, 'propuestas_iniciales_hashes.json'), 'w', encoding='utf-8') as fh:
                _json.dump({'fecha_respaldo_unix': t_corte, 'archivos': {'datos_laconvencion__sinpad_1_001_q2.png': {'sha256': sha_q2}}}, fh)

            os.makedirs(os.path.join(tmp, 'informes'), exist_ok=True)
            with open(os.path.join(tmp, 'informes', 'decisiones_agente.csv'), 'w', newline='', encoding='utf-8') as fd:
                w = csv.DictWriter(fd, fieldnames=['parche', 'decision'])
                w.writeheader()
                w.writerow({'parche': 'sinpad_1_001_q1', 'decision': 'positivo'})

            dir_n1 = os.path.join(tmp, 'mascaras_nivel1')
            # 1) Importar como AGENTE: solo debe entrar q1, NO q2 (no está en decisiones_agente.csv)
            n_ag = importar_128(dir_iris, dir_n1, quien='AGENTE', raiz=tmp)
            self.assertEqual(n_ag, 1)
            self.assertTrue(os.path.exists(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', 'sinpad_1_001_q1.tif')))
            self.assertFalse(os.path.exists(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', 'sinpad_1_001_q2.tif')))
            with rasterio.open(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', 'sinpad_1_001_q1.tif')) as dm:
                self.assertEqual(dm.tags().get('origen'), 'agente en IRIS 128x128')

            # 2) Importar como humano VN: q2 tiene mtime <= t_corte y hash de propuesta -> se rechaza.
            #    q1 tiene mtime > t_corte y hash distinto -> se importa con origen humano.
            n_hum = importar_128(dir_iris, dir_n1, quien='VN', raiz=tmp)
            self.assertEqual(n_hum, 1)
            self.assertFalse(os.path.exists(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', 'sinpad_1_001_q2.tif')))
            with rasterio.open(os.path.join(dir_n1, 'parches_128', 'datos_laconvencion', 'sinpad_1_001_q1.tif')) as dm:
                self.assertEqual(dm.tags().get('origen'), 'marcado humano en IRIS 128x128')

            # 3) Ahora que sinpad_1_001 es de VN en mascaras_nivel1.csv, AGENTE no debe sobrescribirla
            with self.assertRaises(SystemExit):
                importar_128(dir_iris, dir_n1, quien='AGENTE', raiz=tmp)

    def test_iris_128_modo_ciego_sin_vistas_modelo(self):
        """Comprueba que --ciego omite 'Color natural con cajas' y 'Probabilidad del modelo' y pone a 0 bandas 7-8."""
        from iris_128 import exportar_128
        with tempfile.TemporaryDirectory() as tmp:
            _crear_tesela_sintetica(os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_9_001', 'image.tif'))
            with open(os.path.join(tmp, 'particion.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['tesela', 'proyecto', 'zona', 'conjunto', 'uso'])
                w.writeheader()
                w.writerow({'tesela': 'sinpad_9_001', 'proyecto': 'datos_laconvencion', 'zona': 'Z09', 'conjunto': 'prueba', 'uso': 'prueba'})
            dir_ciego = os.path.join(tmp, 'iris_ciego')
            n = exportar_128(tmp, dir_ciego, conjuntos=['prueba'], ciego=True)
            self.assertEqual(n, 4)
            cfg = json.load(open(os.path.join(dir_ciego, 'deslizamientos_128.json'), encoding='utf-8'))
            self.assertNotIn('Color natural con cajas', cfg['views'])
            self.assertNotIn('Probabilidad del modelo', cfg['views'])
            self.assertNotIn('probabilidad', cfg['view_groups'])
            with rasterio.open(os.path.join(dir_ciego, 'images', 'datos_laconvencion__sinpad_9_001_q1', 'image.tif')) as ds:
                self.assertEqual(float(ds.read(7).max()), 0.0)
                self.assertEqual(float(ds.read(8).max()), 0.0)

    def test_acuerdo_anotadores_geojson_e_iris(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            t_path = os.path.join(tmp, 'datos_laconvencion', 'images', 'sinpad_15525_001', 'image.tif')
            _crear_tesela_sintetica(t_path, 750000, 8600000)
            dir_a = os.path.join(tmp, 'iris_a')
            dir_b = os.path.join(tmp, 'iris_b')
            os.makedirs(os.path.join(dir_a, 'masks'), exist_ok=True)
            os.makedirs(os.path.join(dir_b, 'masks'), exist_ok=True)
            for q in ('q1', 'q2', 'q3', 'q4'):
                ma = np.zeros((128, 128), dtype=np.uint8)
                mb = np.zeros((128, 128), dtype=np.uint8)
                if q == 'q1':
                    ma[10:30, 10:30] = 1
                    mb[10:30, 15:35] = 1
                Image.fromarray(ma).save(os.path.join(dir_a, 'masks', f'datos_laconvencion__sinpad_15525_001_{q}.png'))
                Image.fromarray(mb).save(os.path.join(dir_b, 'masks', f'datos_laconvencion__sinpad_15525_001_{q}.png'))
            r = subprocess.run(
                [sys.executable, os.path.join(AQUI, 'acuerdo_anotadores.py'), dir_a, dir_b,
                 '--teselas', 'sinpad_15525_001', '--raiz', tmp],
                capture_output=True, text=True, check=True
            )
            self.assertIn('IoU entre anotadores 0.600', r.stdout)

    def test_proyecto_iris_real_y_vistas(self):
        """Comprueba el proyecto iris_128 real exportado y el renderizado de todas las vistas en IRIS."""
        from iris.project import project
        ruta_json = os.path.join(RAIZ, 'iris_128', 'deslizamientos_128.json')
        self.assertTrue(os.path.exists(ruta_json), 'Falta iris_128/deslizamientos_128.json')
        project.load_from(ruta_json)
        self.assertEqual(len(project.image_ids), 464)
        # El primer parche debe ser la primera tesela de acuerdo entre anotadores (sinpad_15525_001_q1)
        self.assertEqual(project.image_ids[0], 'datos_laconvencion__sinpad_15525_001_q1')
        self.assertEqual(project.get_start_image_id(), 'datos_laconvencion__sinpad_15525_001_q1')
        # Renderizar todas las vistas tipo imagen sobre un parche real
        for vname, v in project['views'].items():
            if v.get('type') == 'image':
                arr = project.render_image(project.image_ids[0], v)
                self.assertEqual(arr.shape, (128, 128, 3), f'Fallo shape en vista {vname}')
                self.assertEqual(arr.dtype, np.uint8, f'Fallo dtype en vista {vname}')

    def test_geomorfologia_tiers_y_poligonos_qgis(self):
        """Comprueba Vía 3 (CURVATURA/TPI/GRVI), pesos por Tier en cargar_locales y suavizado Chaikin/.qml."""
        from nucleo import recetas
        import ajuste_fino
        import exportar_poligonos_qgis as epq

        # 1. Vía 3: CURVATURA, TPI y GRVI en recetas
        r = {
            'nombre': 'test_geom', 'onnx': 'x.onnx',
            'canales': ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion', 'CURVATURA', 'TPI', 'GRVI'],
            'lado_px': 128, 'normalizacion': {'metodo': 'ninguna'}, 'salida': 'sigmoide', 'umbral': 0.5,
        }
        recetas.normalizar_nombres(r)
        recetas.validar(r)
        fuente = {c: np.random.default_rng(7).uniform(0.1, 1.0, (128, 128)).astype(np.float32)
                  for c in ('B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion')}
        t = recetas.armar(fuente, np.ones((128, 128), dtype=bool), r)
        self.assertEqual(t.shape, (9, 128, 128))
        self.assertTrue(np.isfinite(t).all())
        self.assertGreaterEqual(float(t[6:].min()), -1.0)
        self.assertLessEqual(float(t[6:].max()), 1.0)

        # 2. Suavizado Chaikin y estilo .qml para QGIS
        cuadrado = {'type': 'Polygon', 'coordinates': [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]}
        suave = epq._suavizar_geom_geojson(cuadrado)
        self.assertGreater(len(suave['coordinates'][0]), 5)
        self.assertEqual(suave['coordinates'][0][0], suave['coordinates'][0][-1])

        with tempfile.TemporaryDirectory() as tmp:
            ruta_qml = os.path.join(tmp, 'estilo.qml')
            epq._escribir_qml_qgis(ruta_qml)
            self.assertTrue(os.path.exists(ruta_qml))
            self.assertIn('tier_calidad', open(ruta_qml, encoding='utf-8').read())

            # 3. cargar_locales leyendo indice_parches_ampliado.csv + catalogo_parches_tiers.csv
            p_dir = os.path.join(tmp, 'parches_128')
            m_dir = os.path.join(tmp, 'n2', 'parches_128')
            os.makedirs(os.path.join(p_dir, 'datos_laconvencion'), exist_ok=True)
            os.makedirs(os.path.join(m_dir, 'datos_laconvencion'), exist_ok=True)
            tr = from_origin(750000.0, 8600000.0, 10.0, 10.0)
            arr6 = np.full((6, 128, 128), 500.0, dtype=np.float32)
            with rasterio.open(os.path.join(p_dir, 'datos_laconvencion', 't1_c0.tif'), 'w',
                               driver='GTiff', height=128, width=128, count=6, dtype='float32',
                               crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr6)
            mk1 = np.zeros((128, 128), dtype=np.uint8)
            with rasterio.open(os.path.join(m_dir, 'datos_laconvencion', 't1_c0.tif'), 'w',
                               driver='GTiff', height=128, width=128, count=1, dtype='uint8',
                               crs='EPSG:32718', transform=tr) as ds:
                ds.write(mk1, 1)
            with open(os.path.join(p_dir, 'indice_parches_ampliado.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['parche', 'proyecto', 'tesela', 'conjunto', 'zona', 'archivo'])
                w.writeheader()
                w.writerow({'parche': 't1_c0', 'proyecto': 'datos_laconvencion', 'tesela': 't1',
                            'conjunto': 'entrenamiento', 'zona': 'Z1', 'archivo': 'datos_laconvencion/t1_c0.tif'})
            with open(os.path.join(tmp, 'n2', 'catalogo_parches_tiers.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['proyecto', 'parche', 'peso_perdida'])
                w.writeheader()
                w.writerow({'proyecto': 'datos_laconvencion', 'parche': 't1_c0', 'peso_perdida': '0.8'})

            rec_base = {
                'nombre': 'base', 'onnx': 'b.onnx',
                'canales': ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'],
                'lado_px': 128, 'normalizacion': {'metodo': 'ninguna'}, 'salida': 'sigmoide', 'umbral': 0.5,
            }
            recetas.normalizar_nombres(rec_base)
            escala = {b: 1000.0 for b in ('B01', 'B02', 'B03', 'B04')}
            X, Y, V, filas = ajuste_fino.cargar_locales(p_dir, m_dir, rec_base, escala, {'entrenamiento'}, solo_con_positivos=True)
            self.assertEqual(len(X), 1)
            self.assertAlmostEqual(filas[0]['_peso_tier'], 0.8)
            # filtro por Tier: el catálogo de la prueba no trae columna «tier», así que ninguno es «oro»
            X0, *_ = ajuste_fino.cargar_locales(p_dir, m_dir, rec_base, escala, {'entrenamiento'},
                                                solo_con_positivos=True, tiers={'oro'})
            self.assertEqual(len(X0), 0)


    def test_cargar_l4spe_conjuntos_unidades_y_nan(self):
        """cargar_l4spe: filtra por conjunto, pasa pendiente de radianes a unidades L4S y marca los NaN como no válidos."""
        import json as _json
        from nucleo import recetas
        import ajuste_fino
        rng = np.random.default_rng(3)
        x = np.zeros((3, 6, 128, 128), np.float32)
        for k in range(4):
            x[:, k] = rng.uniform(300, 900, (3, 128, 128))
        x[:, 4] = 0.5                      # radianes
        x[:, 5] = 1500.0                   # metros
        x[1, 2, :4, :4] = np.nan
        y = np.zeros((3, 128, 128), np.uint8); y[:, 10:20, 10:20] = 1
        rec = {'nombre': 'base', 'onnx': 'b.onnx', 'canales': ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'],
               'lado_px': 128, 'normalizacion': {'metodo': 'ninguna'}, 'salida': 'sigmoide', 'umbral': 0.5}
        recetas.normalizar_nombres(rec)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, 'l4spe.npz')
            np.savez_compressed(ruta, x=x, y=y, punto=np.array(['point_0001', 'point_0002', 'point_0003']),
                                zona=np.array(['Z1', 'Z1', 'Z2']), conjunto=np.array(['entrenamiento', 'entrenamiento', 'validacion']),
                                escala_json=np.array(_json.dumps({'B01': 600., 'B02': 600., 'B03': 600., 'B04': 600.})))
            X, Y, V, F = ajuste_fino.cargar_l4spe(ruta, rec, {'entrenamiento'})
            self.assertEqual((len(X), len(F)), (2, 2))
            self.assertEqual(float(V[1, :4, :4].max()), 0.0)             # NaN -> no válido
            self.assertEqual(float(V[0].min()), 1.0)
            self.assertAlmostEqual(float(X[0, 4].mean()), float(np.degrees(0.5)) / 20.292, places=3)
            Xv, *_ = ajuste_fino.cargar_l4spe(ruta, rec, {'validacion'})
            self.assertEqual(len(Xv), 1)
            Xn, *_ = ajuste_fino.cargar_l4spe(ruta, rec, {'prueba'})
            self.assertEqual(len(Xn), 0)


    def test_revisar_mascaras_criterios(self):
        """Verifica que revisar_mascaras.py detecta los errores geométricos, físicos y de consistencia."""
        from revisar_mascaras import revisar
        with tempfile.TemporaryDirectory() as tmp:
            p_dir = os.path.join(tmp, 'parches_128', 'datos_laconvencion')
            m_dir = os.path.join(tmp, 'mascaras_nivel1', 'parches_128', 'datos_laconvencion')
            os.makedirs(p_dir, exist_ok=True)
            os.makedirs(m_dir, exist_ok=True)
            tr = from_origin(750000.0, 8600000.0, 10.0, 10.0)
            arr = np.zeros((6, 128, 128), dtype=np.float32)
            arr[1] = 300.0; arr[2] = 350.0; arr[3] = 500.0; arr[4] = 25.0; arr[5] = 1500.0
            arr[4, 20:40, 20:40] = 3.0       # pendiente < 8°
            arr[1, 50:60, 50:60] = 1500.0    # nube blanca (B2>1000, B4=500)
            with rasterio.open(os.path.join(p_dir, 't1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=6, dtype='float32', crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr)
            mk = np.zeros((128, 128), dtype=np.uint8)
            mk[5, 5] = 1                     # mancha < 3 px
            mk[25:35, 25:35] = 1             # sobre pendiente < 8°
            mk[52:58, 52:58] = 1             # sobre nube
            mk[0, 80:100] = 1                # línea recta tocando borde
            with rasterio.open(os.path.join(m_dir, 't1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=1, dtype='uint8', crs='EPSG:32718', transform=tr) as dm:
                dm.write(mk, 1)
            hall = revisar(tmp)
            tipos = {h['tipo'] for h in hall}
            self.assertIn('mancha_menor_3px', tipos)
            self.assertIn('pendiente_menor_8', tipos)
            self.assertIn('positivo_sobre_nube', tipos)
            self.assertIn('borde_linea_recta', tipos)

    def test_revisar_mascaras_ingemmet_bordes_y_grvi(self):
        """Verifica IoU>0.80 contra poligono INGEMMET, bordes geometricos rectos y GRVI>0.05."""
        from revisar_mascaras import revisar
        from pyproj import Transformer
        from shapely.geometry import box as shp_box
        from shapely.ops import transform as shp_transform

        with tempfile.TemporaryDirectory() as tmp:
            p_dir = os.path.join(tmp, 'parches_128', 'datos_ingemmet')
            m_dir = os.path.join(tmp, 'mascaras_nivel1', 'parches_128', 'datos_ingemmet')
            os.makedirs(p_dir, exist_ok=True)
            os.makedirs(m_dir, exist_ok=True)
            tr = from_origin(750000.0, 8600000.0, 10.0, 10.0)
            arr = np.zeros((6, 128, 128), dtype=np.float32)
            # B3 (verde)=700, B4 (rojo)=500 -> GRVI = +0.167 (> 0.05) en todo el parche
            arr[1] = 300.0; arr[2] = 700.0; arr[3] = 500.0; arr[4] = 28.0; arr[5] = 1500.0
            with rasterio.open(os.path.join(p_dir, 'ing_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=6, dtype='float32', crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr)
            mk = np.zeros((128, 128), dtype=np.uint8)
            # Rectangulo de bordes rectos largos (30x30 px) que coincide exactamente con el poligono INGEMMET
            mk[30:60, 40:70] = 1
            with rasterio.open(os.path.join(m_dir, 'ing_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=1, dtype='uint8', crs='EPSG:32718', transform=tr) as dm:
                dm.write(mk, 1)

            tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True).transform
            x0, y1 = tr * (40, 30)
            x1, y0 = tr * (70, 60)
            pol_ll = shp_transform(tr_ll, shp_box(x0 + 1, y0 + 1, x1 - 1, y1 - 1))
            ruta_ing = os.path.join(tmp, 'ing_pols.geojson')
            with open(ruta_ing, 'w', encoding='utf-8') as f:
                json.dump({'type': 'FeatureCollection', 'features': [
                    {'type': 'Feature', 'properties': {}, 'geometry': pol_ll.__geo_interface__}
                ]}, f)

            hall = revisar(tmp, ingemmet_geojson=ruta_ing)
            tipos = {h['tipo'] for h in hall}
            self.assertIn('iou_ingemmet_alto', tipos)
            self.assertIn('bordes_geometricos', tipos)
            self.assertIn('grvi_verde_positivo', tipos)

    def test_revisar_mascaras_cauce_nube_y_carretera(self):
        """Verifica controles v3: positivo_en_cauce, blanco_saturado_cerca_nube y positivo_paralelo_carretera."""
        from revisar_mascaras import revisar
        from pyproj import Transformer
        from shapely.geometry import LineString
        from shapely.ops import transform as shp_transform

        with tempfile.TemporaryDirectory() as tmp:
            p_dir = os.path.join(tmp, 'parches_128', 'datos_prueba')
            m_dir = os.path.join(tmp, 'mascaras_nivel1', 'parches_128', 'datos_prueba')
            os.makedirs(p_dir, exist_ok=True)
            os.makedirs(m_dir, exist_ok=True)
            tr = from_origin(750000.0, 8600000.0, 10.0, 10.0)

            # 1) Parche cauce: valle encajonado horizontal en fila 40..46
            arr1 = np.zeros((6, 128, 128), dtype=np.float32)
            arr1[1] = 350.0; arr1[2] = 450.0; arr1[3] = 550.0; arr1[4] = 22.0; arr1[5] = 1600.0
            yy, _ = np.mgrid[0:128, 0:128]
            arr1[5] = 1600.0 + 4.0 * np.abs(yy - 43.0)
            arr1[4][40:47, :] = 10.0
            with rasterio.open(os.path.join(p_dir, 'cauce_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=6, dtype='float32', crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr1)
            mk1 = np.zeros((128, 128), dtype=np.uint8)
            mk1[41:46, 20:90] = 1
            with rasterio.open(os.path.join(m_dir, 'cauce_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=1, dtype='uint8', crs='EPSG:32718', transform=tr) as dm:
                dm.write(mk1, 1)

            # 2) Parche nube + blanco saturado cercano
            arr2 = np.zeros((6, 128, 128), dtype=np.float32)
            arr2[1] = 300.0; arr2[2] = 450.0; arr2[3] = 600.0; arr2[4] = 32.0; arr2[5] = 1800.0
            arr2[1, 60:72, 20:32] = 1200.0; arr2[2, 60:72, 20:32] = 1250.0; arr2[3, 60:72, 20:32] = 1260.0  # nube
            arr2[1, 55:65, 42:52] = 1100.0; arr2[2, 55:65, 42:52] = 1250.0; arr2[3, 55:65, 42:52] = 1400.0  # blanco sat
            with rasterio.open(os.path.join(p_dir, 'nube_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=6, dtype='float32', crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr2)
            mk2 = np.zeros((128, 128), dtype=np.uint8)
            mk2[55:65, 42:52] = 1
            with rasterio.open(os.path.join(m_dir, 'nube_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=1, dtype='uint8', crs='EPSG:32718', transform=tr) as dm:
                dm.write(mk2, 1)

            # 3) Parche carretera paralela a < 3 px
            arr3 = np.zeros((6, 128, 128), dtype=np.float32)
            arr3[1] = 400.0; arr3[2] = 500.0; arr3[3] = 750.0; arr3[4] = 34.0; arr3[5] = 2000.0
            with rasterio.open(os.path.join(p_dir, 'via_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=6, dtype='float32', crs='EPSG:32718', transform=tr) as ds:
                ds.write(arr3)
            mk3 = np.zeros((128, 128), dtype=np.uint8)
            mk3[20:24, 30:85] = 1
            with rasterio.open(os.path.join(m_dir, 'via_1_q1.tif'), 'w', driver='GTiff',
                               height=128, width=128, count=1, dtype='uint8', crs='EPSG:32718', transform=tr) as dm:
                dm.write(mk3, 1)

            tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True).transform
            x_a, y_a = tr * (25.0, 22.0)
            x_b, y_b = tr * (90.0, 22.0)
            via_ll = shp_transform(tr_ll, LineString([(x_a, y_a), (x_b, y_b)]))
            ruta_vias = os.path.join(tmp, 'vias.geojson')
            with open(ruta_vias, 'w', encoding='utf-8') as f:
                json.dump({'type': 'FeatureCollection', 'features': [
                    {'type': 'Feature', 'properties': {}, 'geometry': via_ll.__geo_interface__}
                ]}, f)

            hall = revisar(tmp, carreteras_geojson=ruta_vias)
            tipos = {h['tipo'] for h in hall}
            self.assertIn('positivo_en_cauce', tipos)
            self.assertIn('blanco_saturado_cerca_nube', tipos)
            self.assertIn('positivo_paralelo_carretera', tipos)

    def test_indice_parches_nivel1_560_304_360_y_nivel2_sin_fuga(self):
        """A.4 y B: Verifica que indice_parches_ampliado.csv cargue exactamente 560 / 304 / 360 en Nivel 1,
        que verificar_conteos_nivel1 aborte con SystemExit si falta algún parche, que 'excluida'/'pendiente'/'sin_zona'
        jamás se carguen, y que ninguna tesela de Nivel 2 en entrenamiento esté fuera de entrenamiento o a < 3 km
        de validacion, prueba o prueba_externa."""
        import ajuste_fino as af
        from shapely.geometry import box as shp_box
        from shapely.ops import unary_union

        part = {(r['proyecto'], r['tesela']): r for r in csv.DictReader(open('particion.csv', encoding='utf-8'))}
        part_tes = {r['tesela']: r for r in part.values()}

        # 1) Verificar que indice_parches_ampliado.csv coincida por tesela con particion.csv y sume 560/304/360/252 en Nivel 1
        filas_amp = list(csv.DictReader(open('parches_128/indice_parches_ampliado.csv', encoding='utf-8')))
        n1_por_c = {'entrenamiento': [], 'validacion': [], 'prueba': [], 'prueba2': []}
        for r in filas_amp:
            if r['tesela'] in part_tes:
                self.assertEqual(r['conjunto'], part_tes[r['tesela']]['conjunto'],
                                 f"Desajuste de conjunto en {r['parche']}: {r['conjunto']} vs {part_tes[r['tesela']]['conjunto']}")
            m1 = os.path.join('mascaras_nivel1', 'parches_128', r['proyecto'], r['parche'] + '.tif')
            if os.path.exists(m1) and r['conjunto'] in n1_por_c:
                n1_por_c[r['conjunto']].append(r)

        metodos_n1 = {}
        for r_d in csv.DictReader(open('informes/decisiones_agente.csv', encoding='utf-8')):
            metodos_n1[r_d['parche']] = r_d['metodo']
            metodos_n1[r_d['id_iris']] = r_d['metodo']

        self.assertTrue(af.verificar_conteos_nivel1(
            n1_por_c['entrenamiento'], n1_por_c['validacion'], n1_por_c['prueba'], metodos_n1,
            fl_pru2=n1_por_c['prueba2']
        ))
        with self.assertRaises(SystemExit):
            af.verificar_conteos_nivel1(
                n1_por_c['entrenamiento'][:-1], n1_por_c['validacion'], n1_por_c['prueba'], metodos_n1,
                fl_pru2=n1_por_c['prueba2']
            )
        with self.assertRaises(SystemExit):
            af.verificar_conteos_nivel1(
                n1_por_c['entrenamiento'], n1_por_c['validacion'], n1_por_c['prueba'], metodos_n1,
                fl_pru2=n1_por_c['prueba2'][:-1]
            )

        # 2) Verificar que Nivel 2 en entrenamiento esté 100% en entrenamiento en particion.csv y a >= 3000m de eval
        geoms_eval = []
        for (proy, tes), r in part.items():
            if r['conjunto'] in ('validacion', 'prueba', 'prueba2', 'prueba_externa'):
                cx, cy = float(r['x']), float(r['y'])
                geoms_eval.append(shp_box(cx - 1280.0, cy - 1280.0, cx + 1280.0, cy + 1280.0))
        u_eval = unary_union(geoms_eval)

        for r_t in csv.DictReader(open('mascaras_provisionales/nivel2/catalogo_parches_tiers.csv', encoding='utf-8')):
            if r_t['conjunto'] == 'entrenamiento' and float(r_t.get('peso_perdida') or 0) > 0:
                info = part.get((r_t['proyecto'], r_t['tesela'])) or part_tes.get(r_t['tesela'])
                self.assertIsNotNone(info, f"Tesela N2 {r_t['tesela']} no existe en particion.csv")
                self.assertEqual(info['conjunto'], 'entrenamiento',
                                 f"Tesela N2 {r_t['tesela']} tiene conjunto {info['conjunto']} en particion.csv")
                cx, cy = float(info['x']), float(info['y'])
                g_t = shp_box(cx - 1280.0, cy - 1280.0, cx + 1280.0, cy + 1280.0)
                self.assertGreaterEqual(g_t.distance(u_eval), 2995.0,
                                        f"Tesela N2 {r_t['tesela']} a < 3 km de eval")

    def test_peso_pequenas_y_copiar_pegar_solo_manual(self):
        """Verifica 10a (peso extra solo en <50 px) y 10b (banco copy-paste exclusivamente manual_v4)."""
        import ajuste_fino as af
        X = np.zeros((3, 6, 32, 32), dtype=np.float32)
        X[:, 4] = 25.0   # pendiente
        X[:, 5] = 1500.0 # elevacion
        Y = np.zeros((3, 32, 32), dtype=np.float32)
        V = np.ones((3, 32, 32), dtype=np.float32)
        # Parche 0 (manual_v4): cicatriz pequeña 4x4 = 16 px
        Y[0, 5:9, 5:9] = 1.0
        X[0, :4, 5:9, 5:9] = 9.0
        # Parche 1 (clasificador_v5, NO elegible): cicatriz pequeña 5x5 = 25 px
        Y[1, 10:15, 10:15] = 1.0
        X[1, :4, 10:15, 10:15] = 7.0
        # Parche 2 (manual_v4): cicatriz grande 10x10 = 100 px (>50 px)
        Y[2, 10:20, 10:20] = 1.0
        W = af.mapa_peso_pequenas(Y, peso=3.0, max_px=50)
        self.assertEqual(float(W[0, 6, 6]), 3.0)
        self.assertEqual(float(W[0, 0, 0]), 1.0)
        self.assertEqual(float(W[2, 12, 12]), 1.0)
        banco = af.banco_cicatrices_pequenas(X, Y, V, np.array([True, False, True]), min_px=10, max_px=60)
        self.assertEqual(len(banco), 1)
        self.assertEqual(banco[0][2], 16)
        # Aplicar copy-paste sobre parche negativo con región dudosa V=0
        xb = np.zeros((1, 6, 32, 32), dtype=np.float32)
        xb[0, 4] = 30.0; xb[0, 5] = 1800.0
        yb = np.zeros((1, 32, 32), dtype=np.float32)
        vb = np.ones((1, 32, 32), dtype=np.float32)
        vb[0, :10, :] = 0.0
        xb2, yb2, vb2 = af.aplicar_copiar_pegar(xb, yb, vb, banco, np.random.default_rng(7), prob=1.0, max_pegar=1, peso_pequenas=2.0)
        self.assertEqual(int((yb2 > 0).sum()), 16)
        self.assertEqual(int(((yb2 > 0) & (vb == 0)).sum()), 0)
        self.assertTrue(np.allclose(xb2[0, 4], 30.0))
        self.assertTrue(np.allclose(xb2[0, 5], 1800.0))
        self.assertEqual(float(vb2[yb2 > 0].min()), 2.0)

    def test_posproceso_v6_filtro_altoandino_y_ensamble(self):
        """Verifica filtro altoandino WorldCover+DEM y ensamble DeepLab gated by U-Net."""
        import posproceso_v6 as pp6
        p_u = np.zeros((1, 16, 16), dtype=np.float32)
        p_d = np.zeros((1, 16, 16), dtype=np.float32)
        p_u[0, 2:5, 2:5] = 0.8
        p_d[0, 2:6, 2:6] = 0.7   # tocado por U-Net -> se conserva
        p_d[0, 10:13, 10:13] = 0.9  # aislado sin U-Net -> se filtra
        g = pp6.combinar_ensamble(p_u, p_d, modo='deeplab_gated_by_unet', u_gate=0.2)
        self.assertAlmostEqual(float(g[0, 3, 3]), 0.7, places=5)
        self.assertEqual(float(g[0, 11, 11]), 0.0)
        elev = np.full((1, 16, 16), 3500.0, dtype=np.float32)
        elev[0, :8, :] = 1500.0
        wc = np.full((1, 16, 16), 30, dtype=np.uint8)  # grassland
        p_in = np.full((1, 16, 16), 0.6, dtype=np.float32)
        p_f = pp6.aplicar_filtro_altoandino(p_in, elev, wc, modo='wc_dem_3200')
        self.assertAlmostEqual(float(p_f[0, 2, 2]), 0.6, places=5)
        self.assertEqual(float(p_f[0, 12, 12]), 0.0)

    def test_mejora_v8_histeresis_y_morfologia(self):
        """Verifica umbral con histeresis 8-conectada y post-proceso por objeto (relleno + min_px)."""
        import mejora_v8_l4spe as mv8
        p = np.zeros((1, 20, 20), dtype=np.float32)
        # Componente 1 (6x6 = 36 px con hueco central): borde a 0.25, semilla 1 px a 0.65
        p[0, 2:8, 2:8] = 0.25
        p[0, 4:6, 4:6] = 0.0   # hueco interno 2x2
        p[0, 2, 2] = 0.65      # semilla >= u_alto
        # Componente 2 aislado (4x4 = 16 px) solo a 0.35 (< u_alto=0.50) -> debe descartarse
        p[0, 12:16, 12:16] = 0.35
        # Componente 3 chico (2x2 = 4 px) a 0.70 -> pasa histeresis pero cae por min_px=10
        p[0, 17:19, 2:4] = 0.70

        mh = mv8.umbral_histeresis(p, u_alto=0.50, u_bajo=0.20)
        self.assertEqual(int(mh[0, 2:8, 2:8].sum()), 32)   # 36 - 4 del hueco
        self.assertEqual(int(mh[0, 12:16, 12:16].sum()), 0)
        self.assertEqual(int(mh[0, 17:19, 2:4].sum()), 4)

        mm = mv8.postproceso_objeto(mh, min_px=10, fill_holes=True, closing_1px=False)
        self.assertEqual(int(mm[0, 2:8, 2:8].sum()), 36)   # hueco rellenado
        self.assertEqual(int(mm[0, 17:19, 2:4].sum()), 0)  # < 10 px eliminado


if __name__ == '__main__':
    unittest.main(verbosity=2)
