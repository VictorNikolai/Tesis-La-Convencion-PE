"""
Exportador Vectorial Consolidado para QGIS y Google Earth (.geojson).

Vectoriza todas las cicatrices de deslizamiento aceptadas (`valor == 1`) de las máscaras
de Nivel 2 ampliado (INGEMMET geológico + Detección Multitemporal + Morfología),
enriqueciendo cada polígono con sus métricas geomorfológicas y su Tier de calidad.

Excluye automáticamente:
  - Las 3 teselas de acuerdo inter-anotador (`sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`).
  - Los proyectos pre-evento (`datos_indeci_antes`, `datos_ingemmet_antes`).

Escribe en `datos_laconvencion/qgis/`:
  - `cicatrices_consolidadas_qgis.geojson` (EPSG:4326 listo para abrir en QGIS)
  - `cicatrices_consolidadas_resumen.csv`  (tabla de atributos y resumen por zona)
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)

TESELAS_ACUERDO = {'sinpad_15525_001', 'sinpad_48878_001', 'sinpad_557_001'}


def _suavizar_anillo_chaikin(coords, iteraciones=2):
    """Suavizado cartográfico de Chaikin para eliminar bordes pixelados de 10 m conservando el área."""
    if len(coords) < 4:
        return coords
    pts = list(coords)
    cerrado = pts[0] == pts[-1]
    if cerrado:
        pts = pts[:-1]
    for _ in range(iteraciones):
        nuevo = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = [round(0.75 * p0[0] + 0.25 * p1[0], 6), round(0.75 * p0[1] + 0.25 * p1[1], 6)]
            r = [round(0.25 * p0[0] + 0.75 * p1[0], 6), round(0.25 * p0[1] + 0.75 * p1[1], 6)]
            nuevo.extend([q, r])
        pts = nuevo
    pts.append(pts[0])
    return pts


def _suavizar_geom_geojson(geom):
    """Aplica suavizado Chaikin a Polygon o MultiPolygon GeoJSON."""
    g = dict(geom)
    if g['type'] == 'Polygon':
        g['coordinates'] = [_suavizar_anillo_chaikin(ring) for ring in g['coordinates']]
    elif g['type'] == 'MultiPolygon':
        g['coordinates'] = [[_suavizar_anillo_chaikin(ring) for ring in poly] for poly in g['coordinates']]
    return g


def _escribir_qml_qgis(ruta_qml):
    """Genera un estilo QGIS (.qml) categorizado por tier_calidad para visualización inmediata."""
    qml = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="tier_calidad" symbollevels="0" enableorderby="0">
    <categories>
      <category value="oro" symbol="0" label="Tier ORO (Alta Certeza &gt;90%)" render="true"/>
      <category value="plata" symbol="1" label="Tier PLATA (Certeza Media 70-90%)" render="true"/>
      <category value="bronce" symbol="2" label="Tier BRONCE (Apoyo)" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="0.65">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="227,26,28,165"/>
            <Option type="QString" name="outline_color" value="140,0,0,255"/>
            <Option type="QString" name="outline_width" value="0.45"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.60">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="255,127,0,150"/>
            <Option type="QString" name="outline_color" value="180,80,0,255"/>
            <Option type="QString" name="outline_width" value="0.35"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="0.50">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="255,215,0,125"/>
            <Option type="QString" name="outline_color" value="160,120,0,255"/>
            <Option type="QString" name="outline_width" value="0.3"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
"""
    with open(ruta_qml, 'w', encoding='utf-8') as fq:
        fq.write(qml)


def ejecutar(raiz='.'):
    dir_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2')
    ruta_n2_comp = os.path.join(dir_n2, 'mascaras_nivel2_completo.csv')
    meta_tes = {}
    if os.path.exists(ruta_n2_comp):
        with open(ruta_n2_comp, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                meta_tes[(r['proyecto'], r['tesela'])] = r

    rutas_m256 = sorted(glob.glob(os.path.join(dir_n2, 'mascaras_256', '*', '*.tif')))
    features = []
    resumen_rows = []
    id_seq = 1

    for ruta_m in rutas_m256:
        proy = os.path.basename(os.path.dirname(ruta_m))
        tes = os.path.splitext(os.path.basename(ruta_m))[0]

        if proy in ('datos_indeci_antes', 'datos_ingemmet_antes'):
            continue
        if tes in TESELAS_ACUERDO:
            continue

        with rasterio.open(ruta_m) as dm:
            m = dm.read(1)
            tr = dm.transform
            crs = dm.crs

        if not (m == 1).any():
            continue

        ruta_img = os.path.join(raiz, proy, 'images', tes, 'image.tif')
        pend = None
        if os.path.exists(ruta_img):
            with rasterio.open(ruta_img) as di:
                pend = di.read(5)

        lab, n_comp = ndimage.label(m == 1)
        info = meta_tes.get((proy, tes), {})
        zona = info.get('zona', 'Z_GENERAL')
        conjunto = info.get('conjunto', 'entrenamiento')
        fuente = 'INGEMMET Geológico' if proy == 'datos_ingemmet' else 'Multitemporal + Morfología'

        for geom_crs, val in shapes(lab.astype(np.int32), mask=(lab > 0), transform=tr):
            k = int(val)
            comp = (lab == k)
            area_px = int(comp.sum())
            if area_px < 5:  # eliminar chispas menores a 0.05 ha
                continue

            area_ha = round(area_px * 0.01, 2)
            erod = ndimage.binary_erosion(comp)
            perim_px = max(int((comp & ~erod).sum()), 1)
            compacidad = round(area_px / perim_px, 2)

            p_med = round(float(pend[comp].mean()), 1) if pend is not None else 0.0
            p_max = round(float(pend[comp].max()), 1) if pend is not None else 0.0

            if area_px >= 20 and p_med >= 22.0 and p_max >= 26.0:
                tier = 'oro'
            elif area_px >= 12 and p_med >= 18.0:
                tier = 'plata'
            else:
                tier = 'bronce'

            geom_wgs84 = _suavizar_geom_geojson(transform_geom(crs, 'EPSG:4326', geom_crs))
            props = {
                'id_poligono': id_seq,
                'proyecto': proy,
                'tesela': tes,
                'zona': zona,
                'conjunto': conjunto,
                'area_ha': area_ha,
                'pend_media': p_med,
                'pend_max': p_max,
                'compacidad': compacidad,
                'tier_calidad': tier,
                'fuente': fuente,
            }
            features.append({
                'type': 'Feature',
                'properties': props,
                'geometry': geom_wgs84,
            })
            resumen_rows.append(props)
            id_seq += 1

    dir_qgis = os.path.join(raiz, 'datos_laconvencion', 'qgis')
    os.makedirs(dir_qgis, exist_ok=True)

    ruta_gj = os.path.join(dir_qgis, 'cicatrices_consolidadas_qgis.geojson')
    gj = {
        'type': 'FeatureCollection',
        'name': 'cicatrices_consolidadas_laconvencion',
        'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
        'features': features,
    }
    with open(ruta_gj, 'w', encoding='utf-8') as f:
        json.dump(gj, f, ensure_ascii=False)

    ruta_qml = os.path.join(dir_qgis, 'cicatrices_consolidadas_qgis.qml')
    _escribir_qml_qgis(ruta_qml)

    ruta_csv = os.path.join(dir_qgis, 'cicatrices_consolidadas_resumen.csv')
    if resumen_rows:
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(resumen_rows[0].keys()))
            w.writeheader()
            w.writerows(resumen_rows)

    tot_ha = sum(r['area_ha'] for r in resumen_rows)
    n_oro = sum(1 for r in resumen_rows if r['tier_calidad'] == 'oro')
    n_plata = sum(1 for r in resumen_rows if r['tier_calidad'] == 'plata')
    n_bronce = sum(1 for r in resumen_rows if r['tier_calidad'] == 'bronce')

    print(f"Capa vectorial consolidada exportada para QGIS:")
    print(f"  • Archivo GeoJSON: {ruta_gj}")
    print(f"  • Total polígonos de deslizamiento: {len(features)} polígonos ({tot_ha:.1f} ha)")
    print(f"  • Distribución por calidad: 🥇 {n_oro} Oro | 🥈 {n_plata} Plata | 🥉 {n_bronce} Bronce")
    return len(features)


if __name__ == '__main__':
    ejecutar()

