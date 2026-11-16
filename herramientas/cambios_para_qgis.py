"""
Busca cicatrices NUEVAS comparando dos temporadas secas de las mismas teselas.

Una ladera que en la imagen antigua tenía vegetación y en la nueva aparece como suelo desnudo,
con pendiente, es un candidato fuerte a deslizamiento reciente (o a una chacra nueva o un
camino nuevo: por eso una persona lo confirma). Es la señal más limpia que tenemos para
encontrar deslizamientos de verdad, porque no depende del modelo.

  python herramientas/cambios_para_qgis.py datos_laconvencion datos_laconvencion_2025

Escribe en <antes>/qgis/:
  cambios.geojson        polígonos nivel = nuevo_desnudo (vegetación → suelo) y revegetado
                         (suelo → vegetación), con area_ha, pend_media, caida_grvi, tesela
  <id>_cambio.tif        diferencia de verdor (GRVI nuevo menos GRVI antiguo): negativo = perdió verde
  cambios_resumen.csv    hectáreas de suelo nuevo por tesela, ordenado
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def grvi(x):
    return (x[2] - x[3]) / np.maximum(x[2] + x[3], 1)


def es_nube_blanca(x):
    b2, b3, b4 = x[1], x[2], x[3]
    suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
    return (b2 > 1000) & ~suelo_claro


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('antes', help='proyecto con la imagen antigua (por ejemplo 2018)')
    ap.add_argument('despues', help='proyecto con la imagen nueva (por ejemplo 2025)')
    ap.add_argument('--pendiente', type=float, default=10)
    ap.add_argument('--caida', type=float, default=0.12, help='caída mínima de GRVI para contar como pérdida de verde')
    ap.add_argument('--min-px', dest='min_px', type=int, default=5)
    ap.add_argument('--salida', help='carpeta de salida (por defecto <antes>/qgis)')
    a = ap.parse_args()

    import rasterio
    from rasterio.features import shapes
    from rasterio.io import MemoryFile
    from rasterio.warp import Resampling, reproject, transform_geom
    from scipy import ndimage

    salida = a.salida or os.path.join(a.antes, 'qgis')
    os.makedirs(salida, exist_ok=True)
    feats, filas = [], []
    for ruta in sorted(glob.glob(os.path.join(a.antes, 'images', '*', 'image.tif'))):
        idx = os.path.basename(os.path.dirname(ruta))
        ruta2 = os.path.join(a.despues, 'images', idx, 'image.tif')
        if not os.path.exists(ruta2):
            continue
        with rasterio.open(ruta) as d1:
            x1 = d1.read(); perfil = d1.profile; tr = d1.transform; crs = d1.crs
        with rasterio.open(ruta2) as d2:
            if d2.transform == tr and d2.shape == x1.shape[1:] and d2.crs == crs:
                x2 = d2.read()
            else:                                   # misma zona pero otra malla: se remuestrea
                x2 = np.zeros_like(x1)
                for k in range(x1.shape[0]):
                    reproject(rasterio.band(d2, k + 1), x2[k], dst_transform=tr, dst_crs=crs,
                              resampling=Resampling.bilinear)
        g1, g2 = grvi(x1), grvi(x2)
        pend = x1[4]
        ok = (x1[1] > 0) & (x2[1] > 0) & ~es_nube_blanca(x1) & ~es_nube_blanca(x2)      # con dato y sin nube blanca
        dif = np.where(ok, g2 - g1, np.nan).astype('float32')
        p1 = dict(perfil, count=1, dtype='float32', compress='deflate', nodata=-9)
        with MemoryFile() as mem:
            with mem.open(**p1) as d:
                d.write(np.nan_to_num(dif, nan=-9), 1)
            with open(os.path.join(salida, f'{idx}_cambio.tif'), 'wb') as f_cam_tif:
                f_cam_tif.write(mem.read())
        nuevo = ok & (g1 > 0.05) & (g2 <= 0.0) & ((g1 - g2) >= a.caida) & (x2[3] > 600) & (pend >= a.pendiente)
        reveg = ok & (g1 <= 0.0) & (g2 > 0.05) & ((g2 - g1) >= a.caida) & (pend >= a.pendiente)
        ha_nuevo = 0.0
        for mask, nivel in ((nuevo, 'nuevo_desnudo'), (reveg, 'revegetado')):
            mask = ndimage.binary_opening(mask, iterations=1)
            lab, n = ndimage.label(mask)
            if not n:
                continue
            tam = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
            for k, t in enumerate(tam, 1):
                if t < a.min_px:
                    lab[lab == k] = 0
            for geo, k in shapes(lab.astype('int32'), mask=lab > 0, transform=tr):
                z = lab == int(k)
                props = {'tesela': idx, 'nivel': nivel, 'area_ha': round(float(z.sum()) * 0.01, 2),
                         'pend_media': round(float(pend[z].mean()), 1),
                         'caida_grvi': round(float((g1[z] - g2[z]).mean()), 2),
                         'prob_modelo_max': round(float(x2[6][z].max()), 2)}
                if nivel == 'nuevo_desnudo':
                    ha_nuevo += props['area_ha']
                feats.append({'type': 'Feature', 'properties': props,
                              'geometry': transform_geom(crs, 'EPSG:4326', geo)})
        n_nuevo = sum(1 for f in feats if f['properties']['tesela'] == idx and f['properties']['nivel'] == 'nuevo_desnudo')
        filas.append({'tesela': idx, 'zonas_nuevo_desnudo': n_nuevo, 'ha_nuevo_desnudo': round(ha_nuevo, 2),
                      'sin_dato_o_nube_pct': round(float((~ok).mean()) * 100, 1)})
        print(f'{idx}: {n_nuevo} zonas nuevas de suelo desnudo en ladera, {ha_nuevo:.1f} ha'
              + (f' (sin comparar {(~ok).mean()*100:.0f} % por nubes)' if (~ok).mean() > 0.05 else ''))
    if not filas:
        raise SystemExit('no hay teselas en común entre los dos proyectos')
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(os.path.join(salida, 'cambios.geojson'), 'w'))
    filas.sort(key=lambda r: -r['ha_nuevo_desnudo'])
    with open(os.path.join(salida, 'cambios_resumen.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    print(f'\n{len(feats)} zonas de cambio en {os.path.join(os.path.abspath(salida), "cambios.geojson")}')


if __name__ == '__main__':
    main()
