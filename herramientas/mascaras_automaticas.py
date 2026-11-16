"""
Máscaras AUTOMÁTICAS de nivel 3: solo pistas para buscar cicatrices. NO USAR PARA ENTRENAR.

Pasa a ráster lo que ya calcularon candidatos_para_qgis.py y cambios_para_qgis.py, sin
revisión de nadie. Mezclan cicatrices con chacras nuevas, caminos, cauces y pastos secos.
No incluye la probabilidad de la U-Net: entrenar con ella sería enseñarle al modelo sus
propios errores.

Valores (uint8, misma malla que la tesela):
    0  nada
    1  suelo desnudo en ladera (regla de color y pendiente)
    2  cambio: tenía vegetación antes y ahora es suelo desnudo en ladera (pista más fuerte)

  python herramientas/mascaras_automaticas.py

Escribe mascaras_provisionales/nivel3/<proyecto>/<tesela>.tif y mascaras_nivel3.csv.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

PROYECTOS = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    a = ap.parse_args()
    import rasterio
    from rasterio.features import rasterize
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_geom

    sal = os.path.join(a.raiz, 'mascaras_provisionales', 'nivel3')
    filas = []
    for proy in PROYECTOS:
        capas = collections.defaultdict(lambda: {1: [], 2: []})
        for archivo, nivel, valor in (('candidatos.geojson', 'suelo_desnudo', 1), ('cambios.geojson', 'nuevo_desnudo', 2)):
            ruta = os.path.join(a.raiz, proy, 'qgis', archivo)
            if not os.path.exists(ruta):
                continue
            for f in json.load(open(ruta, encoding='utf-8'))['features']:
                if f['properties'].get('nivel') == nivel:
                    capas[f['properties']['tesela']][valor].append(f['geometry'])
        carpeta = os.path.join(a.raiz, proy, 'images')
        if not os.path.isdir(carpeta):
            continue
        os.makedirs(os.path.join(sal, proy), exist_ok=True)
        for tes in sorted(os.listdir(carpeta)):
            ruta = os.path.join(carpeta, tes, 'image.tif')
            if not os.path.exists(ruta):
                continue
            with rasterio.open(ruta) as ds:
                tr, crs, forma = ds.transform, ds.crs, ds.shape
            m = np.zeros(forma, 'uint8')
            for valor in (1, 2):
                geos = [transform_geom('EPSG:4326', crs, g) for g in capas[tes][valor]]
                if geos:
                    m[rasterize([(g, 1) for g in geos], out_shape=forma, transform=tr, fill=0, dtype='uint8') == 1] = valor
            p = dict(driver='GTiff', height=forma[0], width=forma[1], count=1, dtype='uint8', crs=crs,
                     transform=tr, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p) as d:
                    d.write(m, 1)
                    d.update_tags(nivel='3', valores='0 nada, 1 suelo desnudo en ladera, 2 cambio a suelo desnudo',
                                  aviso='automática, sin revisar: NO USAR PARA ENTRENAR')
                open(os.path.join(sal, proy, tes + '.tif'), 'wb').write(mem.read())
            filas.append({'proyecto': proy, 'tesela': tes, 'ha_suelo_desnudo': round(float((m == 1).sum()) * 0.01, 2),
                          'ha_cambio': round(float((m == 2).sum()) * 0.01, 2)})
    with open(os.path.join(sal, 'mascaras_nivel3.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(sorted(filas, key=lambda r: -r['ha_cambio']))
    print(f'{len(filas)} máscaras de nivel 3; suelo desnudo {sum(r["ha_suelo_desnudo"] for r in filas):.0f} ha, '
          f'cambio {sum(r["ha_cambio"] for r in filas):.0f} ha')


if __name__ == '__main__':
    main()
