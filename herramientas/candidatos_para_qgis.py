"""
Dibuja, para cada tesela, dónde PUEDE haber un deslizamiento, como ayuda para marcar en QGIS.

No son etiquetas: son candidatos que una persona confirma, corrige o borra. Salen de dos fuentes:

  modelo     la probabilidad de la U-Net: ≥ umbral alto (0,45) = «probable»,
             entre el umbral bajo (0,30) y el alto = «posible»
  suelo      una regla sin modelo: suelo desnudo (más rojo que verde y claro, sin ser nube)
             sobre una ladera con pendiente ≥ 10°. Atrapa cicatrices que el modelo no ve,
             pero también chacras recién abiertas y caminos: hay que mirar.

Escribe en <proyecto>/qgis/:
  candidatos.geojson      polígonos con campos tesela, fuente (modelo | suelo), nivel
                          (probable | posible | suelo_desnudo), prob_max y area_ha
  <id>_prob.tif           la probabilidad del modelo (0 a 1), para verla como mapa de calor
  prioridad.csv           teselas ordenadas por cuánto hay que revisar (más candidatos primero)

En QGIS, si un candidato está bien, se copia a la capa «marcas»: seleccionarlo, Ctrl+C,
activar «marcas» en edición, Ctrl+V, y corregir la clase si hace falta.

  python herramientas/candidatos_para_qgis.py iris_laconvencion
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def es_nube_blanca(b2, b3, b4):
    """Nube blanca que la máscara SCL no quitó (B2 > 1000).
    No cuenta como nube el suelo desnudo muy claro (rojo > verde y B4 > 1,2 * B2, con B2 <= 2000),
    como el centro de las cicatrices de Sillapata y Mesacancha."""
    suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
    return (b2 > 1000) & ~suelo_claro


def poligonos(mascara, valores, pend, tr, crs, props_base, transform_geom, shapes, min_px, pend_min):
    """Un polígono por zona conexa. Las zonas sobre terreno llano (pendiente media < pend_min)
    se marcan nivel «<nivel>_plano»: casi siempre cauces, pastos secos o pueblos, no laderas."""
    from scipy import ndimage
    out = []
    lab, n = ndimage.label(mascara)
    if not n:
        return out
    tam = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    chicos = {k + 1 for k, t in enumerate(tam) if t < min_px}
    for k in chicos:
        lab[lab == k] = 0
    for geo, k in shapes(lab.astype('int32'), mask=lab > 0, transform=tr):
        k = int(k)
        zona = lab == k
        props = dict(props_base)
        props['prob_max'] = round(float(valores[zona].max()), 2) if valores is not None else None
        props['area_ha'] = round(float(zona.sum()) * 0.01, 2)      # píxel de 10 m = 0,01 ha
        props['pend_media'] = round(float(pend[zona].mean()), 1)
        props['pend_max'] = round(float(pend[zona].max()), 1)
        if props['pend_media'] < pend_min:
            props['nivel'] = props['nivel'] + '_plano'
        out.append({'type': 'Feature', 'properties': props,
                    'geometry': transform_geom(crs, 'EPSG:4326', geo)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('proyecto')
    ap.add_argument('--alto', type=float, default=0.45, help='probabilidad desde la que es «probable»')
    ap.add_argument('--bajo', type=float, default=0.30, help='probabilidad desde la que es «posible»')
    ap.add_argument('--pendiente', type=float, default=10, help='pendiente mínima (grados) para la regla de suelo')
    ap.add_argument('--min-px', dest='min_px', type=int, default=4, help='zonas más chicas se descartan (4 px = 0,04 ha)')
    a = ap.parse_args()

    import rasterio
    from rasterio.features import shapes
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_geom
    from scipy import ndimage

    salida = os.path.join(a.proyecto, 'qgis')
    os.makedirs(salida, exist_ok=True)
    feats, filas = [], []
    for ruta in sorted(glob.glob(os.path.join(a.proyecto, 'images', '*', 'image.tif'))):
        idx = os.path.basename(os.path.dirname(ruta))
        with rasterio.open(ruta) as ds:
            x = ds.read(); perfil = ds.profile; tr = ds.transform; crs = ds.crs
        b2, b3, b4, pend, prob = x[1], x[2], x[3], x[4], x[6]
        con_dato = b2 > 0
        nube = con_dato & es_nube_blanca(b2, b3, b4)
        # probabilidad como ráster aparte, para el mapa de calor
        p1 = dict(perfil, count=1, dtype='float32', compress='deflate', nodata=-1)
        with MemoryFile() as mem:
            with mem.open(**p1) as d:
                d.write(np.where(con_dato, prob, -1).astype('float32'), 1)
            with open(os.path.join(salida, f'{idx}_prob.tif'), 'wb') as f_prob_tif:
                f_prob_tif.write(mem.read())
        base = {'tesela': idx}
        args = (transform_geom, shapes, a.min_px, a.pendiente)
        f_prob = poligonos(prob >= a.alto, prob, pend, tr, crs, dict(base, fuente='modelo', nivel='probable'), *args)
        f_posi = poligonos((prob >= a.bajo) & (prob < a.alto), prob, pend, tr, crs,
                           dict(base, fuente='modelo', nivel='posible'), *args)
        # regla de suelo desnudo en ladera: rojo ≥ verde (GRVI ≤ 0), claro, sin nube, pendiente alta
        grvi = (b3 - b4) / np.maximum(b3 + b4, 1)
        suelo = con_dato & ~nube & (grvi <= 0.0) & (b4 > 600) & (pend >= a.pendiente)
        suelo = ndimage.binary_opening(suelo, iterations=1)
        f_suelo = poligonos(suelo, prob, pend, tr, crs, dict(base, fuente='suelo', nivel='suelo_desnudo'), *args)
        feats += f_prob + f_posi + f_suelo
        en_ladera = lambda fs: [f for f in fs if not f['properties']['nivel'].endswith('_plano')]
        pl, po, su = en_ladera(f_prob), en_ladera(f_posi), f_suelo
        filas.append({'tesela': idx, 'probables_ladera': len(pl), 'probables_plano': len(f_prob) - len(pl),
                      'posibles_ladera': len(po), 'suelo_desnudo': len(su),
                      'ha_probable_ladera': round(sum(f['properties']['area_ha'] for f in pl), 2),
                      'nubes_pct': round(float(nube.mean()) * 100, 1),
                      'puntaje': len(pl) * 3 + len(su) + len(po) * 0.5})
        print(f'{idx}: {len(pl)} probables en ladera ({len(f_prob) - len(pl)} en llano), {len(po)} posibles, '
              f'{len(su)} de suelo desnudo' + (f', nubes {nube.mean()*100:.0f} %' if nube.mean() > 0.02 else ''))
    if not filas:
        raise SystemExit(f'no hay teselas en {a.proyecto}/images')
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open(os.path.join(salida, 'candidatos.geojson'), 'w'))
    filas.sort(key=lambda r: -r['puntaje'])
    with open(os.path.join(salida, 'prioridad.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f'\n{len(feats)} candidatos en {os.path.join(os.path.abspath(salida), "candidatos.geojson")}')
    print(f'orden sugerido para revisar: {os.path.join(os.path.abspath(salida), "prioridad.csv")}')


if __name__ == '__main__':
    main()
