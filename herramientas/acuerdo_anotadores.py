"""
Mide el acuerdo entre dos anotadores que marcaron las MISMAS teselas sin verse.

Cada uno exporta su capa como GeoJSON (EPSG:4326) con el campo clase. Se pasa cada capa
a la malla de la tesela y se compara píxel a píxel la clase deslizamiento. Los píxeles
que cualquiera de los dos marcó como dudoso, y las nubes, no cuentan.

  python herramientas/acuerdo_anotadores.py marcas_NH.geojson marcas_LM.geojson \
      --teselas sinpad_15525_001,sinpad_48878_001,sinpad_557_001

Da por tesela y en total: IoU entre anotadores, kappa de Cohen, hectáreas de cada uno y
las cicatrices que uno marcó y el otro no tocó. Escribe acuerdo_anotadores.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

BASES = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']


def capas(ruta, crs, tr, forma, rasterize, transform_geom):
    des, dud = np.zeros(forma, bool), np.zeros(forma, bool)
    for f in json.load(open(ruta, encoding='utf-8')).get('features', []):
        if not f.get('geometry'):
            continue
        clase = ((f.get('properties') or {}).get('clase') or '').strip().lower()
        z = rasterize([(transform_geom('EPSG:4326', crs, f['geometry']), 1)], out_shape=forma, transform=tr,
                      fill=0, dtype='uint8') == 1
        if clase == 'deslizamiento':
            des |= z
        elif clase == 'dudoso':
            dud |= z
    return des, dud


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('marcas_a')
    ap.add_argument('marcas_b')
    ap.add_argument('--teselas', required=True, help='separadas por coma')
    ap.add_argument('--raiz', default='.')
    a = ap.parse_args()
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from scipy import ndimage

    filas, tot = [], np.zeros(4, np.int64)
    for tes in [t.strip() for t in a.teselas.split(',') if t.strip()]:
        ruta = next((os.path.join(a.raiz, b, 'images', tes, 'image.tif') for b in BASES
                     if os.path.exists(os.path.join(a.raiz, b, 'images', tes, 'image.tif'))), None)
        if not ruta:
            print(f'no encuentro la tesela {tes}')
            continue
        with rasterio.open(ruta) as ds:
            b2, tr, crs, forma = ds.read(2), ds.transform, ds.crs, ds.shape
        da, ua = capas(a.marcas_a, crs, tr, forma, rasterize, transform_geom)
        db, ub = capas(a.marcas_b, crs, tr, forma, rasterize, transform_geom)
        ok = ~(ua | ub) & (b2 > 0) & (b2 <= 1000)
        tp, fa, fb, tn = (int((da & db & ok).sum()), int((da & ~db & ok).sum()),
                          int((~da & db & ok).sum()), int((~da & ~db & ok).sum()))
        tot += [tp, fa, fb, tn]
        la, na = ndimage.label(da & ok)
        lb, nb = ndimage.label(db & ok)
        solo_a = sum(1 for k in range(1, na + 1) if not (db & (la == k)).any())
        solo_b = sum(1 for k in range(1, nb + 1) if not (da & (lb == k)).any())
        filas.append({'tesela': tes, 'iou': round(tp / max(tp + fa + fb, 1), 3), 'ha_a': round((tp + fa) * 0.01, 2),
                      'ha_b': round((tp + fb) * 0.01, 2), 'cicatrices_a': na, 'cicatrices_b': nb,
                      'solo_a': solo_a, 'solo_b': solo_b})
        print(filas[-1])
    tp, fa, fb, tn = (float(v) for v in tot)
    n = tp + fa + fb + tn
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fa) * (tp + fb) + (tn + fb) * (tn + fa)) / max(n * n, 1)
    kappa = (po - pe) / max(1 - pe, 1e-9)
    print(f'TOTAL: IoU entre anotadores {tp / max(tp + fa + fb, 1):.3f}; kappa de Cohen {kappa:.3f}')
    print('Como referencia: un IoU por debajo de 0,5 pide discutir las reglas antes de seguir marcando.')
    if filas:
        with open(os.path.join(a.raiz, 'acuerdo_anotadores.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)


if __name__ == '__main__':
    main()
