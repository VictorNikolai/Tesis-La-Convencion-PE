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


def _crs_de(gj):
    nombre = ((gj.get('crs') or {}).get('properties') or {}).get('name', '')
    if 'EPSG' in nombre:
        cod = nombre.replace('::', ':').split(':')[-1]
        return 'EPSG:4326' if cod in ('CRS84', '4326') else f'EPSG:{cod}'
    return 'EPSG:4326'


def capas(ruta, crs, tr, forma, rasterize, transform_geom, tes=None, proy=None):
    des, dud = np.zeros(forma, bool), np.zeros(forma, bool)
    if os.path.isdir(ruta) and tes:
        from PIL import Image
        from cortar_parches_128 import LADO, cuadrantes
        for q, f0, c0 in cuadrantes(forma[0], forma[1], LADO):
            posibles = [
                os.path.join(ruta, 'masks', f'{proy}__{tes}_{q}.png') if proy else '',
                os.path.join(ruta, 'masks', f'datos_laconvencion__{tes}_{q}.png'),
                os.path.join(ruta, 'masks', f'datos_indeci__{tes}_{q}.png'),
                os.path.join(ruta, 'masks', f'datos_ingemmet__{tes}_{q}.png'),
                os.path.join(ruta, 'masks', f'datos_azar__{tes}_{q}.png'),
            ]
            p_png = next((p for p in posibles if p and os.path.exists(p)), None)
            if not p_png:
                dud[f0:f0 + LADO, c0:c0 + LADO] = True
                continue
            raw = np.asarray(Image.open(p_png))
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            vals = set(np.unique(raw).tolist())
            if 3 in vals:
                q_des, q_dud = (raw == 2), (raw == 3) | (raw == 0)
            else:
                q_des, q_dud = (raw == 1), (raw == 2) | (raw == 255)
            des[f0:f0 + LADO, c0:c0 + LADO] |= q_des
            dud[f0:f0 + LADO, c0:c0 + LADO] |= q_dud
        return des, dud

    with open(ruta, encoding='utf-8') as f_gj:
        gj = json.load(f_gj)
    crs_m = _crs_de(gj)
    for f in gj.get('features', []):
        if not f.get('geometry'):
            continue
        clase = ((f.get('properties') or {}).get('clase') or '').strip().lower()
        z = rasterize([(transform_geom(crs_m, crs, f['geometry']), 1)], out_shape=forma, transform=tr,
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
    ap.add_argument('--teselas', default='sinpad_15525_001,sinpad_48878_001,sinpad_557_001', help='separadas por coma')
    ap.add_argument('--raiz', default='.')
    a = ap.parse_args()
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from scipy import ndimage

    filas, tot = [], np.zeros(4, np.int64)
    for tes in [t.strip() for t in a.teselas.split(',') if t.strip()]:
        par_proy = next(((b, os.path.join(a.raiz, b, 'images', tes, 'image.tif')) for b in BASES
                         if os.path.exists(os.path.join(a.raiz, b, 'images', tes, 'image.tif'))), None)
        if not par_proy:
            print(f'no encuentro la tesela {tes}')
            continue
        proy, ruta = par_proy
        with rasterio.open(ruta) as ds:
            b2, b3, b4 = ds.read(2), ds.read(3), ds.read(4)
            tr, crs, forma = ds.transform, ds.crs, ds.shape
        da, ua = capas(a.marcas_a, crs, tr, forma, rasterize, transform_geom, tes=tes, proy=proy)
        db, ub = capas(a.marcas_b, crs, tr, forma, rasterize, transform_geom, tes=tes, proy=proy)
        suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
        nube = (b2 > 1000) & ~suelo_claro & ~(da | db)
        ok = ~(ua | ub) & (b2 > 0) & ~nube
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
