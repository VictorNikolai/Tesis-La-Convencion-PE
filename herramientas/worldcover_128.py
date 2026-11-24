"""Descarga y alinea ESA WorldCover 10 m (2021 v200) desde Planetary Computer a parches 128x128 locales.

Guarda cada parche como GeoTIFF uint8 en parches_128/worldcover/<proyecto>/<parche>.tif.
Clases ESA WorldCover 2021:
  10 = Bosque (Tree cover)
  20 = Matorral (Shrubland)
  30 = Pastizal / pajonal (Grassland)
  40 = Cultivo / chacra (Cropland)
  50 = Urbano / poblado (Built-up)
  60 = Suelo desnudo / escasa vegetacion (Bare / sparse vegetation)
  70 = Nieve / hielo (Snow and ice)
  80 = Agua permanente / rio (Permanent water bodies)
  90 = Humedal herbaceo (Herbaceous wetland)
  95 = Manglar
  100 = Musgo y liquen
"""
import argparse
import csv
import os
import sys

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds


def abrir_fuentes_worldcover(bbox=(-74.0, -13.6, -72.0, -11.2)):
    cat = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace,
    )
    search = cat.search(
        collections=['esa-worldcover'],
        bbox=list(bbox),
        query={'esa_worldcover:product_version': {'eq': '2.0.0'}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError('No se encontraron teselas ESA WorldCover 2021 v200 en Planetary Computer')
    fuentes = []
    for it in items:
        href = it.assets['map'].href
        fuentes.append((it.id, tuple(it.bbox), rasterio.open(href)))
    return fuentes


def extraer_worldcover_parche(ruta_parche, fuentes):
    with rasterio.open(ruta_parche) as ds:
        dst_crs = ds.crs
        dst_tr = ds.transform
        h, w = ds.height, ds.width
        b_ll = transform_bounds(dst_crs, 'EPSG:4326', *ds.bounds, densify_pts=8)
    out = np.zeros((h, w), dtype=np.uint8)
    for _, fb, src in fuentes:
        if b_ll[2] < fb[0] or b_ll[0] > fb[2] or b_ll[3] < fb[1] or b_ll[1] > fb[3]:
            continue
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_tr,
            width=w,
            height=h,
            resampling=Resampling.nearest,
            nodata=0,
        ) as vrt:
            arr = vrt.read(1)
            mask = (out == 0) & (arr > 0)
            out[mask] = arr[mask]
    return out, dst_crs, dst_tr


def descargar_conjuntos(conjuntos, carpeta_parches='parches_128', salida='parches_128/worldcover', solo_faltantes=True):
    indice = os.path.join(carpeta_parches, 'indice_parches_ampliado.csv')
    with open(indice, encoding='utf-8') as f:
        filas = [r for r in csv.DictReader(f) if r['conjunto'] in conjuntos]
    pendientes = []
    for r in filas:
        dst = os.path.join(salida, r['proyecto'], r['parche'] + '.tif')
        if solo_faltantes and os.path.exists(dst):
            continue
        pendientes.append((r, dst))
    print(f'WorldCover {sorted(conjuntos)}: total={len(filas)}, por descargar={len(pendientes)}', flush=True)
    if not pendientes:
        return len(filas)
    fuentes = abrir_fuentes_worldcover()
    try:
        for i, (r, dst) in enumerate(pendientes, 1):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src_tif = os.path.join(carpeta_parches, r['archivo'])
            arr, crs, tr = extraer_worldcover_parche(src_tif, fuentes)
            with rasterio.open(
                dst, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
                count=1, dtype='uint8', crs=crs, transform=tr, compress='deflate'
            ) as out_ds:
                out_ds.write(arr, 1)
            if i % 100 == 0 or i == len(pendientes):
                print(f'  [{i}/{len(pendientes)}] descargados', flush=True)
    finally:
        for _, _, s in fuentes:
            s.close()
    return len(filas)


def main():
    ap = argparse.ArgumentParser(description='Descarga ESA WorldCover 10m alineado a parches 128x128')
    ap.add_argument('--conjuntos', required=True, help='Lista separada por comas (ej. validacion,prueba2)')
    ap.add_argument('--parches', default='parches_128')
    ap.add_argument('--salida', default='parches_128/worldcover')
    ap.add_argument('--forzar', action='store_true')
    a = ap.parse_args()
    conjs = {c.strip() for c in a.conjuntos.split(',') if c.strip()}
    descargar_conjuntos(conjs, carpeta_parches=a.parches, salida=a.salida, solo_faltantes=not a.forzar)


if __name__ == '__main__':
    main()
