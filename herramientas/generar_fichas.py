"""
Genera fichas visuales PNG por parche (128x128 px escalado 2x) para inspección detallada
según la Sección 6.1 del protocolo de tesis:
  1. Color natural B4/B3/B2 (estirado p2-p98) + contorno INGEMMET (si cae en el parche)
  2. Imagen anterior al evento (si existe variante *_antes)
  3. Índice de verdor GRVI = (B3 - B4) / (B3 + B4)
  4. Pendiente en grados
  5. Relieve sombreado (Hillshade del DEM)
  6. Solo en `entrenamiento`: Probabilidad del ensamble (`unet_peru` + `deeplabv3plus_peru`).
     En `validacion` y `prueba` (MODO CIEGO): muestra Bordes Sobel sin ninguna salida de modelo.

Uso:
  python herramientas/generar_fichas.py --parche sinpad_005_001_q1 --proyecto datos_laconvencion --salida fichas_png
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)


def _estirar_rgb(b4, b3, b2):
    rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32)
    valid = (b2 > 0) & np.isfinite(rgb).all(axis=-1)
    if valid.any():
        lo = np.percentile(rgb[valid], 2)
        hi = np.percentile(rgb[valid], 98)
        hi = max(hi, lo + 50.0)
    else:
        lo, hi = 0.0, 1500.0
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def _color_mapa(arr, vmin, vmax, cmap_name):
    import matplotlib
    cm = matplotlib.colormaps.get_cmap(cmap_name)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = cm(norm)
    return (rgba[:, :, :3] * 255.0).astype(np.uint8)


def _hillshade(elev, dx=10.0, azimut=315.0, altitud=45.0):
    gy, gx = np.gradient(elev.astype(np.float64), dx, dx)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(azimut)
    alt = np.radians(altitud)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    hs = np.clip((hs + 0.1) / 1.1, 0.0, 1.0)
    gray = (hs * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _sobel_rgb(b4, b3, b2):
    from scipy import ndimage
    g = (b4.astype(np.float32) + b3 + b2) / 3.0
    sx = ndimage.sobel(g, axis=1)
    sy = ndimage.sobel(g, axis=0)
    mag = np.hypot(sx, sy)
    hi = max(float(np.percentile(mag, 98)), 1.0)
    v = np.clip(mag / hi * 255.0, 0, 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def _cargar_ingemmet(raiz):
    ruta_pol = os.path.join(raiz, 'ingemmet', 'ingemmet_cartografia_peligros_poligonos_bbox.geojson')
    ruta_pt = os.path.join(raiz, 'ingemmet', 'ingemmet_peligros_puntos_bbox.geojson')
    pols, pts = [], []
    if os.path.exists(ruta_pol):
        from shapely.geometry import shape
        gj = json.load(open(ruta_pol, encoding='utf-8'))
        for f in gj['features']:
            if f.get('geometry'):
                pols.append((shape(f['geometry']), f.get('properties', {})))
    if os.path.exists(ruta_pt):
        from shapely.geometry import Point
        gj = json.load(open(ruta_pt, encoding='utf-8'))
        for f in gj['features']:
            if f.get('geometry') and f['geometry']['type'] == 'Point':
                lon, lat = f['geometry']['coordinates'][:2]
                pts.append((Point(lon, lat), f.get('properties', {})))
    return pols, pts


def generar_ficha(ruta_parche, conjunto, zona, ruta_salida, ruta_antes=None,
                  prob_ensamble=None, ingemmet_data=None, info_extra=''):
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    from PIL import Image, ImageDraw

    with rasterio.open(ruta_parche) as ds:
        x = ds.read().astype(np.float32)
        tr, crs, bounds = ds.transform, ds.crs, ds.bounds

    b1, b2, b3, b4, pend, elev = x[0], x[1], x[2], x[3], x[4], x[5]
    rgb = _estirar_rgb(b4, b3, b2)

    # Contorno INGEMMET si cae en el parche
    n_ing_pol, n_ing_pt = 0, 0
    if ingemmet_data is not None:
        pols, pts = ingemmet_data
        b_ll = transform_bounds(crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        caja_ll = box(*b_ll)
        tr_utm = Transformer.from_crs('EPSG:4326', str(crs), always_xy=True).transform
        geoms_utm = []
        for s_pol, _ in pols:
            if s_pol.intersects(caja_ll):
                geoms_utm.append(shp_transform(tr_utm, s_pol.boundary))
                n_ing_pol += 1
        if geoms_utm:
            from rasterio.features import rasterize as rz
            mask_b = rz([(g, 1) for g in geoms_utm], out_shape=(128, 128), transform=tr, all_touched=True)
            rgb[mask_b > 0] = [0, 255, 255]
        for s_pt, _ in pts:
            if caja_ll.contains(s_pt):
                px, py = tr_utm(s_pt.x, s_pt.y)
                c_p, f_p = ~tr * (px, py)
                ci, fi = int(round(c_p)), int(round(f_p))
                if 2 <= fi < 126 and 2 <= ci < 126:
                    rgb[fi - 2:fi + 3, ci - 2:ci + 3] = [255, 255, 0]
                    n_ing_pt += 1

    # Antes
    if ruta_antes and os.path.exists(ruta_antes):
        with rasterio.open(ruta_antes) as da:
            xa = da.read().astype(np.float32)
        panel_antes = _estirar_rgb(xa[3], xa[2], xa[1])
        tit_antes = 'Antes del evento'
    else:
        panel_antes = np.full((128, 128, 3), 40, dtype=np.uint8)
        tit_antes = 'Sin imagen previa'

    # GRVI = (B3 - B4) / (B3 + B4)
    den = b3 + b4
    grvi = np.where(den != 0, (b3 - b4) / np.where(den != 0, den, 1.0), 0.0)
    panel_grvi = _color_mapa(grvi, -0.35, 0.35, 'RdYlGn')
    panel_pend = _color_mapa(pend, 0.0, 50.0, 'viridis')
    panel_hs = _hillshade(elev)

    es_ciego = conjunto in ('validacion', 'prueba')
    if not es_ciego and prob_ensamble is not None:
        panel_6 = _color_mapa(prob_ensamble, 0.0, 1.0, 'magma')
        tit_6 = f'Ensamble (max {prob_ensamble.max():.2f})'
    else:
        panel_6 = _sobel_rgb(b4, b3, b2)
        tit_6 = 'Bordes Sobel (MODO CIEGO)'

    ESC = 2
    W_P = 128 * ESC
    GAP = 8
    TOP = 48
    paneles = [
        ('Color natural + INGEMMET', rgb),
        (tit_antes, panel_antes),
        ('GRVI (rojo=suelo, verde=veg)', panel_grvi),
        (f'Pendiente (med {pend.mean():.1f} deg)', panel_pend),
        ('Relieve sombreado DEM', panel_hs),
        (tit_6, panel_6),
    ]
    lienzo = Image.new('RGB', (len(paneles) * (W_P + GAP) + GAP, TOP + W_P + 26), (18, 22, 28))
    dib = ImageDraw.Draw(lienzo)
    cx, cy = tr * (64, 64)
    lon_c, lat_c = Transformer.from_crs(str(crs), 'EPSG:4326', always_xy=True).transform(cx, cy)
    nom_p = os.path.splitext(os.path.basename(ruta_parche))[0]
    dib.text((10, 8), f'Parche: {nom_p} | Conjunto: {conjunto.upper()} | Zona: {zona} | ({lat_c:.4f}, {lon_c:.4f}) | INGEMMET: {n_ing_pol} pol, {n_ing_pt} pts', fill=(255, 235, 140))
    if info_extra:
        dib.text((10, 24), info_extra, fill=(180, 220, 255))

    for i, (tit, arr) in enumerate(paneles):
        x0 = GAP + i * (W_P + GAP)
        dib.text((x0 + 4, TOP - 14), tit, fill=(220, 225, 235))
        im_p = Image.fromarray(arr).resize((W_P, W_P), Image.NEAREST)
        lienzo.paste(im_p, (x0, TOP))

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
    lienzo.save(ruta_salida)
    return {'grvi_min': round(float(grvi.min()), 3), 'pend_media': round(float(pend.mean()), 1),
            'pend_max': round(float(pend.max()), 1), 'ing_pol': n_ing_pol, 'ing_pt': n_ing_pt}
