"""
Descarga una tesela (o mosaico de teselas 256x256 px a 10 m en UTM 18S) desde Planetary Computer:
  - Sentinel-2 L2A (B01, B02, B03, B04), mediana de temporada seca filtrando nubes SCL y offset +1000.
  - NASADEM (elevación en metros) y pendiente en grados sobre malla de 10 m.
  - Opcionalmente bandas 7 y 8 (probabilidad y cajas de guía del modelo; o ceros con --modelo ninguno
    para marcado ciego en prueba/validación).

Llamado por `descargar_desde_puntos.py` o directamente por CLI.
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)

from nucleo import config


NOMBRES_8 = [
    'B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)',
    'probabilidad del modelo', 'cajas del modelo (contorno)'
]


def _predecir_modelo_en_mosaico(arr6, nombre_modelo, umbral_cajas):
    """Devuelve (prob_2d, cajas_2d) sobre bandas (6, H, W) usando un modelo de modelos_ajustados/ o modelos/."""
    import glob
    from nucleo import modelos, recetas
    if not nombre_modelo or nombre_modelo.lower() in ('ninguno', 'none', 'ciego'):
        h, w = arr6.shape[1], arr6.shape[2]
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    # Buscar recetas ONNX disponibles
    rutas_json = []
    for carpeta in ('modelos_ajustados', 'modelos'):
        d = os.path.join(RAIZ, carpeta)
        if os.path.isdir(d):
            rutas_json.extend(sorted(glob.glob(os.path.join(d, '*.json'))))

    objetivos = []
    if nombre_modelo == 'ensamble':
        for rj in rutas_json:
            bn = os.path.basename(rj)
            if bn in ('unet_peru.json', 'deeplabv3plus_peru.json'):
                objetivos.append(rj)
    else:
        for rj in rutas_json:
            if nombre_modelo in os.path.basename(rj):
                objetivos.append(rj)
                break
    if not objetivos:
        h, w = arr6.shape[1], arr6.shape[2]
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    import onnxruntime as ort
    h, w = arr6.shape[1], arr6.shape[2]
    b1, b2, b3, b4, pend, elev = arr6[0], arr6[1], arr6[2], arr6[3], arr6[4], arr6[5]
    valido = b2 > 0
    probs_modelos = []
    for rj in objetivos:
        r = recetas.leer(rj)
        ruta_onnx = os.path.join(os.path.dirname(rj), r['onnx'])
        if not os.path.exists(ruta_onnx):
            continue
        ses = ort.InferenceSession(ruta_onnx, providers=['CPUExecutionProvider'])
        in_name = ses.get_inputs()[0].name
        prob_m = np.zeros((h, w), dtype=np.float32)
        for f0 in range(0, h - 127, 128):
            for c0 in range(0, w - 127, 128):
                bdn = {
                    'B01': b1[f0:f0 + 128, c0:c0 + 128],
                    'B02': b2[f0:f0 + 128, c0:c0 + 128],
                    'B03': b3[f0:f0 + 128, c0:c0 + 128],
                    'B04': b4[f0:f0 + 128, c0:c0 + 128],
                }
                terr = {
                    'pendiente': pend[f0:f0 + 128, c0:c0 + 128],
                    'elevacion': elev[f0:f0 + 128, c0:c0 + 128],
                }
                v_sub = valido[f0:f0 + 128, c0:c0 + 128]
                ten = recetas.desde_l2a(bdn, terr, v_sub, r)
                out = ses.run(None, {in_name: ten[None]})[0]
                prob_m[f0:f0 + 128, c0:c0 + 128] = recetas.probabilidad(out, r['salida'])[0]
        probs_modelos.append(prob_m)

    if not probs_modelos:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    prob = np.mean(probs_modelos, axis=0).astype(np.float32)
    from scipy import ndimage
    lab, n = ndimage.label(prob >= umbral_cajas)
    cajas = np.zeros((h, w), dtype=np.float32)
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < 4:
            continue
        y0, y1 = max(0, int(ys.min()) - 1), min(h - 1, int(ys.max()) + 1)
        x0, x1 = max(0, int(xs.min()) - 1), min(w - 1, int(xs.max()) + 1)
        cajas[y0, x0:x1 + 1] = 1.0
        cajas[y1, x0:x1 + 1] = 1.0
        cajas[y0:y1 + 1, x0] = 1.0
        cajas[y0:y1 + 1, x1] = 1.0
    return prob, cajas


def descargar_y_exportar(sur, oeste, norte, este, desde, hasta, salida, prefijo,
                         nubes=20.0, modelo='ninguno', umbral_cajas=0.45):
    import pystac_client
    import planetary_computer
    import stackstac
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from pyproj import Transformer
    from scipy import ndimage

    tr_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32718', always_xy=True)
    tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True)
    xs_b, ys_b = tr_utm.transform([oeste, este, oeste, este], [sur, sur, norte, norte])
    # Alinear a múltiplos de 10 m y ancho/alto múltiplos de 256 px (2560 m por tesela)
    cx = (min(xs_b) + max(xs_b)) / 2.0
    cy = (min(ys_b) + max(ys_b)) / 2.0
    ancho_m = max(2560.0, max(xs_b) - min(xs_b))
    alto_m = max(2560.0, max(ys_b) - min(ys_b))
    nx = max(1, int(round(ancho_m / 2560.0)))
    ny = max(1, int(round(alto_m / 2560.0)))
    W = nx * 256
    H = ny * 256
    x0 = round((cx - W * 5.0) / 10.0) * 10.0
    y1 = round((cy + H * 5.0) / 10.0) * 10.0
    x1 = x0 + W * 10.0
    y0 = y1 - H * 10.0

    cat = pystac_client.Client.open(config.STAC, modifier=planetary_computer.sign_inplace)
    items_s2 = list(cat.search(
        collections=[config.COLECCION_S2],
        bbox=[oeste - 0.01, sur - 0.01, este + 0.01, norte + 0.01],
        datetime=f'{desde}T00:00:00Z/{hasta}T23:59:59Z',
        query={'eo:cloud_cover': {'lt': nubes}},
        limit=50,
    ).items())
    if not items_s2:
        # Relajar nubosidad si no hubo escenas con < nubes
        items_s2 = list(cat.search(
            collections=[config.COLECCION_S2],
            bbox=[oeste - 0.01, sur - 0.01, este + 0.01, norte + 0.01],
            datetime=f'{desde}T00:00:00Z/{hasta}T23:59:59Z',
            query={'eo:cloud_cover': {'lt': 60}},
            limit=50,
        ).items())
    if not items_s2:
        raise RuntimeError(f'Sin escenas Sentinel-2 para {desde}..{hasta}')

    items_s2.sort(key=lambda it: it.properties.get('eo:cloud_cover', 100.0))
    items_s2 = items_s2[:config.MAX_ESCENAS]

    # Bounds exactos en EPSG:32718 para stackstac (x0, y0, x1-10, y1-10)
    bounds_utm = (x0, y0, x1, y1)
    da = stackstac.stack(
        items_s2,
        assets=['B01', 'B02', 'B03', 'B04', 'SCL'],
        epsg=32718,
        resolution=10,
        bounds=bounds_utm,
        fill_value=np.nan,
    ).compute()

    # Asegurar tamaño exacto (H, W) recortando/paddeando si stackstac da +-1 px por redondeo
    arr_np = da.values  # (T, 5, Hy, Wx)
    hy, wx = arr_np.shape[2], arr_np.shape[3]
    if hy != H or wx != W:
        out_pad = np.full((arr_np.shape[0], 5, H, W), np.nan, dtype=np.float32)
        hh, ww = min(hy, H), min(wx, W)
        out_pad[:, :, :hh, :ww] = arr_np[:, :, :hh, :ww]
        arr_np = out_pad

    bandas_raw = arr_np[:, :4].astype(np.float32)
    scl = arr_np[:, 4]
    es_mala = np.isnan(scl) | np.isin(scl, config.SCL_MALAS)
    # Restar desplazamiento baseline 04.00 (enero 2022+)
    for t_idx, it in enumerate(items_s2):
        proc_base = float(it.properties.get('s2:processing_baseline', '04.00') or 4.0)
        dt_str = (it.properties.get('datetime') or '')[:10]
        if proc_base >= 4.0 or dt_str >= '2022-01-25':
            bandas_raw[t_idx] = np.maximum(0.0, bandas_raw[t_idx] - config.DESPLAZAMIENTO_L2A)

    bandas_raw = np.where(es_mala[:, None, :, :], np.nan, bandas_raw)
    comp = np.nanmedian(bandas_raw, axis=0)
    comp = np.nan_to_num(comp, nan=0.0).astype(np.float32)

    # NASADEM
    items_dem = list(cat.search(
        collections=['nasadem'],
        bbox=[oeste - 0.02, sur - 0.02, este + 0.02, norte + 0.02],
        limit=10,
    ).items())
    if not items_dem:
        raise RuntimeError('Sin teselas NASADEM en la zona')
    da_dem = stackstac.stack(
        items_dem,
        assets=['elevation'],
        epsg=32718,
        resolution=10,
        bounds=bounds_utm,
        resampling=rasterio.enums.Resampling.bilinear,
        fill_value=np.nan,
    ).compute()
    elev_raw = np.nanmedian(da_dem.values[:, 0], axis=0)
    elev = np.zeros((H, W), dtype=np.float32)
    hh, ww = min(elev_raw.shape[0], H), min(elev_raw.shape[1], W)
    elev[:hh, :ww] = np.nan_to_num(elev_raw[:hh, :ww], nan=0.0)
    # Suavizado ligero equivalente a pasos de 30->10 m para gradiente estable y pendiente en grados
    gy, gx = np.gradient(elev.astype(np.float64), 10.0, 10.0)
    pend = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    arr6 = np.concatenate([comp, pend[None], elev[None]], axis=0)
    prob_m, cajas_m = _predecir_modelo_en_mosaico(arr6, modelo, umbral_cajas)
    arr8 = np.concatenate([arr6, prob_m[None], cajas_m[None]], axis=0).astype(np.float32)

    os.makedirs(os.path.join(salida, 'images'), exist_ok=True)
    k = 0
    rutas_escritas = []
    for fy in range(ny):
        for cx_i in range(nx):
            k += 1
            tes_id = f'{prefijo}_{k:03d}'
            f0, c0 = fy * 256, cx_i * 256
            sub = arr8[:, f0:f0 + 256, c0:c0 + 256]
            # Saltar sub-teselas con >70% sin dato en cajas grandes
            if float((sub[1] > 0).mean()) < 0.30 and (nx * ny > 1):
                continue
            tr_sub = from_origin(x0 + c0 * 10.0, y1 - f0 * 10.0, 10.0, 10.0)
            carp = os.path.join(salida, 'images', tes_id)
            os.makedirs(carp, exist_ok=True)
            p8 = dict(driver='GTiff', height=256, width=256, count=8, dtype='float32',
                      crs='EPSG:32718', transform=tr_sub, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p8) as dst:
                    dst.write(sub)
                    dst.descriptions = tuple(NOMBRES_8)
                with open(os.path.join(carp, 'image.tif'), 'wb') as fo:
                    fo.write(mem.read())
            lon_c, lat_c = tr_ll.transform(x0 + (c0 + 128) * 10.0, y1 - (f0 + 128) * 10.0)
            meta = {
                'location': [round(lon_c, 5), round(lat_c, 5)],
                'escenas': [it.id for it in items_s2],
                'fechas': sorted({(it.properties.get('datetime') or '')[:10] for it in items_s2}),
                'modelo_cajas': modelo,
            }
            with open(os.path.join(carp, 'metadata.json'), 'w', encoding='utf-8') as fm:
                json.dump(meta, fm, indent=1, ensure_ascii=False)
            rutas_escritas.append(tes_id)

    print(f'{len(rutas_escritas)} tesela(s) exportadas en {salida}/images ({prefijo})')
    return rutas_escritas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sur', type=float, required=True)
    ap.add_argument('--oeste', type=float, required=True)
    ap.add_argument('--norte', type=float, required=True)
    ap.add_argument('--este', type=float, required=True)
    ap.add_argument('--desde', required=True)
    ap.add_argument('--hasta', required=True)
    ap.add_argument('--modelo', default='ninguno')
    ap.add_argument('--nubes', type=float, default=20.0)
    ap.add_argument('--umbral_cajas', type=float, default=0.45)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--prefijo', required=True)
    a = ap.parse_args()
    descargar_y_exportar(a.sur, a.oeste, a.norte, a.este, a.desde, a.hasta,
                         a.salida, a.prefijo, nubes=a.nubes, modelo=a.modelo,
                         umbral_cajas=a.umbral_cajas)


if __name__ == '__main__':
    main()
