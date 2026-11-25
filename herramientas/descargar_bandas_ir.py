"""Descarga B08 (10m), B11 (20m->10m) y B12 (20m->10m) para todas las teselas de La Convencion (447 teselas),
usando las mismas fechas y compuesto mediano de temporada seca que las bandas actuales (B01-B04).

Guarda cada tesela 256x256 (3 bandas: B08, B11, B12 en float32) en:
  parches_128/bandas_ir/<proyecto>/<tesela>.tif
con exactamente el mismo CRS y transformada afin de la tesela fuente.
"""
import argparse
import concurrent.futures
import csv
import json
import os
import sys

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
import stackstac

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..'))
from nucleo import config


def construir_mapa_anios():
    """Lee anio_imagen de puntos_resumen.csv cuando existe."""
    mapa = {}
    for p in ('datos_azar', 'datos_indeci', 'datos_ingemmet', 'datos_laconvencion'):
        pr = os.path.join(p, 'puntos_resumen.csv')
        if not os.path.exists(pr):
            continue
        for r in csv.DictReader(open(pr, encoding='utf-8')):
            anio = (r.get('anio_imagen') or '').strip()
            pid = (r.get('id') or '').strip()
            if anio.isdigit() and pid:
                mapa[(p, pid)] = int(anio)
    return mapa


def _anio_de_tesela(proy, tesela, mapa_anios):
    # Extraer id del punto de tesela (ej: ZN04_VILCABAMBA_001 -> buscar por proyecto o subcadena)
    for (p, pid), anio in mapa_anios.items():
        if p == proy and (pid in tesela or tesela.endswith(f'_{pid}')):
            return anio
    return None


def _descargar_compuesto(cat, bbox_ll, bounds_utm, anio):
    desde = f'{anio}-06-01T00:00:00Z'
    hasta = f'{anio}-09-30T23:59:59Z'
    items = list(cat.search(
        collections=[config.COLECCION_S2],
        bbox=bbox_ll,
        datetime=f'{desde}/{hasta}',
        query={'eo:cloud_cover': {'lt': 20}},
        limit=50,
    ).items())
    if not items:
        items = list(cat.search(
            collections=[config.COLECCION_S2],
            bbox=bbox_ll,
            datetime=f'{desde}/{hasta}',
            query={'eo:cloud_cover': {'lt': 60}},
            limit=50,
        ).items())
    if not items:
        return None
    items.sort(key=lambda it: it.properties.get('eo:cloud_cover', 100.0))
    items = items[:config.MAX_ESCENAS]

    try:
        da = stackstac.stack(
            items,
            assets=['B04', 'B08', 'B11', 'B12', 'SCL'],
            epsg=32718,
            resolution=10,
            bounds=bounds_utm,
            resampling=Resampling.bilinear,
            fill_value=np.nan,
        ).compute()
    except Exception:
        return None
    arr = da.values.astype(np.float32)  # (T, 5, H, W)
    hy, wx = arr.shape[2], arr.shape[3]
    if hy != 256 or wx != 256:
        pad = np.full((arr.shape[0], 5, 256, 256), np.nan, dtype=np.float32)
        hh, ww = min(hy, 256), min(wx, 256)
        pad[:, :, :hh, :ww] = arr[:, :, :hh, :ww]
        arr = pad

    bandas = arr[:, :4]
    scl = arr[:, 4]
    es_mala = np.isnan(scl) | np.isin(scl, config.SCL_MALAS)
    for t_idx in range(min(len(items), len(bandas))):
        it = items[t_idx]
        proc_base = float(it.properties.get('s2:processing_baseline', '04.00') or 4.0)
        dt_str = (it.properties.get('datetime') or '')[:10]
        if proc_base >= 4.0 or dt_str >= '2022-01-25':
            bandas[t_idx] = np.maximum(0.0, bandas[t_idx] - config.DESPLAZAMIENTO_L2A)
    bandas = np.where(es_mala[:, None, :, :], np.nan, bandas)
    comp = np.nanmedian(bandas, axis=0)
    return np.nan_to_num(comp, nan=0.0).astype(np.float32)  # (4, 256, 256): B04, B08, B11, B12


def procesar_tesela(proy, tesela, anio_hint=None):
    out_dir = os.path.join('parches_128', 'bandas_ir', proy)
    out_path = os.path.join(out_dir, f'{tesela}.tif')
    if os.path.exists(out_path):
        return tesela, 'existe', 1.0

    # Leer geometria y B04 local de la tesela 256x256
    p_src = os.path.join(proy, 'images', tesela, 'image.tif')
    if not os.path.exists(p_src):
        return tesela, 'sin_fuente', 0.0
    with rasterio.open(p_src) as ds:
        b4_loc = ds.read(4).astype(np.float32)
        bounds = ds.bounds
        crs = ds.crs
        tr = ds.transform

    from pyproj import Transformer
    tr_ll = Transformer.from_crs(str(crs), 'EPSG:4326', always_xy=True)
    lon_w, lat_s = tr_ll.transform(bounds.left, bounds.bottom)
    lon_e, lat_n = tr_ll.transform(bounds.right, bounds.top)
    bbox_ll = [lon_w - 0.01, lat_s - 0.01, lon_e + 0.01, lat_n + 0.01]
    bounds_utm = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    cat = pystac_client.Client.open(config.STAC, modifier=planetary_computer.sign_inplace)
    cands = [anio_hint] if anio_hint else []
    for y in (2024, 2025, 2023, 2022, 2021, 2020, 2019, 2018):
        if y not in cands:
            cands.append(y)

    best_corr = -1.0
    best_comp = None
    best_anio = None
    val_loc = (b4_loc > 0)
    for y in cands[:4]:
        comp = _descargar_compuesto(cat, bbox_ll, bounds_utm, y)
        if comp is None:
            continue
        b4_new = comp[0]
        vm = val_loc & (b4_new > 0)
        if int(vm.sum()) < 100:
            continue
        corr = float(np.corrcoef(b4_loc[vm], b4_new[vm])[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_comp = comp
            best_anio = y
        if corr >= 0.96:
            break

    if best_comp is None:
        return tesela, 'sin_compuesto', 0.0

    os.makedirs(out_dir, exist_ok=True)
    perfil = dict(driver='GTiff', height=256, width=256, count=3, dtype='float32',
                  crs=crs, transform=tr, compress='lzw')
    with rasterio.open(out_path, 'w', **perfil) as dst:
        dst.write(best_comp[1:4])  # B08, B11, B12
        dst.descriptions = ('B08', 'B11', 'B12')
        dst.update_tags(anio_usado=str(best_anio), corr_b04=f'{best_corr:.4f}')
    return tesela, f'ok_{best_anio}', best_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limite', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    n1 = list(csv.DictReader(open('mascaras_nivel1/mascaras_nivel1.csv', encoding='utf-8')))
    mapa_anios = construir_mapa_anios()
    tareas = []
    for r in n1:
        proy, tes = r['proyecto'].strip(), r['tesela'].strip()
        tareas.append((proy, tes, _anio_de_tesela(proy, tes, mapa_anios)))
    if a.limite > 0:
        tareas = tareas[:a.limite]

    ok_n = err_n = 0
    corrs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(procesar_tesela, p, t, y) for p, t, y in tareas]
        for k, f in enumerate(concurrent.futures.as_completed(futs), 1):
            tes, st, c = f.result()
            if st.startswith('ok') or st == 'existe':
                ok_n += 1
                if st.startswith('ok'):
                    corrs.append(c)
            else:
                err_n += 1
            if k % 25 == 0 or k == len(tareas):
                med_c = float(np.mean(corrs)) if corrs else 1.0
                print(f'[{k}/{len(tareas)}] ok={ok_n} err={err_n} corr_B04_media={med_c:.4f}', flush=True)


if __name__ == '__main__':
    main()
