"""Posprocesamiento y ensamble del protocolo v6 (Etapa 4, evaluado en validacion).

Incluye:
  1. TTA de 4 espejos (original, horizontal, vertical, ambos).
  2. Filtro altoandino basado en ESA WorldCover 10 m + DEM (elevacion en metros).
  3. Ensamble ponderado U-Net + DeepLabV3+ (1.0/0.0, 0.8/0.2, 0.7/0.3, 0.5/0.5 y DeepLab filtrado por U-Net).

Uso en validacion (sin tocar ningun conjunto de prueba):
  python herramientas/posproceso_v6.py --unet <ckpt_unet.pt> --stats-unet <stats.json> \
      --deeplab <ckpt_dl.pt> --stats-deeplab <stats.json> \
      --salida informes/etapa4_validacion_v6.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
from entrenar import modelo_con_indices, predecir, receta_de


def cargar_modelo_ckpt(ruta_ckpt, ruta_stats, disp='cpu'):
    import torch
    import segmentation_models_pytorch as smp
    ck = torch.load(ruta_ckpt, map_location='cpu', weights_only=False)
    cfg0 = ck.get('config') or {}
    arq = ck.get('arquitectura', 'unet')
    canales = list(cfg0.get('canales') or ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'])
    stats = json.load(open(ruta_stats, encoding='utf-8'))
    receta = receta_de(canales, stats)
    indices = bool((cfg0.get('v4') or {}).get('indices_verdor')) or any(k.startswith('base.') for k in ck['modelo'])
    clase = {'unet': smp.Unet, 'deeplabv3plus': smp.DeepLabV3Plus}[arq]
    modelo = clase(encoder_name=cfg0.get('encoder', 'resnet34'), encoder_weights=None,
                   in_channels=len(canales) + (2 if indices else 0), classes=1)
    if indices:
        modelo = modelo_con_indices(modelo, canales)
    modelo.load_state_dict(ck['modelo'], strict=True)
    modelo = modelo.to(disp).eval()
    return modelo, receta, ck


def predecir_tta(modelo, X, disp='cpu', usar_tta=True):
    """Prediccion con TTA de 4 espejos (original, flip H, flip V, flip HV)."""
    if not usar_tta:
        return predecir(modelo, X, disp)
    p0 = predecir(modelo, X, disp)
    ph = predecir(modelo, np.ascontiguousarray(X[:, :, :, ::-1]), disp)[:, :, ::-1]
    pv = predecir(modelo, np.ascontiguousarray(X[:, :, ::-1, :]), disp)[:, ::-1, :]
    phv = predecir(modelo, np.ascontiguousarray(X[:, :, ::-1, ::-1]), disp)[:, ::-1, ::-1]
    return (p0 + ph + pv + phv) * 0.25


def cargar_dem_y_worldcover(filas, carpeta_parches='parches_128', carpeta_wc='parches_128/worldcover'):
    """Carga elevacion DEM (m) y mapa ESA WorldCover 10m (uint8) alineados a cada parche."""
    elevs, wcs = [], []
    for r in filas:
        p_img = os.path.join(carpeta_parches, r['archivo'])
        with rasterio.open(p_img) as ds:
            elevs.append(ds.read(6).astype(np.float32))
        p_wc = os.path.join(carpeta_wc, r['proyecto'], r['parche'] + '.tif')
        if os.path.exists(p_wc):
            with rasterio.open(p_wc) as dsw:
                wcs.append(dsw.read(1))
        else:
            wcs.append(np.zeros((128, 128), dtype=np.uint8))
    return np.stack(elevs), np.stack(wcs)


def aplicar_filtro_altoandino(prob, elev, wc, modo='ninguno'):
    """Aplica filtro altoandino (ESA WorldCover 10 m + DEM) sobre probabilidades (N, 128, 128)."""
    if modo == 'ninguno':
        return prob
    out = prob.copy()
    if modo == 'wc_dem_3000':
        # Suprime pajonal/desnudo/nieve/musgo a >= 3000 m
        mask = (elev >= 3000.0) & np.isin(wc, [30, 60, 70, 100])
        out[mask] = 0.0
    elif modo == 'wc_dem_3200':
        # Suprime matorral/pajonal/desnudo/nieve/musgo a >= 3200 m
        mask = (elev >= 3200.0) & np.isin(wc, [20, 30, 60, 70, 100])
        out[mask] = 0.0
    elif modo == 'wc_dem_3400':
        mask = (elev >= 3400.0) & np.isin(wc, [20, 30, 60, 70, 100])
        out[mask] = 0.0
    elif modo == 'wc_dem_3200_suave':
        # Atenuacion suave (x0.25) en altoandino >= 3200 m
        mask = (elev >= 3200.0) & np.isin(wc, [20, 30, 60, 70, 100])
        out[mask] *= 0.25
    elif modo == 'wc_dem_3200_agua_urb':
        # Altoandino >= 3200 m + agua permanente (80) + urbano (50)
        mask = ((elev >= 3200.0) & np.isin(wc, [20, 30, 60, 70, 100])) | np.isin(wc, [50, 80])
        out[mask] = 0.0
    else:
        raise ValueError(f'Modo de filtro desconocido: {modo}')
    return out


def combinar_ensamble(p_unet, p_dl, modo='1.0_0.0', u_gate=0.20):
    """Combina probabilidades de U-Net y DeepLabV3+."""
    if modo == '1.0_0.0':
        return p_unet
    if modo == '0.0_1.0':
        return p_dl
    if modo == '0.8_0.2':
        return 0.8 * p_unet + 0.2 * p_dl
    if modo == '0.7_0.3':
        return 0.7 * p_unet + 0.3 * p_dl
    if modo == '0.5_0.5':
        return 0.5 * p_unet + 0.5 * p_dl
    if modo == 'deeplab_gated_by_unet':
        # Conserva probabilidades de DeepLab solo en componentes 8-conectados (>= u_gate) tocados por U-Net (>= u_gate)
        out = np.zeros_like(p_dl)
        st8 = np.ones((3, 3), dtype=bool)
        for i in range(len(p_dl)):
            cand = p_dl[i] >= u_gate
            if not cand.any():
                continue
            guia = p_unet[i] >= u_gate
            lab, n = ndimage.label(cand, structure=st8)
            for cid in range(1, n + 1):
                cm = (lab == cid)
                if (cm & guia).any():
                    out[i][cm] = p_dl[i][cm]
        return out
    raise ValueError(f'Modo de ensamble desconocido: {modo}')


def mejor_umbral_07_03(p_pe, Ypv, Vpv, p_lc, Yv, Vv, w_pe=0.7):
    b_pe = af.barrido(p_pe, Ypv, Vpv)
    b_lc = af.barrido(p_lc, Yv, Vv)
    pe_u = {round(float(r['umbral']), 2): r['iou'] for r in b_pe}
    lc_u = {round(float(r['umbral']), 2): r['iou'] for r in b_lc}
    comunes = sorted(set(pe_u) & set(lc_u))
    u_best = max(comunes, key=lambda u: w_pe * pe_u[u] + (1.0 - w_pe) * lc_u[u])
    return u_best, pe_u[u_best], lc_u[u_best], w_pe * pe_u[u_best] + (1.0 - w_pe) * lc_u[u_best]


def main():
    ap = argparse.ArgumentParser(description='Evaluacion de Etapa 4 (TTA, filtro altoandino, ensamble) SOLO en validacion')
    ap.add_argument('--unet', required=True)
    ap.add_argument('--stats-unet', required=True)
    ap.add_argument('--deeplab', required=True)
    ap.add_argument('--stats-deeplab', required=True)
    ap.add_argument('--parches', default='parches_128')
    ap.add_argument('--mascaras', default='mascaras_nivel1/parches_128')
    ap.add_argument('--l4spe', default='l4spe_profesor.npz')
    ap.add_argument('--hilos-cpu', type=int, default=12)
    ap.add_argument('--salida', default='informes/etapa4_validacion_v6.csv')
    a = ap.parse_args()

    import torch
    from nucleo import config as cfg_srv
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    if disp == 'cpu' and a.hilos_cpu > 0:
        torch.set_num_threads(a.hilos_cpu)

    m_unet, rec_u, _ = cargar_modelo_ckpt(a.unet, a.stats_unet, disp)
    m_dl, rec_d, _ = cargar_modelo_ckpt(a.deeplab, a.stats_deeplab, disp)
    escala = cfg_srv.ESCALA_L4S

    # Cargar SOLO validacion (La Convencion + L4SPe)
    Xv_u, Yv, Vv, Fv = af.cargar_locales(a.parches, a.mascaras, rec_u, escala, {'validacion'})
    Xv_d, _, _, _ = af.cargar_locales(a.parches, a.mascaras, rec_d, escala, {'validacion'})
    Xpv_u, Ypv, Vpv, Fpv = af.cargar_l4spe(a.l4spe, rec_u, {'validacion'})
    Xpv_d, _, _, _ = af.cargar_l4spe(a.l4spe, rec_d, {'validacion'})
    elev_v, wc_v = cargar_dem_y_worldcover(Fv, carpeta_parches=a.parches)

    # Precalcular predicciones sin TTA y con TTA (4 espejos)
    preds = {}
    for tta in (False, True):
        preds[(tta, 'unet_lc')] = predecir_tta(m_unet, Xv_u, disp, usar_tta=tta)
        preds[(tta, 'dl_lc')] = predecir_tta(m_dl, Xv_d, disp, usar_tta=tta)
        preds[(tta, 'unet_pe')] = predecir_tta(m_unet, Xpv_u, disp, usar_tta=tta)
        preds[(tta, 'dl_pe')] = predecir_tta(m_dl, Xpv_d, disp, usar_tta=tta)

    filtros = ['ninguno', 'wc_dem_3000', 'wc_dem_3200', 'wc_dem_3400', 'wc_dem_3200_suave', 'wc_dem_3200_agua_urb']
    ensambles = ['1.0_0.0', '0.0_1.0', '0.8_0.2', '0.7_0.3', '0.5_0.5', 'deeplab_gated_by_unet']

    rows = []
    for tta in (False, True):
        for ens in ensambles:
            p_lc_raw = combinar_ensamble(preds[(tta, 'unet_lc')], preds[(tta, 'dl_lc')], modo=ens)
            p_pe = combinar_ensamble(preds[(tta, 'unet_pe')], preds[(tta, 'dl_pe')], modo=ens)
            for flt in filtros:
                p_lc = aplicar_filtro_altoandino(p_lc_raw, elev_v, wc_v, modo=flt)
                u, iou_pe, iou_lc, punt = mejor_umbral_07_03(p_pe, Ypv, Vpv, p_lc, Yv, Vv)
                res_lc = af.resumen_local(p_lc, Yv, Vv, Fv, u)
                d10 = res_lc['deteccion_cicatriz_30pct']['10-50']
                dall = res_lc['deteccion_cicatriz_30pct']['total_ge10']
                rows.append({
                    'tta_4': int(tta), 'ensamble': ens, 'filtro_altoandino': flt, 'umbral': u,
                    'iou_val_l4spe': round(iou_pe, 4), 'iou_val_lc': round(iou_lc, 4),
                    'puntaje_07_03': round(punt, 4),
                    'iou_tol1px_8c_lc': res_lc['iou_tol1px_8c'],
                    'det_10_50_lc': f'{d10["detectadas"]}/{d10["total"]}',
                    'det_10_50_pct': d10['pct'],
                    'det_total_lc': f'{dall["detectadas"]}/{dall["total"]}',
                    'fa_ha_km2_lc': res_lc['falsas_alarmas_ha_por_km2_en_negativos'],
                })

    rows.sort(key=lambda r: (-r['puntaje_07_03'], r['fa_ha_km2_lc']))
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'Etapa 4 completa ({len(rows)} combinaciones en validacion) -> {a.salida}')
    print('Top 10 por puntaje 0.7/0.3:')
    for r in rows[:10]:
        print(' ', r)


if __name__ == '__main__':
    main()
