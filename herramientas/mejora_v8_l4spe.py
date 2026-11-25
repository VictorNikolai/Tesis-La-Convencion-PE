"""Pipeline de Mejora v8 (herramientas/mejora_v8_l4spe.py):
Acercar las metricas en L4SPe sin empeorar las falsas alarmas en prueba2.

Incluye:
  - Tarea 1: Umbral con histeresis (u_alto in 0.40..0.70, u_bajo in 0.15..0.40) en validacion.
  - Tarea 2: Post-proceso por objeto (min_px in {1,5,10,20}, fill_holes, closing 1 px) en validacion.
  - Tarea 3: Ensamble U-Net (B1_v5 o v6) + P3 con TTA 4 espejos ({0.5/0.5, 0.6/0.4, 0.7/0.3}) en validacion,
    mas histeresis y morfologia encima.
  - Verificacion de condicion de Tarea 4 (si det >= 2 ha en L4SPe val < 85.0%).
  - Tarea 5: Evaluacion unica (post hoc) en L4SPe test (124), prueba sellada (360) y prueba2 (349 negativos),
    con IC 95% por bootstrap de zonas (2,000 replicas), tabla de metas y 4 laminas en _fichas/parada_r_v8/.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import evaluar_etapa5_v6 as e5
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv


ST8 = np.ones((3, 3), dtype=bool)
U_ALTOS = [round(x, 2) for x in np.arange(0.40, 0.71, 0.05)]
U_BAJOS = [round(x, 2) for x in np.arange(0.15, 0.41, 0.05)]


def umbral_histeresis(prob, u_alto, u_bajo):
    """Mascara booleana (N, H, W) o (H, W): pixel positivo si prob >= u_bajo y su componente
    8-conectado contiene al menos un pixel con prob >= u_alto."""
    semilla = prob >= float(u_alto)
    mascara_baja = prob >= float(u_bajo)
    if prob.ndim == 2:
        if not semilla.any():
            return np.zeros_like(semilla, dtype=bool)
        return ndimage.binary_propagation(semilla, mask=mascara_baja, structure=ST8)
    out = np.zeros_like(semilla, dtype=bool)
    for i in range(len(prob)):
        if semilla[i].any():
            out[i] = ndimage.binary_propagation(semilla[i], mask=mascara_baja[i], structure=ST8)
    return out


def postproceso_objeto(mask_bool, min_px=1, fill_holes=False, closing_1px=False):
    """Aplica cierre morfologico 1 px, relleno de huecos y filtro de tamano minimo sobre mascara booleana."""
    if mask_bool.ndim == 2:
        m = mask_bool.copy()
        if closing_1px and m.any():
            m = ndimage.binary_closing(m, structure=ST8, iterations=1)
        if fill_holes and m.any():
            m = ndimage.binary_fill_holes(m)
        if min_px > 1 and m.any():
            lab, nc = ndimage.label(m, structure=ST8)
            if nc > 0:
                tams = np.bincount(lab.ravel())
                keep = tams >= int(min_px)
                keep[0] = False
                m = keep[lab]
        return m
    out = np.zeros_like(mask_bool, dtype=bool)
    for i in range(len(mask_bool)):
        if mask_bool[i].any():
            out[i] = postproceso_objeto(mask_bool[i], min_px=min_px, fill_holes=fill_holes, closing_1px=closing_1px)
    return out


def metricas_rapidas_mascara(mask_pred, Y, V, filas):
    """Calcula metricas clave en validacion a partir de una mascara binaria booleana (N, H, W):
    IoU/F1 exacto, IoU/F1 tol 1 y 2 px, metricas por objeto (>=10, >=50, >=200 px) y FA ha/km2 en teselas negativas."""
    prob_bin = mask_pred.astype(np.float32)
    _, p0, p1, p2, ob = mr.stats_por_zona(prob_bin, Y, V, filas, u=0.5, solo_ge50=False)
    m_p0 = mr.calc_metrics_from_counts(p0.sum(axis=0), is_obj=False)
    m_p1 = mr.calc_metrics_from_counts(p1.sum(axis=0), is_obj=False)
    m_p2 = mr.calc_metrics_from_counts(p2.sum(axis=0), is_obj=False)
    m_o10 = mr.calc_metrics_from_counts(ob[10].sum(axis=0), is_obj=True)
    m_o50 = mr.calc_metrics_from_counts(ob[50].sum(axis=0), is_obj=True)
    m_o200 = mr.calc_metrics_from_counts(ob[200].sum(axis=0), is_obj=True)
    c200 = [int(round(x)) for x in ob[200].sum(axis=0)]  # [tp, fn, fp]

    # FA en teselas negativas
    teselas = {}
    for i, r in enumerate(filas):
        t = teselas.setdefault((r['proyecto'], r['tesela']), [0.0, 0.0, 0.0])
        vi = V[i] > 0
        t[0] += float(((Y[i] > 0) & vi).sum())
        t[1] += float((mask_pred[i] & vi).sum())
        t[2] += float(vi.sum())
    neg = [t for t in teselas.values() if t[0] == 0]
    px_neg, fa_neg = sum(t[2] for t in neg), sum(t[1] for t in neg)
    fa_ha_km2 = round(fa_neg / max(px_neg, 1.0) * 100.0, 4)

    return {
        'iou_0px': round(float(m_p0['iou']), 4),
        'f1_0px': round(float(m_p0['f1']), 4),
        'iou_1px': round(float(m_p1['iou']), 4),
        'f1_1px': round(float(m_p1['f1']), 4),
        'iou_2px': round(float(m_p2['iou']), 4),
        'f1_2px': round(float(m_p2['f1']), 4),
        'obj10_f1': round(float(m_o10['f1']), 4),
        'obj10_prec': round(float(m_o10['prec']), 4),
        'obj10_rec': round(float(m_o10['rec']), 4),
        'obj50_f1': round(float(m_o50['f1']), 4),
        'obj50_rec': round(float(m_o50['rec']), 4),
        'obj200_f1': round(float(m_o200['f1']), 4),
        'obj200_rec': round(float(m_o200['rec']), 4),
        'obj200_tp': c200[0],
        'obj200_tot': c200[0] + c200[1],
        'fa_ha_km2': fa_ha_km2,
    }


def evaluar_mascara_bootstrap_completo(mask_pred, Y, V, filas):
    """Evalua una mascara booleana (N, H, W) con IC 95% por bootstrap de zonas (2,000 replicas)."""
    prob_bin = mask_pred.astype(np.float32)
    res = mr.evaluar_modelo_conjunto(prob_bin, Y, V, filas, u=0.5)
    # Agregar FA ha/km2 en teselas negativas
    teselas = {}
    for i, r in enumerate(filas):
        t = teselas.setdefault((r['proyecto'], r['tesela']), [0.0, 0.0, 0.0])
        vi = V[i] > 0
        t[0] += float(((Y[i] > 0) & vi).sum())
        t[1] += float((mask_pred[i] & vi).sum())
        t[2] += float(vi.sum())
    neg = [t for t in teselas.values() if t[0] == 0]
    res['fa_ha_km2_negativos'] = round(sum(t[1] for t in neg) / max(sum(t[2] for t in neg), 1.0) * 100.0, 4)
    return res


def rgb_l4spe(x_raw_6ch):
    """Construye vista RGB natural desde parche crudo L4SPe (B04, B03, B02 en indices 3, 2, 1)."""
    rgb = np.stack([x_raw_6ch[3], x_raw_6ch[2], x_raw_6ch[1]], axis=-1).astype(np.float32)
    v = np.isfinite(rgb) & (rgb > 0)
    if v.any():
        p2, p98 = np.percentile(rgb[v], [2, 98])
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return np.nan_to_num(rgb, nan=0.0)


def overlay(rgb, m, color=(1.0, 0.15, 0.15), alpha=0.55):
    out = rgb.copy()
    mb = m > 0
    for c in range(3):
        out[..., c] = np.where(mb, out[..., c] * (1.0 - alpha) + color[c] * alpha, out[..., c])
    return out


def ejecutar_tareas_1_a_3(ckpt_p3, stats_p3, ckpt_t4=None, stats_t4=None):
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S

    modelos_ck = {
        'B1_v5': ('resultados_ajuste/B1_unet_base/mejor_unet_b1b4b13b14_dfbbe6695037.pt',
                  'resultados_ajuste/B1_unet_base/stats_norm_dfbbe6695037.json', 0.30, 'ninguno'),
        'U-Net_v6': ('resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
                     'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', 0.45, 'wc_dem_3200_agua_urb'),
        'P3_prof': (ckpt_p3, stats_p3, 0.45, 'wc_dem_3200_agua_urb'),
    }
    if ckpt_t4 and stats_t4:
        modelos_ck['T4_unet'] = (ckpt_t4, stats_t4, 0.45, 'wc_dem_3200_agua_urb')

    mods = {}
    rec_u = None
    for k, (cp, sp, u_ref, flt_ref) in modelos_ck.items():
        m, rec, _ = pp6.cargar_modelo_ckpt(cp, sp, disp)
        mods[k] = (m, rec, u_ref, flt_ref)
        if rec_u is None:
            rec_u = rec

    # Cargar SOLO validacion (L4SPe val 127 + La Convencion val 304)
    Xv_lc, Yv_lc, Vv_lc, Fv_lc = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    ev_v, wc_v = pp6.cargar_dem_y_worldcover(Fv_lc)
    Xv_pe, Yv_pe, Vv_pe, Fv_pe = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'validacion'})

    # Probabilidades en validacion (sin TTA para T1/T2; con TTA 4 para T3)
    p_val_pe = {}
    p_val_lc = {}
    p_val_pe_tta = {}
    p_val_lc_tta = {}
    for k, (m, _, _, flt_ref) in mods.items():
        p_val_pe[k] = pp6.predecir_tta(m, Xv_pe, disp, usar_tta=False)
        plc = pp6.predecir_tta(m, Xv_lc, disp, usar_tta=False)
        if flt_ref != 'ninguno':
            plc = pp6.aplicar_filtro_altoandino(plc, ev_v, wc_v, modo=flt_ref)
        p_val_lc[k] = plc

        p_val_pe_tta[k] = pp6.predecir_tta(m, Xv_pe, disp, usar_tta=True)
        plc_t = pp6.predecir_tta(m, Xv_lc, disp, usar_tta=True)
        # En LC del ensamble usamos wc_dem_3200_agua_urb para proteger prueba2
        plc_t = pp6.aplicar_filtro_altoandino(plc_t, ev_v, wc_v, modo='wc_dem_3200_agua_urb')
        p_val_lc_tta[k] = plc_t

    # ==================== TAREA 1: Histeresis en validacion ====================
    t1_rows = []
    t1_best_por_modelo = {}
    # Cota de FA de referencia en LC val para cada modelo (su umbral base +20%, con piso minimo 0.001)
    for k, (_, _, u_ref, _) in mods.items():
        base_pe = metricas_rapidas_mascara((p_val_pe[k] >= u_ref) & (Vv_pe > 0), Yv_pe, Vv_pe, Fv_pe)
        base_lc = metricas_rapidas_mascara((p_val_lc[k] >= u_ref) & (Vv_lc > 0), Yv_lc, Vv_lc, Fv_lc)
        fa_lim = round(base_lc['fa_ha_km2'] * 1.20 + 1e-6, 4)
        t1_rows.append({
            'modelo': k, 'tipo': 'umbral_fijo_base', 'u_alto': u_ref, 'u_bajo': u_ref,
            'pe_obj200_rec': base_pe['obj200_rec'], 'pe_obj200_hits': f"{base_pe['obj200_tp']}/{base_pe['obj200_tot']}",
            'pe_f1_2px': base_pe['f1_2px'], 'pe_iou_0px': base_pe['iou_0px'], 'pe_obj10_f1': base_pe['obj10_f1'],
            'lc_iou_0px': base_lc['iou_0px'], 'lc_fa_ha_km2': base_lc['fa_ha_km2'], 'fa_limite_20pct': fa_lim,
            'admisible_fa': 1,
        })
        best = None
        for ua in U_ALTOS:
            for ub in U_BAJOS:
                if ub > ua:
                    continue
                m_pe = umbral_histeresis(p_val_pe[k], ua, ub) & (Vv_pe > 0)
                m_lc = umbral_histeresis(p_val_lc[k], ua, ub) & (Vv_lc > 0)
                r_pe = metricas_rapidas_mascara(m_pe, Yv_pe, Vv_pe, Fv_pe)
                r_lc = metricas_rapidas_mascara(m_lc, Yv_lc, Vv_lc, Fv_lc)
                adm = int(r_lc['fa_ha_km2'] <= fa_lim)
                row = {
                    'modelo': k, 'tipo': 'histeresis', 'u_alto': ua, 'u_bajo': ub,
                    'pe_obj200_rec': r_pe['obj200_rec'], 'pe_obj200_hits': f"{r_pe['obj200_tp']}/{r_pe['obj200_tot']}",
                    'pe_f1_2px': r_pe['f1_2px'], 'pe_iou_0px': r_pe['iou_0px'], 'pe_obj10_f1': r_pe['obj10_f1'],
                    'lc_iou_0px': r_lc['iou_0px'], 'lc_fa_ha_km2': r_lc['fa_ha_km2'], 'fa_limite_20pct': fa_lim,
                    'admisible_fa': adm,
                }
                t1_rows.append(row)
                if adm:
                    key_cmp = (r_pe['obj200_rec'], r_pe['f1_2px'], r_pe['iou_0px'], -r_lc['fa_ha_km2'])
                    if best is None or key_cmp > best[0]:
                        best = (key_cmp, row)
        t1_best_por_modelo[k] = best[1]
        print(f"[T1] {k:10s} base: det>=2ha={base_pe['obj200_rec']*100:.1f}% F1_2px={base_pe['f1_2px']*100:.1f}% FA_LC={base_lc['fa_ha_km2']:.4f} "
              f"-> mejor hist ({best[1]['u_alto']}/{best[1]['u_bajo']}): det>=2ha={best[1]['pe_obj200_rec']*100:.1f}% ({best[1]['pe_obj200_hits']}) "
              f"F1_2px={best[1]['pe_f1_2px']*100:.1f}% FA_LC={best[1]['lc_fa_ha_km2']:.4f}", flush=True)

    with open('informes/v8_tarea1_histeresis_val.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys())); w.writeheader(); w.writerows(t1_rows)

    # Elegir el mejor modelo de Tarea 1 entre los que mantienen FA_LC <= 1.20 * FA_B1_v5 (para no empeorar prueba2)
    fa_b1_lim = round(metricas_rapidas_mascara((p_val_lc['B1_v5'] >= 0.30) & (Vv_lc > 0), Yv_lc, Vv_lc, Fv_lc)['fa_ha_km2'] * 1.20 + 1e-6, 4)
    cands_t1 = [r for k, r in t1_best_por_modelo.items() if r['lc_fa_ha_km2'] <= fa_b1_lim]
    ganador_t1 = max(cands_t1, key=lambda r: (r['pe_obj200_rec'], r['pe_f1_2px'], r['pe_iou_0px']))
    print(f"[T1 GANADOR GLOBAL (FA_LC <= {fa_b1_lim})]: {ganador_t1}", flush=True)

    # ==================== TAREA 2: Post-proceso por objeto en validacion ====================
    mk_t1_pe = umbral_histeresis(p_val_pe[ganador_t1['modelo']], ganador_t1['u_alto'], ganador_t1['u_bajo']) & (Vv_pe > 0)
    mk_t1_lc = umbral_histeresis(p_val_lc[ganador_t1['modelo']], ganador_t1['u_alto'], ganador_t1['u_bajo']) & (Vv_lc > 0)
    t2_rows = []
    best_t2 = None
    for min_px in (1, 5, 10, 20):
        for fh in (False, True):
            for cl in (False, True):
                m_pe = postproceso_objeto(mk_t1_pe, min_px=min_px, fill_holes=fh, closing_1px=cl) & (Vv_pe > 0)
                m_lc = postproceso_objeto(mk_t1_lc, min_px=min_px, fill_holes=fh, closing_1px=cl) & (Vv_lc > 0)
                r_pe = metricas_rapidas_mascara(m_pe, Yv_pe, Vv_pe, Fv_pe)
                r_lc = metricas_rapidas_mascara(m_lc, Yv_lc, Vv_lc, Fv_lc)
                score_t2 = round(0.5 * r_pe['obj10_f1'] + 0.5 * r_pe['f1_2px'], 4)
                row = {
                    'modelo_base_t1': ganador_t1['modelo'],
                    'u_alto': ganador_t1['u_alto'], 'u_bajo': ganador_t1['u_bajo'],
                    'min_px': min_px, 'fill_holes': int(fh), 'closing_1px': int(cl),
                    'pe_obj10_f1': r_pe['obj10_f1'], 'pe_f1_2px': r_pe['f1_2px'],
                    'score_f1obj_f1tol2': score_t2,
                    'pe_obj200_rec': r_pe['obj200_rec'], 'pe_obj200_hits': f"{r_pe['obj200_tp']}/{r_pe['obj200_tot']}",
                    'pe_iou_0px': r_pe['iou_0px'], 'lc_fa_ha_km2': r_lc['fa_ha_km2'],
                }
                t2_rows.append(row)
                key_t2 = (score_t2, r_pe['obj200_rec'], -r_lc['fa_ha_km2'])
                if best_t2 is None or key_t2 > best_t2[0]:
                    best_t2 = (key_t2, row)
    ganador_t2 = best_t2[1]
    with open('informes/v8_tarea2_morfologia_val.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(t2_rows[0].keys())); w.writeheader(); w.writerows(t2_rows)
    print(f"[T2 GANADOR]: {ganador_t2}", flush=True)

    # ==================== TAREA 3: Ensamble con P3 + TTA 4 espejos ====================
    t3_rows = []
    best_t3 = None
    pares_ens = [('B1_v5', 'P3_prof'), ('U-Net_v6', 'P3_prof')]
    if 'T4_unet' in mods:
        pares_ens.append(('T4_unet', 'P3_prof'))
    for m_loc, m_p3 in pares_ens:
        for w_loc in (0.5, 0.6, 0.7):
            w_p3 = round(1.0 - w_loc, 2)
            pe_ens = w_loc * p_val_pe_tta[m_loc] + w_p3 * p_val_pe_tta[m_p3]
            lc_ens = w_loc * p_val_lc_tta[m_loc] + w_p3 * p_val_lc_tta[m_p3]
            u_opt, iou_pe, iou_lc, punt_07_03 = pp6.mejor_umbral_07_03(pe_ens, Yv_pe, Vv_pe, lc_ens, Yv_lc, Vv_lc)
            # Evaluar tambien aplicando encima histeresis (barrida respetando FA <= fa_v6_lim o fa_b1_lim) + morfologia T2
            # Cota estricta de FA en LC val para no empeorar prueba2: <= max(1.20 * FA(U-Net_v6), 0.045)
            fa_v6_base = metricas_rapidas_mascara((p_val_lc['U-Net_v6'] >= 0.45) & (Vv_lc > 0), Yv_lc, Vv_lc, Fv_lc)['fa_ha_km2']
            fa_cap_ens = round(max(fa_v6_base * 1.20, 0.035), 4)

            best_h_ens = None
            for ua in U_ALTOS:
                for ub in U_BAJOS:
                    if ub > ua:
                        continue
                    m_pe = postproceso_objeto(
                        umbral_histeresis(pe_ens, ua, ub) & (Vv_pe > 0),
                        min_px=ganador_t2['min_px'],
                        fill_holes=bool(ganador_t2['fill_holes']),
                        closing_1px=bool(ganador_t2['closing_1px']),
                    ) & (Vv_pe > 0)
                    m_lc = postproceso_objeto(
                        umbral_histeresis(lc_ens, ua, ub) & (Vv_lc > 0),
                        min_px=ganador_t2['min_px'],
                        fill_holes=bool(ganador_t2['fill_holes']),
                        closing_1px=bool(ganador_t2['closing_1px']),
                    ) & (Vv_lc > 0)
                    r_pe = metricas_rapidas_mascara(m_pe, Yv_pe, Vv_pe, Fv_pe)
                    r_lc = metricas_rapidas_mascara(m_lc, Yv_lc, Vv_lc, Fv_lc)
                    if r_lc['fa_ha_km2'] <= fa_cap_ens:
                        punt_h = 0.7 * r_pe['iou_0px'] + 0.3 * r_lc['iou_0px']
                        kc = (r_pe['obj200_rec'], r_pe['f1_2px'], punt_h, -r_lc['fa_ha_km2'])
                        if best_h_ens is None or kc > best_h_ens[0]:
                            best_h_ens = (kc, ua, ub, r_pe, r_lc, round(punt_h, 4))

            if best_h_ens is not None:
                _, ua_e, ub_e, r_pe_e, r_lc_e, punt_h_e = best_h_ens
                row = {
                    'par': f'{m_loc}+{m_p3}', 'm_loc': m_loc, 'w_loc': w_loc, 'w_p3': w_p3,
                    'u_simple': u_opt, 'puntaje_07_03_simple': round(punt_07_03, 4),
                    'u_alto': ua_e, 'u_bajo': ub_e,
                    'min_px': ganador_t2['min_px'], 'fill_holes': ganador_t2['fill_holes'], 'closing_1px': ganador_t2['closing_1px'],
                    'puntaje_07_03_hist_morf': punt_h_e,
                    'pe_iou_0px': r_pe_e['iou_0px'], 'pe_f1_2px': r_pe_e['f1_2px'],
                    'pe_obj10_f1': r_pe_e['obj10_f1'], 'pe_obj200_rec': r_pe_e['obj200_rec'],
                    'pe_obj200_hits': f"{r_pe_e['obj200_tp']}/{r_pe_e['obj200_tot']}",
                    'lc_iou_0px': r_lc_e['iou_0px'], 'lc_fa_ha_km2': r_lc_e['fa_ha_km2'],
                }
                t3_rows.append(row)
                # Elegir peso por puntaje 0.7 L4SPe + 0.3 LC y desempatar por det>=2ha / F1_2px
                k_sel = (row['puntaje_07_03_simple'], row['pe_obj200_rec'], row['pe_f1_2px'])
                if best_t3 is None or k_sel > best_t3[0]:
                    best_t3 = (k_sel, row)

    with open('informes/v8_tarea3_ensamble_val.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(t3_rows[0].keys())); w.writeheader(); w.writerows(t3_rows)

    # Elegir tanto el ganador por puntaje 0.7/0.3 como el de maxima det >=2ha en T3
    ganador_t3_punt = best_t3[1]
    ganador_t3_obj200 = max(t3_rows, key=lambda r: (r['pe_obj200_rec'], r['pe_f1_2px'], r['puntaje_07_03_hist_morf']))
    print(f"[T3 GANADOR 0.7/0.3]: {ganador_t3_punt}", flush=True)
    print(f"[T3 MAX DET>=2HA   ]: {ganador_t3_obj200}", flush=True)

    summary = {
        'tarea1_mejor_por_modelo': t1_best_por_modelo,
        'tarea1_ganador': ganador_t1,
        'tarea2_ganador': ganador_t2,
        'tarea3_ganador_puntaje': ganador_t3_punt,
        'tarea3_max_obj200': ganador_t3_obj200,
        'llega_85pct_obj200_val': bool(max(ganador_t2['pe_obj200_rec'], ganador_t3_obj200['pe_obj200_rec']) >= 0.85),
    }
    json.dump(summary, open('informes/v8_validacion_resumen.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    return summary


def ejecutar_tarea5_posthoc(ckpt_p3, stats_p3, cfg_ganadora, ckpt_extra=None, stats_extra=None):
    """Evalua UNA SOLA VEZ (post hoc) B1_v5, U-Net v6 y la configuracion ganadora v8 en:
    L4SPe test (124), prueba sellada (360) y prueba2 (349 negativos por tipo), y genera las 4 laminas."""
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S

    m_b1, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste/B1_unet_base/mejor_unet_b1b4b13b14_dfbbe6695037.pt',
        'resultados_ajuste/B1_unet_base/stats_norm_dfbbe6695037.json', disp)
    m_v6, _, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', disp)
    m_p3, _, _ = pp6.cargar_modelo_ckpt(ckpt_p3, stats_p3, disp)
    m_ex = None
    if ckpt_extra and stats_extra:
        m_ex, _, _ = pp6.cargar_modelo_ckpt(ckpt_extra, stats_extra, disp)

    # Cargar conjuntos de prueba UNA vez
    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ev_t, wc_t = pp6.cargar_dem_y_worldcover(Ft)
    Xp2, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    ev_p2, wc_p2 = pp6.cargar_dem_y_worldcover(Fp2)

    # 1) B1 v5 (referencia: sin TTA, u=0.30, sin filtro)
    pb1_pe = pp6.predecir_tta(m_b1, Xpt, disp, usar_tta=False) >= 0.30
    pb1_lc = pp6.predecir_tta(m_b1, Xt, disp, usar_tta=False) >= 0.30
    pb1_p2 = pp6.predecir_tta(m_b1, Xp2, disp, usar_tta=False)

    # 2) U-Net v6 (sin TTA + wc_dem_3200_agua_urb, u=0.45)
    pv6_pe = pp6.predecir_tta(m_v6, Xpt, disp, usar_tta=False) >= 0.45
    pv6_lc = pp6.aplicar_filtro_altoandino(pp6.predecir_tta(m_v6, Xt, disp, usar_tta=False), ev_t, wc_t, 'wc_dem_3200_agua_urb') >= 0.45
    pv6_p2 = pp6.aplicar_filtro_altoandino(pp6.predecir_tta(m_v6, Xp2, disp, usar_tta=False), ev_p2, wc_p2, 'wc_dem_3200_agua_urb')

    # 3) Ganador v8 (ensamble TTA 4 + wc_dem_3200_agua_urb en LC + histeresis + morfologia)
    m_loc_key = cfg_ganadora['m_loc']
    m_loc_obj = {'B1_v5': m_b1, 'U-Net_v6': m_v6, 'T4_unet': m_ex}[m_loc_key]
    wl, wp = float(cfg_ganadora['w_loc']), float(cfg_ganadora['w_p3'])
    ua, ub = float(cfg_ganadora['u_alto']), float(cfg_ganadora['u_bajo'])
    mpx = int(cfg_ganadora['min_px'])
    fh = bool(cfg_ganadora['fill_holes'])
    cl = bool(cfg_ganadora['closing_1px'])

    prob_v8_pe = wl * pp6.predecir_tta(m_loc_obj, Xpt, disp, True) + wp * pp6.predecir_tta(m_p3, Xpt, disp, True)
    prob_v8_lc = pp6.aplicar_filtro_altoandino(
        wl * pp6.predecir_tta(m_loc_obj, Xt, disp, True) + wp * pp6.predecir_tta(m_p3, Xt, disp, True),
        ev_t, wc_t, 'wc_dem_3200_agua_urb')
    prob_v8_p2 = pp6.aplicar_filtro_altoandino(
        wl * pp6.predecir_tta(m_loc_obj, Xp2, disp, True) + wp * pp6.predecir_tta(m_p3, Xp2, disp, True),
        ev_p2, wc_p2, 'wc_dem_3200_agua_urb')

    mv8_pe = postproceso_objeto(umbral_histeresis(prob_v8_pe, ua, ub) & (Vpt > 0), mpx, fh, cl) & (Vpt > 0)
    mv8_lc = postproceso_objeto(umbral_histeresis(prob_v8_lc, ua, ub) & (Vt > 0), mpx, fh, cl) & (Vt > 0)
    mv8_p2 = postproceso_objeto(umbral_histeresis(prob_v8_p2, ua, ub) & (Vp2 > 0), mpx, fh, cl) & (Vp2 > 0)

    etiqueta_v8 = f"Ganador v8 ({m_loc_key} {wl:.1f} + P3 {wp:.1f}, TTA4, hist {ua:.2f}/{ub:.2f}, min{mpx})"

    modelos_eval = {
        'U-Net B1 v5 (ref)': (pb1_pe & (Vpt > 0), pb1_lc & (Vt > 0), (pb1_p2 >= 0.30).astype(np.float32), 0.5),
        'U-Net v6 (sin TTA+flt)': (pv6_pe & (Vpt > 0), pv6_lc & (Vt > 0), (pv6_p2 >= 0.45).astype(np.float32), 0.5),
        etiqueta_v8: (mv8_pe, mv8_lc, mv8_p2.astype(np.float32), 0.5),
    }

    res_final = {'config_ganadora_v8': cfg_ganadora, 'modelos': {}}
    for nombre, (m_pe, m_lc, p_p2_bin, u_bin) in modelos_eval.items():
        print(f'[T5 POST HOC] Evaluando {nombre}...', flush=True)
        res_final['modelos'][nombre] = {
            'l4spe_test': evaluar_mascara_bootstrap_completo(m_pe, Ypt, Vpt, Fpt),
            'prueba_360': evaluar_mascara_bootstrap_completo(m_lc, Yt, Vt, Ft),
            'prueba2_fa': e5.eval_prueba2_fa(p_p2_bin, Vp2, Fp2, u_bin),
        }

    json.dump(res_final, open('informes/evaluacion_etapa5_v8.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # Generar 4 laminas en _fichas/parada_r_v8/ (2 objetos grandes recuperados en L4SPe + 2 errores)
    fdir = '_fichas/parada_r_v8'
    os.makedirs(fdir, exist_ok=True)
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_te = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        Xpt_raw = z['x'][sel_te]

    recuperados = []
    errores_fn = []
    errores_fp = []
    for i in range(len(Xpt)):
        yi = (Ypt[i] > 0) & (Vpt[i] > 0)
        b_old = pb1_pe[i] & (Vpt[i] > 0)
        b_v6 = pv6_pe[i] & (Vpt[i] > 0)
        b_new = mv8_pe[i]
        if yi.any():
            lab, nc = ndimage.label(yi, structure=ST8)
            for cid in range(1, nc + 1):
                cm = (lab == cid)
                sz = int(cm.sum())
                cov_b1 = float((cm & b_old).sum()) / sz
                cov_v6 = float((cm & b_v6).sum()) / sz
                cov_v8 = float((cm & b_new).sum()) / sz
                # Recuperado: <30% antes (en B1 o v6) y >=30% en v8 (o gran salto de cobertura en >=100 px)
                if sz >= 50 and cov_v8 >= 0.30 and (cov_b1 < 0.30 or cov_v6 < 0.30):
                    recuperados.append((sz, cov_v8 - min(cov_b1, cov_v6), i, cov_b1, cov_v6, cov_v8))
                elif sz >= 150 and (cov_v8 - min(cov_b1, cov_v6)) >= 0.15:
                    recuperados.append((sz, cov_v8 - min(cov_b1, cov_v6), i, cov_b1, cov_v6, cov_v8))
                if sz >= 20 and cov_v8 < 0.30:
                    errores_fn.append((sz, i, cov_v8))
        # Falsa alarma
        fp_px = int((b_new & (~yi) & (Vpt[i] > 0)).sum())
        if fp_px >= 20:
            errores_fp.append((fp_px, i))

    recuperados.sort(key=lambda x: (-x[0], -x[1]))
    usados = set()
    rec_sel = []
    for item in recuperados:
        if item[2] not in usados:
            rec_sel.append(item); usados.add(item[2])
        if len(rec_sel) == 2:
            break
    # Fallback si hiciera falta completar 2 con mayores mejoras de IoU en objetos >=200 px
    if len(rec_sel) < 2:
        cand_alt = []
        for i in range(len(Xpt)):
            if i in usados:
                continue
            yi = (Ypt[i] > 0) & (Vpt[i] > 0)
            if int(yi.sum()) >= 200:
                cov_v6 = float((yi & pv6_pe[i]).sum()) / int(yi.sum())
                cov_v8 = float((yi & mv8_pe[i]).sum()) / int(yi.sum())
                cand_alt.append((cov_v8 - cov_v6, int(yi.sum()), i, float((yi & pb1_pe[i]).sum()) / int(yi.sum()), cov_v6, cov_v8))
        cand_alt.sort(reverse=True)
        for d_cov, sz, i, cb1, cv6, cv8 in cand_alt:
            rec_sel.append((sz, d_cov, i, cb1, cv6, cv8)); usados.add(i)
            if len(rec_sel) == 2:
                break

    def plot_l4spe(idx, out_path, subtitulo):
        rgb = rgb_l4spe(Xpt_raw[idx])
        gt = (Ypt[idx] > 0) & (Vpt[idx] > 0)
        paneles = [
            (f"L4SPe {Fpt[idx]['parche']}\nRGB (S2 L1C)", rgb),
            (f"Experto GT\n({int(gt.sum())} px = {gt.sum()*0.01:.2f} ha)", overlay(rgb, gt, (0.1, 0.9, 0.2), 0.55)),
            (f"U-Net B1 v5 (u=0.30)\n({int((pb1_pe[idx]&(Vpt[idx]>0)).sum())} px)", overlay(rgb, pb1_pe[idx] & (Vpt[idx] > 0))),
            (f"U-Net v6 (u=0.45)\n({int((pv6_pe[idx]&(Vpt[idx]>0)).sum())} px)", overlay(rgb, pv6_pe[idx] & (Vpt[idx] > 0))),
            (f"Ganador v8 (hist+morf)\n({int(mv8_pe[idx].sum())} px)", overlay(rgb, mv8_pe[idx])),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=140)
        for ax, (tit, im) in zip(axes, paneles):
            ax.imshow(im); ax.set_title(tit, fontsize=8.8, fontweight='bold'); ax.axis('off')
        fig.suptitle(f"{Fpt[idx]['parche']} (zona {Fpt[idx]['zona']}) — {subtitulo}", fontsize=10.5, fontweight='bold', y=1.05)
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    for k, (sz, _, idx, cb1, cv6, cv8) in enumerate(rec_sel, 1):
        plot_l4spe(idx, os.path.join(fdir, f"0{k}_recuperado_{Fpt[idx]['parche']}.png"),
                   f"Objeto grande recuperado ({sz} px): cob. B1={cb1*100:.0f}%, v6={cv6*100:.0f}% -> v8={cv8*100:.0f}%")

    errores_fn.sort(reverse=True)
    errores_fp.sort(reverse=True)
    err_sel = []
    for sz, idx, cv8 in errores_fn:
        if idx not in usados:
            err_sel.append((idx, f"Error FN en cicatriz de {sz} px (cob. v8={cv8*100:.0f}%)"))
            usados.add(idx)
            break
    for fp_px, idx in errores_fp:
        if idx not in usados:
            err_sel.append((idx, f"Error FP: {fp_px} px fuera de la mascara del experto"))
            usados.add(idx)
            break
    for sz, idx, cv8 in errores_fn:
        if len(err_sel) >= 2:
            break
        if idx not in usados:
            err_sel.append((idx, f"Error FN en cicatriz de {sz} px (cob. v8={cv8*100:.0f}%)"))
            usados.add(idx)
    for fp_px, idx in errores_fp:
        if len(err_sel) >= 2:
            break
        if idx not in usados:
            err_sel.append((idx, f"Error FP: {fp_px} px fuera de la mascara del experto"))
            usados.add(idx)
    for k, (idx, desc) in enumerate(err_sel[:2], 3):
        plot_l4spe(idx, os.path.join(fdir, f"0{k}_error_{Fpt[idx]['parche']}.png"), desc)

    # Escribir informes/resultado_mejora_v8.md con la tabla final de metas (como metricas_reformuladas.md)
    escribir_informe_v8(res_final, etiqueta_v8)
    print('Tarea 5 post hoc completada -> informes/resultado_mejora_v8.md y laminas en', fdir, flush=True)


def escribir_informe_v8(res_final, etiqueta_v8):
    fmt = mr.fmt_ic
    md = [
        '# Resultados Mejora v8 — Evaluación Única «Post Hoc» (Tarea 5)',
        '',
        '> [!IMPORTANT]',
        '> **Regla metodológica:** Todos los hiperparámetros de histéresis, post-proceso morfológico y pesos de ensamble fueron elegidos **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Todas las cifras sobre `L4SPe test`, `prueba` (360) y `prueba2` son **«post hoc»**, con **IC 95 % por bootstrap de zonas (`2 000` réplicas)**.',
        '',
        '## 1. L4SPe Test (`124` parches, `68` zonas) — «post hoc»',
        '',
        '### 1a. Por píxel (tolerancia `0`, `1` y `2 px`)',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | F1 0 px [IC 95 %] | IoU 1 px [IC 95 %] | F1 1 px [IC 95 %] | IoU 2 px [IC 95 %] | **F1 2 px [IC 95 %]** | FA neg. (`ha/km²`) |',
        '|---|---|---|---|---|---|---|---:|',
    ]
    for nm, dm in res_final['modelos'].items():
        t = dm['l4spe_test']['todas']
        fa = dm['l4spe_test']['fa_ha_km2_negativos']
        md.append(
            f"| {nm} | {fmt(t['pixel_exacto'],'iou')} | {fmt(t['pixel_exacto'],'f1',True)} | {fmt(t['pixel_tol1px'],'iou')} | {fmt(t['pixel_tol1px'],'f1',True)} | {fmt(t['pixel_tol2px'],'iou')} | **{fmt(t['pixel_tol2px'],'f1',True)}** | {fa:.3f} |"
        )
    md.extend([
        '',
        '### 1b. Por objeto en L4SPe Test (`total ≥ 10 px`, `≥ 0.5 ha`, `≥ 2 ha`)',
        '',
        '| Modelo | Tramo | TP / FN / FP | Precisión [IC 95 %] | **Exhaustividad [IC 95 %]** | **F1 objeto [IC 95 %]** |',
        '|---|---|---|---:|---:|---:|',
    ])
    for nm, dm in res_final['modelos'].items():
        t = dm['l4spe_test']['todas']
        for bk, bl in [('obj_ge10', 'Total (`≥ 0.1 ha`)'), ('obj_ge50', '`≥ 0.5 ha` (`50 px`)'), ('obj_ge200', '`≥ 2 ha` (`200 px`)')]:
            ob = t[bk]; c = ob['counts']
            md.append(f"| {nm} | {bl} | `{c[0]} / {c[1]} / {c[2]}` | {fmt(ob,'prec',True)} | **{fmt(ob,'rec',True)}** | **{fmt(ob,'f1',True)}** |")

    md.extend([
        '',
        '## 2. Prueba Sellada de La Convención (`360` parches, `62` zonas) — «post hoc»',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | IoU 1 px [IC 95 %] | IoU 2 px [IC 95 %] | F1 2 px [IC 95 %] | Recall obj `≥ 0.5 ha` | Recall obj `≥ 2 ha` | FA neg. (`ha/km²`) |',
        '|---|---|---|---|---|---|---|---:|',
    ])
    for nm, dm in res_final['modelos'].items():
        t = dm['prueba_360']['todas']
        fa = dm['prueba_360']['fa_ha_km2_negativos']
        md.append(
            f"| {nm} | {fmt(t['pixel_exacto'],'iou')} | {fmt(t['pixel_tol1px'],'iou')} | {fmt(t['pixel_tol2px'],'iou')} | {fmt(t['pixel_tol2px'],'f1',True)} | {fmt(t['obj_ge50'],'rec',True)} | {fmt(t['obj_ge200'],'rec',True)} | {fa:.3f} |"
        )

    md.extend([
        '',
        '## 3. `prueba2` Sellada (`349` negativos confirmados por `tipo_negativo`) — Verificación de Falsas Alarmas',
        '',
        '| Modelo | **TOTAL (`ha/km²` / `%`)** | `altoandino` (49) | `chacra_o_pasto` (9) | `rio` (34) | `carretera` (8) | `poblado` (6) | `bosque_ladera` (243) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for nm, dm in res_final['modelos'].items():
        p2 = dm['prueba2_fa']
        md.append(
            f"| {nm} | **{p2['TOTAL_349']['fa_ha_km2']:.4f} ({p2['TOTAL_349']['pct_con_marca']} %)** | "
            f"{p2['altoandino']['fa_ha_km2']:.4f} | {p2['chacra_o_pasto']['fa_ha_km2']:.4f} | "
            f"{p2['rio']['fa_ha_km2']:.4f} | {p2['carretera']['fa_ha_km2']:.4f} | "
            f"{p2['poblado']['fa_ha_km2']:.4f} | {p2['bosque_ladera']['fa_ha_km2']:.4f} |"
        )

    # Tabla final contra metas del proyecto (TODAS las formulaciones)
    md.extend([
        '',
        '## 4. Meta del Proyecto frente al Mejor Valor (Todas las Formulaciones, «post hoc»)',
        '',
        '| Meta del proyecto | Formulación evaluada | Mejor valor en `prueba` (360) *(modelo)* | Mejor valor en `L4SPe` test *(modelo)* | ¿Cumple la meta? |',
        '|---|---|---|---|---|',
    ])
    def best_hi(ck, fn):
        return max(((fn(dm[ck])[0], nm, fn(dm[ck])[1]) for nm, dm in res_final['modelos'].items()), key=lambda x: x[0])
    def best_lo(ck, fn):
        return min(((fn(dm[ck])[0], nm, fn(dm[ck])[1]) for nm, dm in res_final['modelos'].items()), key=lambda x: x[0])

    metas = [
        ('**IoU ≥ 0.70**', 'Píxel exacto (0 px)', lambda c: (c['todas']['pixel_exacto']['iou'], fmt(c['todas']['pixel_exacto'],'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 1 px', lambda c: (c['todas']['pixel_tol1px']['iou'], fmt(c['todas']['pixel_tol1px'],'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 2 px', lambda c: (c['todas']['pixel_tol2px']['iou'], fmt(c['todas']['pixel_tol2px'],'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 2 px (solo `≥ 0.5 ha`)', lambda c: (c['solo_ge50']['pixel_tol2px']['iou'], fmt(c['solo_ge50']['pixel_tol2px'],'iou')), False, 0.70),
        ('**F1 > 75.5 %**', 'Píxel exacto (0 px)', lambda c: (c['todas']['pixel_exacto']['f1'], fmt(c['todas']['pixel_exacto'],'f1',True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Píxel tol. 1 px', lambda c: (c['todas']['pixel_tol1px']['f1'], fmt(c['todas']['pixel_tol1px'],'f1',True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Píxel tol. 2 px', lambda c: (c['todas']['pixel_tol2px']['f1'], fmt(c['todas']['pixel_tol2px'],'f1',True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto total (`≥ 0.1 ha`)', lambda c: (c['todas']['obj_ge10']['f1'], fmt(c['todas']['obj_ge10'],'f1',True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto `≥ 0.5 ha` (`50 px`)', lambda c: (c['todas']['obj_ge50']['f1'], fmt(c['todas']['obj_ge50'],'f1',True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto `≥ 2 ha` (`200 px`)', lambda c: (c['todas']['obj_ge200']['f1'], fmt(c['todas']['obj_ge200'],'f1',True)), False, 0.755),
        ('**Recall > 85 %**', 'Píxel exacto (0 px)', lambda c: (c['todas']['pixel_exacto']['rec'], fmt(c['todas']['pixel_exacto'],'rec',True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto total (`≥ 0.1 ha`)', lambda c: (c['todas']['obj_ge10']['rec'], fmt(c['todas']['obj_ge10'],'rec',True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto `≥ 0.5 ha` (`50 px`)', lambda c: (c['todas']['obj_ge50']['rec'], fmt(c['todas']['obj_ge50'],'rec',True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto `≥ 2 ha` (`200 px`)', lambda c: (c['todas']['obj_ge200']['rec'], fmt(c['todas']['obj_ge200'],'rec',True)), False, 0.85),
        ('**FP ≤ 15 %**', 'Tasa FP píxel (`1 − Prec`)', lambda c: (c['todas']['pixel_exacto']['fp_rate'], fmt(c['todas']['pixel_exacto'],'fp_rate',True)), True, 0.15),
        ('**FP ≤ 15 %**', 'Tasa FP objeto (`≥ 0.1 ha`)', lambda c: (c['todas']['obj_ge10']['fp_rate'], fmt(c['todas']['obj_ge10'],'fp_rate',True)), True, 0.15),
        ('**FP ≤ 15 %**', 'Tasa FP objeto (`≥ 0.5 ha`)', lambda c: (c['todas']['obj_ge50']['fp_rate'], fmt(c['todas']['obj_ge50'],'fp_rate',True)), True, 0.15),
    ]
    for meta, form, fn, low, thr in metas:
        if low:
            vlc, mlc, slc = best_lo('prueba_360', fn)
            vpe, mpe, spe = best_lo('l4spe_test', fn)
            ok = 'Sí (ambos)' if (vlc <= thr and vpe <= thr) else ('Sí (L4SPe)' if vpe <= thr else ('Sí (LC)' if vlc <= thr else 'No'))
        else:
            vlc, mlc, slc = best_hi('prueba_360', fn)
            vpe, mpe, spe = best_hi('l4spe_test', fn)
            ok = 'Sí (ambos)' if (vlc >= thr and vpe >= thr) else ('Sí (L4SPe)' if vpe >= thr else ('Sí (LC)' if vlc >= thr else 'No'))
        md.append(f'| {meta} | {form} | {slc} *({mlc})* | {spe} *({mpe})* | **{ok}** |')

    with open('informes/resultado_mejora_v8.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', choices=['val', 'posthoc'], required=True)
    ap.add_argument('--ckpt-p3', required=True)
    ap.add_argument('--stats-p3', required=True)
    ap.add_argument('--ckpt-t4', default=None)
    ap.add_argument('--stats-t4', default=None)
    ap.add_argument('--usar-max-obj200', action='store_true')
    a = ap.parse_args()

    if a.fase == 'val':
        ejecutar_tareas_1_a_3(a.ckpt_p3, a.stats_p3, a.ckpt_t4, a.stats_t4)
    else:
        res_v = json.load(open('informes/v8_validacion_resumen.json', encoding='utf-8'))
        cfg = res_v['tarea3_max_obj200'] if a.usar_max_obj200 else res_v['tarea3_ganador_puntaje']
        ejecutar_tarea5_posthoc(a.ckpt_p3, a.stats_p3, cfg, a.ckpt_t4, a.stats_t4)


if __name__ == '__main__':
    main()
