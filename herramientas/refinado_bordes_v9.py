"""Tarea 3 (mejora-iou): Refinado de bordes sobre la salida de v9 (validacion + post hoc).

Prueba en validacion (L4SPe val 127 + La Convencion val 304) dos metodos:
  a) Crecimiento de region guiado por probabilidad (umbral bajo u_crec dentro de <= 2 px de lo ya detectado).
  b) Filtro guiado (Guided Filter) usando las bandas RGB (B04, B03, B02).

Si ninguno mejora en validacion frente a v9, no se aplica y se documenta en informes/refinado_bordes.md.
"""
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
import mejora_v8_l4spe as v8
import mejora_v9 as v9
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv

ST8 = np.ones((3, 3), dtype=bool)


def crecimiento_region_prob(mask_base, prob, u_crec, dist_px=2):
    """Agrega pixeles con prob >= u_crec dentro de dist_px (8-conectado) de mask_base."""
    out = mask_base.copy()
    for i in range(len(mask_base)):
        if not mask_base[i].any():
            continue
        halo = ndimage.binary_dilation(mask_base[i], structure=ST8, iterations=int(dist_px))
        out[i] = mask_base[i] | (halo & (prob[i] >= float(u_crec)))
    return out


def guided_filter_rgb(prob_2d, rgb_2d, radius=3, eps=1e-2):
    """Filtro guiado rapido (He et al.) sobre guia RGB (H, W, 3) normalizada en [0, 1]."""
    k = 2 * int(radius) + 1
    I = rgb_2d.astype(np.float64)
    p = prob_2d.astype(np.float64)
    # Promedios locales via uniform_filter
    mean_I = np.stack([ndimage.uniform_filter(I[..., c], size=k) for c in range(3)], axis=-1)
    mean_p = ndimage.uniform_filter(p, size=k)
    mean_Ip = np.stack([ndimage.uniform_filter(I[..., c] * p, size=k) for c in range(3)], axis=-1)
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]

    # Matriz de covarianza 3x3 por pixel + eps*I3
    rr, rg, rb = I[..., 0], I[..., 1], I[..., 2]
    mr_i, mg_i, mb_i = mean_I[..., 0], mean_I[..., 1], mean_I[..., 2]
    var_rr = ndimage.uniform_filter(rr * rr, size=k) - mr_i * mr_i + eps
    var_rg = ndimage.uniform_filter(rr * rg, size=k) - mr_i * mg_i
    var_rb = ndimage.uniform_filter(rr * rb, size=k) - mr_i * mb_i
    var_gg = ndimage.uniform_filter(rg * rg, size=k) - mg_i * mg_i + eps
    var_gb = ndimage.uniform_filter(rg * rb, size=k) - mg_i * mb_i
    var_bb = ndimage.uniform_filter(rb * rb, size=k) - mb_i * mb_i + eps

    # Resolver Sigma * a = cov_Ip por regla de Cramer 3x3 vectorizada
    c00 = var_gg * var_bb - var_gb * var_gb
    c01 = var_rb * var_gb - var_rg * var_bb
    c02 = var_rg * var_gb - var_rb * var_gg
    c11 = var_rr * var_bb - var_rb * var_rb
    c12 = var_rb * var_rg - var_rr * var_gb
    c22 = var_rr * var_gg - var_rg * var_rg
    det = var_rr * c00 + var_rg * c01 + var_rb * c02
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    cr, cg, cb = cov_Ip[..., 0], cov_Ip[..., 1], cov_Ip[..., 2]
    a0 = (c00 * cr + c01 * cg + c02 * cb) / det
    a1 = (c01 * cr + c11 * cg + c12 * cb) / det
    a2 = (c02 * cr + c12 * cg + c22 * cb) / det
    b = mean_p - (a0 * mr_i + a1 * mg_i + a2 * mb_i)

    q = (
        ndimage.uniform_filter(a0, size=k) * rr
        + ndimage.uniform_filter(a1, size=k) * rg
        + ndimage.uniform_filter(a2, size=k) * rb
        + ndimage.uniform_filter(b, size=k)
    )
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def rgb_norm_desde_raw(x_raw_6ch):
    rgb = np.stack([x_raw_6ch[3], x_raw_6ch[2], x_raw_6ch[1]], axis=-1).astype(np.float32)
    v = np.isfinite(rgb) & (rgb > 0)
    if v.any():
        p2, p98 = np.percentile(rgb[v], [2, 98])
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return np.nan_to_num(rgb, nan=0.0)


def main():
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S
    cfg_v9 = json.load(open('modelos_ajustados/v9.json', encoding='utf-8'))

    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', disp)
    m_p3, _, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v8/P3_solo_profesor/mejor_unet_b1b4b13b14_da008a8d0de3.pt',
        'resultados_ajuste_v8/P3_solo_profesor/stats_norm_da008a8d0de3.json', disp)

    # Cargar VALIDACION
    Xv_lc, Yv_lc, Vv_lc, Fv_lc = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    ctx_lc = v9.cargar_contexto_espacial_locales(Fv_lc)
    Xraw_lc = v9._cargar_raw_locales(Fv_lc)
    Xv_pe, Yv_pe, Vv_pe, Fv_pe = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'validacion'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_v = [i for i, c in enumerate(z['conjunto']) if str(c) == 'validacion']
        Xraw_pe = z['x'][sel_v]
        elev_pe = Xraw_pe[:, 5, :, :].astype(np.float32)

    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    pv9_lc = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xv_lc, disp, True), pp6.predecir_tta(m_p3, Xv_lc, disp, True), ctx_lc['elev'], ctx_lc['wc'], corte, cond_wc)
    pv9_lc[np.isin(ctx_lc['wc'], [50, 80])] = 0.0
    pv9_pe = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xv_pe, disp, True), pp6.predecir_tta(m_p3, Xv_pe, disp, True), elev_pe, None, corte, cond_wc)

    mv9_lc = v8.postproceso_objeto(
        v9.aplicar_e4_filtros(v8.umbral_histeresis(pv9_lc, ua, ub) & (Vv_lc > 0), ctx_lc, False, True, True) & (Vv_lc > 0),
        mpx, False, cl1) & (Vv_lc > 0)
    mv9_pe = v8.postproceso_objeto(v8.umbral_histeresis(pv9_pe, ua, ub) & (Vv_pe > 0), mpx, False, cl1) & (Vv_pe > 0)

    r_base_lc = v8.metricas_rapidas_mascara(mv9_lc, Yv_lc, Vv_lc, Fv_lc)
    r_base_pe = v8.metricas_rapidas_mascara(mv9_pe, Yv_pe, Vv_pe, Fv_pe)
    punt_base = round(0.7 * r_base_pe['iou_0px'] + 0.3 * r_base_lc['iou_0px'], 4)
    print(f'[BASE v9 val] pe_iou={r_base_pe["iou_0px"]} lc_iou={r_base_lc["iou_0px"]} punt={punt_base} FA_LC={r_base_lc["fa_ha_km2"]}', flush=True)

    # Metodo (a): Crecimiento de region guiado por probabilidad dentro de <= 2 px
    filas_val = []
    for u_c in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        mlc = crecimiento_region_prob(mv9_lc, pv9_lc, u_c, dist_px=2) & (Vv_lc > 0)
        mlc[np.isin(ctx_lc['wc'], [50, 80])] = False
        mpe = crecimiento_region_prob(mv9_pe, pv9_pe, u_c, dist_px=2) & (Vv_pe > 0)
        rlc = v8.metricas_rapidas_mascara(mlc, Yv_lc, Vv_lc, Fv_lc)
        rpe = v8.metricas_rapidas_mascara(mpe, Yv_pe, Vv_pe, Fv_pe)
        punt = round(0.7 * rpe['iou_0px'] + 0.3 * rlc['iou_0px'], 4)
        filas_val.append({
            'metodo': f'crecimiento_2px(u={u_c:.2f})', 'tipo': 'a',
            'pe_iou': rpe['iou_0px'], 'pe_f1': rpe['f1_0px'],
            'lc_iou': rlc['iou_0px'], 'lc_f1': rlc['f1_0px'],
            'fa_lc': rlc['fa_ha_km2'], 'puntaje_07_03': punt,
            'mejora_punt': round(punt - punt_base, 4),
        })

    # Metodo (b): Filtro guiado RGB sobre probabilidad v9 (conservando componentes tocados por mv9)
    rgbs_lc = [rgb_norm_desde_raw(Xraw_lc[i]) for i in range(len(Xraw_lc))]
    rgbs_pe = [rgb_norm_desde_raw(Xraw_pe[i]) for i in range(len(Xraw_pe))]
    for rad in (2, 3, 5):
        for eps in (1e-3, 1e-2, 5e-2):
            q_lc = np.stack([guided_filter_rgb(pv9_lc[i], rgbs_lc[i], radius=rad, eps=eps) for i in range(len(pv9_lc))])
            q_pe = np.stack([guided_filter_rgb(pv9_pe[i], rgbs_pe[i], radius=rad, eps=eps) for i in range(len(pv9_pe))])
            for u_g in (0.30, 0.35, 0.40, 0.45):
                # Conservar solo pixeles >= u_g en componentes 8-con tocados por mv9 (o dentro de 2 px)
                mlc = np.zeros_like(mv9_lc)
                for i in range(len(mv9_lc)):
                    if mv9_lc[i].any():
                        halo = ndimage.binary_dilation(mv9_lc[i], structure=ST8, iterations=2)
                        mlc[i] = halo & (q_lc[i] >= u_g) & (~np.isin(ctx_lc['wc'][i], [50, 80]))
                mpe = np.zeros_like(mv9_pe)
                for i in range(len(mv9_pe)):
                    if mv9_pe[i].any():
                        halo = ndimage.binary_dilation(mv9_pe[i], structure=ST8, iterations=2)
                        mpe[i] = halo & (q_pe[i] >= u_g)
                mlc = mlc & (Vv_lc > 0)
                mpe = mpe & (Vv_pe > 0)
                rlc = v8.metricas_rapidas_mascara(mlc, Yv_lc, Vv_lc, Fv_lc)
                rpe = v8.metricas_rapidas_mascara(mpe, Yv_pe, Vv_pe, Fv_pe)
                punt = round(0.7 * rpe['iou_0px'] + 0.3 * rlc['iou_0px'], 4)
                filas_val.append({
                    'metodo': f'guided_rgb(r={rad},eps={eps},u={u_g:.2f})', 'tipo': 'b',
                    'pe_iou': rpe['iou_0px'], 'pe_f1': rpe['f1_0px'],
                    'lc_iou': rlc['iou_0px'], 'lc_f1': rlc['f1_0px'],
                    'fa_lc': rlc['fa_ha_km2'], 'puntaje_07_03': punt,
                    'mejora_punt': round(punt - punt_base, 4),
                })

    # Seleccion SOLO en validacion: mayor puntaje 0.7/0.3 sujeto a FA_LC <= FA_base * 1.05
    cands_ok = [r for r in filas_val if r['fa_lc'] <= r_base_lc['fa_ha_km2'] * 1.05 + 1e-6]
    best = max(cands_ok, key=lambda r: (r['puntaje_07_03'], -r['fa_lc'])) if cands_ok else max(filas_val, key=lambda r: r['puntaje_07_03'])
    mejora_real = best['mejora_punt'] > 0.001
    print(f'[MEJOR VAL]: {best}, mejora_real={mejora_real}', flush=True)

    # Evaluar en post hoc SOLO si mejora en validacion (o reportar comparativa post hoc del mejor candidato)
    # Evaluamos el mejor candidato en post hoc con bootstrap 1000 para documentar su efecto real frente a v9
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = v9.cargar_contexto_espacial_locales(Ft)
    Xraw_t = v9._cargar_raw_locales(Ft)
    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_t = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        Xraw_pt = z['x'][sel_t]
        elev_pt = Xraw_pt[:, 5, :, :].astype(np.float32)

    pv9_t = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xt, disp, True), pp6.predecir_tta(m_p3, Xt, disp, True), ctx_t['elev'], ctx_t['wc'], corte, cond_wc)
    pv9_t[np.isin(ctx_t['wc'], [50, 80])] = 0.0
    pv9_pt = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xpt, disp, True), pp6.predecir_tta(m_p3, Xpt, disp, True), elev_pt, None, corte, cond_wc)

    mv9_t = v8.postproceso_objeto(
        v9.aplicar_e4_filtros(v8.umbral_histeresis(pv9_t, ua, ub) & (Vt > 0), ctx_t, False, True, True) & (Vt > 0),
        mpx, False, cl1) & (Vt > 0)
    mv9_pt = v8.postproceso_objeto(v8.umbral_histeresis(pv9_pt, ua, ub) & (Vpt > 0), mpx, False, cl1) & (Vpt > 0)

    # Aplicar metodo ganador de validacion sobre prueba
    if best['tipo'] == 'a':
        u_c = float(best['metodo'].split('u=')[1].rstrip(')'))
        mref_t = crecimiento_region_prob(mv9_t, pv9_t, u_c, 2) & (Vt > 0)
        mref_t[np.isin(ctx_t['wc'], [50, 80])] = False
        mref_pt = crecimiento_region_prob(mv9_pt, pv9_pt, u_c, 2) & (Vpt > 0)
    else:
        parts = best['metodo'].split('(')[1].rstrip(')').split(',')
        rad = int(parts[0].split('=')[1]); eps = float(parts[1].split('=')[1]); u_g = float(parts[2].split('=')[1])
        mref_t = np.zeros_like(mv9_t)
        for i in range(len(mv9_t)):
            if mv9_t[i].any():
                qi = guided_filter_rgb(pv9_t[i], rgb_norm_desde_raw(Xraw_t[i]), radius=rad, eps=eps)
                mref_t[i] = ndimage.binary_dilation(mv9_t[i], structure=ST8, iterations=2) & (qi >= u_g) & (~np.isin(ctx_t['wc'][i], [50, 80]))
        mref_pt = np.zeros_like(mv9_pt)
        for i in range(len(mv9_pt)):
            if mv9_pt[i].any():
                qi = guided_filter_rgb(pv9_pt[i], rgb_norm_desde_raw(Xraw_pt[i]), radius=rad, eps=eps)
                mref_pt[i] = ndimage.binary_dilation(mv9_pt[i], structure=ST8, iterations=2) & (qi >= u_g)
        mref_t &= (Vt > 0); mref_pt &= (Vpt > 0)

    def boot_iou(mask_p, Y, V, F):
        _, p0, _, _, _ = mr.stats_por_zona(mask_p.astype(np.float32), Y, V, F, u=0.5, solo_ge50=False)
        return mr.bootstrap_metricas(p0, is_obj=False, n_boot=1000, seed=7)

    b_v9_pe = boot_iou(mv9_pt, Ypt, Vpt, Fpt)
    b_rf_pe = boot_iou(mref_pt, Ypt, Vpt, Fpt)
    b_v9_lc = boot_iou(mv9_t, Yt, Vt, Ft)
    b_rf_lc = boot_iou(mref_t, Yt, Vt, Ft)
    diff_pe = v9._bootstrap_pareado_diff(mref_pt, mv9_pt, Ypt, Vpt, Fpt, n_boot=1000)
    diff_lc = v9._bootstrap_pareado_diff(mref_t, mv9_t, Yt, Vt, Ft, n_boot=1000)

    fmt = mr.fmt_ic
    md = [
        '# Tarea 3 — Refinado de Bordes sobre `v9`',
        '',
        '> [!IMPORTANT]',
        '> Selección realizada **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Los resultados sobre `L4SPe test` (`124`) y `La Convención sellada` (`360`) son **«post hoc»** con bootstrap por zonas (`1 000` réplicas).',
        '',
        '## 1. Barrido en Validación',
        '',
        f'- **Referencia `v9` en validación:** `L4SPe IoU = {r_base_pe["iou_0px"]:.4f}`, `LC IoU = {r_base_lc["iou_0px"]:.4f}`, puntaje `0.7/0.3 = {punt_base:.4f}`, `FA_LC = {r_base_lc["fa_ha_km2"]:.4f} ha/km²`.',
        '',
        '| Método | L4SPe val IoU | LC val IoU | FA LC (`ha/km²`) | Puntaje `0.7/0.3` | Δ vs `v9` |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    top_a = sorted([r for r in filas_val if r['tipo'] == 'a'], key=lambda x: -x['puntaje_07_03'])[:4]
    top_b = sorted([r for r in filas_val if r['tipo'] == 'b'], key=lambda x: -x['puntaje_07_03'])[:4]
    for r in top_a + top_b:
        md.append(f"| `{r['metodo']}` | {r['pe_iou']:.4f} | {r['lc_iou']:.4f} | {r['fa_lc']:.4f} | {r['puntaje_07_03']:.4f} | `{r['mejora_punt']:+.4f}` |")

    dec_txt = '**SE APLICA**' if mejora_real else '**NO SE APLICA** (mejora nula o negativa en validación; diferencia dentro del ruido)'
    md.extend([
        '',
        f'## 2. Decisión en Validación: {dec_txt}',
        f'- Mejor candidato en validación: `{best["metodo"]}` (`puntaje 0.7/0.3 = {best["puntaje_07_03"]:.4f}`, `Δ = {best["mejora_punt"]:+.4f}`, `FA_LC = {best["fa_lc"]:.4f}`).',
        '',
        '## 3. Comparación «Post Hoc» frente a `v9` (IC 95 %, `1 000` réplicas)',
        '',
        '| Conjunto | `v9` IoU [IC 95 %] | Mejor refinado IoU [IC 95 %] | Δ IoU [IC 95 %] |',
        '|---|---|---|---|',
        f"| `L4SPe test` | {fmt(b_v9_pe,'iou')} | {fmt(b_rf_pe,'iou')} | `{diff_pe['iou_0px']['diff']:+.4f}` [{diff_pe['iou_0px']['ic95'][0]:+.4f}, {diff_pe['iou_0px']['ic95'][1]:+.4f}] |",
        f"| `La Convención sellada` | {fmt(b_v9_lc,'iou')} | {fmt(b_rf_lc,'iou')} | `{diff_lc['iou_0px']['diff']:+.4f}` [{diff_lc['iou_0px']['ic95'][0]:+.4f}, {diff_lc['iou_0px']['ic95'][1]:+.4f}] |",
    ])
    with open('informes/refinado_bordes.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('Tarea 3 completada -> informes/refinado_bordes.md', flush=True)


if __name__ == '__main__':
    main()
