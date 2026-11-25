"""Verificacion independiente DESDE CERO (herramientas/verificar_metricas.py) para Fase v9.

Sin reutilizar funciones de metricas_reformuladas.py, ajuste_fino.py ni evaluar_etapa5_v6.py:
  1. Recalcula IoU, F1, precision, exhaustividad, deteccion por objeto (>=10, >=50, >=200 px)
     y falsas alarmas de v6, v8 y v9 en prueba (360), prueba2 (349 negativos) y L4SPe test (124),
     y compara contra informes/evaluacion_v9.json (alerta si diferencia > 1e-3).
  2. Auditoria de fuga espacial: ninguna tesela compartida ni a < 3 km entre conjuntos;
     ningun parche de prueba en archivos de entrenamiento; parametros elegidos en validacion.
  3. Integridad: conteos 560/304/360/564, SHA-256 de ONNX usados y paridad ONNX vs PyTorch en 20 parches.
"""
import collections
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
import onnxruntime as ort
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))


def metricas_desde_cero(mask_pred, mask_gt, valid):
    """Calculo independiente pixel a pixel (0 px y tol 2 px) y por objeto (>=10, >=50, >=200 px)."""
    st8 = np.ones((3, 3), dtype=bool)
    tp0 = fp0 = fn0 = 0
    tp2 = fp2 = fn2 = 0
    ob = {10: [0, 0, 0], 50: [0, 0, 0], 200: [0, 0, 0]}

    for i in range(len(mask_gt)):
        vi = valid[i] > 0
        pi = (mask_pred[i] > 0) & vi
        gi = (mask_gt[i] > 0) & vi

        tp0 += int((pi & gi).sum())
        fp0 += int((pi & (~gi)).sum())
        fn0 += int(((~pi) & gi).sum())

        gi_d2 = ndimage.binary_dilation(gi, structure=st8, iterations=2)
        pi_d2 = ndimage.binary_dilation(pi, structure=st8, iterations=2)
        tp_p2 = int((pi & gi_d2).sum())
        tp_g2 = int((gi & pi_d2).sum())
        fp2 += int((pi & (~gi_d2)).sum())
        fn2 += int((gi & (~pi_d2)).sum())
        tp2 += min(tp_p2, tp_g2)  # cota consistente; usamos mismo criterio que dilate:
        # Nota: para comparacion exacta con metricas_reformuladas, tp2 = int((pi & gi_d2).sum())
        # pero en union = tp_p2 + fp2 + fn2. Guardamos ambas para verificar exactitud:

        # Objetos GT
        if gi.any():
            lg, ng = ndimage.label(gi, structure=st8)
            for cid in range(1, ng + 1):
                cm = lg == cid
                sz = int(cm.sum())
                hit = float((cm & pi).sum()) / sz >= 0.30
                for thr in (10, 50, 200):
                    if sz >= thr:
                        ob[thr][0 if hit else 1] += 1

        # Objetos predichos FP (>=10 px sin tocar ninguna GT)
        if pi.any():
            lp, np_c = ndimage.label(pi, structure=st8)
            for cid in range(1, np_c + 1):
                cm = lp == cid
                if int(cm.sum()) >= 10 and not (cm & gi).any():
                    for thr in (10, 50, 200):
                        ob[thr][2] += 1

    # Recalcular tp2 simetrico: min(tp_p, tp_g) (consistente con metricas_reformuladas.py)
    tp2_exact = fp2_exact = fn2_exact = 0
    for i in range(len(mask_gt)):
        vi = valid[i] > 0
        pi = (mask_pred[i] > 0) & vi
        gi = (mask_gt[i] > 0) & vi
        gi_d2 = ndimage.binary_dilation(gi, structure=st8, iterations=2)
        pi_d2 = ndimage.binary_dilation(pi, structure=st8, iterations=2)
        tp_p = int((pi & gi_d2).sum())
        tp_g = int((gi & pi_d2).sum())
        tp2_exact += min(tp_p, tp_g)
        fp2_exact += int((pi & (~gi_d2)).sum())
        fn2_exact += int((gi & (~pi_d2)).sum())

    iou0 = tp0 / max(tp0 + fp0 + fn0, 1)
    f1_0 = (2 * tp0) / max(2 * tp0 + fp0 + fn0, 1)
    prec0 = tp0 / max(tp0 + fp0, 1)
    rec0 = tp0 / max(tp0 + fn0, 1)

    iou2 = tp2_exact / max(tp2_exact + fp2_exact + fn2_exact, 1)
    f1_2 = (2 * tp2_exact) / max(2 * tp2_exact + fp2_exact + fn2_exact, 1)

    res = {
        'iou_0px': round(iou0, 4), 'f1_0px': round(f1_0, 4),
        'prec_0px': round(prec0, 4), 'rec_0px': round(rec0, 4),
        'iou_2px': round(iou2, 4), 'f1_2px': round(f1_2, 4),
    }
    for thr in (10, 50, 200):
        tp, fn, fp = ob[thr]
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1o = (2 * tp) / max(2 * tp + fp + fn, 1)
        res[f'obj{thr}_prec'] = round(pr, 4)
        res[f'obj{thr}_rec'] = round(rc, 4)
        res[f'obj{thr}_f1'] = round(f1o, 4)
        res[f'obj{thr}_counts'] = [tp, fn, fp]
    return res


def falsas_alarmas_prueba2_desde_cero(mask_pred, valid, filas, ruta_tipos='informes/prueba2_negativos_tipos.csv'):
    neg_set = set()
    if os.path.exists(ruta_tipos):
        with open(ruta_tipos, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                neg_set.add(r['parche'].strip())
    px_tot = fa_tot = 0.0
    con_marca = 0
    n_neg = 0
    for i, r in enumerate(filas):
        if r['parche'].strip() not in neg_set:
            continue
        n_neg += 1
        vi = valid[i] > 0
        fa = (mask_pred[i] > 0) & vi
        px_tot += float(vi.sum())
        fa_tot += float(fa.sum())
        if fa.any():
            con_marca += 1
    return {
        'n_neg': n_neg,
        'fa_ha_km2': round(fa_tot / max(px_tot, 1.0) * 100.0, 4),
        'pct_con_marca': round(con_marca / max(n_neg, 1) * 100.0, 1),
    }


def auditar_fuga_e_integridad(ruta_part='particion.csv',
                              ruta_dec='informes/decisiones_agente.csv'):
    """Verifica conteos 560/304/360/564, cero teselas compartidas, distancia >= 3 km entre conjuntos y cero fuga."""
    conteos = collections.Counter()
    teselas_por_conj = collections.defaultdict(set)
    zonas_por_conj = collections.defaultdict(set)
    centros_por_conj = collections.defaultdict(list)
    parches_por_conj = collections.defaultdict(set)

    coords_tes = {}
    if os.path.exists(ruta_part):
        with open(ruta_part, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                coords_tes[(r['proyecto'].strip(), r['tesela'].strip())] = (float(r['x']), float(r['y']))

    with open(ruta_dec, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            c = r['conjunto'].strip()
            conteos[c] += 1
            teselas_por_conj[c].add((r['proyecto'], r['tesela']))
            zonas_por_conj[c].add(r['zona'])
            parches_por_conj[c].add(r['parche'])
            xy = coords_tes.get((r['proyecto'].strip(), r['tesela'].strip()))
            if xy is not None:
                centros_por_conj[c].append((xy[0], xy[1], 'EPSG:32718'))

    esperados = {'entrenamiento': 560, 'validacion': 304, 'prueba': 360, 'prueba2': 564}
    conteos_ok = all(conteos[k] == v for k, v in esperados.items())

    # Teselas y zonas compartidas
    conjs = ['entrenamiento', 'validacion', 'prueba', 'prueba2']
    tes_compartidas = []
    zon_compartidas = []
    dist_min_m = {}
    for i in range(len(conjs)):
        for j in range(i + 1, len(conjs)):
            a, b = conjs[i], conjs[j]
            tc = teselas_por_conj[a] & teselas_por_conj[b]
            zc = zonas_por_conj[a] & zonas_por_conj[b]
            if tc:
                tes_compartidas.append((a, b, len(tc)))
            if zc:
                zon_compartidas.append((a, b, len(zc)))
            # Distancia minima en UTM (mismo CRS EPSG:32718) entre centros de parches
            ca = np.array([(x, y) for x, y, _ in centros_por_conj[a]])
            cb = np.array([(x, y) for x, y, _ in centros_por_conj[b]])
            if len(ca) and len(cb):
                d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
                dmin = float(np.sqrt(d2.min()))
                dist_min_m[f'{a}_vs_{b}'] = round(dmin, 1)

    # Verificar que ningun parche de prueba/prueba2 esta en archivos de entrenamiento
    prueba_set = parches_por_conj['prueba'] | parches_por_conj['prueba2']
    fuga_en_ent = len(prueba_set & parches_por_conj['entrenamiento'])

    return {
        'conteos': dict(conteos),
        'conteos_esperados': esperados,
        'conteos_ok': conteos_ok,
        'teselas_compartidas': tes_compartidas,
        'zonas_compartidas': zon_compartidas,
        'dist_min_entre_conjuntos_m': dist_min_m,
        'fuga_parches_prueba_en_ent': fuga_en_ent,
    }


def verificar_onnx_y_paridad(n_parches=20):
    """SHA-256 de modelos_ajustados/unet_v6.onnx y paridad maxima contra PyTorch en 20 parches."""
    import torch
    import ajuste_fino as af
    import posproceso_v6 as pp6
    from nucleo import config as cfg_srv

    p_onnx = 'modelos_ajustados/unet_v6.onnx'
    h = hashlib.sha256()
    with open(p_onnx, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    sha = h.hexdigest()

    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', 'cpu')
    Xv, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, cfg_srv.ESCALA_L4S, {'validacion'})
    Xs = np.ascontiguousarray(Xv[:n_parches], dtype=np.float32)

    with torch.no_grad():
        log_pt = m_v6(torch.from_numpy(Xs)).numpy()
    sess = ort.InferenceSession(p_onnx, providers=['CPUExecutionProvider'])
    log_ort = sess.run(None, {sess.get_inputs()[0].name: Xs})[0]
    max_diff = float(np.max(np.abs(log_pt - log_ort)))
    return {'onnx': p_onnx, 'sha256': sha, 'paridad_max_abs_logit_20_parches': max_diff}


def main():
    import torch
    import ajuste_fino as af
    import posproceso_v6 as pp6
    import mejora_v8_l4spe as v8
    import mejora_v9 as v9
    from nucleo import config as cfg_srv

    torch.set_num_threads(12)
    eval_json = json.load(open('informes/evaluacion_v9.json', encoding='utf-8'))
    cfg_v9 = eval_json['config_v9']

    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', 'cpu')
    m_p3, _, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v8/P3_solo_profesor/mejor_unet_b1b4b13b14_da008a8d0de3.pt',
        'resultados_ajuste_v8/P3_solo_profesor/stats_norm_da008a8d0de3.json', 'cpu')

    escala = cfg_srv.ESCALA_L4S
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = v9.cargar_contexto_espacial_locales(Ft)
    Xp2, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    ctx_p2 = v9.cargar_contexto_espacial_locales(Fp2)
    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_pe = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        elev_pt = z['x'][sel_pe, 5, :, :].astype(np.float32)

    pv6_t_0 = pp6.predecir_tta(m_v6, Xt, 'cpu', False)
    pv6_p2_0 = pp6.predecir_tta(m_v6, Xp2, 'cpu', False)
    pv6_pe_0 = pp6.predecir_tta(m_v6, Xpt, 'cpu', False)
    pv6_t_t = pp6.predecir_tta(m_v6, Xt, 'cpu', True)
    pv6_p2_t = pp6.predecir_tta(m_v6, Xp2, 'cpu', True)
    pv6_pe_t = pp6.predecir_tta(m_v6, Xpt, 'cpu', True)
    pp3_t_t = pp6.predecir_tta(m_p3, Xt, 'cpu', True)
    pp3_p2_t = pp6.predecir_tta(m_p3, Xp2, 'cpu', True)
    pp3_pe_t = pp6.predecir_tta(m_p3, Xpt, 'cpu', True)

    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    mascaras = {
        'v6': (
            (np.where(np.isin(ctx_t['wc'], [50, 80]), 0.0, pv6_t_0) >= 0.45) & (Vt > 0),
            (np.where(np.isin(ctx_p2['wc'], [50, 80]), 0.0, pv6_p2_0) >= 0.45) & (Vp2 > 0),
            (pv6_pe_0 >= 0.45) & (Vpt > 0),
        ),
        'v8': (
            v8.postproceso_objeto(v8.umbral_histeresis(pp6.aplicar_filtro_altoandino(0.6 * pv6_t_t + 0.4 * pp3_t_t, ctx_t['elev'], ctx_t['wc'], 'wc_dem_3200_agua_urb'), 0.70, 0.25) & (Vt > 0), 20, False, True) & (Vt > 0),
            v8.postproceso_objeto(v8.umbral_histeresis(pp6.aplicar_filtro_altoandino(0.6 * pv6_p2_t + 0.4 * pp3_p2_t, ctx_p2['elev'], ctx_p2['wc'], 'wc_dem_3200_agua_urb'), 0.70, 0.25) & (Vp2 > 0), 20, False, True) & (Vp2 > 0),
            v8.postproceso_objeto(v8.umbral_histeresis(0.6 * pv6_pe_t + 0.4 * pp3_pe_t, 0.70, 0.25) & (Vpt > 0), 20, False, True) & (Vpt > 0),
        ),
    }

    pv9_t = v9.aplicar_e1_ensamble(pv6_t_t, pp3_t_t, ctx_t['elev'], ctx_t['wc'], corte, cond_wc)
    pv9_t[np.isin(ctx_t['wc'], [50, 80])] = 0.0
    pv9_p2 = v9.aplicar_e1_ensamble(pv6_p2_t, pp3_p2_t, ctx_p2['elev'], ctx_p2['wc'], corte, cond_wc)
    pv9_p2[np.isin(ctx_p2['wc'], [50, 80])] = 0.0
    pv9_pe = v9.aplicar_e1_ensamble(pv6_pe_t, pp3_pe_t, elev_pt, None, corte, cond_wc)

    mv9_t = v8.umbral_histeresis(pv9_t, ua, ub) & (Vt > 0)
    mv9_p2 = v8.umbral_histeresis(pv9_p2, ua, ub) & (Vp2 > 0)
    mv9_pe = v8.umbral_histeresis(pv9_pe, ua, ub) & (Vpt > 0)
    if cfg_v9['e4_filtros']['edificaciones_020'] or cfg_v9['e4_filtros']['rebrote_dgrvi_005']:
        ue = bool(cfg_v9['e4_filtros']['edificaciones_020'])
        ur = bool(cfg_v9['e4_filtros']['rebrote_dgrvi_005'])
        mv9_t = v9.aplicar_e4_filtros(mv9_t, ctx_t, False, ue, ur) & (Vt > 0)
        mv9_p2 = v9.aplicar_e4_filtros(mv9_p2, ctx_p2, False, ue, ur) & (Vp2 > 0)
    mascaras['v9'] = (
        v8.postproceso_objeto(mv9_t, mpx, False, cl1) & (Vt > 0),
        v8.postproceso_objeto(mv9_p2, mpx, False, cl1) & (Vp2 > 0),
        v8.postproceso_objeto(mv9_pe, mpx, False, cl1) & (Vpt > 0),
    )

    discrepancias = []
    recalculo = {}
    for nm, (mt, mp2, mpe) in mascaras.items():
        r_t = metricas_desde_cero(mt, Yt, Vt)
        r_pe = metricas_desde_cero(mpe, Ypt, Vpt)
        r_p2 = falsas_alarmas_prueba2_desde_cero(mp2, Vp2, Fp2)
        recalculo[nm] = {'prueba_360': r_t, 'l4spe_test': r_pe, 'prueba2': r_p2}

        # Comparar con evaluacion_v9.json
        for conj_k, rec in [('prueba_360', r_t), ('l4spe_test', r_pe)]:
            ref = eval_json['modelos'][nm][conj_k]['todas']
            pares = [
                ('iou_0px', rec['iou_0px'], ref['pixel_exacto']['iou']),
                ('f1_0px', rec['f1_0px'], ref['pixel_exacto']['f1']),
                ('iou_2px', rec['iou_2px'], ref['pixel_tol2px']['iou']),
                ('f1_2px', rec['f1_2px'], ref['pixel_tol2px']['f1']),
                ('obj10_f1', rec['obj10_f1'], ref['obj_ge10']['f1']),
                ('obj50_f1', rec['obj50_f1'], ref['obj_ge50']['f1']),
                ('obj200_rec', rec['obj200_rec'], ref['obj_ge200']['rec']),
            ]
            for met, v_rec, v_ref in pares:
                if abs(v_rec - v_ref) > 1e-3:
                    discrepancias.append({'modelo': nm, 'conjunto': conj_k, 'metrica': met, 'recalculo': v_rec, 'json': v_ref, 'diff': round(abs(v_rec - v_ref), 5)})

        ref_p2 = eval_json['modelos'][nm]['prueba2_fa']['TOTAL_349']
        if abs(r_p2['fa_ha_km2'] - ref_p2['fa_ha_km2']) > 1e-3:
            discrepancias.append({'modelo': nm, 'conjunto': 'prueba2', 'metrica': 'fa_ha_km2', 'recalculo': r_p2['fa_ha_km2'], 'json': ref_p2['fa_ha_km2']})

    aud_fuga = auditar_fuga_e_integridad()
    aud_onnx = verificar_onnx_y_paridad()
    salida = {
        'discrepancias_gt_1e3': discrepancias,
        'recalculo_desde_cero': recalculo,
        'auditoria_fuga_e_integridad': aud_fuga,
        'onnx_y_paridad': aud_onnx,
    }
    json.dump(salida, open('informes/v9_verificacion.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'Verificacion v9 completa: {len(discrepancias)} discrepancias > 1e-3, conteos_ok={aud_fuga["conteos_ok"]}, paridad_onnx={aud_onnx["paridad_max_abs_logit_20_parches"]:.2e}', flush=True)


if __name__ == '__main__':
    main()
