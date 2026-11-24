"""Tarea 5: Metricas reformuladas (sin reentrenar; evaluacion post hoc sobre prueba 360 y L4SPe test).

Calcula para U-Net B1 v5, U-Net v6 (sin TTA + filtro wc_dem_3200_agua_urb) y DeepLabV3+ v6
(sin TTA + filtro wc_dem_3200_agua_urb):
  a) Por objeto (8-conectado): precision, exhaustividad, F1 y tasa de FP por objeto
     (acierto = solape >= 30%; FA = componente predicho >= 10 px que no toca ninguna cicatriz GT),
     en total (>= 10 px), >= 50 px (0.5 ha) y >= 200 px (2 ha).
  b) IoU por pixel exacto, con tolerancia de 1 px (8c) y de 2 px (8c).
  c) Mismas metricas solo sobre cicatrices >= 50 px (las cicatrices GT < 50 px pasan a 255 / ignoradas).
  d) IC 95% por bootstrap de zonas (2,000 remuestreos) en todas las metricas.
Entrega informes/metricas_reformuladas.md y informes/metricas_reformuladas.json.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import posproceso_v6 as pp6
from nucleo import config as cfg_srv


ST8 = np.ones((3, 3), dtype=bool)
ST_TOL2 = ndimage.generate_binary_structure(2, 2)  # 8-con, iteraciones=2 -> Chebyshev dist <= 2 px


def predecir_onnx(p_onnx, X):
    sess = ort.InferenceSession(p_onnx, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(X), 32):
        xb = np.ascontiguousarray(X[i:i+32], dtype=np.float32)
        logits = sess.run(None, {inp: xb})[0][:, 0]
        out.append(1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))))
    return np.concatenate(out, axis=0)


def enmascarar_menores(Y, V, min_px=50):
    """Pasa a 255 (V=0, Y=0) todas las cicatrices GT 8-conectadas de tamano < min_px."""
    Y2 = Y.copy()
    V2 = V.copy()
    for i in range(len(Y)):
        yi = (Y[i] > 0) & (V[i] > 0)
        if not yi.any():
            continue
        lab, nc = ndimage.label(yi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = (lab == cid)
            if int(cm.sum()) < min_px:
                Y2[i][cm] = 0.0
                V2[i][cm] = 0.0
    return Y2, V2


def stats_por_zona(prob, Y, V, filas, u, min_px_obj=10, solo_ge50=False):
    """Precalcula conteos por zona para metricas puntuales y bootstrap de zonas (2,000 replicas)."""
    if solo_ge50:
        Y_use, V_use = enmascarar_menores(Y, V, min_px=50)
    else:
        Y_use, V_use = Y, V

    zonas = sorted({r['zona'] for r in filas})
    z_idx = {z: k for k, z in enumerate(zonas)}
    nz = len(zonas)
    # Por zona:
    # pix0: [tp, fp, fn], pix1: [tp, fp, fn], pix2: [tp, fp, fn]
    # obj_all (>=10), obj_50 (>=50), obj_200 (>=200): [tp_obj, fn_obj, fp_obj]
    pix0 = np.zeros((nz, 3), dtype=np.float64)
    pix1 = np.zeros((nz, 3), dtype=np.float64)
    pix2 = np.zeros((nz, 3), dtype=np.float64)
    obj_bins = {b: np.zeros((nz, 3), dtype=np.float64) for b in (10, 50, 200)}

    for i, r in enumerate(filas):
        zk = z_idx[r['zona']]
        vi = V_use[i] > 0
        yi = (Y_use[i] > 0) & vi
        pi = (prob[i] >= u) & vi

        # Pixel exacto
        pix0[zk, 0] += float((yi & pi).sum())
        pix0[zk, 1] += float((pi & ~yi).sum())
        pix0[zk, 2] += float((yi & ~pi).sum())

        # Tolerancia 1 px y 2 px (8-conectado)
        if yi.any() or pi.any():
            yd1 = ndimage.binary_dilation(yi, structure=ST8, iterations=1) & vi
            pd1 = ndimage.binary_dilation(pi, structure=ST8, iterations=1) & vi
            tp1 = 0.5 * (float((pi & yd1).sum()) + float((yi & pd1).sum()))
            pix1[zk, 0] += tp1
            pix1[zk, 1] += float((pi & ~yd1).sum())
            pix1[zk, 2] += float((yi & ~pd1).sum())

            yd2 = ndimage.binary_dilation(yi, structure=ST8, iterations=2) & vi
            pd2 = ndimage.binary_dilation(pi, structure=ST8, iterations=2) & vi
            tp2 = 0.5 * (float((pi & yd2).sum()) + float((yi & pd2).sum()))
            pix2[zk, 0] += tp2
            pix2[zk, 1] += float((pi & ~yd2).sum())
            pix2[zk, 2] += float((yi & ~pd2).sum())

        # Por objeto:
        # GT cicatrices (8c)
        gt_all = (Y[i] > 0) & (V[i] > 0)  # cualquier cicatriz GT real para juzgar si un pred toca algo
        if yi.any():
            lab_g, nc_g = ndimage.label(yi, structure=ST8)
            for cid in range(1, nc_g + 1):
                cm = (lab_g == cid)
                sz = int(cm.sum())
                hit = (float((cm & pi).sum()) / max(sz, 1)) >= 0.30
                for b in (10, 50, 200):
                    if sz >= b:
                        if hit:
                            obj_bins[b][zk, 0] += 1.0
                        else:
                            obj_bins[b][zk, 1] += 1.0

        # FP objeto: componente predicho >= 10 px que no toca ninguna cicatriz GT
        if pi.any():
            lab_p, nc_p = ndimage.label(pi, structure=ST8)
            for cid in range(1, nc_p + 1):
                cm = (lab_p == cid)
                sz_p = int(cm.sum())
                if sz_p < 10:
                    continue
                if not (cm & gt_all).any():
                    # Cuenta como FP en todos los umbrales de tamano GT (en >=50 / >=200 tamb. es FA >=10 px)
                    for b in (10, 50, 200):
                        obj_bins[b][zk, 2] += 1.0

    return zonas, pix0, pix1, pix2, obj_bins


def calc_metrics_from_counts(arr, is_obj=False):
    tp, fp_or_fn, fn_or_fp = arr[0], arr[1], arr[2]
    if is_obj:
        tp, fn, fp = arr[0], arr[1], arr[2]
    else:
        tp, fp, fn = arr[0], arr[1], arr[2]
    iou = tp / max(tp + fp + fn, 1e-9)
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-9)
    fpr_obj = fp / max(tp + fp, 1e-9)  # tasa de falsos positivos entre detecciones (1 - precision)
    return {'iou': iou, 'f1': f1, 'prec': prec, 'rec': rec, 'fp_rate': fpr_obj}


def bootstrap_metricas(arr_z, is_obj=False, n_boot=2000, seed=7):
    tot = arr_z.sum(axis=0)
    pt = calc_metrics_from_counts(tot, is_obj=is_obj)
    nz = len(arr_z)
    rng = np.random.default_rng(seed)
    keys = ['iou', 'f1', 'prec', 'rec', 'fp_rate']
    samples = {k: np.empty(n_boot, dtype=np.float64) for k in keys}
    for b in range(n_boot):
        s = arr_z[rng.integers(0, nz, nz)].sum(axis=0)
        m = calc_metrics_from_counts(s, is_obj=is_obj)
        for k in keys:
            samples[k][b] = m[k]
    out = {'counts': [int(round(x)) for x in tot]}
    for k in keys:
        lo, hi = np.percentile(samples[k], [2.5, 97.5])
        out[k] = round(float(pt[k]), 4)
        out[k + '_ic95'] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def evaluar_modelo_conjunto(prob, Y, V, filas, u):
    res = {}
    for modo, solo50 in [('todas', False), ('solo_ge50', True)]:
        zonas, p0, p1, p2, ob = stats_por_zona(prob, Y, V, filas, u, solo_ge50=solo50)
        res[modo] = {
            'n_zonas': len(zonas),
            'pixel_exacto': bootstrap_metricas(p0, is_obj=False),
            'pixel_tol1px': bootstrap_metricas(p1, is_obj=False),
            'pixel_tol2px': bootstrap_metricas(p2, is_obj=False),
            'obj_ge10': bootstrap_metricas(ob[10], is_obj=True),
            'obj_ge50': bootstrap_metricas(ob[50], is_obj=True),
            'obj_ge200': bootstrap_metricas(ob[200], is_obj=True),
        }
    return res


def fmt_ic(d, k, pct=False):
    v = d[k] * (100.0 if pct else 1.0)
    lo, hi = d[k + '_ic95'][0] * (100.0 if pct else 1.0), d[k + '_ic95'][1] * (100.0 if pct else 1.0)
    if pct:
        return f'{v:.1f} % [{lo:.1f}, {hi:.1f}]'
    return f'{v:.4f} [{lo:.4f}, {hi:.4f}]'


def main():
    escala = cfg_srv.ESCALA_L4S
    rec_u = json.load(open('modelos_ajustados/unet_v6.json', encoding='utf-8'))
    rec_d = json.load(open('modelos_ajustados/deeplabv3plus_v6.json', encoding='utf-8'))

    Xt_u, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    Xt_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'prueba'})
    elev_t, wc_t = pp6.cargar_dem_y_worldcover(Ft)

    Xpt_u, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    Xpt_d, _, _, _ = af.cargar_l4spe('l4spe_profesor.npz', rec_d, {'prueba'})

    SPECS = [
        ('U-Net B1 v5', 'modelos_ajustados/unet_b1_v5.onnx', 0.30, 'ninguno', False),
        ('U-Net v6 (sin TTA+flt)', 'modelos_ajustados/unet_v6.onnx', 0.45, 'wc_dem_3200_agua_urb', False),
        ('DeepLabV3+ v6 (sin TTA+flt)', 'modelos_ajustados/deeplabv3plus_v6.onnx', 0.70, 'wc_dem_3200_agua_urb', True),
    ]

    resultados = {'nota': 'Evaluacion POST HOC sobre prueba sellada (360) y L4SPe test (124)', 'modelos': {}}
    for nombre, p_onnx, u, flt, is_dl in SPECS:
        print(f'Evaluando {nombre}...', flush=True)
        p_lc = predecir_onnx(p_onnx, Xt_d if is_dl else Xt_u)
        if flt != 'ninguno':
            p_lc = pp6.aplicar_filtro_altoandino(p_lc, elev_t, wc_t, modo=flt)
        p_pe = predecir_onnx(p_onnx, Xpt_d if is_dl else Xpt_u)

        resultados['modelos'][nombre] = {
            'umbral': u, 'filtro': flt,
            'prueba_360': evaluar_modelo_conjunto(p_lc, Yt, Vt, Ft, u),
            'l4spe_test': evaluar_modelo_conjunto(p_pe, Ypt, Vpt, Fpt, u),
        }

    with open('informes/metricas_reformuladas.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # Generar informes/metricas_reformuladas.md
    md = [
        '# Métricas Reformuladas — Evaluación «Post Hoc» (Tarea 5)',
        '',
        '> [!IMPORTANT]',
        '> **Nota metodológica:** Todas las cifras de este informe son **«post hoc»** (calculadas sobre conjuntos ya vistos en v5/v6, sin reentrenar y con los umbrales fijados exclusivamente en validación). Todos los intervalos son **IC 95 % por bootstrap de zonas (`2 000` remuestreos)**.',
        '> ',
        '> **Definición de métricas por objeto (8-conectado):** acierto (`TP`) = cicatriz GT con solape `≥ 30 %`; falsa alarma (`FP`) = componente predicho `≥ 10 px` que no toca ninguna cicatriz GT; tasa de falsos positivos por objeto = `FP / (TP + FP) = 1 − Precisión`.',
        '',
    ]

    for conj_key, conj_tit in [('prueba_360', '1. Prueba Sellada de La Convención (`360` parches, `62` zonas) — «post hoc»'),
                               ('l4spe_test', '2. Prueba L4SPe del Profesor (`124` parches, `68` zonas) — «post hoc»')]:
        md.append(f'## {conj_tit}')
        md.append('')
        md.append('### 1a. Métricas por píxel (exacto, tol. 1 px y tol. 2 px) — Todas las cicatrices vs. Solo `≥ 50 px`')
        md.append('')
        md.append('| Modelo | Evaluación | IoU exacto [IC 95 %] | IoU tol. 1 px [IC 95 %] | IoU tol. 2 px [IC 95 %] | F1 píxel exacto [IC 95 %] |')
        md.append('|---|---|---|---|---|---|')
        for nombre in resultados['modelos']:
            for modo, m_lbl in [('todas', 'Todas (`≥ 10 px`)'), ('solo_ge50', 'Solo `≥ 50 px` (`<50→255`)')]:
                d = resultados['modelos'][nombre][conj_key][modo]
                md.append(
                    f"| {nombre} | {m_lbl} | {fmt_ic(d['pixel_exacto'], 'iou')} | {fmt_ic(d['pixel_tol1px'], 'iou')} | {fmt_ic(d['pixel_tol2px'], 'iou')} | {fmt_ic(d['pixel_exacto'], 'f1', pct=True)} |"
                )
        md.append('')
        md.append('### 1b. Métricas por objeto (acierto `≥ 30 %`, FA componente `≥ 10 px` sin tocar GT)')
        md.append('')
        md.append('| Modelo | Tramo GT | TP / FN / FP | Precisión obj. [IC 95 %] | Exhaustividad (Recall) obj. [IC 95 %] | F1 objeto [IC 95 %] | Tasa FP obj. [IC 95 %] |')
        md.append('|---|---|---|---:|---:|---:|---:|')
        for nombre in resultados['modelos']:
            d_all = resultados['modelos'][nombre][conj_key]['todas']
            for b_key, b_lbl in [('obj_ge10', 'Total (`≥ 10 px`)'), ('obj_ge50', '`≥ 50 px` (`0.5 ha`)'), ('obj_ge200', '`≥ 200 px` (`2 ha`)')]:
                ob = d_all[b_key]
                c = ob['counts']
                md.append(
                    f"| {nombre} | {b_lbl} | `{c[0]} / {c[1]} / {c[2]}` | {fmt_ic(ob, 'prec', pct=True)} | {fmt_ic(ob, 'rec', pct=True)} | {fmt_ic(ob, 'f1', pct=True)} | {fmt_ic(ob, 'fp_rate', pct=True)} |"
                )
        md.append('')

    # Tabla resumen contra metas del proyecto (TODAS las formulaciones)
    md.extend([
        '## 3. Contraste frente a las Metas del Proyecto (Todas las Formulaciones, «post hoc»)',
        '',
        'Metas originales del proyecto: **`IoU ≥ 0.70`**, **`F1 > 75.5 %`**, **`Recall > 85 %`**, **`Falsos positivos ≤ 15 %`**.',
        '',
        '| Meta del proyecto | Formulación evaluada | Mejor valor en `prueba` (360) *(modelo)* | Mejor valor en `L4SPe` test *(modelo)* | ¿Cumple la meta? |',
        '|---|---|---|---|---|',
    ])

    def best_across(conj_key, fn_val):
        best_v, best_m, best_str = -1e9, '', ''
        for nm, md_m in resultados['modelos'].items():
            v, s = fn_val(md_m[conj_key])
            if v > best_v:
                best_v, best_m, best_str = v, nm, s
        return best_v, best_m, best_str

    def best_low_across(conj_key, fn_val):
        best_v, best_m, best_str = 1e9, '', ''
        for nm, md_m in resultados['modelos'].items():
            v, s = fn_val(md_m[conj_key])
            if v < best_v:
                best_v, best_m, best_str = v, nm, s
        return best_v, best_m, best_str

    filas_metas = [
        ('**IoU ≥ 0.70**', 'Píxel exacto (todas)', lambda c: (c['todas']['pixel_exacto']['iou'], fmt_ic(c['todas']['pixel_exacto'], 'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 1 px (todas)', lambda c: (c['todas']['pixel_tol1px']['iou'], fmt_ic(c['todas']['pixel_tol1px'], 'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 2 px (todas)', lambda c: (c['todas']['pixel_tol2px']['iou'], fmt_ic(c['todas']['pixel_tol2px'], 'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel exacto (solo `≥ 50 px`)', lambda c: (c['solo_ge50']['pixel_exacto']['iou'], fmt_ic(c['solo_ge50']['pixel_exacto'], 'iou')), False, 0.70),
        ('**IoU ≥ 0.70**', 'Píxel tol. 2 px (solo `≥ 50 px`)', lambda c: (c['solo_ge50']['pixel_tol2px']['iou'], fmt_ic(c['solo_ge50']['pixel_tol2px'], 'iou')), False, 0.70),
        ('**F1 > 75.5 %**', 'Píxel exacto (todas)', lambda c: (c['todas']['pixel_exacto']['f1'], fmt_ic(c['todas']['pixel_exacto'], 'f1', True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Píxel tol. 2 px (todas)', lambda c: (c['todas']['pixel_tol2px']['f1'], fmt_ic(c['todas']['pixel_tol2px'], 'f1', True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto total (`≥ 10 px`)', lambda c: (c['todas']['obj_ge10']['f1'], fmt_ic(c['todas']['obj_ge10'], 'f1', True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto `≥ 50 px` (`0.5 ha`)', lambda c: (c['todas']['obj_ge50']['f1'], fmt_ic(c['todas']['obj_ge50'], 'f1', True)), False, 0.755),
        ('**F1 > 75.5 %**', 'Objeto `≥ 200 px` (`2 ha`)', lambda c: (c['todas']['obj_ge200']['f1'], fmt_ic(c['todas']['obj_ge200'], 'f1', True)), False, 0.755),
        ('**Recall > 85 %**', 'Píxel exacto (todas)', lambda c: (c['todas']['pixel_exacto']['rec'], fmt_ic(c['todas']['pixel_exacto'], 'rec', True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto total (`≥ 10 px`)', lambda c: (c['todas']['obj_ge10']['rec'], fmt_ic(c['todas']['obj_ge10'], 'rec', True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto `≥ 50 px` (`0.5 ha`)', lambda c: (c['todas']['obj_ge50']['rec'], fmt_ic(c['todas']['obj_ge50'], 'rec', True)), False, 0.85),
        ('**Recall > 85 %**', 'Objeto `≥ 200 px` (`2 ha`)', lambda c: (c['todas']['obj_ge200']['rec'], fmt_ic(c['todas']['obj_ge200'], 'rec', True)), False, 0.85),
        ('**FP ≤ 15 %**', 'Tasa FP píxel (`1 − Prec`, todas)', lambda c: (c['todas']['pixel_exacto']['fp_rate'], fmt_ic(c['todas']['pixel_exacto'], 'fp_rate', True)), True, 0.15),
        ('**FP ≤ 15 %**', 'Tasa FP objeto (total `≥ 10 px`)', lambda c: (c['todas']['obj_ge10']['fp_rate'], fmt_ic(c['todas']['obj_ge10'], 'fp_rate', True)), True, 0.15),
        ('**FP ≤ 15 %**', 'Tasa FP objeto (`≥ 50 px`)', lambda c: (c['todas']['obj_ge50']['fp_rate'], fmt_ic(c['todas']['obj_ge50'], 'fp_rate', True)), True, 0.15),
    ]
    for meta, form, fn, is_low, thr in filas_metas:
        if is_low:
            v_lc, m_lc, s_lc = best_low_across('prueba_360', fn)
            v_pe, m_pe, s_pe = best_low_across('l4spe_test', fn)
            ok = 'Sí (L4SPe)' if v_pe <= thr and v_lc > thr else ('Sí (ambos)' if (v_lc <= thr and v_pe <= thr) else ('Sí (LC)' if v_lc <= thr else 'No'))
        else:
            v_lc, m_lc, s_lc = best_across('prueba_360', fn)
            v_pe, m_pe, s_pe = best_across('l4spe_test', fn)
            ok = 'Sí (L4SPe)' if v_pe >= thr and v_lc < thr else ('Sí (ambos)' if (v_lc >= thr and v_pe >= thr) else ('Sí (LC)' if v_lc >= thr else 'No'))
        md.append(f'| {meta} | {form} | {s_lc} *({m_lc})* | {s_pe} *({m_pe})* | **{ok}** |')

    with open('informes/metricas_reformuladas.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('Escrito informes/metricas_reformuladas.md y informes/metricas_reformuladas.json')


if __name__ == '__main__':
    main()
