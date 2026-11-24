"""Tarea 6: Filtro de cambio entre fechas dGRVI = GRVI_actual - GRVI_anterior.

- Barre t en [-0.10, -0.08, -0.06, -0.04, -0.02, 0.00, +0.02, +0.05] SOLO en validacion,
  solo y combinado con WorldCover (sin corte de altitud: wc_urb_agua; y con corte: wc_dem_3200_agua_urb).
- Reporta cuantas cicatrices de validacion borra el filtro.
- Evalua UNA SOLA VEZ (post hoc) en prueba sellada (360) y prueba2 (349 negativos por tipo_negativo).
- Documenta que L4SPe (l4spe_profesor.npz) no tiene imagen anterior.
"""
import csv
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import posproceso_v6 as pp6
import evaluar_etapa5_v6 as e5
from nucleo import config as cfg_srv


ST8 = np.ones((3, 3), dtype=bool)
TS = [None, -0.10, -0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.05]
WC_MODOS = ['ninguno', 'wc_urb_agua', 'wc_dem_3200_agua_urb']


def aplicar_wc(prob, elev, wc, modo):
    if modo == 'ninguno':
        return prob
    if modo == 'wc_urb_agua':
        out = prob.copy()
        out[np.isin(wc, [50, 80])] = 0.0
        return out
    return pp6.aplicar_filtro_altoandino(prob, elev, wc, modo=modo)


def cargar_dgrvi_conjunto(filas, carpeta_parches='parches_128'):
    """Calcula dGRVI = GRVI_actual - GRVI_anterior y mascara valida_anterior (despejada) para cada parche."""
    dgrvis = []
    valid_ant = []
    n_ok = 0
    for r in filas:
        p_act = os.path.join(carpeta_parches, r['archivo'])
        with rasterio.open(p_act) as ds:
            a_act = ds.read().astype(np.float32)
        g_act = (a_act[2] - a_act[3]) / (np.abs(a_act[2] + a_act[3]) + 1e-3)

        p1 = os.path.join(r['proyecto'] + '_antes', 'images', r['tesela'], 'image.tif')
        p2 = os.path.join(r['proyecto'] + '_antes', r['tesela'] + '.tif')
        p_ant = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        if p_ant is None:
            dgrvis.append(np.full((128, 128), np.nan, dtype=np.float32))
            valid_ant.append(np.zeros((128, 128), dtype=bool))
            continue
        with rasterio.open(p_ant) as da:
            if da.height == 256 and da.width == 256:
                q = r['parche'].rsplit('_', 1)[-1]
                r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
                a_ant = da.read()[:, r0:r0+128, c0:c0+128].astype(np.float32)
            else:
                a_ant = da.read().astype(np.float32)
        g_ant = (a_ant[2] - a_ant[3]) / (np.abs(a_ant[2] + a_ant[3]) + 1e-3)
        ok = (a_ant[1] > 0) & (a_ant[1] < 1000) & np.isfinite(g_ant) & (a_act[1] > 0)
        dgrvi = np.where(ok, g_act - g_ant, np.nan).astype(np.float32)
        dgrvis.append(dgrvi)
        valid_ant.append(ok)
        if ok.mean() >= 0.05:
            n_ok += 1
    return np.stack(dgrvis), np.stack(valid_ant), n_ok


def aplicar_dgrvi(prob, dgrvi, valid_ant, t):
    """Quita predicciones donde la escena anterior esta despejada y dGRVI > t (sin perdida de verdor)."""
    if t is None:
        return prob
    out = prob.copy()
    suprimir = valid_ant & (dgrvi > float(t))
    out[suprimir] = 0.0
    return out


def predecir_onnx(p_onnx, X):
    sess = ort.InferenceSession(p_onnx, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(X), 32):
        xb = np.ascontiguousarray(X[i:i+32], dtype=np.float32)
        logits = sess.run(None, {inp: xb})[0][:, 0]
        out.append(1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))))
    return np.concatenate(out, axis=0)


def main():
    escala = cfg_srv.ESCALA_L4S
    rec_u = json.load(open('modelos_ajustados/unet_v6.json', encoding='utf-8'))
    rec_d = json.load(open('modelos_ajustados/deeplabv3plus_v6.json', encoding='utf-8'))

    # Cargar validacion, prueba (360) y prueba2 (564)
    Xv_u, Yv, Vv, Fv = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    Xv_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'validacion'})
    ev_v, wc_v = pp6.cargar_dem_y_worldcover(Fv)
    dg_v, ok_v, n_ok_v = cargar_dgrvi_conjunto(Fv)

    Xt_u, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    Xt_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'prueba'})
    ev_t, wc_t = pp6.cargar_dem_y_worldcover(Ft)
    dg_t, ok_t, n_ok_t = cargar_dgrvi_conjunto(Ft)

    Xp2_u, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    Xp2_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'prueba2'})
    ev_p2, wc_p2 = pp6.cargar_dem_y_worldcover(Fp2)
    dg_p2, ok_p2, n_ok_p2 = cargar_dgrvi_conjunto(Fp2)

    print(f'Escenas anteriores despejadas: val={n_ok_v}/{len(Fv)}, prueba={n_ok_t}/{len(Ft)}, prueba2={n_ok_p2}/{len(Fp2)}', flush=True)

    SPECS = [
        ('U-Net v6 (sin TTA)', 'modelos_ajustados/unet_v6.onnx', 0.45, False),
        ('U-Net B1 v5', 'modelos_ajustados/unet_b1_v5.onnx', 0.30, False),
        ('DeepLabV3+ v6 (sin TTA)', 'modelos_ajustados/deeplabv3plus_v6.onnx', 0.70, True),
    ]

    val_rows = []
    posthoc_rows = []
    best_by_model_wc = {}

    for nombre, p_onnx, u, is_dl in SPECS:
        pv_raw = predecir_onnx(p_onnx, Xv_d if is_dl else Xv_u)
        pt_raw = predecir_onnx(p_onnx, Xt_d if is_dl else Xt_u)
        pp2_raw = predecir_onnx(p_onnx, Xp2_d if is_dl else Xp2_u)

        # Base sin filtro en validacion para contar cicatrices borradas
        base_val = af.resumen_local(pv_raw, Yv, Vv, Fv, u)
        toc_base = base_val['cicatrices_tocadas']
        det30_base = base_val['deteccion_cicatriz_30pct']['total_ge10']['detectadas']

        for wc_m in WC_MODOS:
            pv_wc = aplicar_wc(pv_raw, ev_v, wc_v, wc_m)
            best_row = None
            for t in TS:
                pv_f = aplicar_dgrvi(pv_wc, dg_v, ok_v, t)
                res_v = af.resumen_local(pv_f, Yv, Vv, Fv, u)
                d30 = res_v['deteccion_cicatriz_30pct']['total_ge10']
                row = {
                    'modelo': nombre, 'u': u, 'wc_modo': wc_m,
                    't_dgrvi': 'sin_dgrvi' if t is None else f'{t:+.2f}',
                    'val_iou': res_v['iou'], 'val_f1': res_v['f1'],
                    'val_prec': res_v['precision'], 'val_rec': res_v['recall'],
                    'val_fa_ha_km2': res_v['falsas_alarmas_ha_por_km2_en_negativos'],
                    'val_cic_tocadas': f"{res_v['cicatrices_tocadas']}/{res_v['cicatrices']}",
                    'val_tocadas_borradas': toc_base - res_v['cicatrices_tocadas'],
                    'val_det30': f"{d30['detectadas']}/{d30['total']}",
                    'val_det30_borradas': det30_base - d30['detectadas'],
                }
                val_rows.append(row)
                if t is not None:
                    if best_row is None or (row['val_iou'], -row['val_fa_ha_km2']) > (best_row['val_iou'], -best_row['val_fa_ha_km2']):
                        best_row = dict(row, _t=t)

            # Evaluar UNA SOLA VEZ (post hoc) para: (1) sin dGRVI (referencia del wc_modo) y (2) mejor t elegido en validacion
            for tag_eval, t_val in [('sin_dgrvi', None), (f"dgrvi_opt({best_row['_t']:+.2f})", best_row['_t'])]:
                pt_f = aplicar_dgrvi(aplicar_wc(pt_raw, ev_t, wc_t, wc_m), dg_t, ok_t, t_val)
                pp2_f = aplicar_dgrvi(aplicar_wc(pp2_raw, ev_p2, wc_p2, wc_m), dg_p2, ok_p2, t_val)
                res_t = af.resumen_local(pt_f, Yt, Vt, Ft, u)
                fa_p2 = e5.eval_prueba2_fa(pp2_f, Vp2, Fp2, u)
                d30_t = res_t['deteccion_cicatriz_30pct']['total_ge10']
                posthoc_rows.append({
                    'modelo': nombre, 'u': u, 'wc_modo': wc_m, 'filtro_dgrvi': tag_eval,
                    'pru_iou': res_t['iou'], 'pru_tol1_8c': res_t['iou_tol1px_8c'],
                    'pru_f1': res_t['f1'], 'pru_prec': res_t['precision'], 'pru_rec': res_t['recall'],
                    'pru_det30': f"{d30_t['detectadas']}/{d30_t['total']}",
                    'pru_fa_ha_km2': res_t['falsas_alarmas_ha_por_km2_en_negativos'],
                    'p2_total_ha_km2': fa_p2['TOTAL_349']['fa_ha_km2'],
                    'p2_total_pct': fa_p2['TOTAL_349']['pct_con_marca'],
                    'p2_altoandino_ha': fa_p2['altoandino']['fa_ha_km2'],
                    'p2_chacra_ha': fa_p2['chacra_o_pasto']['fa_ha_km2'],
                    'p2_rio_ha': fa_p2['rio']['fa_ha_km2'],
                    'p2_carretera_ha': fa_p2['carretera']['fa_ha_km2'],
                    'p2_poblado_ha': fa_p2['poblado']['fa_ha_km2'],
                    'p2_bosque_ha': fa_p2['bosque_ladera']['fa_ha_km2'],
                })

    with open('informes/filtro_dgrvi_v7_validacion.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys())); w.writeheader(); w.writerows(val_rows)
    with open('informes/filtro_dgrvi_v7_posthoc.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(posthoc_rows[0].keys())); w.writeheader(); w.writerows(posthoc_rows)

    # Escribir resumen Markdown informes/filtro_dgrvi_v7.md
    md = [
        '# Filtro de Cambio entre Fechas ΔGRVI — Tarea 6 (Validación y Evaluación «Post Hoc»)',
        '',
        '> [!IMPORTANT]',
        '> **Regla aplicada:** El umbral `t` de `ΔGRVI = GRVI_actual − GRVI_anterior` (se suprimen píxeles con escena anterior despejada `B02 < 1000` y `ΔGRVI > t`) se barrió en `[-0.10, +0.05]` **exclusivamente en validación** (`304` parches). La evaluación en `prueba` (`360`) y `prueba2` (`349` negativos) se realizó **una sola vez («post hoc»)**.',
        '> ',
        '> **Cobertura de escenas anteriores (`2018–2020`):** `validacion`: `' + f'{n_ok_v}/{len(Fv)}' + '`, `prueba`: `' + f'{n_ok_t}/{len(Ft)}' + '`, `prueba2`: `' + f'{n_ok_p2}/{len(Fp2)}' + '`. **L4SPe (`l4spe_profesor.npz`) NO dispone de fecha ni imagen anterior**, por lo que este filtro no aplica sobre L4SPe.',
        '',
        '## 1. Barrido en Validación (`U-Net v6 sin TTA`, `u = 0.45`) — Efecto de `t` y WorldCover (con y sin corte de altitud)',
        '',
        '| Filtro WorldCover | `t` (`ΔGRVI > t` se borra) | IoU val LC | F1 val LC | FA val (`ha/km²`) | Cicatrices tocadas (borradas) | Det. `≥ 30 %` (borradas) |',
        '|---|---|---:|---:|---:|---|---|',
    ]
    for r in val_rows:
        if r['modelo'] == 'U-Net v6 (sin TTA)':
            md.append(
                f"| `{r['wc_modo']}` | `{r['t_dgrvi']}` | {r['val_iou']:.4f} | {r['val_f1']:.4f} | {r['val_fa_ha_km2']:.3f} | {r['val_cic_tocadas']} (−{r['val_tocadas_borradas']}) | {r['val_det30']} (−{r['val_det30_borradas']}) |"
            )

    md.extend([
        '',
        '## 2. Evaluación Única «Post Hoc» en `prueba` (360) y `prueba2` (349 negativos por tipo)',
        '',
        '| Modelo | WorldCover | Filtro ΔGRVI (elegido en val.) | `prueba` IoU (tol 1 px) | `prueba` Det. `≥ 30 %` | `prueba` FA (`ha/km²`) | **`prueba2` TOTAL (`ha/km²` / `%`)** | `altoandino` | `chacra` | `rio` | `poblado` |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|',
    ])
    for r in posthoc_rows:
        md.append(
            f"| {r['modelo']} | `{r['wc_modo']}` | `{r['filtro_dgrvi']}` | {r['pru_iou']:.4f} ({r['pru_tol1_8c']:.4f}) | {r['pru_det30']} | {r['pru_fa_ha_km2']:.3f} | **{r['p2_total_ha_km2']:.4f} ({r['p2_total_pct']} %)** | {r['p2_altoandino_ha']:.4f} | {r['p2_chacra_ha']:.4f} | {r['p2_rio_ha']:.4f} | {r['p2_poblado_ha']:.4f} |"
        )
    md.append('')
    with open('informes/filtro_dgrvi_v7.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('Escrito informes/filtro_dgrvi_v7.md, _validacion.csv y _posthoc.csv', flush=True)


if __name__ == '__main__':
    main()
