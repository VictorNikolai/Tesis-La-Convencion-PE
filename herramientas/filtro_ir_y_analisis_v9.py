"""Tareas A y B (mejora-iou):
  - Tarea A: Filtro infrarrojo (NDVI, NDMI, BSI, B11/B12) sobre la salida de v9.
    Seleccion EXCLUSIVA en validacion (sin perder > 1 cicatriz real en val).
    Reporte post hoc en prueba (360), prueba2 (349 neg) y las 39 FA de fa_clasificacion_v9.csv.
  - Tarea B: Analisis de errores de la version final (tamaño, pendiente, altitud en rangos de 500 m,
    y falsas alarmas por tipo de terreno) + figura de 3 paneles lista para la tesis.
"""
import collections
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import evaluar_etapa5_v6 as e5
import generar_revision_profesor as grp
import mejora_v8_l4spe as v8
import mejora_v9 as v9
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv

ST8 = np.ones((3, 3), dtype=bool)


def cargar_indices_ir_locales(filas):
    """Carga B02, B04 locales + B08, B11, B12 de parches_128/bandas_ir/ y calcula:
      NDVI, NDMI, BSI, B11/B12 y ok_ir (donde B08 > 0)."""
    N = len(filas)
    ndvi = np.zeros((N, 128, 128), dtype=np.float32)
    ndmi = np.zeros((N, 128, 128), dtype=np.float32)
    bsi = np.zeros((N, 128, 128), dtype=np.float32)
    r11_12 = np.zeros((N, 128, 128), dtype=np.float32)
    ok_ir = np.zeros((N, 128, 128), dtype=bool)
    cache_ir = {}

    for i, r in enumerate(filas):
        with rasterio.open(os.path.join('parches_128', r['archivo'])) as ds:
            b2 = ds.read(2).astype(np.float32)
            b4 = ds.read(4).astype(np.float32)
        key = (r['proyecto'], r['tesela'])
        if key not in cache_ir:
            p_ir = os.path.join('parches_128', 'bandas_ir', r['proyecto'], r['tesela'] + '.tif')
            if os.path.exists(p_ir):
                with rasterio.open(p_ir) as dsi:
                    cache_ir[key] = dsi.read().astype(np.float32)
            else:
                cache_ir[key] = None
        if cache_ir[key] is None:
            continue
        q = r['parche'].rsplit('_', 1)[-1]
        r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
        ir = cache_ir[key][:, r0:r0 + 128, c0:c0 + 128]
        b8, b11, b12 = ir[0], ir[1], ir[2]
        ok = (b8 > 0) & (b11 > 0) & (b12 > 0) & (b4 > 0) & (b2 > 0)
        ok_ir[i] = ok
        ndvi[i] = np.where(ok, (b8 - b4) / (b8 + b4 + 1e-3), 0.0)
        ndmi[i] = np.where(ok, (b8 - b11) / (b8 + b11 + 1e-3), 0.0)
        bsi[i] = np.where(ok, ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + 1e-3), 0.0)
        r11_12[i] = np.where(ok, b11 / (b12 + 1e-3), 0.0)

    return {'ndvi': ndvi, 'ndmi': ndmi, 'bsi': bsi, 'r11_12': r11_12, 'ok_ir': ok_ir}


def predecir_v9_conjunto(X, V, ctx, cfg_v9, m_v6, m_p3, disp='cpu'):
    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    pv9 = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, X, disp, True), pp6.predecir_tta(m_p3, X, disp, True),
                                 ctx['elev'], ctx['wc'], corte, cond_wc)
    pv9[np.isin(ctx['wc'], [50, 80])] = 0.0
    mv9 = v9.aplicar_e4_filtros(v8.umbral_histeresis(pv9, ua, ub) & (V > 0), ctx, False, True, True) & (V > 0)
    return v8.postproceso_objeto(mv9, mpx, False, cl1) & (V > 0)


def extraer_objetos_v9(mask_v9, Y, V, ir):
    objs = []
    for i in range(len(mask_v9)):
        pi = mask_v9[i] & (V[i] > 0)
        if not pi.any():
            continue
        yi = (Y[i] > 0) & (V[i] > 0)
        lp, nc = ndimage.label(pi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lp == cid
            ok = cm & ir['ok_ir'][i]
            if not ok.any():
                continue
            objs.append({
                'i': i, 'cid': cid, 'cm': cm, 'px': int(cm.sum()),
                'es_tp': bool((cm & yi).any()),
                'ndvi': float(ir['ndvi'][i][ok].mean()),
                'ndmi': float(ir['ndmi'][i][ok].mean()),
                'bsi': float(ir['bsi'][i][ok].mean()),
                'r11_12': float(ir['r11_12'][i][ok].mean()),
            })
    return objs


def aplicar_regla_ir(mask_v9, V, ir, regla_fn):
    out = mask_v9.copy()
    for i in range(len(out)):
        pi = out[i] & (V[i] > 0)
        if not pi.any():
            continue
        lp, nc = ndimage.label(pi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lp == cid
            ok = cm & ir['ok_ir'][i]
            if float(ok.sum()) / max(int(cm.sum()), 1) < 0.30:
                continue
            ndvi_m = float(ir['ndvi'][i][ok].mean())
            ndmi_m = float(ir['ndmi'][i][ok].mean())
            bsi_m = float(ir['bsi'][i][ok].mean())
            r_m = float(ir['r11_12'][i][ok].mean())
            if regla_fn(ndvi_m, ndmi_m, bsi_m, r_m):
                out[i][cm] = False
    return out


def _desglose_errores(mask_pred, Y, V, pend, elev):
    """Calcula TP y FN de cicatrices GT por tamano (<0.5, 0.5-2, >2 ha), pendiente y altitud (500 m)."""
    tam_bins = {'< 0.5 ha': [0, 0], '0.5–2 ha': [0, 0], '> 2 ha': [0, 0]}
    pend_bins = {'< 20°': [0, 0], '20–30°': [0, 0], '30–40°': [0, 0], '> 40°': [0, 0]}
    alt_bins = {'< 1500 m': [0, 0], '1500–2000 m': [0, 0], '2000–2500 m': [0, 0],
                '2500–3000 m': [0, 0], '3000–3500 m': [0, 0], '> 3500 m': [0, 0]}

    for i in range(len(Y)):
        yi = (Y[i] > 0) & (V[i] > 0)
        if not yi.any():
            continue
        pi = (mask_pred[i] > 0) & (V[i] > 0)
        lg, nc = ndimage.label(yi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lg == cid
            sz = int(cm.sum())
            if sz < 10:
                continue
            hit = int(float((cm & pi).sum()) / sz >= 0.30)
            pm = float(pend[i][cm].mean())
            em = float(elev[i][cm].mean())

            kt = '< 0.5 ha' if sz < 50 else ('0.5–2 ha' if sz <= 200 else '> 2 ha')
            tam_bins[kt][0 if hit else 1] += 1

            kp = '< 20°' if pm < 20 else ('20–30°' if pm < 30 else ('30–40°' if pm < 40 else '> 40°'))
            pend_bins[kp][0 if hit else 1] += 1

            if em < 1500: ka = '< 1500 m'
            elif em < 2000: ka = '1500–2000 m'
            elif em < 2500: ka = '2000–2500 m'
            elif em < 3000: ka = '2500–3000 m'
            elif em < 3500: ka = '3000–3500 m'
            else: ka = '> 3500 m'
            alt_bins[ka][0 if hit else 1] += 1
    return tam_bins, pend_bins, alt_bins


def main():
    import torch
    torch.set_num_threads(12)
    disp = 'cpu'
    escala = cfg_srv.ESCALA_L4S
    cfg_v9 = json.load(open('modelos_ajustados/v9.json', encoding='utf-8'))

    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', disp)
    m_p3, _, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v8/P3_solo_profesor/mejor_unet_b1b4b13b14_da008a8d0de3.pt',
        'resultados_ajuste_v8/P3_solo_profesor/stats_norm_da008a8d0de3.json', disp)

    # ==================== TAREA A: VALIDACION ====================
    Xv_lc, Yv_lc, Vv_lc, Fv_lc = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    ctx_v = v9.cargar_contexto_espacial_locales(Fv_lc)
    ir_v = cargar_indices_ir_locales(Fv_lc)
    mv9_v = predecir_v9_conjunto(Xv_lc, Vv_lc, ctx_v, cfg_v9, m_v6, m_p3, disp)

    objs_v = extraer_objetos_v9(mv9_v, Yv_lc, Vv_lc, ir_v)
    tps_v = [o for o in objs_v if o['es_tp']]
    fas_v = [o for o in objs_v if not o['es_tp']]
    print(f'[VAL] Objetos v9 en LC val: {len(objs_v)} (TP={len(tps_v)}, FA={len(fas_v)})', flush=True)

    def resumen_idx(lst):
        if not lst:
            return {'n': 0, 'ndvi': 0, 'ndmi': 0, 'bsi': 0, 'r11_12': 0}
        return {
            'n': len(lst),
            'ndvi': round(float(np.mean([o['ndvi'] for o in lst])), 3),
            'ndmi': round(float(np.mean([o['ndmi'] for o in lst])), 3),
            'bsi': round(float(np.mean([o['bsi'] for o in lst])), 3),
            'r11_12': round(float(np.mean([o['r11_12'] for o in lst])), 3),
        }

    st_tp_v, st_fa_v = resumen_idx(tps_v), resumen_idx(fas_v)
    print('[VAL] Medias TP:', st_tp_v, '| Medias FA:', st_fa_v, flush=True)

    # Catalogo de reglas simples (max 2 reglas combinadas)
    reglas = []
    for t_nd in (0.25, 0.30, 0.35, 0.40, 0.45):
        reglas.append((f'NDVI > {t_nd:.2f}', lambda nd, nm, bs, r, t=t_nd: nd > t))
    for t_bs in (0.05, 0.10, 0.15, 0.20):
        reglas.append((f'BSI > {t_bs:.2f}', lambda nd, nm, bs, r, t=t_bs: bs > t))
    for t_nm in (-0.10, -0.05, 0.00):
        reglas.append((f'NDMI < {t_nm:+.2f}', lambda nd, nm, bs, r, t=t_nm: nm < t))
    for t_r in (1.5, 1.7, 1.9):
        reglas.append((f'B11/B12 < {t_r:.1f}', lambda nd, nm, bs, r, t=t_r: r < t))
    # Combinaciones de 2 reglas
    for t_nd in (0.30, 0.35, 0.40):
        for t_bs in (0.05, 0.10, 0.15):
            reglas.append((f'NDVI > {t_nd:.2f} o BSI > {t_bs:.2f}',
                           lambda nd, nm, bs, r, tn=t_nd, tb=t_bs: (nd > tn) or (bs > tb)))
    for t_bs in (0.05, 0.10, 0.15):
        for t_nm in (-0.05, 0.00, 0.05):
            reglas.append((f'BSI > {t_bs:.2f} y NDMI < {t_nm:+.2f}',
                           lambda nd, nm, bs, r, tb=t_bs, tm=t_nm: (bs > tb) and (nm < tm)))

    base_val = v8.metricas_rapidas_mascara(mv9_v, Yv_lc, Vv_lc, Fv_lc)
    res_reglas = []
    for nom_r, fn_r in reglas:
        mk_r = aplicar_regla_ir(mv9_v, Vv_lc, ir_v, fn_r)
        mr_v = v8.metricas_rapidas_mascara(mk_r, Yv_lc, Vv_lc, Fv_lc)
        _, perd = v9.contar_ganadas_perdidas(mv9_v, mk_r, Yv_lc, Vv_lc)
        cumple = (mr_v['fa_ha_km2'] < base_val['fa_ha_km2'] - 1e-5) and (perd <= 1)
        res_reglas.append({
            'regla': nom_r, 'fn': fn_r, 'lc_iou': mr_v['iou_0px'],
            'fa_lc': mr_v['fa_ha_km2'], 'perdidas_val': perd, 'cumple': cumple,
        })

    cands_ok = [r for r in res_reglas if r['cumple']]
    mejor_r = max(cands_ok, key=lambda r: (r['lc_iou'], -r['fa_lc'])) if cands_ok else None
    se_aplica = mejor_r is not None
    print(f'[TAREA A] Mejor regla en val: {mejor_r["regla"] if mejor_r else "Ninguna"} (se_aplica={se_aplica})', flush=True)

    # ==================== TAREA A: EVALUACION POST HOC ====================
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = v9.cargar_contexto_espacial_locales(Ft)
    ir_t = cargar_indices_ir_locales(Ft)
    mv9_t = predecir_v9_conjunto(Xt, Vt, ctx_t, cfg_v9, m_v6, m_p3, disp)

    Xp2, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    ctx_p2 = v9.cargar_contexto_espacial_locales(Fp2)
    ir_p2 = cargar_indices_ir_locales(Fp2)
    mv9_p2 = predecir_v9_conjunto(Xp2, Vp2, ctx_p2, cfg_v9, m_v6, m_p3, disp)

    if se_aplica:
        mir_t = aplicar_regla_ir(mv9_t, Vt, ir_t, mejor_r['fn']) & (Vt > 0)
        mir_p2 = aplicar_regla_ir(mv9_p2, Vp2, ir_p2, mejor_r['fn']) & (Vp2 > 0)
    else:
        # Evaluar la mejor candidata de validacion solo como referencia documental
        top_cand = max(res_reglas, key=lambda r: (r['lc_iou'] - base_val['iou_0px'], -r['fa_lc']))
        mir_t = aplicar_regla_ir(mv9_t, Vt, ir_t, top_cand['fn']) & (Vt > 0)
        mir_p2 = aplicar_regla_ir(mv9_p2, Vp2, ir_p2, top_cand['fn']) & (Vp2 > 0)

    def boot_res(mk, Y, V, F):
        _, p0, _, _, ob = mr.stats_por_zona(mk.astype(np.float32), Y, V, F, u=0.5, solo_ge50=False)
        return {
            'iou': mr.bootstrap_metricas(p0, is_obj=False, n_boot=1000, seed=7),
            'o50': mr.bootstrap_metricas(ob[50], is_obj=True, n_boot=1000, seed=7),
            'o200': mr.bootstrap_metricas(ob[200], is_obj=True, n_boot=1000, seed=7),
        }

    b_v9_t = boot_res(mv9_t, Yt, Vt, Ft)
    b_ir_t = boot_res(mir_t, Yt, Vt, Ft)
    p2_v9 = e5.eval_prueba2_fa(mv9_p2.astype(np.float32), Vp2, Fp2, 0.5)
    p2_ir = e5.eval_prueba2_fa(mir_p2.astype(np.float32), Vp2, Fp2, 0.5)

    # Revisar las 39 FA de fa_clasificacion_v9.csv
    idx_pch = {r['parche']: i for i, r in enumerate(Ft)}
    fa_rows = list(csv.DictReader(open('informes/fa_clasificacion_v9.csv', encoding='utf-8')))
    fa39_por_tipo = collections.defaultdict(lambda: {'n_v9': 0, 'px_v9': 0, 'n_elim': 0, 'px_elim': 0})
    fa_elim_cands = []
    for r in fa_rows:
        i = idx_pch[r['parche']]
        lp, _ = ndimage.label(mv9_t[i] & (Vt[i] > 0), structure=ST8)
        cm = (lp == int(r['cid']))
        px_10 = int((cm & mir_t[i]).sum())
        d = fa39_por_tipo[r['clase']]
        d['n_v9'] += 1; d['px_v9'] += int(r['px'])
        if px_10 == 0:
            d['n_elim'] += 1; d['px_elim'] += int(r['px'])
            fa_elim_cands.append((int(r['px']), i, cm, r['parche'], r['clase']))

    # Figura Tarea A: 4 ejemplos (2 FA eliminadas + 2 cicatrices mantenidas)
    fa_elim_cands.sort(key=lambda x: (-x[0], x[1]))
    tp_mant_cands = []
    for i in range(len(Ft)):
        yi = (Yt[i] > 0) & (Vt[i] > 0)
        if int(yi.sum()) >= 50 and int((yi & mir_t[i]).sum()) >= 30:
            tp_mant_cands.append((int((yi & mir_t[i]).sum()), i, yi, Ft[i]['parche']))
    tp_mant_cands.sort(key=lambda x: (-x[0], x[1]))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    ejemplos = []
    for sz, i, cm, pch, cl in fa_elim_cands[:2]:
        ejemplos.append((i, f'FA eliminada (tipo {cl}, {sz} px)', cm, mir_t[i], (Yt[i] > 0) & (Vt[i] > 0)))
    for sz, i, yi, pch in tp_mant_cands[:2]:
        ejemplos.append((i, f'Cicatriz mantenida ({sz} px TP)', mv9_t[i], mir_t[i], yi))
    if ejemplos:
        fig, axes = plt.subplots(len(ejemplos), 3, figsize=(10.5, 3.0 * len(ejemplos)), dpi=180, facecolor='white')
        if len(ejemplos) == 1:
            axes = axes[None, :]
        for ri, (i, sub, m_ant, m_des, gt) in enumerate(ejemplos):
            rgb, _, _ = grp.s2_rgb(os.path.join('parches_128', Ft[i]['archivo']))
            axes[ri, 0].imshow(v8.overlay(rgb, gt, (0.1, 0.85, 0.2), 0.55))
            axes[ri, 1].imshow(v8.overlay(rgb, m_ant, (1.0, 0.25, 0.15), 0.55))
            axes[ri, 2].imshow(v8.overlay(rgb, m_des, (1.0, 0.15, 0.15), 0.55))
            if ri == 0:
                axes[ri, 0].set_title('Referencia', fontsize=9.5, fontweight='bold')
                axes[ri, 1].set_title('Modelo v9', fontsize=9.5, fontweight='bold')
                axes[ri, 2].set_title('v9 + filtro infrarrojo', fontsize=9.5, fontweight='bold')
            axes[ri, 0].set_ylabel(f"{Ft[i]['parche']}\n{sub}", fontsize=8, rotation=0, labelpad=58, va='center')
            for ax in axes[ri]:
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig('informes/figuras_finales/filtro_ir_4_ejemplos.png', dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fmt = mr.fmt_ic
    regla_usada_nom = mejor_r['regla'] if se_aplica else top_cand['regla']
    md_a = [
        '# Tarea A — Filtro Infrarrojo sobre la `v9`',
        '',
        '> [!IMPORTANT]',
        '> **Limitación:** `L4SPe` no dispone de bandas `B08, B11, B12`, por lo que el filtro infrarrojo se evalúa y aplica **únicamente sobre La Convención** (`validacion` 304, `prueba` 360 y `prueba2` 349 negativos). Toda elección de regla se hizo **exclusivamente en validación**.',
        '',
        '## 1. Comparación de Índices en Validación (`La Convención val`, `304` parches)',
        '',
        f'- **Objetos `v9` en validación:** `TP = {st_tp_v["n"]}`, `FA = {st_fa_v["n"]}`.',
        f'- **Medias en cicatrices reales (`TP`):** `NDVI = {st_tp_v["ndvi"]:.3f}`, `NDMI = {st_tp_v["ndmi"]:.3f}`, `BSI = {st_tp_v["bsi"]:.3f}`, `B11/B12 = {st_tp_v["r11_12"]:.3f}`.',
        f'- **Medias en falsas alarmas (`FA`):** `NDVI = {st_fa_v["ndvi"]:.3f}`, `NDMI = {st_fa_v["ndmi"]:.3f}`, `BSI = {st_fa_v["bsi"]:.3f}`, `B11/B12 = {st_fa_v["r11_12"]:.3f}`.',
        '',
        '### Reglas evaluadas en validación (top 6)',
        '| Regla | `LC val` IoU | `FA LC val` (`ha/km²`) | Cicatrices perdidas | ¿Cumple? |',
        '|---|---:|---:|---:|---|',
    ]
    for r in sorted(res_reglas, key=lambda x: (-int(x['cumple']), -x['lc_iou'], x['fa_lc']))[:6]:
        md_a.append(f"| `{r['regla']}` | {r['lc_iou']:.4f} | {r['fa_lc']:.4f} | {r['perdidas_val']} | {'Sí' if r['cumple'] else 'No'} |")
    md_a.extend([
        '',
        f"**Decisión en validación:** **{'SE APLICA' if se_aplica else 'NO SE APLICA'}** (`{regla_usada_nom}`).",
        '',
        '## 2. Comparación «Post Hoc» en `La Convención sellada` (`360`) y `prueba2` (`349` negativos)',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] | `prueba2` TOTAL (`ha/km²` / `%`) |',
        '|---|---|---|---|---:|',
        f"| **v9** | {fmt(b_v9_t['iou'],'iou')} | {fmt(b_v9_t['o50'],'f1',True)} | {fmt(b_v9_t['o200'],'rec',True)} | **{p2_v9['TOTAL_349']['fa_ha_km2']:.4f} ({p2_v9['TOTAL_349']['pct_con_marca']} %)** |",
        f"| **v9 + filtro IR (`{regla_usada_nom}`)** | {fmt(b_ir_t['iou'],'iou')} | {fmt(b_ir_t['o50'],'f1',True)} | {fmt(b_ir_t['o200'],'rec',True)} | **{p2_ir['TOTAL_349']['fa_ha_km2']:.4f} ({p2_ir['TOTAL_349']['pct_con_marca']} %)** |",
        '',
        '### `prueba2` por tipo de terreno («post hoc»)',
        '| Modelo | `bosque_ladera` | `altoandino` | `rio` | `chacra_o_pasto` | `carretera` | `poblado` |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| **v9** | {p2_v9['bosque_ladera']['fa_ha_km2']:.4f} | {p2_v9['altoandino']['fa_ha_km2']:.4f} | {p2_v9['rio']['fa_ha_km2']:.4f} | {p2_v9['chacra_o_pasto']['fa_ha_km2']:.4f} | {p2_v9['carretera']['fa_ha_km2']:.4f} | {p2_v9['poblado']['fa_ha_km2']:.4f} |",
        f"| **v9 + filtro IR** | {p2_ir['bosque_ladera']['fa_ha_km2']:.4f} | {p2_ir['altoandino']['fa_ha_km2']:.4f} | {p2_ir['rio']['fa_ha_km2']:.4f} | {p2_ir['chacra_o_pasto']['fa_ha_km2']:.4f} | {p2_ir['carretera']['fa_ha_km2']:.4f} | {p2_ir['poblado']['fa_ha_km2']:.4f} |",
        '',
        '## 3. Eliminación sobre las `39` falsas alarmas completas de `fa_clasificacion_v9.csv`',
        '',
        '| Tipo | Comp. `v9` | Eliminadas por filtro IR | Área `v9` (`ha`) | Área eliminada (`ha`) |',
        '|---|---:|---:|---:|---:|',
    ])
    for cl in ('a', 'b', 'c', 'd', 'e', 'f'):
        d = fa39_por_tipo[cl]
        md_a.append(f"| `{cl}` | {d['n_v9']} | **{d['n_elim']}** | {d['px_v9']*0.01:.2f} | {d['px_elim']*0.01:.2f} |")
    md_a.extend(['', '![Ejemplos del filtro infrarrojo](figuras_finales/filtro_ir_4_ejemplos.png)'])
    with open('informes/filtro_infrarrojo.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_a) + '\n')

    # ==================== TAREA B: ANALISIS DE ERRORES VERSION FINAL ====================
    m_fin_lc = mir_t if se_aplica else mv9_t
    p2_fin = p2_ir if se_aplica else p2_v9
    nombre_fin = f'v9 + filtro IR ({regla_usada_nom})' if se_aplica else 'v9 (sin filtro IR adicional)'

    # L4SPe test
    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_t = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        Xraw_pt = z['x'][sel_t]
        pend_pt = np.nan_to_num(Xraw_pt[:, 4, :, :] * 90.0, nan=30.0).astype(np.float32)
        elev_pt = np.nan_to_num(Xraw_pt[:, 5, :, :], nan=2000.0).astype(np.float32)
    pv9_pt = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xpt, disp, True), pp6.predecir_tta(m_p3, Xpt, disp, True),
                                    elev_pt, None, float(cfg_v9['e1_corte_altitud_m']), bool(cfg_v9['e1_condicion_wc_30_60']))
    m_fin_pe = v8.postproceso_objeto(v8.umbral_histeresis(pv9_pt, float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])) & (Vpt > 0),
                                     int(cfg_v9['morfologia']['min_px']), False, True) & (Vpt > 0)

    tam_lc, pend_lc, alt_lc = _desglose_errores(m_fin_lc, Yt, Vt, ctx_t['pend'], ctx_t['elev'])
    tam_pe, pend_pe, alt_pe = _desglose_errores(m_fin_pe, Ypt, Vpt, pend_pt, elev_pt)

    # Figura 3 paneles lista para tesis (en espanol)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=200, facecolor='white')
    # Panel 1: Tamano
    kt = ['< 0.5 ha', '0.5–2 ha', '> 2 ha']
    x1 = np.arange(len(kt)); w = 0.35
    r_lc = [tam_lc[k][0] / max(sum(tam_lc[k]), 1) * 100 for k in kt]
    r_pe = [tam_pe[k][0] / max(sum(tam_pe[k]), 1) * 100 for k in kt]
    axes[0].bar(x1 - w / 2, r_pe, w, label='L4SPe (prueba)', color='#2b7bba')
    axes[0].bar(x1 + w / 2, r_lc, w, label='La Convención (sellada)', color='#1f9d55')
    axes[0].set_xticks(x1); axes[0].set_xticklabels(kt, fontsize=9)
    axes[0].set_ylabel('Aciertos de cicatrices (%)', fontsize=9.5)
    axes[0].set_title('Aciertos por tamaño de cicatriz', fontsize=10.5, fontweight='bold')
    axes[0].set_ylim(0, 100); axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.25)

    # Panel 2: Altitud (rangos de 500 m)
    ka = ['< 1500 m', '1500–2000 m', '2000–2500 m', '2500–3000 m', '3000–3500 m', '> 3500 m']
    x2 = np.arange(len(ka))
    a_lc = [alt_lc[k][0] / max(sum(alt_lc[k]), 1) * 100 for k in ka]
    a_pe = [alt_pe[k][0] / max(sum(alt_pe[k]), 1) * 100 for k in ka]
    axes[1].bar(x2 - w / 2, a_pe, w, label='L4SPe (prueba)', color='#2b7bba')
    axes[1].bar(x2 + w / 2, a_lc, w, label='La Convención (sellada)', color='#1f9d55')
    axes[1].set_xticks(x2); axes[1].set_xticklabels(['<1.5k', '1.5–2k', '2–2.5k', '2.5–3k', '3–3.5k', '>3.5k'], fontsize=8.5)
    axes[1].set_ylabel('Aciertos de cicatrices (%)', fontsize=9.5)
    axes[1].set_title('Aciertos por altitud (rangos de 500 m)', fontsize=10.5, fontweight='bold')
    axes[1].set_ylim(0, 100); axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=0.25)

    # Panel 3: Falsas alarmas por tipo de terreno (prueba2)
    kf = ['bosque_ladera', 'altoandino', 'rio', 'chacra_o_pasto', 'carretera', 'poblado']
    kl = ['Bosque', 'Altoandino', 'Río', 'Chacra/pasto', 'Carretera', 'Poblado']
    vals_fa = [p2_fin[k]['fa_ha_km2'] for k in kf]
    axes[2].bar(np.arange(len(kf)), vals_fa, color='#d95f02', edgecolor='black', linewidth=0.6)
    axes[2].set_xticks(np.arange(len(kf))); axes[2].set_xticklabels(kl, fontsize=8.5, rotation=15)
    axes[2].set_ylabel('Falsas alarmas (ha/km²)', fontsize=9.5)
    axes[2].set_title('Falsas alarmas en prueba 2 por terreno', fontsize=10.5, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.25)

    plt.tight_layout()
    fig.savefig('informes/figuras_finales/analisis_errores_tesis_3paneles.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    md_b = [
        f'# Tarea B — Análisis de Errores de la Versión Final (`{nombre_fin}`)',
        '',
        '> [!IMPORTANT]',
        '> Evaluación **«post hoc»** sobre `La Convención sellada` (`360` parches), `L4SPe test` (`124` parches) y `prueba2` (`349` negativos confirmados).',
        '',
        '## a) Aciertos y omisiones por tamaño de cicatriz',
        '| Tamaño | `L4SPe test` (Aciertos / Omisiones / `%`) | `La Convención sellada` (Aciertos / Omisiones / `%`) |',
        '|---|---:|---:|',
    ]
    for k in kt:
        tp_p, fn_p = tam_pe[k]; tp_l, fn_l = tam_lc[k]
        md_b.append(f"| {k} | {tp_p} / {fn_p} ({tp_p/max(tp_p+fn_p,1)*100:.1f} %) | {tp_l} / {fn_l} ({tp_l/max(tp_l+fn_l,1)*100:.1f} %) |")

    md_b.extend([
        '',
        '## b1) Aciertos y omisiones por pendiente',
        '| Pendiente | `L4SPe test` (Aciertos / Omisiones / `%`) | `La Convención sellada` (Aciertos / Omisiones / `%`) |',
        '|---|---:|---:|',
    ])
    for k in ('< 20°', '20–30°', '30–40°', '> 40°'):
        tp_p, fn_p = pend_pe[k]; tp_l, fn_l = pend_lc[k]
        md_b.append(f"| {k} | {tp_p} / {fn_p} ({tp_p/max(tp_p+fn_p,1)*100:.1f} %) | {tp_l} / {fn_l} ({tp_l/max(tp_l+fn_l,1)*100:.1f} %) |")

    md_b.extend([
        '',
        '## b2) Aciertos y omisiones por altitud (rangos de `500 m`)',
        '| Altitud | `L4SPe test` (Aciertos / Omisiones / `%`) | `La Convención sellada` (Aciertos / Omisiones / `%`) |',
        '|---|---:|---:|',
    ])
    for k in ka:
        tp_p, fn_p = alt_pe[k]; tp_l, fn_l = alt_lc[k]
        md_b.append(f"| {k} | {tp_p} / {fn_p} ({tp_p/max(tp_p+fn_p,1)*100:.1f} %) | {tp_l} / {fn_l} ({tp_l/max(tp_l+fn_l,1)*100:.1f} %) |")

    md_b.extend([
        '',
        '## c) Falsas alarmas por tipo de terreno (`prueba2`, `349` negativos)',
        '| Tipo de terreno | `N` | Falsas alarmas (`ha/km²`) | `%` parches con marca |',
        '|---|---:|---:|---:|',
    ])
    for k, nom in zip(kf, kl):
        n_p = p2_fin[k].get('n_parches', p2_fin[k].get('n', '-'))
        md_b.append(f"| {nom} | {n_p} | {p2_fin[k]['fa_ha_km2']:.4f} | {p2_fin[k]['pct_con_marca']} % |")
    md_b.extend(['', '![Análisis de errores en 3 paneles](figuras_finales/analisis_errores_tesis_3paneles.png)'])

    with open('informes/analisis_errores_final.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_b) + '\n')

    json.dump({
        'se_aplica_filtro_ir': se_aplica,
        'regla_ir': regla_usada_nom,
        'val_base_v9': base_val,
        'val_mejor_ir': {k: v for k, v in (mejor_r or top_cand).items() if k != 'fn'},
        'posthoc_v9': b_v9_t, 'posthoc_ir': b_ir_t,
        'prueba2_v9': p2_v9, 'prueba2_ir': p2_ir,
        'fa39_por_tipo': dict(fa39_por_tipo),
    }, open('informes/filtro_ir_resumen.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Tareas A y B completadas.', flush=True)


if __name__ == '__main__':
    main()
