"""Diagnostico del IoU (Tarea 1) y metricas por cicatrices medibles (Tarea 2) en rama mejora-iou.

Evalua v6, v8 y v9 (sin reentrenar) sobre L4SPe test (124) y La Convencion sellada (360)
como «post hoc», con bootstrap por zonas (1 000 replicas).

Salidas:
  - informes/iou_diagnostico.md
  - informes/figuras_finales/iou_diagnostico_barras.png
  - informes/iou_reformulado.md
"""
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
import mejora_v8_l4spe as v8
import mejora_v9 as v9
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv

ST8 = np.ones((3, 3), dtype=bool)


def descomponer_error_iou(mask_pred, Y, V):
    """Descompone el error (FP + FN) en:
      a) borde (1-2 px del contorno de referencia)
      b) cicatrices omitidas completas (>= 50 px sin ningun pixel detectado)
      c) falsas alarmas completas (componentes predichos que no tocan GT)
      d) cicatrices pequenas (< 0.5 ha = < 50 px)
      e) interior (> 2 px en cicatrices grandes parcialmente detectadas)
    y calcula el IoU oraculo al corregir solo cada componente."""
    tp_tot = fp_tot = fn_tot = 0
    comp_fp = {k: 0 for k in ('borde', 'omitidas', 'fa_completa', 'pequenas', 'interior')}
    comp_fn = {k: 0 for k in ('borde', 'omitidas', 'fa_completa', 'pequenas', 'interior')}

    for i in range(len(Y)):
        vi = V[i] > 0
        yi = (Y[i] > 0) & vi
        pi = (mask_pred[i] > 0) & vi

        tp_i = yi & pi
        fp_i = pi & (~yi)
        fn_i = yi & (~pi)
        tp_tot += int(tp_i.sum())
        fp_tot += int(fp_i.sum())
        fn_tot += int(fn_i.sum())

        # Contorno 1-2 px alrededor de yi (dilatacion 2 px menos erosion 2 px)
        yd2 = ndimage.binary_dilation(yi, structure=ST8, iterations=2) & vi
        ye2 = ndimage.binary_erosion(yi, structure=ST8, iterations=2, border_value=0) & vi
        borde_band = yd2 & (~ye2)

        # Etiquetar GT para distinguir pequenas (< 50 px) y omitidas completas (>= 50 px, 0 px tocados)
        fn_usado = np.zeros_like(fn_i)
        if yi.any():
            lg, ng = ndimage.label(yi, structure=ST8)
            for cid in range(1, ng + 1):
                cm = (lg == cid)
                sz = int(cm.sum())
                c_fn = cm & fn_i
                if not c_fn.any():
                    continue
                if sz < 50:
                    comp_fn['pequenas'] += int(c_fn.sum())
                    fn_usado |= c_fn
                elif not (cm & pi).any():
                    comp_fn['omitidas'] += int(c_fn.sum())
                    fn_usado |= c_fn

        fn_rest = fn_i & (~fn_usado)
        fn_borde = fn_rest & borde_band
        fn_int = fn_rest & (~borde_band)
        comp_fn['borde'] += int(fn_borde.sum())
        comp_fn['interior'] += int(fn_int.sum())

        # Etiquetar prediccion para falsas alarmas completas (no tocan ninguna GT)
        fp_usado = np.zeros_like(fp_i)
        if pi.any():
            lp, np_c = ndimage.label(pi, structure=ST8)
            for cid in range(1, np_c + 1):
                cm = (lp == cid)
                if not (cm & yi).any():
                    comp_fp['fa_completa'] += int((cm & fp_i).sum())
                    fp_usado |= (cm & fp_i)

        fp_rest = fp_i & (~fp_usado)
        fp_borde = fp_rest & borde_band
        fp_int = fp_rest & (~borde_band)
        comp_fp['borde'] += int(fp_borde.sum())
        comp_fp['interior'] += int(fp_int.sum())

    err_tot = fp_tot + fn_tot
    iou_base = tp_tot / max(tp_tot + fp_tot + fn_tot, 1)

    desglose = {}
    for k in ('borde', 'omitidas', 'fa_completa', 'pequenas', 'interior'):
        err_k = comp_fp[k] + comp_fn[k]
        # Oraculo: quitar FP_k y convertir FN_k en TP
        tp_o = tp_tot + comp_fn[k]
        fp_o = fp_tot - comp_fp[k]
        fn_o = fn_tot - comp_fn[k]
        iou_o = tp_o / max(tp_o + fp_o + fn_o, 1)
        desglose[k] = {
            'fp': comp_fp[k], 'fn': comp_fn[k], 'err': err_k,
            'pct_err': round(err_k / max(err_tot, 1) * 100.0, 1),
            'iou_oraculo': round(iou_o, 4),
            'ganancia_iou': round(iou_o - iou_base, 4),
        }

    return {
        'tp': tp_tot, 'fp': fp_tot, 'fn': fn_tot, 'err_tot': err_tot,
        'iou_base': round(iou_base, 4),
        'componentes': desglose,
    }


def evaluar_reformulado(mask_pred, Y, V, filas, n_boot=1000):
    """Calcula metricas con bootstrap de zonas (1000 replicas) para:
      - todas (>=10 px): 0 px, tol 1 px, tol 2 px
      - solo >= 0.5 ha (>=50 px, menores a 255): 0 px, tol 1 px, tol 2 px"""
    prob_bin = mask_pred.astype(np.float32)
    out = {}
    for tag, solo50 in [('todas', False), ('ge05ha', True)]:
        _, p0, p1, p2, _ = mr.stats_por_zona(prob_bin, Y, V, filas, u=0.5, solo_ge50=solo50)
        out[tag] = {
            'tol0': mr.bootstrap_metricas(p0, is_obj=False, n_boot=n_boot, seed=7),
            'tol1': mr.bootstrap_metricas(p1, is_obj=False, n_boot=n_boot, seed=7),
            'tol2': mr.bootstrap_metricas(p2, is_obj=False, n_boot=n_boot, seed=7),
        }
    return out


def plot_barras_diagnostico(diag_data, out_path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    etiquetas = ['L4SPe\nv6', 'L4SPe\nv9', 'La Convención\nv6', 'La Convención\nv9']
    pares = [
        diag_data[('l4spe', 'v6')],
        diag_data[('l4spe', 'v9')],
        diag_data[('lc', 'v6')],
        diag_data[('lc', 'v9')],
    ]
    claves = [
        ('borde', 'a) Borde (1–2 px)', '#4c78a8'),
        ('omitidas', 'b) Cicatrices omitidas (≥ 0.5 ha)', '#f58518'),
        ('fa_completa', 'c) Falsas alarmas completas', '#e45756'),
        ('pequenas', 'd) Cicatrices pequeñas (< 0.5 ha)', '#72b7b2'),
        ('interior', 'e) Interior parcial (> 2 px)', '#bab0ac'),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200, facecolor='white')
    x = np.arange(len(etiquetas))
    bottom = np.zeros(len(etiquetas), dtype=np.float64)
    for k, nom, col in claves:
        vals = np.array([d['componentes'][k]['pct_err'] for d in pares], dtype=np.float64)
        bars = ax.bar(x, vals, bottom=bottom, width=0.55, label=nom, color=col, edgecolor='white', linewidth=0.8)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 5.5:
                ax.text(xi, b + v * 0.5, f'{v:.1f} %', ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=10, fontweight='bold')
    ax.set_ylabel('Porcentaje de píxeles con error (FP + FN, %)', fontsize=10)
    ax.set_ylim(0, 106)
    ax.set_title('Descomposición del error de píxeles en IoU («post hoc»)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, fontsize=8.8)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


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

    # Cargar conjuntos de prueba (evaluacion post hoc)
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = v9.cargar_contexto_espacial_locales(Ft)
    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_pe = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        elev_pt = z['x'][sel_pe, 5, :, :].astype(np.float32)

    pv6_t_0 = pp6.predecir_tta(m_v6, Xt, disp, False)
    pv6_pe_0 = pp6.predecir_tta(m_v6, Xpt, disp, False)
    pv6_t_t = pp6.predecir_tta(m_v6, Xt, disp, True)
    pv6_pe_t = pp6.predecir_tta(m_v6, Xpt, disp, True)
    pp3_t_t = pp6.predecir_tta(m_p3, Xt, disp, True)
    pp3_pe_t = pp6.predecir_tta(m_p3, Xpt, disp, True)

    mv6_t = (np.where(np.isin(ctx_t['wc'], [50, 80]), 0.0, pv6_t_0) >= 0.45) & (Vt > 0)
    mv6_pe = (pv6_pe_0 >= 0.45) & (Vpt > 0)

    pv8_t = pp6.aplicar_filtro_altoandino(0.6 * pv6_t_t + 0.4 * pp3_t_t, ctx_t['elev'], ctx_t['wc'], 'wc_dem_3200_agua_urb')
    pv8_pe = 0.6 * pv6_pe_t + 0.4 * pp3_pe_t
    mv8_t = v8.postproceso_objeto(v8.umbral_histeresis(pv8_t, 0.70, 0.25) & (Vt > 0), 20, False, True) & (Vt > 0)
    mv8_pe = v8.postproceso_objeto(v8.umbral_histeresis(pv8_pe, 0.70, 0.25) & (Vpt > 0), 20, False, True) & (Vpt > 0)

    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    pv9_t = v9.aplicar_e1_ensamble(pv6_t_t, pp3_t_t, ctx_t['elev'], ctx_t['wc'], corte, cond_wc)
    pv9_t[np.isin(ctx_t['wc'], [50, 80])] = 0.0
    pv9_pe = v9.aplicar_e1_ensamble(pv6_pe_t, pp3_pe_t, elev_pt, None, corte, cond_wc)
    mv9_t = v8.umbral_histeresis(pv9_t, ua, ub) & (Vt > 0)
    mv9_pe = v8.umbral_histeresis(pv9_pe, ua, ub) & (Vpt > 0)
    mv9_t = v9.aplicar_e4_filtros(mv9_t, ctx_t, False, True, True) & (Vt > 0)
    mv9_t = v8.postproceso_objeto(mv9_t, mpx, False, cl1) & (Vt > 0)
    mv9_pe = v8.postproceso_objeto(mv9_pe, mpx, False, cl1) & (Vpt > 0)

    # TAREA 1: Diagnostico del IoU (v6 y v9)
    diag = {
        ('l4spe', 'v6'): descomponer_error_iou(mv6_pe, Ypt, Vpt),
        ('l4spe', 'v9'): descomponer_error_iou(mv9_pe, Ypt, Vpt),
        ('lc', 'v6'): descomponer_error_iou(mv6_t, Yt, Vt),
        ('lc', 'v9'): descomponer_error_iou(mv9_t, Yt, Vt),
    }
    plot_barras_diagnostico(diag, 'informes/figuras_finales/iou_diagnostico_barras.png')

    md1 = [
        '# Tarea 1 — Diagnóstico del IoU («post hoc»)',
        '',
        '> [!IMPORTANT]',
        '> Todas las cifras se miden sobre los conjuntos de prueba ya sellados (`L4SPe` test `124` parches y `La Convención` sellada `360` parches), por lo que se reportan como **«post hoc»**.',
        '',
        '## Descomposición del error de píxeles (`FP + FN`) e IoU «oráculo» por componente',
        '',
        'Cada píxel con error se asigna de forma excluyente a: `a)` borde (`1–2 px` del contorno de la referencia), `b)` cicatrices omitidas completas (`≥ 0.5 ha`, `0 px` tocados), `c)` falsas alarmas completas (componentes predichos que no tocan referencia), `d)` cicatrices pequeñas (`< 0.5 ha`), y `e)` interior parcial (`> 2 px` dentro de cicatrices grandes parcialmente detectadas).',
        '',
        '| Conjunto | Modelo | IoU base | Error total (`FP + FN`) | `a)` Borde 1–2 px (`%` \| IoU oráculo) | `b)` Omitidas ≥ 0.5 ha (`%` \| IoU oráculo) | `c)` FA completas (`%` \| IoU oráculo) | `d)` Pequeñas < 0.5 ha (`%` \| IoU oráculo) | `e)` Interior > 2 px (`%` \| IoU oráculo) |',
        '|---|---|---:|---:|---|---|---|---|---|',
    ]
    for (cj_k, cj_nom) in [('l4spe', 'L4SPe test'), ('lc', 'La Convención sellada')]:
        for mod in ('v6', 'v9'):
            d = diag[(cj_k, mod)]
            c = d['componentes']
            md1.append(
                f"| {cj_nom} | **{mod}** | `{d['iou_base']:.4f}` | `{d['err_tot']:,}` px | "
                f"{c['borde']['pct_err']:.1f} % \\| `{c['borde']['iou_oraculo']:.4f}` (`+{c['borde']['ganancia_iou']:.4f}`) | "
                f"{c['omitidas']['pct_err']:.1f} % \\| `{c['omitidas']['iou_oraculo']:.4f}` (`+{c['omitidas']['ganancia_iou']:.4f}`) | "
                f"{c['fa_completa']['pct_err']:.1f} % \\| `{c['fa_completa']['iou_oraculo']:.4f}` (`+{c['fa_completa']['ganancia_iou']:.4f}`) | "
                f"{c['pequenas']['pct_err']:.1f} % \\| `{c['pequenas']['iou_oraculo']:.4f}` (`+{c['pequenas']['ganancia_iou']:.4f}`) | "
                f"{c['interior']['pct_err']:.1f} % \\| `{c['interior']['iou_oraculo']:.4f}` (`+{c['interior']['ganancia_iou']:.4f}`) |"
            )
    md1.extend([
        '',
        '![Descomposición del error de píxeles en IoU](figuras_finales/iou_diagnostico_barras.png)',
        '',
        '### Hallazgos principales del diagnóstico',
        f"- En **L4SPe test**, el mayor cuello de botella es el **borde a `1–2 px`** (`{diag[('l4spe','v6')]['componentes']['borde']['pct_err']:.1f} %` de todo el error en `v6` y `{diag[('l4spe','v9')]['componentes']['borde']['pct_err']:.1f} %` en `v9`): corregir solo el borde de `1–2 px` elevaría el IoU oráculo de `{diag[('l4spe','v6')]['iou_base']:.4f}` a **`{diag[('l4spe','v6')]['componentes']['borde']['iou_oraculo']:.4f}`** en `v6` y de `{diag[('l4spe','v9')]['iou_base']:.4f}` a **`{diag[('l4spe','v9')]['componentes']['borde']['iou_oraculo']:.4f}`** en `v9`.",
        '- En **La Convención sellada**, el error está dominado por **cicatrices grandes omitidas** y **falsas alarmas completas**, reflejo directo tanto de omisiones en sustratos pardo-rojizos/revegetados como de discrepancias de criterio y desalineación en la referencia de `10 m`.',
    ])
    with open('informes/iou_diagnostico.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md1) + '\n')

    # TAREA 2: IoU por cicatrices medibles (v6, v8, v9, bootstrap 1000)
    ref_t2 = {}
    for cj_k, Y, V, F, ms in [
        ('l4spe', Ypt, Vpt, Fpt, {'v6': mv6_pe, 'v8': mv8_pe, 'v9': mv9_pe}),
        ('lc', Yt, Vt, Ft, {'v6': mv6_t, 'v8': mv8_t, 'v9': mv9_t}),
    ]:
        ref_t2[cj_k] = {m: evaluar_reformulado(mp, Y, V, F, n_boot=1000) for m, mp in ms.items()}

    fmt = mr.fmt_ic
    md2 = [
        '# Tarea 2 — IoU por Cicatrices Medibles («post hoc»)',
        '',
        '> [!IMPORTANT]',
        '> Todas las métricas corresponden a una evaluación **«post hoc»** sobre `L4SPe test` (`124` parches, `68` zonas) y `La Convención sellada` (`360` parches, `62` zonas), con **IC 95 % por bootstrap de zonas (`1 000` réplicas)**.',
        '',
        '## ¿Por qué se reformula la métrica?',
        '- **Resolución espacial de `10 m` (Sentinel-2):** en cicatrices pequeñas o estrechas (p. ej. `10–49 px`, ancho de `1–3 px`), un desplazamiento de apenas `1–2 px` (`10–20 m`) por corregistro o suavizado de bordes destruye el IoU exacto aunque el deslizamiento esté correctamente localizado.',
        '- **Concordancia de la referencia humana (`0.30–0.39`):** en la auditoría de acuerdo entre anotadores (profesor vs rondas independientes en `acuerdo_anotadores.py`), el IoU inter-anotador sobre la misma imagen Sentinel-2 a `10 m` se sitúa entre `0.30` y `0.39`; exigir `IoU ≥ 0.70` a píxel exacto sobre todas las cicatrices `≥ 10 px` supera el techo de reproducibilidad de la propia etiqueta humana.',
        '- Por ello se evalúan explícitamente: **a)** solo cicatrices medibles `≥ 0.5 ha` (`≥ 50 px`, pasando las menores a `255` dudoso/ignorado), **b)** tolerancia de borde de `1 px` y `2 px` (`8`-conectada), y **c)** la combinación de **a) y b)**.',
        '',
    ]
    for cj_k, cj_nom in [('l4spe', 'L4SPe Test (`124` parches, `68` zonas)'), ('lc', 'La Convención Sellada (`360` parches, `62` zonas)')]:
        md2.extend([
            f'## {cj_nom} — «post hoc»',
            '',
            '| Modelo | Formulación | IoU [IC 95 %] | F1 [IC 95 %] | Recall [IC 95 %] |',
            '|---|---|---|---|---|',
        ])
        for mod in ('v6', 'v8', 'v9'):
            d = ref_t2[cj_k][mod]
            filas_m = [
                ('Original (`≥ 0.1 ha`, `0 px`)', d['todas']['tol0']),
                ('a) Solo `≥ 0.5 ha` (`0 px`)', d['ge05ha']['tol0']),
                ('b1) Todas + tol. `1 px`', d['todas']['tol1']),
                ('b2) Todas + tol. `2 px`', d['todas']['tol2']),
                ('c1) `≥ 0.5 ha` + tol. `1 px`', d['ge05ha']['tol1']),
                ('c2) `≥ 0.5 ha` + tol. `2 px`', d['ge05ha']['tol2']),
            ]
            for f_nom, m in filas_m:
                md2.append(f"| **{mod}** | {f_nom} | {fmt(m,'iou')} | {fmt(m,'f1',True)} | {fmt(m,'rec',True)} |")
        md2.append('')

    with open('informes/iou_reformulado.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md2) + '\n')

    json.dump({'diagnostico': {f'{k[0]}_{k[1]}': v for k, v in diag.items()}, 'reformulado': ref_t2},
              open('informes/iou_diagnostico_y_reformulado.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Tareas 1 y 2 completadas -> informes/iou_diagnostico.md, iou_reformulado.md y figura.', flush=True)


if __name__ == '__main__':
    main()
