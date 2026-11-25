"""Tarea Extra (clasificacion visual de FA completas de v9 en prueba 360) y
Tarea 5 (material para remarcado de prueba 360: lista CSV, docs/GUIA_REMARCADO.md, PDF).

Uso:
  1) python herramientas/preparar_remarcado_y_fa.py --fase fa
     -> Genera laminas de TODAS las FA completas de v9 en prueba (360) para inspeccion visual.
  2) python herramientas/preparar_remarcado_y_fa.py --fase pdf --prioridad-alta informes/fa_clasificacion_v9.csv
     -> Genera lista de 40-60 parches con mayor desacuerdo + prioridad alta de FA tipo (a)
        y arma informes/remarcado_laminas.pdf (sin modificar ninguna mascara).
"""
import argparse
import concurrent.futures
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import generar_revision_profesor as grp
import mejora_v8_l4spe as v8
import mejora_v9 as v9
import posproceso_v6 as pp6
from nucleo import config as cfg_srv

ST8 = np.ones((3, 3), dtype=bool)


def calcular_v9_prueba_360():
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

    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = v9.cargar_contexto_espacial_locales(Ft)

    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    pv9_t = v9.aplicar_e1_ensamble(pp6.predecir_tta(m_v6, Xt, disp, True), pp6.predecir_tta(m_p3, Xt, disp, True),
                                   ctx_t['elev'], ctx_t['wc'], corte, cond_wc)
    pv9_t[np.isin(ctx_t['wc'], [50, 80])] = 0.0
    mv9_t = v9.aplicar_e4_filtros(v8.umbral_histeresis(pv9_t, ua, ub) & (Vt > 0), ctx_t, False, True, True) & (Vt > 0)
    mv9_t = v8.postproceso_objeto(mv9_t, mpx, False, cl1) & (Vt > 0)
    return Ft, Yt, Vt, mv9_t, ctx_t


def descargar_bings(filas):
    cache = {}
    def uno(r):
        p_tif = os.path.join('parches_128', r['archivo'])
        with rasterio.open(p_tif) as ds:
            b, crs = ds.bounds, ds.crs
        return r['parche'], grp.fetch_bing_patch(b, str(crs), z=16, out_px=256)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for pch, im in ex.map(uno, filas):
            cache[pch] = im
    return cache


def fase_fa():
    Ft, Yt, Vt, mv9_t, ctx_t = calcular_v9_prueba_360()
    fa_comps = []
    for i, r in enumerate(Ft):
        yi = (Yt[i] > 0) & (Vt[i] > 0)
        pi = mv9_t[i] & (Vt[i] > 0)
        if not pi.any():
            continue
        lp, nc = ndimage.label(pi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lp == cid
            if not (cm & yi).any():
                sz = int(cm.sum())
                ok = cm & ctx_t['ok_ant'][i]
                dg_m = float(ctx_t['dgrvi'][i][ok].mean()) if ok.any() else float('nan')
                el_m = float(ctx_t['elev'][i][cm].mean())
                pe_m = float(ctx_t['pend'][i][cm].mean())
                wc_vals, wc_cnts = np.unique(ctx_t['wc'][i][cm], return_counts=True)
                wc_dom = int(wc_vals[np.argmax(wc_cnts)])
                via_f = float((cm & ctx_t['cerca_via'][i]).mean())
                cauce_f = float((cm & ctx_t['tpi_cauce'][i]).mean())
                edif_f = float((cm & ctx_t['edif_mask'][i]).mean())
                ns_f = float((cm & ctx_t['cerca_ns'][i]).mean())
                fa_comps.append({
                    'idx': i, 'cid': cid, 'parche': r['parche'], 'zona': r['zona'],
                    'px': sz, 'ha': round(sz * 0.01, 2), 'elev_m': round(el_m, 0),
                    'pend_deg': round(pe_m, 1), 'wc_dom': wc_dom,
                    'dgrvi': None if np.isnan(dg_m) else round(dg_m, 3),
                    'via_pct': round(via_f * 100, 0), 'cauce_pct': round(cauce_f * 100, 0),
                    'edif_pct': round(edif_f * 100, 0), 'ns_pct': round(ns_f * 100, 0),
                    'cm': cm, 'r': r,
                })

    fa_comps.sort(key=lambda d: (-d['px'], d['zona'], d['parche']))
    print(f'Total componentes FA completas v9 en prueba (360): {len(fa_comps)}, area total = {sum(d["px"] for d in fa_comps)} px ({sum(d["px"] for d in fa_comps)*0.01:.2f} ha)', flush=True)

    # Descargar Bing para los parches unicos con FA completa
    unicos = []
    vistos = set()
    for d in fa_comps:
        if d['parche'] not in vistos:
            vistos.add(d['parche'])
            unicos.append(d['r'])
    bings = descargar_bings(unicos)

    # Guardar CSV crudo para clasificacion
    os.makedirs('informes/figuras_finales', exist_ok=True)
    with open('informes/fa_completas_v9_crudo.csv', 'w', newline='', encoding='utf-8') as f:
        cols = ['id_fa', 'parche', 'zona', 'cid', 'px', 'ha', 'elev_m', 'pend_deg', 'wc_dom', 'dgrvi', 'via_pct', 'cauce_pct', 'edif_pct', 'ns_pct']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for k, d in enumerate(fa_comps, 1):
            row = {c: d[c] for c in cols if c != 'id_fa'}
            row['id_fa'] = k
            w.writerow(row)

    # Dibujar paginas de 6 componentes por lamina (antes, despues, aerea, v9)
    per_page = 6
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    for p0 in range(0, len(fa_comps), per_page):
        lote = fa_comps[p0:p0 + per_page]
        fig, axes = plt.subplots(len(lote), 4, figsize=(13.5, 2.55 * len(lote)), dpi=150, facecolor='white')
        if len(lote) == 1:
            axes = axes[None, :]
        for row_i, d in enumerate(lote):
            id_fa = p0 + row_i + 1
            r = d['r']
            rgb_desp, _, _ = grp.s2_rgb(os.path.join('parches_128', r['archivo']))
            rgb_ant = grp.s2_antes_rgb(r)
            rgb_bing = bings[r['parche']]
            im_v9 = v8.overlay(rgb_desp, d['cm'], (1.0, 0.15, 0.15), 0.65)
            im_v9 = grp.contorno_amarillo(im_v9, d['cm']) if hasattr(grp, 'contorno_amarillo') else im_v9

            if rgb_ant is not None:
                axes[row_i, 0].imshow(rgb_ant)
            else:
                axes[row_i, 0].text(0.5, 0.5, 'Sin escena', ha='center', va='center')
            axes[row_i, 1].imshow(rgb_desp)
            axes[row_i, 2].imshow(rgb_bing)
            axes[row_i, 3].imshow(im_v9)
            if row_i == 0:
                axes[row_i, 0].set_title('Antes (2018–2020)', fontsize=9, fontweight='bold')
                axes[row_i, 1].set_title('Después (2023–2025)', fontsize=9, fontweight='bold')
                axes[row_i, 2].set_title('Aérea (Bing)', fontsize=9, fontweight='bold')
                axes[row_i, 3].set_title('Falsa alarma v9', fontsize=9, fontweight='bold')
            axes[row_i, 0].set_ylabel(
                f"#{id_fa} {d['parche']}\n{d['px']} px ({d['ha']:.2f} ha)\nWC={d['wc_dom']} {d['elev_m']:.0f}m\ndG={d['dgrvi']}",
                fontsize=7.2, rotation=0, labelpad=48, va='center'
            )
            for ax in axes[row_i]:
                ax.set_xticks([]); ax.set_yticks([])
        pag = p0 // per_page + 1
        fig.suptitle(f'Falsas alarmas completas de v9 en prueba (360) — Lámina {pag}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'informes/figuras_finales/fa_v9_prueba_pag{pag:02d}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f'Laminas FA guardadas en informes/figuras_finales/fa_v9_prueba_pag*.png', flush=True)


def fase_pdf(ruta_prio_alta):
    Ft, Yt, Vt, mv9_t, _ = calcular_v9_prueba_360()
    prio_alta = set()
    if ruta_prio_alta and os.path.exists(ruta_prio_alta):
        with open(ruta_prio_alta, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('clase') == 'a':
                    prio_alta.add(r['parche'].strip())

    # IoU por parche entre referencia y v9
    filas_des = []
    for i, r in enumerate(Ft):
        vi = Vt[i] > 0
        yi = (Yt[i] > 0) & vi
        pi = mv9_t[i] & vi
        inter = int((yi & pi).sum())
        uni = int((yi | pi).sum())
        if uni == 0:
            continue
        iou = inter / uni
        err_px = int((yi ^ pi).sum())
        es_prio = r['parche'] in prio_alta
        if es_prio or err_px >= 10:
            filas_des.append({
                'parche': r['parche'], 'zona': r['zona'],
                'prioridad': 'ALTA (FA v9 tipo a)' if es_prio else 'normal',
                'iou_parche': round(iou, 4),
                'px_ref': int(yi.sum()), 'px_v9': int(pi.sum()), 'px_error': err_px,
                '_idx': i, '_r': r, '_prio_num': 0 if es_prio else 1,
            })

    # Seleccionar todos los prioridad ALTA + top desacuerdo (mayor error / menor IoU) hasta 55 parches, ordenados por prioridad y zona
    filas_des.sort(key=lambda d: (d['_prio_num'], d['iou_parche'], -d['px_error']))
    sel = filas_des[:55]
    sel.sort(key=lambda d: (d['_prio_num'], d['zona'], d['parche']))

    with open('informes/remarcado_lista.csv', 'w', newline='', encoding='utf-8') as f:
        cols = ['orden', 'prioridad', 'zona', 'parche', 'iou_parche', 'px_ref', 'px_v9', 'px_error']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for k, d in enumerate(sel, 1):
            w.writerow({'orden': k, 'prioridad': d['prioridad'], 'zona': d['zona'], 'parche': d['parche'],
                        'iou_parche': d['iou_parche'], 'px_ref': d['px_ref'], 'px_v9': d['px_v9'], 'px_error': d['px_error']})

    bings = descargar_bings([d['_r'] for d in sel])

    # Generar PDF informes/remarcado_laminas.pdf (5 paneles: antes, despues, aerea, referencia actual, v9)
    os.makedirs('informes', exist_ok=True)
    with PdfPages('informes/remarcado_laminas.pdf') as pdf:
        # Pagina 1: resumen e instrucciones
        fig = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor='white')
        ax = fig.add_axes([0.08, 0.1, 0.84, 0.8]); ax.axis('off')
        ax.text(0.0, 0.94, 'LÁMINAS PARA REMARCADO DE LA PRUEBA SELLADA (55 PARCHES)', fontsize=15, fontweight='bold')
        ax.text(0.0, 0.85,
                'Criterio común (ver docs/GUIA_REMARCADO.md):\n'
                '  1. Marcar únicamente cicatrices claras de 0.5 ha o más (>= 50 píxeles a 10 m).\n'
                '  2. Los bordes inciertos y fragmentos menores de 0.5 ha se marcan como DUDOSO (255).\n'
                '  3. No marcar cauces activos, playas de río, chacras, pastos, carreteras ni construcciones.\n'
                f'  4. Incluye {len(prio_alta)} parches con PRIORIDAD ALTA (posible cicatriz real en FA de v9).',
                fontsize=11, va='top', linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)

        # 2 parches por pagina (5 paneles por parche)
        for p0 in range(0, len(sel), 2):
            lote = sel[p0:p0 + 2]
            fig, axes = plt.subplots(len(lote), 5, figsize=(11.69, 8.27), dpi=150, facecolor='white')
            if len(lote) == 1:
                axes = axes[None, :]
            for ri, d in enumerate(lote):
                idx = d['_idx']; r = d['_r']
                rgb_desp, _, _ = grp.s2_rgb(os.path.join('parches_128', r['archivo']))
                rgb_ant = grp.s2_antes_rgb(r)
                rgb_bing = bings[r['parche']]
                gt = (Yt[idx] > 0) & (Vt[idx] > 0)
                p9 = mv9_t[idx] & (Vt[idx] > 0)
                if rgb_ant is not None:
                    axes[ri, 0].imshow(rgb_ant)
                else:
                    axes[ri, 0].text(0.5, 0.5, 'Sin escena', ha='center', va='center')
                axes[ri, 1].imshow(rgb_desp)
                axes[ri, 2].imshow(rgb_bing)
                axes[ri, 3].imshow(v8.overlay(rgb_desp, gt, (0.1, 0.85, 0.2), 0.55))
                axes[ri, 4].imshow(v8.overlay(rgb_desp, p9, (1.0, 0.15, 0.15), 0.55))
                tits = ['Antes (2018–2020)', 'Después (2023–2025)', 'Aérea (Bing)',
                        f"Referencia ({d['px_ref']} px)", f"v9 ({d['px_v9']} px)"]
                for ax_c, t in zip(axes[ri], tits):
                    ax_c.set_title(t, fontsize=8.5, fontweight='bold')
                    ax_c.set_xticks([]); ax_c.set_yticks([])
                axes[ri, 0].set_ylabel(
                    f"#{p0+ri+1} [{d['prioridad']}]\n{d['zona']}\n{d['parche']}\nIoU={d['iou_parche']:.3f}",
                    fontsize=7.2, rotation=0, labelpad=58, va='center'
                )
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f'Lista ({len(sel)} parches) -> informes/remarcado_lista.csv y PDF -> informes/remarcado_laminas.pdf', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', choices=['fa', 'pdf'], required=True)
    ap.add_argument('--prioridad-alta', default='informes/fa_clasificacion_v9.csv')
    a = ap.parse_args()
    if a.fase == 'fa':
        fase_fa()
    else:
        fase_pdf(a.prioridad_alta)


if __name__ == '__main__':
    main()
