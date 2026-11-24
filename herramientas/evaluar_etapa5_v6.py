"""Evaluacion unica de Etapa 5 (protocolo v6).

Evalua UNA SOLA VEZ con umbrales fijados exclusivamente en validacion:
  - prueba2 (principal): falsas alarmas en ha/km2 y % de parches negativos con marca, por tipo_negativo
    (altoandino, chacra_o_pasto, rio, carretera, poblado, bosque_ladera), mas lamina de los 4 positivos cualitativos.
  - prueba sellada antigua (360) y L4SPe test (124), rotuladas «post hoc»:
    IoU, IoU tol 1 px (4c/8c), precision, exhaustividad, deteccion por tamano (10-19, 20-50, 50-200, >200),
    falsas alarmas e IC 95 % por zonas frente a B1_v5 y B2_v5 con el mismo posproceso.
  - Tiempo de inferencia por parche en CPU (ms/parche) con y sin TTA.
  - 6 laminas de prueba sellada (3 aciertos + 3 errores) + lamina de los 4 positivos cualitativos de prueba2.
"""
import csv
import json
import os
import sys
import time

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
import posproceso_v6 as pp6
from nucleo import config as cfg_srv


FILTRO = 'wc_dem_3200_agua_urb'
TIPOS_NEG = ['altoandino', 'chacra_o_pasto', 'rio', 'carretera', 'poblado', 'bosque_ladera']
KEEP_POS_P2 = [
    'ZP2_26_PICHARI_001_q1',
    'ZP2_128_ECHARATE_001_q4',
    'ZP2_102_ECHARATE_001_q3',
    'ZP2_14_KIMBIRI_001_q3',
]


def det_por_tamano_4(prob, Y, V, u):
    """Deteccion (>=30% solape) en 4 tramos: 10-19, 20-50, 50-200, >200 y total >=10."""
    st8 = np.ones((3, 3), dtype=bool)
    det = {'10-19': [0, 0], '20-50': [0, 0], '50-200': [0, 0], '>200': [0, 0], 'total_ge10': [0, 0]}
    for i in range(len(prob)):
        vi = V[i] > 0
        yi = (Y[i] > 0) & vi
        pi = (prob[i] >= u) & vi
        if not yi.any():
            continue
        lab, nc = ndimage.label(yi, structure=st8)
        for cid in range(1, nc + 1):
            cm = (lab == cid)
            sz = int(cm.sum())
            if sz < 10:
                continue
            bk = '10-19' if sz < 20 else ('20-50' if sz <= 50 else ('50-200' if sz <= 200 else '>200'))
            hit = float((cm & pi).sum()) / sz >= 0.30
            det[bk][1] += 1; det['total_ge10'][1] += 1
            if hit:
                det[bk][0] += 1; det['total_ge10'][0] += 1
    return {k: {'detectadas': v[0], 'total': v[1], 'pct': round(v[0] / max(v[1], 1) * 100.0, 1)} for k, v in det.items()}


def eval_prueba2_fa(prob, V, filas, u):
    """Metricas principales de prueba2: FA en ha/km2 y % de parches negativos con marca, por tipo_negativo."""
    out = {}
    grupos = {'TOTAL_349': [i for i, r in enumerate(filas) if (r.get('tipo_negativo') or '').strip()]}
    for t in TIPOS_NEG:
        grupos[t] = [i for i, r in enumerate(filas) if (r.get('tipo_negativo') or '').strip() == t]
    for g, idxs in grupos.items():
        n = len(idxs)
        if n == 0:
            out[g] = {'n_parches': 0, 'fa_ha_km2': 0.0, 'parches_con_marca': 0, 'pct_con_marca': 0.0, 'px_fa': 0}
            continue
        px_val = 0.0
        px_fa = 0
        n_hit = 0
        for i in idxs:
            vi = V[i] > 0
            pi = (prob[i] >= u) & vi
            px_val += float(vi.sum())
            c_fa = int(pi.sum())
            px_fa += c_fa
            if c_fa > 0:
                n_hit += 1
        out[g] = {
            'n_parches': n,
            'fa_ha_km2': round(px_fa / max(px_val, 1.0) * 100.0, 4),
            'parches_con_marca': n_hit,
            'pct_con_marca': round(n_hit / n * 100.0, 1),
            'px_fa': px_fa,
        }
    return out


def rgb_vis(ruta_tif):
    with rasterio.open(ruta_tif) as ds:
        b2 = ds.read(2).astype(np.float32)
        b3 = ds.read(3).astype(np.float32)
        b4 = ds.read(4).astype(np.float32)
    rgb = np.stack([b4, b3, b2], axis=-1)
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 2000)
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return rgb


def rgb_antes_parche(r):
    p_ant = os.path.join(r['proyecto'] + '_antes', r['tesela'] + '.tif')
    if not os.path.exists(p_ant):
        return None
    q = r['parche'].rsplit('_', 1)[-1]
    r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
    with rasterio.open(p_ant) as ds:
        a = ds.read()[:, r0:r0+128, c0:c0+128].astype(np.float32)
    rgb = np.stack([a[3], a[2], a[1]], axis=-1)
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 2000)
    return np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)


def overlay_mask(rgb, mask, color=(1.0, 0.15, 0.15), alpha=0.55):
    out = rgb.copy()
    m = mask > 0
    for c in range(3):
        out[..., c] = np.where(m, out[..., c] * (1.0 - alpha) + color[c] * alpha, out[..., c])
    return out


def guardar_lamina_parche(ruta_out, r, y_gt, v_gt, preds_dict, titulo_extra=''):
    """Lamina 1x6: RGB post, RGB antes, GT, B1_v5+flt, U-Net v6 (TTA+flt), Ensamble v6 (TTA+flt)."""
    rgb = rgb_vis(os.path.join('parches_128', r['archivo']))
    rgb_a = rgb_antes_parche(r)
    cols = [
        ('RGB Post', rgb),
        ('RGB Antes', rgb_a if rgb_a is not None else np.zeros_like(rgb)),
        (f'GT ({int((y_gt>0).sum())} px)', overlay_mask(rgb, y_gt > 0, (0.1, 0.9, 0.2), 0.55)),
    ]
    for nombre, (pm, u) in preds_dict.items():
        bm = (pm >= u) & (v_gt > 0)
        cols.append((f'{nombre}\nu={u:.2f} ({int(bm.sum())} px)', overlay_mask(rgb, bm, (1.0, 0.15, 0.15), 0.55)))
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6), dpi=140)
    for ax, (tit, img) in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(tit, fontsize=8.5)
        ax.axis('off')
    fig.suptitle(f"{r['parche']} ({r['zona']}) {titulo_extra}", fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_out), exist_ok=True)
    fig.savefig(ruta_out, bbox_inches='tight')
    plt.close(fig)


def guardar_lamina_prueba2_4(ruta_out, filas_p2, Y_p2, V_p2, preds_p2_dict):
    """Lamina 4x7 con los 4 positivos cualitativos de prueba2 y la prediccion de cada modelo."""
    idx_map = {r['parche']: i for i, r in enumerate(filas_p2)}
    n_cols = 3 + len(preds_p2_dict)
    fig, axes = plt.subplots(len(KEEP_POS_P2), n_cols, figsize=(2.9 * n_cols, 3.0 * len(KEEP_POS_P2)), dpi=140)
    for row_i, parche in enumerate(KEEP_POS_P2):
        idx = idx_map[parche]
        r = filas_p2[idx]
        rgb = rgb_vis(os.path.join('parches_128', r['archivo']))
        rgb_a = rgb_antes_parche(r)
        y_gt = Y_p2[idx] > 0
        v_gt = V_p2[idx] > 0
        paneles = [
            (f'{parche}\nRGB Post', rgb),
            ('RGB Antes', rgb_a if rgb_a is not None else np.zeros_like(rgb)),
            (f'GT cualit. ({int(y_gt.sum())} px)', overlay_mask(rgb, y_gt, (0.1, 0.9, 0.2), 0.55)),
        ]
        for nombre, (prob_arr, u) in preds_p2_dict.items():
            bm = (prob_arr[idx] >= u) & v_gt
            ov = float((bm & y_gt).sum()) / max(int(y_gt.sum()), 1) * 100.0
            paneles.append((f'{nombre} (u={u:.2f})\n{int(bm.sum())} px | cov {ov:.0f}%',
                            overlay_mask(rgb, bm, (1.0, 0.15, 0.15), 0.55)))
        for col_i, (tit, img) in enumerate(paneles):
            ax = axes[row_i, col_i]
            ax.imshow(img)
            ax.set_title(tit, fontsize=7.5)
            ax.axis('off')
    fig.suptitle('prueba2 — 4 positivos cualitativos (solo inspeccion visual; no se reporta IoU)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_out), exist_ok=True)
    fig.savefig(ruta_out, bbox_inches='tight')
    plt.close(fig)


def medir_tiempo_cpu(m_unet, X_u, m_dl, X_d, n=32):
    """Mide tiempo promedio por parche en ms en CPU (PyTorch), con y sin TTA 4."""
    xu = X_u[:n]; xd = X_d[:n]
    # Warmup
    _ = pp6.predecir_tta(m_unet, xu[:4], 'cpu', usar_tta=False)
    _ = pp6.predecir_tta(m_dl, xd[:4], 'cpu', usar_tta=False)
    res = {}
    for nombre, m, x in [('unet', m_unet, xu), ('deeplabv3plus', m_dl, xd)]:
        for tta in (False, True):
            t0 = time.perf_counter()
            _ = pp6.predecir_tta(m, x, 'cpu', usar_tta=tta)
            dt_ms = (time.perf_counter() - t0) / len(x) * 1000.0
            res[f'{nombre}_{"tta4" if tta else "sin_tta"}_ms'] = round(dt_ms, 2)
    res['ensamble_sin_tta_ms'] = round(res['unet_sin_tta_ms'] + res['deeplabv3plus_sin_tta_ms'], 2)
    res['ensamble_tta4_ms'] = round(res['unet_tta4_ms'] + res['deeplabv3plus_tta4_ms'], 2)
    return res


def main():
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S

    # Cargar todos los checkpoints necesarios
    ckpts = {
        'unet_v6_s42': ('resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
                        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json'),
        'unet_v6_s123': ('resultados_ajuste_v6/Control_B1_unet_s123/mejor_unet_b1b4b13b14_93eeb70a6d8d.pt',
                         'resultados_ajuste_v6/Control_B1_unet_s123/stats_norm_93eeb70a6d8d.json'),
        'unet_v6_s777': ('resultados_ajuste_v6/Control_B1_unet_s777/mejor_unet_b1b4b13b14_d892dd6b87df.pt',
                         'resultados_ajuste_v6/Control_B1_unet_s777/stats_norm_d892dd6b87df.json'),
        'c1_p2_s123': ('resultados_ajuste_v6/C1_unet_p2_cp_s123/mejor_unet_b1b4b13b14_b29f37240ff9.pt',
                       'resultados_ajuste_v6/C1_unet_p2_cp_s123/stats_norm_b29f37240ff9.json'),
        'dl_v6_s42': ('resultados_ajuste_v6/C2_deeplab_p2_cp_s42/mejor_deeplabv3plus_b1b4b13b14_ac82bc42d339.pt',
                      'resultados_ajuste_v6/C2_deeplab_p2_cp_s42/stats_norm_ac82bc42d339.json'),
        'b1_v5': ('resultados_ajuste/B1_unet_base/mejor_unet_b1b4b13b14_dfbbe6695037.pt',
                  'resultados_ajuste/B1_unet_base/stats_norm_dfbbe6695037.json'),
        'b2_v5': ('resultados_ajuste/B2_deeplabv3plus_base/mejor_deeplabv3plus_b1b4b13b14_3539764d8289.pt',
                  'resultados_ajuste/B2_deeplabv3plus_base/stats_norm_3539764d8289.json'),
    }
    modelos = {}
    recetas = {}
    for k, (cp, sp) in ckpts.items():
        m, rec, _ = pp6.cargar_modelo_ckpt(cp, sp, disp)
        modelos[k] = m
        recetas[k] = rec

    rec_u = recetas['unet_v6_s42']
    rec_d = recetas['dl_v6_s42']

    print('Cargando validacion, prueba2, prueba (360) y L4SPe...', flush=True)
    # Validacion (para fijar umbrales exactos de cada variante)
    Xv_u, Yv, Vv, Fv = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    Xv_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'validacion'})
    Xpv_u, Ypv, Vpv, Fpv = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'validacion'})
    Xpv_d, _, _, _ = af.cargar_l4spe('l4spe_profesor.npz', rec_d, {'validacion'})
    elev_v, wc_v = pp6.cargar_dem_y_worldcover(Fv)

    # prueba2 (564 parches)
    Xp2_u, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    Xp2_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'prueba2'})
    elev_p2, wc_p2 = pp6.cargar_dem_y_worldcover(Fp2)

    # prueba sellada antigua (360 parches, post hoc)
    Xt_u, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    Xt_d, _, _, _ = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_d, escala, {'prueba'})
    elev_t, wc_t = pp6.cargar_dem_y_worldcover(Ft)

    # L4SPe test (124 parches, post hoc)
    Xpt_u, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    Xpt_d, _, _, _ = af.cargar_l4spe('l4spe_profesor.npz', rec_d, {'prueba'})

    # Tiempo de inferencia en CPU
    print('Midiendo tiempos de inferencia CPU...', flush=True)
    tiempos_cpu = medir_tiempo_cpu(modelos['unet_v6_s42'], Xt_u, modelos['dl_v6_s42'], Xt_d, n=32)
    print('Tiempos CPU (PyTorch):', tiempos_cpu, flush=True)

    # Precalcular probabilidades crudas (sin TTA y con TTA 4) por modelo base y conjunto
    raw = {}
    def get_pred(mkey, split, tta):
        key = (mkey, split, tta)
        if key not in raw:
            is_dl = mkey in ('dl_v6_s42', 'b2_v5')
            X_map = {
                'val': Xv_d if is_dl else Xv_u,
                'val_pe': Xpv_d if is_dl else Xpv_u,
                'p2': Xp2_d if is_dl else Xp2_u,
                'pru': Xt_d if is_dl else Xt_u,
                'pru_pe': Xpt_d if is_dl else Xpt_u,
            }
            raw[key] = pp6.predecir_tta(modelos[mkey], X_map[split], disp, usar_tta=tta)
        return raw[key]

    def build_probs(spec, split):
        kind, tta, flt = spec['kind'], spec['tta'], spec['filtro']
        if kind == 'single':
            p = get_pred(spec['mkey'], split, tta)
        elif kind == 'ens_05_05':
            p = 0.5 * get_pred(spec['m1'], split, tta) + 0.5 * get_pred(spec['m2'], split, tta)
        elif kind == 'avg_3seeds':
            p = (get_pred('unet_v6_s42', split, tta) + get_pred('unet_v6_s123', split, tta) + get_pred('unet_v6_s777', split, tta)) / 3.0
        else:
            raise ValueError(kind)
        if flt != 'ninguno' and split in ('val', 'p2', 'pru'):
            elev, wc = {'val': (elev_v, wc_v), 'p2': (elev_p2, wc_p2), 'pru': (elev_t, wc_t)}[split]
            p = pp6.aplicar_filtro_altoandino(p, elev, wc, modo=flt)
        return p

    # Definicion de todas las variantes solicitadas en PARADA (l) aprobada
    SPECS = [
        # 2a) Modelos v6 (con TTA+filtro y sin TTA+filtro)
        {'id': 'unet_v6_tta_flt',      'label': 'U-Net v6 (TTA+flt)',            'kind': 'single',    'mkey': 'unet_v6_s42', 'tta': True,  'filtro': FILTRO,    'ref_cmp': 'b1_v5_tta_flt'},
        {'id': 'unet_v6_sin_tta_flt',  'label': 'U-Net v6 (sin TTA+flt)',        'kind': 'single',    'mkey': 'unet_v6_s42', 'tta': False, 'filtro': FILTRO,    'ref_cmp': 'b1_v5_flt'},
        {'id': 'dl_v6_tta_flt',        'label': 'DeepLabV3+ v6 (TTA+flt)',       'kind': 'single',    'mkey': 'dl_v6_s42',   'tta': True,  'filtro': FILTRO,    'ref_cmp': 'b2_v5_tta_flt'},
        {'id': 'dl_v6_sin_tta_flt',    'label': 'DeepLabV3+ v6 (sin TTA+flt)',   'kind': 'single',    'mkey': 'dl_v6_s42',   'tta': False, 'filtro': FILTRO,    'ref_cmp': 'b2_v5_flt'},
        {'id': 'ens_v6_tta_flt',       'label': 'Ensamble v6 0.5/0.5 (TTA+flt)', 'kind': 'ens_05_05', 'm1': 'unet_v6_s42', 'm2': 'dl_v6_s42', 'tta': True,  'filtro': FILTRO, 'ref_cmp': 'b1_v5_tta_flt'},
        {'id': 'ens_v6_sin_tta_flt',   'label': 'Ensamble v6 0.5/0.5 (sin TTA+flt)', 'kind': 'ens_05_05', 'm1': 'unet_v6_s42', 'm2': 'dl_v6_s42', 'tta': False, 'filtro': FILTRO, 'ref_cmp': 'b1_v5_flt'},
        # 2b) Comparadores B1_v5 y B2_v5
        {'id': 'b1_v5_raw',            'label': 'B1_v5 (sin posproceso)',        'kind': 'single',    'mkey': 'b1_v5',       'tta': False, 'filtro': 'ninguno', 'ref_cmp': None},
        {'id': 'b1_v5_flt',            'label': 'B1_v5 (solo filtro)',           'kind': 'single',    'mkey': 'b1_v5',       'tta': False, 'filtro': FILTRO,    'ref_cmp': None},
        {'id': 'b1_v5_tta_flt',        'label': 'B1_v5 (TTA+filtro)',            'kind': 'single',    'mkey': 'b1_v5',       'tta': True,  'filtro': FILTRO,    'ref_cmp': None},
        {'id': 'b2_v5_raw',            'label': 'B2_v5 (sin posproceso)',        'kind': 'single',    'mkey': 'b2_v5',       'tta': False, 'filtro': 'ninguno', 'ref_cmp': None},
        {'id': 'b2_v5_flt',            'label': 'B2_v5 (solo filtro)',           'kind': 'single',    'mkey': 'b2_v5',       'tta': False, 'filtro': FILTRO,    'ref_cmp': None},
        {'id': 'b2_v5_tta_flt',        'label': 'B2_v5 (TTA+filtro)',            'kind': 'single',    'mkey': 'b2_v5',       'tta': True,  'filtro': FILTRO,    'ref_cmp': None},
        # 2c) Promedio 3 semillas U-Net v6
        {'id': 'unet_v6_3sem_tta_flt', 'label': 'U-Net v6 prom. 3 sem (TTA+flt)','kind': 'avg_3seeds','tta': True,          'filtro': FILTRO,    'ref_cmp': 'b1_v5_tta_flt'},
        # 2d) C1_unet_p2_cp (resultado negativo de 10a/10b)
        {'id': 'c1_p2_tta_flt',        'label': 'C1_unet_p2_cp s123 (TTA+flt)',  'kind': 'single',    'mkey': 'c1_p2_s123',  'tta': True,  'filtro': FILTRO,    'ref_cmp': 'b1_v5_tta_flt'},
    ]

    resultados = {'filtro': FILTRO, 'tiempos_cpu_pytorch': tiempos_cpu, 'modelos': {}}
    cache_pru = {}
    cache_pru_pe = {}
    cache_p2 = {}
    umbrales = {}

    for sp in SPECS:
        sid = sp['id']
        # 1) Elegir umbral SOLO en validacion (0.70 L4SPe + 0.30 LC)
        pv_lc = build_probs(sp, 'val')
        pv_pe = build_probs(sp, 'val_pe')
        u, iou_v_pe, iou_v_lc, punt_v = pp6.mejor_umbral_07_03(pv_pe, Ypv, Vpv, pv_lc, Yv, Vv)
        umbrales[sid] = u

        # 2) Evaluacion UNICA en prueba2 (principal: FA por tipo_negativo)
        pp2 = build_probs(sp, 'p2')
        cache_p2[sid] = pp2
        fa_p2 = eval_prueba2_fa(pp2, Vp2, Fp2, u)

        # 3) Evaluacion UNICA post hoc en prueba (360) y L4SPe test (124)
        ppru = build_probs(sp, 'pru')
        ppru_pe = build_probs(sp, 'pru_pe')
        cache_pru[sid] = ppru
        cache_pru_pe[sid] = ppru_pe

        res_pru = af.resumen_local(ppru, Yt, Vt, Ft, u)
        res_pru['deteccion_4tramos'] = det_por_tamano_4(ppru, Yt, Vt, u)
        res_pe = af.resumen_local(ppru_pe, Ypt, Vpt, Fpt, u)
        res_pe['deteccion_4tramos'] = det_por_tamano_4(ppru_pe, Ypt, Vpt, u)

        resultados['modelos'][sid] = {
            'label': sp['label'],
            'tta': sp['tta'],
            'filtro': sp['filtro'],
            'umbral_val': u,
            'validacion': {'iou_l4spe': round(iou_v_pe, 4), 'iou_lc': round(iou_v_lc, 4), 'puntaje_07_03': round(punt_v, 4)},
            'prueba2_fa': fa_p2,
            'posthoc_prueba_360': res_pru,
            'posthoc_l4spe_test': res_pe,
        }
        print(f"[EVAL] {sp['label']:34s} | u={u:.2f} | P2_FA={fa_p2['TOTAL_349']['fa_ha_km2']:.4f} ({fa_p2['TOTAL_349']['pct_con_marca']}%) | "
              f"PH_LC IoU={res_pru['iou']:.4f} FA={res_pru['falsas_alarmas_ha_por_km2_en_negativos']} | PH_L4SPe IoU={res_pe['iou']:.4f}", flush=True)

    # 4) IC 95 % por zonas (post hoc) frente a B1_v5 / B2_v5 con el mismo posproceso
    for sp in SPECS:
        sid = sp['id']
        ref = sp.get('ref_cmp')
        if ref:
            ic_lc = af.bootstrap_zonas(cache_pru[ref], umbrales[ref], cache_pru[sid], umbrales[sid], Yt, Vt, Ft)
            ic_pe = af.bootstrap_zonas(cache_pru_pe[ref], umbrales[ref], cache_pru_pe[sid], umbrales[sid], Ypt, Vpt, Fpt)
            resultados['modelos'][sid]['posthoc_ic95_vs_' + ref] = {'prueba_360': ic_lc, 'l4spe_test': ic_pe}
            # Para ensamble tambien vs B2_v5 con mismo posproceso
            if sid.startswith('ens_v6'):
                ref2 = 'b2_v5_tta_flt' if sp['tta'] else 'b2_v5_flt'
                ic_lc2 = af.bootstrap_zonas(cache_pru[ref2], umbrales[ref2], cache_pru[sid], umbrales[sid], Yt, Vt, Ft)
                ic_pe2 = af.bootstrap_zonas(cache_pru_pe[ref2], umbrales[ref2], cache_pru_pe[sid], umbrales[sid], Ypt, Vpt, Fpt)
                resultados['modelos'][sid]['posthoc_ic95_vs_' + ref2] = {'prueba_360': ic_lc2, 'l4spe_test': ic_pe2}

    # Guardar JSON y CSVs
    with open('informes/evaluacion_etapa5_v6.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # CSV prueba2 FA
    fa_rows = []
    for sp in SPECS:
        m = resultados['modelos'][sp['id']]
        row = {'id': sp['id'], 'modelo': m['label'], 'u': m['umbral_val']}
        for g in ['TOTAL_349'] + TIPOS_NEG:
            d = m['prueba2_fa'][g]
            row[f'{g}_ha_km2'] = d['fa_ha_km2']
            row[f'{g}_pct'] = d['pct_con_marca']
        fa_rows.append(row)
    with open('informes/etapa5_prueba2_fa.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys())); w.writeheader(); w.writerows(fa_rows)

    # CSV post hoc
    ph_rows = []
    for sp in SPECS:
        sid = sp['id']
        m = resultados['modelos'][sid]
        lc = m['posthoc_prueba_360']; pe = m['posthoc_l4spe_test']
        d_lc = lc['deteccion_4tramos']; d_pe = pe['deteccion_4tramos']
        ref = sp.get('ref_cmp')
        ic_str_lc = ic_str_pe = ''
        if ref:
            ic_d = m['posthoc_ic95_vs_' + ref]
            ic_str_lc = f"[{ic_d['prueba_360']['ic95'][0]:+.4f}, {ic_d['prueba_360']['ic95'][1]:+.4f}]"
            ic_str_pe = f"[{ic_d['l4spe_test']['ic95'][0]:+.4f}, {ic_d['l4spe_test']['ic95'][1]:+.4f}]"
        ph_rows.append({
            'id': sid, 'modelo': m['label'], 'u': m['umbral_val'],
            'lc_iou': lc['iou'], 'lc_tol1_8c': lc['iou_tol1px_8c'], 'lc_f1': lc['f1'],
            'lc_prec': lc['precision'], 'lc_rec': lc['recall'],
            'lc_det_10_19': f"{d_lc['10-19']['detectadas']}/{d_lc['10-19']['total']}",
            'lc_det_20_50': f"{d_lc['20-50']['detectadas']}/{d_lc['20-50']['total']}",
            'lc_det_50_200': f"{d_lc['50-200']['detectadas']}/{d_lc['50-200']['total']}",
            'lc_det_gt200': f"{d_lc['>200']['detectadas']}/{d_lc['>200']['total']}",
            'lc_det_tot': f"{d_lc['total_ge10']['detectadas']}/{d_lc['total_ge10']['total']} ({d_lc['total_ge10']['pct']}%)",
            'lc_fa_ha_km2': lc['falsas_alarmas_ha_por_km2_en_negativos'],
            'lc_ic95_vs_ref': ic_str_lc,
            'pe_iou': pe['iou'], 'pe_tol1_8c': pe['iou_tol1px_8c'], 'pe_f1': pe['f1'],
            'pe_prec': pe['precision'], 'pe_rec': pe['recall'],
            'pe_det_tot': f"{d_pe['total_ge10']['detectadas']}/{d_pe['total_ge10']['total']} ({d_pe['total_ge10']['pct']}%)",
            'pe_fa_ha_km2': pe['falsas_alarmas_ha_por_km2_en_negativos'],
            'pe_ic95_vs_ref': ic_str_pe,
        })
    with open('informes/etapa5_posthoc.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(ph_rows[0].keys())); w.writeheader(); w.writerows(ph_rows)

    # 5) Generar laminas: 3 aciertos + 3 errores de prueba (360) y 4 positivos cualitativos de prueba2
    fdir = '_fichas/parada_m_v6'
    os.makedirs(fdir, exist_ok=True)
    p_ens = cache_pru['ens_v6_tta_flt']; u_ens = umbrales['ens_v6_tta_flt']
    scores_pos = []
    errors_neg = []
    for i, r in enumerate(Ft):
        vi = Vt[i] > 0
        yi = (Yt[i] > 0) & vi
        pi = (p_ens[i] >= u_ens) & vi
        if yi.any():
            inter = float((yi & pi).sum())
            union = float((yi | pi).sum())
            iou_i = inter / max(union, 1.0)
            scores_pos.append((iou_i, int(yi.sum()), i))
        elif pi.any():
            errors_neg.append((int(pi.sum()), i))
    scores_pos.sort(key=lambda x: (-x[0], -x[1]))
    # 3 mejores aciertos (zonas distintas)
    aciertos, zonas_u = [], set()
    for iou_i, sz_i, i in scores_pos:
        if Ft[i]['zona'] not in zonas_u and sz_i >= 50:
            aciertos.append((iou_i, i)); zonas_u.add(Ft[i]['zona'])
        if len(aciertos) == 3:
            break
    # 3 errores representativos: 2 falsos negativos grandes + 1 falsa alarma grande en negativo (si existe)
    errores, zonas_e = [], set()
    for iou_i, sz_i, i in sorted(scores_pos, key=lambda x: (x[0], -x[1])):
        if Ft[i]['zona'] not in zonas_e and sz_i >= 80:
            errores.append((f'FN IoU={iou_i:.2f}', i)); zonas_e.add(Ft[i]['zona'])
        if len(errores) == 2:
            break
    errors_neg.sort(reverse=True)
    for fa_px, i in errors_neg:
        if Ft[i]['zona'] not in zonas_e:
            errores.append((f'FP {fa_px} px', i)); zonas_e.add(Ft[i]['zona'])
            break
    while len(errores) < 3:
        for iou_i, sz_i, i in sorted(scores_pos, key=lambda x: (x[0], -x[1])):
            if i not in [e[1] for e in errores]:
                errores.append((f'FN IoU={iou_i:.2f}', i)); break

    for k, (iou_i, idx) in enumerate(aciertos, 1):
        r = Ft[idx]
        pdict = {
            'B1_v5+flt': (cache_pru['b1_v5_tta_flt'][idx], umbrales['b1_v5_tta_flt']),
            'U-Net v6': (cache_pru['unet_v6_tta_flt'][idx], umbrales['unet_v6_tta_flt']),
            'DeepLab v6': (cache_pru['dl_v6_tta_flt'][idx], umbrales['dl_v6_tta_flt']),
            'Ensamble v6': (cache_pru['ens_v6_tta_flt'][idx], umbrales['ens_v6_tta_flt']),
        }
        guardar_lamina_parche(os.path.join(fdir, f'0{k}_acierto_{r["parche"]}.png'), r, Yt[idx], Vt[idx], pdict, f'[Acierto, IoU ens={iou_i:.2f}]')

    for k, (tag, idx) in enumerate(errores, 4):
        r = Ft[idx]
        pdict = {
            'B1_v5+flt': (cache_pru['b1_v5_tta_flt'][idx], umbrales['b1_v5_tta_flt']),
            'U-Net v6': (cache_pru['unet_v6_tta_flt'][idx], umbrales['unet_v6_tta_flt']),
            'DeepLab v6': (cache_pru['dl_v6_tta_flt'][idx], umbrales['dl_v6_tta_flt']),
            'Ensamble v6': (cache_pru['ens_v6_tta_flt'][idx], umbrales['ens_v6_tta_flt']),
        }
        guardar_lamina_parche(os.path.join(fdir, f'0{k}_error_{r["parche"]}.png'), r, Yt[idx], Vt[idx], pdict, f'[Error: {tag}]')

    guardar_lamina_prueba2_4(
        os.path.join(fdir, '07_prueba2_4_positivos.png'),
        Fp2, Yp2, Vp2,
        {
            'B1_v5+flt': (cache_p2['b1_v5_tta_flt'], umbrales['b1_v5_tta_flt']),
            'U-Net v6': (cache_p2['unet_v6_tta_flt'], umbrales['unet_v6_tta_flt']),
            'DeepLab v6': (cache_p2['dl_v6_tta_flt'], umbrales['dl_v6_tta_flt']),
            'Ensamble v6': (cache_p2['ens_v6_tta_flt'], umbrales['ens_v6_tta_flt']),
        },
    )
    print('Laminas generadas en', fdir, '->', sorted(os.listdir(fdir)), flush=True)


if __name__ == '__main__':
    main()
