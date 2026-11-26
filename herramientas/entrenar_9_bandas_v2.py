"""Tarea C (opcional, mejora-iou): U-Net 9 canales v2 desde v6 mezclando lotes 8+8
(La Convencion con 9 bandas reales + dropout de canal IR p=0.5, y L4SPe con las 3 bandas en 0).

Seleccion de epoca y umbral SOLO en validacion (0.7 * IoU L4SPe + 0.3 * IoU LC).
Si no supera a v9 en validacion, se documenta y no se usa.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
from entrenar import perdida_mixta, predecir, receta_de
import entrenar_9_bandas as e9
import evaluar_etapa5_v6 as e5
import mejora_v8_l4spe as v8
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--epocas', type=int, default=15)
    ap.add_argument('--pasos', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--semilla', type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(12)
    torch.manual_seed(a.semilla)
    rng = np.random.default_rng(a.semilla)
    disp = 'cpu'
    escala = cfg_srv.ESCALA_L4S

    ck_v6 = torch.load('resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
                       map_location='cpu', weights_only=False)
    stats_6 = json.load(open('resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', encoding='utf-8'))
    stats_9 = json.load(open('modelos_ajustados/unet_v10_9b_stats.json', encoding='utf-8'))
    receta_6 = receta_de(['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'], stats_6)
    stats_ir = {k: stats_9['por_canal'][k] for k in ('B08', 'B11', 'B12')}

    # Cargar entrenamiento y validacion (LC 9b + L4SPe con 3 bandas IR en cero)
    X_lc_ent, Y_lc_ent, V_lc_ent, _ = e9.cargar_locales_9b(receta_6, stats_ir, {'entrenamiento'})
    X_lc_val, Y_lc_val, V_lc_val, F_lc_val = e9.cargar_locales_9b(receta_6, stats_ir, {'validacion'})
    _, wc_lc_val = pp6.cargar_dem_y_worldcover(F_lc_val)

    X_pe_ent6, Y_pe_ent, V_pe_ent, _ = af.cargar_l4spe('l4spe_profesor.npz', receta_6, {'entrenamiento'})
    X_pe_val6, Y_pe_val, V_pe_val, F_pe_val = af.cargar_l4spe('l4spe_profesor.npz', receta_6, {'validacion'})
    X_pe_ent = np.concatenate([X_pe_ent6, np.zeros((len(X_pe_ent6), 3, 128, 128), np.float32)], axis=1)
    X_pe_val = np.concatenate([X_pe_val6, np.zeros((len(X_pe_val6), 3, 128, 128), np.float32)], axis=1)

    mod = e9.crear_unet_9b_desde_v6(ck_v6, stats_6, disp=disp)
    e9.congelar_bn(mod)
    opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)

    def eval_val():
        mod.eval()
        p_lc = predecir(mod, X_lc_val, disp); p_lc[np.isin(wc_lc_val, [50, 80])] = 0.0
        p_pe = predecir(mod, X_pe_val, disp)
        u, iou_pe, iou_lc, punt = pp6.mejor_umbral_07_03(p_pe, Y_pe_val, V_pe_val, p_lc, Y_lc_val, V_lc_val)
        return u, round(iou_pe, 4), round(iou_lc, 4), round(punt, 4)

    u0, ipe0, ilc0, pt0 = eval_val()
    best = {'ep': 0, 'u': u0, 'iou_pe': ipe0, 'iou_lc': ilc0, 'punt': pt0,
            'sd': {k: v.cpu().clone() for k, v in mod.state_dict().items()}}
    hist = [{'epoca': 0, 'u': u0, 'iou_pe_val': ipe0, 'iou_lc_val': ilc0, 'punt_07_03': pt0, 'loss': 0.0}]
    print(f'[Ep 00] pe={ipe0:.4f} lc={ilc0:.4f} punt={pt0:.4f} (u={u0})', flush=True)

    t_ini = time.time()
    for ep in range(1, a.epocas + 1):
        mod.train()
        e9.congelar_bn(mod)
        losses = []
        for _ in range(a.pasos):
            i_lc = rng.integers(0, len(X_lc_ent), size=8)
            i_pe = rng.integers(0, len(X_pe_ent), size=8)
            xb = np.concatenate([X_lc_ent[i_lc].copy(), X_pe_ent[i_pe].copy()], axis=0)
            yb = np.concatenate([Y_lc_ent[i_lc].copy(), Y_pe_ent[i_pe].copy()], axis=0)
            vb = np.concatenate([V_lc_ent[i_lc].copy(), V_pe_ent[i_pe].copy()], axis=0)
            # Dropout de canal IR (p=0.5) en los 8 parches de La Convencion
            for j in range(8):
                if rng.random() < 0.5:
                    xb[j, 6:9] = 0.0
            for j in range(16):
                kr = int(rng.integers(0, 4))
                if kr:
                    xb[j] = np.rot90(xb[j], kr, axes=(-2, -1))
                    yb[j] = np.rot90(yb[j], kr, axes=(-2, -1))
                    vb[j] = np.rot90(vb[j], kr, axes=(-2, -1))
                if rng.random() < 0.5:
                    xb[j] = xb[j, :, :, ::-1]; yb[j] = yb[j, :, ::-1]; vb[j] = vb[j, :, ::-1]
            xt = torch.from_numpy(np.ascontiguousarray(xb))
            yt = torch.from_numpy(np.ascontiguousarray(yb))
            vt = torch.from_numpy(np.ascontiguousarray(vb))
            opt.zero_grad(set_to_none=True)
            loss = perdida_mixta(mod(xt)[:, 0], yt, vt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        u_e, ipe_e, ilc_e, pt_e = eval_val()
        hist.append({'epoca': ep, 'u': u_e, 'iou_pe_val': ipe_e, 'iou_lc_val': ilc_e,
                     'punt_07_03': pt_e, 'loss': round(float(np.mean(losses)), 4)})
        if pt_e > best['punt']:
            best = {'ep': ep, 'u': u_e, 'iou_pe': ipe_e, 'iou_lc': ilc_e, 'punt': pt_e,
                    'sd': {k: v.cpu().clone() for k, v in mod.state_dict().items()}}
        if ep % 5 == 0 or ep == a.epocas or pt_e == best['punt']:
            print(f'[Ep {ep:02d}] loss={np.mean(losses):.4f} pe={ipe_e:.4f} lc={ilc_e:.4f} punt={pt_e:.4f} [mejor={best["punt"]:.4f} ep={best["ep"]}]', flush=True)

    dur_min = round((time.time() - t_ini) / 60.0, 1)
    # Comparar en validacion contra v9 (puntaje 0.7/0.3 = 0.2987)
    supera_v9 = best['punt'] > 0.2987
    mod.load_state_dict(best['sd']); mod.eval()

    # Evaluar post hoc frente a v9 (con IC 1000 replicas)
    Xt, Yt, Vt, Ft = e9.cargar_locales_9b(receta_6, stats_ir, {'prueba'})
    _, wc_t = pp6.cargar_dem_y_worldcover(Ft)
    pt_t = predecir(mod, Xt, disp); pt_t[np.isin(wc_t, [50, 80])] = 0.0
    mt_lc = (pt_t >= best['u']) & (Vt > 0)

    Xpt6, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', receta_6, {'prueba'})
    Xpt = np.concatenate([Xpt6, np.zeros((len(Xpt6), 3, 128, 128), np.float32)], axis=1)
    mt_pe = (predecir(mod, Xpt, disp) >= best['u']) & (Vpt > 0)

    Xp2, _, Vp2, Fp2 = e9.cargar_locales_9b(receta_6, stats_ir, {'prueba2'})
    _, wc_p2 = pp6.cargar_dem_y_worldcover(Fp2)
    pp2 = predecir(mod, Xp2, disp); pp2[np.isin(wc_p2, [50, 80])] = 0.0
    fa_p2 = e5.eval_prueba2_fa((pp2 >= best['u']).astype(np.float32), Vp2, Fp2, 0.5)

    def boot_m(mk, Y, V, F):
        _, p0, _, _, ob = mr.stats_por_zona(mk.astype(np.float32), Y, V, F, u=0.5, solo_ge50=False)
        return {
            'iou': mr.bootstrap_metricas(p0, is_obj=False, n_boot=1000, seed=7),
            'o50': mr.bootstrap_metricas(ob[50], is_obj=True, n_boot=1000, seed=7),
            'o200': mr.bootstrap_metricas(ob[200], is_obj=True, n_boot=1000, seed=7),
        }

    b_lc = boot_m(mt_lc, Yt, Vt, Ft)
    b_pe = boot_m(mt_pe, Ypt, Vpt, Fpt)
    eval_v9 = json.load(open('informes/evaluacion_v9.json', encoding='utf-8'))
    fmt = mr.fmt_ic

    md = [
        '# Tarea C (Opcional) — Modelo U-Net 9 Bandas v2 (`8 + 8` + Dropout IR `0.5`)',
        '',
        '> [!IMPORTANT]',
        f'> Entrenado desde `U-Net v6` con lotes mixtos `8 + 8` (`La Convención` 9 bandas con dropout de canal IR `p = 0.5` + `L4SPe` con bandas IR en `0`), `lr = 3e-5`, `{a.epocas}` épocas de `{a.pasos}` lotes en CPU (`{dur_min} min`).',
        f'> **Selección en validación:** época **`{best["ep"]}`**, umbral **`{best["u"]:.2f}`**, `L4SPe val IoU = {best["iou_pe"]:.4f}`, `LC val IoU = {best["iou_lc"]:.4f}`, puntaje `0.7/0.3 = {best["punt"]:.4f}` (vs **`0.2987`** de `v9`).',
        f'> **Decisión:** **{"SUPERA a v9" if supera_v9 else "NO SUPERA a v9 en validación — se documenta y no se usa"}**.',
        '',
        '## Comparación «Post Hoc» frente a `v9` (`1 000` réplicas)',
        '',
        '| Conjunto | Modelo | IoU 0 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |',
        '|---|---|---|---|---|',
        f"| `L4SPe test` | **v9** | {fmt(eval_v9['modelos']['v9']['l4spe_test']['todas']['pixel_exacto'],'iou')} | {fmt(eval_v9['modelos']['v9']['l4spe_test']['todas']['obj_ge50'],'f1',True)} | {fmt(eval_v9['modelos']['v9']['l4spe_test']['todas']['obj_ge200'],'rec',True)} |",
        f"| `L4SPe test` | **9 bandas v2** | {fmt(b_pe['iou'],'iou')} | {fmt(b_pe['o50'],'f1',True)} | {fmt(b_pe['o200'],'rec',True)} |",
        f"| `La Convención sellada` | **v9** | {fmt(eval_v9['modelos']['v9']['prueba_360']['todas']['pixel_exacto'],'iou')} | {fmt(eval_v9['modelos']['v9']['prueba_360']['todas']['obj_ge50'],'f1',True)} | {fmt(eval_v9['modelos']['v9']['prueba_360']['todas']['obj_ge200'],'rec',True)} |",
        f"| `La Convención sellada` | **9 bandas v2** | {fmt(b_lc['iou'],'iou')} | {fmt(b_lc['o50'],'f1',True)} | {fmt(b_lc['o200'],'rec',True)} |",
        '',
        f"- **`prueba2` (`349` negativos):** `9 bandas v2` = **`{fa_p2['TOTAL_349']['fa_ha_km2']:.4f} ha/km²` ({fa_p2['TOTAL_349']['pct_con_marca']} %)** frente a `v9` = **`0.0157 ha/km²` (2.3 %)**.",
    ]
    with open('informes/modelo_9_bandas_v2.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    json.dump({'best_val': {k: v for k, v in best.items() if k != 'sd'}, 'supera_v9': supera_v9,
               'posthoc_lc': b_lc, 'posthoc_pe': b_pe, 'prueba2': fa_p2},
              open('informes/modelo_9_bandas_v2.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Tarea C completada -> informes/modelo_9_bandas_v2.md', flush=True)


if __name__ == '__main__':
    main()
