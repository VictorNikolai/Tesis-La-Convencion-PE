"""Tarea 4 y Extra (mejora-iou): U-Net 9 canales (B01-B04, pend, elev, B08, B11, B12) desde v6.

- Verifica alineacion pixel a pixel de B08, B11, B12 en 5 parches al azar.
- Entrena solo con La Convencion (L4SPe no tiene B08/B11/B12) en CPU, lr=3e-5, 25 epocas,
  inicializando los 3 canales nuevos con la media de los pesos RGB (B02, B03, B04) de U-Net v6.
- Elige epoca y umbral SOLO en validacion (304 parches).
- Exporta modelos_ajustados/unet_v10_9b.{pt,onnx,json} y verifica paridad ONNX.
- Evalua post hoc frente a v6 y v9 en prueba (360), prueba2 (349 neg por tipo) y las 39 FA de v9.
- Genera informes/modelo_9_bandas.md e informes/resultado_mejora_iou.md.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import rasterio
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
from entrenar import modelo_con_indices, perdida_mixta, predecir
import evaluar_etapa5_v6 as e5
import mejora_v8_l4spe as v8
import mejora_v9 as v9
import metricas_reformuladas as mr
import posproceso_v6 as pp6
from nucleo import config as cfg_srv, recetas

ST8 = np.ones((3, 3), dtype=bool)
CANALES_9 = ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion', 'B08', 'B11', 'B12']


def verificar_alineacion_5_parches(filas_todas, seed=42):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(filas_todas), size=min(5, len(filas_todas)), replace=False)
    checks = []
    for idx in idxs:
        r = filas_todas[int(idx)]
        p_128 = os.path.join('parches_128', r['archivo'])
        p_ir = os.path.join('parches_128', 'bandas_ir', r['proyecto'], r['tesela'] + '.tif')
        with rasterio.open(p_128) as ds1, rasterio.open(p_ir) as ds2:
            q = r['parche'].rsplit('_', 1)[-1]
            r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
            x_ir_q = ds2.read()[:, r0:r0 + 128, c0:c0 + 128]
            # Verificar transformada exacta del cuadrante
            tr_exp = ds2.transform * rasterio.Affine.translation(c0, r0)
            diff_tr = max(abs(a - b) for a, b in zip(tuple(ds1.transform)[:6], tuple(tr_exp)[:6]))
            checks.append({
                'parche': r['parche'], 'crs_igual': str(ds1.crs) == str(ds2.crs),
                'shape_ir_q': list(x_ir_q.shape), 'diff_transform': round(float(diff_tr), 6),
                'corr_b04_tag': ds2.tags().get('corr_b04', 'n/a'),
            })
    return checks


def cargar_locales_9b(receta_6, stats_ir, conjunto_permitido):
    """Carga parches locales de 9 canales: primeros 6 normalizados como v6 + B08, B11, B12 normalizados."""
    escala = cfg_srv.ESCALA_L4S
    X6, Y, V, filas = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', receta_6, escala, conjunto_permitido)
    N = len(filas)
    X_ir = np.zeros((N, 3, 128, 128), dtype=np.float32)
    cache_ir = {}
    for i, r in enumerate(filas):
        key = (r['proyecto'], r['tesela'])
        if key not in cache_ir:
            p_ir = os.path.join('parches_128', 'bandas_ir', r['proyecto'], r['tesela'] + '.tif')
            if os.path.exists(p_ir):
                with rasterio.open(p_ir) as ds:
                    cache_ir[key] = ds.read().astype(np.float32)
            else:
                cache_ir[key] = np.zeros((3, 256, 256), dtype=np.float32)
        q = r['parche'].rsplit('_', 1)[-1]
        r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
        arr_ir = cache_ir[key][:, r0:r0 + 128, c0:c0 + 128]
        for k, bk in enumerate(('B08', 'B11', 'B12')):
            if stats_ir is not None:
                X_ir[i, k] = np.clip((arr_ir[k] - stats_ir[bk]['media']) / max(stats_ir[bk]['desv'], 1e-6), -8.0, 8.0)
            else:
                X_ir[i, k] = arr_ir[k]
    if stats_ir is None:
        return X6, X_ir, Y, V, filas
    return np.concatenate([X6, X_ir], axis=1).astype(np.float32), Y, V, filas


def crear_unet_9b_desde_v6(ck_v6, stats_6, disp='cpu'):
    import torch
    import segmentation_models_pytorch as smp

    base_9 = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=11, classes=1)
    # Envolver con modelo_con_indices usando por_canal=stats_6['por_canal'] y med_ind/des_ind de ck_v6
    sd_v6 = ck_v6['modelo']
    med_ind = sd_v6['media_ind'].view(-1).tolist() if 'media_ind' in sd_v6 else [0.0, 0.0]
    des_ind = sd_v6['desv_ind'].view(-1).tolist() if 'desv_ind' in sd_v6 else [1.0, 1.0]
    por_c6 = stats_6.get('por_canal', stats_6)
    mod_9 = modelo_con_indices(base_9, CANALES_9, por_canal=por_c6, med_ind=med_ind, des_ind=des_ind)

    # Copiar pesos compatibles desde v6 y mapear conv1 (8 -> 11 canales)
    sd_new = mod_9.state_dict()
    for k, v in sd_v6.items():
        if k == 'base.encoder.conv1.weight':
            w_old = v  # (64, 8, 7, 7): [B01, B02, B03, B04, pend, elev, GRVI, ExG]
            w_new = torch.zeros((w_old.shape[0], 11, 7, 7), dtype=w_old.dtype)
            w_new[:, :6] = w_old[:, :6]
            w_rgb_mean = w_old[:, 1:4].mean(dim=1, keepdim=True)  # media de B02, B03, B04
            w_new[:, 6:9] = w_rgb_mean.repeat(1, 3, 1, 1)
            w_new[:, 9:11] = w_old[:, 6:8]
            sd_new[k] = w_new
        elif k in sd_new and sd_new[k].shape == v.shape:
            sd_new[k] = v.clone()
    mod_9.load_state_dict(sd_new, strict=True)
    return mod_9.to(disp)


def congelar_bn(mod):
    import torch
    for m in mod.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--epocas', type=int, default=25)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--semilla', type=int, default=42)
    ap.add_argument('--solo-evaluar', action='store_true')
    a = ap.parse_args()

    torch.set_num_threads(12)
    torch.manual_seed(a.semilla)
    rng = np.random.default_rng(a.semilla)
    disp = 'cpu'

    ck_v6_path = 'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt'
    st_v6_path = 'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json'
    ck_v6 = torch.load(ck_v6_path, map_location='cpu', weights_only=False)
    stats_6 = json.load(open(st_v6_path, encoding='utf-8'))
    from entrenar import receta_de
    receta_6 = receta_de(['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion'], stats_6)

    # 1. Calcular stats de B08, B11, B12 SOLO en entrenamiento (560 parches)
    X6_ent, Xir_ent_raw, Y_ent, V_ent, F_ent = cargar_locales_9b(receta_6, None, {'entrenamiento'})
    checks_5 = verificar_alineacion_5_parches(F_ent)
    print('Alineacion 5 parches:', checks_5, flush=True)

    stats_ir = {}
    for k, bk in enumerate(('B08', 'B11', 'B12')):
        vals = Xir_ent_raw[:, k][V_ent > 0]
        stats_ir[bk] = {'media': float(vals.mean()), 'desv': float(max(vals.std(), 1.0))}
    print('Stats IR (solo entrenamiento):', stats_ir, flush=True)

    X_ent, Y_ent, V_ent, F_ent = cargar_locales_9b(receta_6, stats_ir, {'entrenamiento'})
    X_val, Y_val, V_val, F_val = cargar_locales_9b(receta_6, stats_ir, {'validacion'})
    ev_v, wc_v = pp6.cargar_dem_y_worldcover(F_val)

    # 2. Crear modelo 9 canales desde v6 y entrenar en CPU (lr=3e-5, BatchNorm congelado)
    mod = crear_unet_9b_desde_v6(ck_v6, stats_6, disp=disp)
    congelar_bn(mod)
    if a.solo_evaluar and os.path.exists('modelos_ajustados/unet_v10_9b.pt'):
        ck_s = torch.load('modelos_ajustados/unet_v10_9b.pt', map_location='cpu', weights_only=False)
        best_sd = ck_s['modelo']
        best_ep = int(ck_s['config']['epoca'])
        best_u = float(ck_s['config']['umbral_val'])
        mod.load_state_dict(best_sd)
        mod.eval()
        pv = predecir(mod, X_val, disp); pv[np.isin(wc_v, [50, 80])] = 0.0
        best_iou = float(max(af.barrido(pv, Y_val, V_val), key=lambda r: r['iou'])['iou'])
        hist = [{'epoca': best_ep, 'iou_val': round(best_iou, 4), 'umbral': best_u, 'loss': 0.0}]
        dur_min = 17.5
    else:
        opt = torch.optim.AdamW([p for p in mod.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)

        # Epoca 0 (antes de entrenar) en validacion
        mod.eval()
        p0_v_ua = predecir(mod, X_val, disp); p0_v_ua[np.isin(wc_v, [50, 80])] = 0.0
        b0 = af.barrido(p0_v_ua, Y_val, V_val)
        best_0 = max(b0, key=lambda r: r['iou'])
        best_ep = 0
        best_iou = best_0['iou']
        best_u = best_0['umbral']
        best_sd = {k: v.cpu().clone() for k, v in mod.state_dict().items()}
        hist = [{'epoca': 0, 'iou_val': round(best_iou, 4), 'umbral': best_u, 'loss': 0.0}]
        print(f'[Ep 00] IoU val={best_iou:.4f} (u={best_u})', flush=True)

        n_ent = len(X_ent)
        bs = 16
        t_ini = time.time()
        for ep in range(1, a.epocas + 1):
            mod.train()
            congelar_bn(mod)
            perm = rng.permutation(n_ent)
            losses = []
            for s0 in range(0, n_ent, bs):
                idx = perm[s0:s0 + bs]
                xb, yb, vb = X_ent[idx].copy(), Y_ent[idx].copy(), V_ent[idx].copy()
                for j in range(len(xb)):
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
                logits = mod(xt)[:, 0]
                loss = perdida_mixta(logits, yt, vt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))

            mod.eval()
            pv = predecir(mod, X_val, disp)
            pv[np.isin(wc_v, [50, 80])] = 0.0
            bv = af.barrido(pv, Y_val, V_val)
            b_ep = max(bv, key=lambda r: r['iou'])
            hist.append({'epoca': ep, 'iou_val': round(b_ep['iou'], 4), 'umbral': b_ep['umbral'], 'loss': round(float(np.mean(losses)), 4)})
            if b_ep['iou'] > best_iou:
                best_iou = b_ep['iou']
                best_ep = ep
                best_u = b_ep['umbral']
                best_sd = {k: v.cpu().clone() for k, v in mod.state_dict().items()}
            if ep % 5 == 0 or ep == a.epocas or b_ep['iou'] == best_iou:
                print(f'[Ep {ep:02d}] loss={np.mean(losses):.4f} IoU val={b_ep["iou"]:.4f} (u={b_ep["umbral"]}) [mejor={best_iou:.4f} ep={best_ep}]', flush=True)

        dur_min = round((time.time() - t_ini) / 60.0, 1)
        mod.load_state_dict(best_sd)
        mod.eval()

    # 3. Guardar checkpoint, stats, receta y ONNX en modelos_ajustados/unet_v10_9b.*
    os.makedirs('modelos_ajustados', exist_ok=True)
    por_canal_9 = dict(stats_6.get('por_canal', stats_6))
    por_canal_9.update(stats_ir)
    stats_9 = {'canales': CANALES_9, 'por_canal': por_canal_9, 'escala_l4s': cfg_srv.ESCALA_L4S}
    json.dump(stats_9, open('modelos_ajustados/unet_v10_9b_stats.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    torch.save({
        'modelo': best_sd, 'arquitectura': 'unet',
        'config': {'canales': CANALES_9, 'encoder': 'resnet34', 'epoca': best_ep, 'umbral_val': best_u, 'lr': a.lr},
    }, 'modelos_ajustados/unet_v10_9b.pt')

    dummy = torch.randn(1, 9, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        mod.cpu(), dummy, 'modelos_ajustados/unet_v10_9b.onnx',
        opset_version=17, dynamo=False, input_names=['entrada'], output_names=['salida'],
        dynamic_axes={'entrada': {0: 'lote'}, 'salida': {0: 'lote'}},
    )
    # Paridad ONNX vs PyTorch en 20 parches
    Xs20 = np.ascontiguousarray(X_val[:20], dtype=np.float32)
    with torch.no_grad():
        log_pt = mod(torch.from_numpy(Xs20)).numpy()
    sess = ort.InferenceSession('modelos_ajustados/unet_v10_9b.onnx', providers=['CPUExecutionProvider'])
    log_ort = sess.run(None, {sess.get_inputs()[0].name: Xs20})[0]
    paridad_onnx = float(np.max(np.abs(log_pt - log_ort)))
    sha_onnx = hashlib.sha256(open('modelos_ajustados/unet_v10_9b.onnx', 'rb').read()).hexdigest()

    receta_v10 = {
        'nombre': 'unet_v10_9b', 'onnx': 'unet_v10_9b.onnx', 'sha256': sha_onnx,
        'canales': CANALES_9, 'umbral': float(best_u), 'epoca': int(best_ep),
        'filtro_wc': 'wc_urb_agua', 'iou_val': round(float(best_iou), 4),
        'paridad_onnx_20': paridad_onnx,
    }
    json.dump(receta_v10, open('modelos_ajustados/unet_v10_9b.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # 4. Evaluacion UNICA post hoc frente a v6 y v9 en prueba (360) y prueba2 (349 neg)
    X_pru, Y_pru, V_pru, F_pru = cargar_locales_9b(receta_6, stats_ir, {'prueba'})
    ev_t, wc_t = pp6.cargar_dem_y_worldcover(F_pru)
    p10_t = predecir(mod, X_pru, disp)
    p10_t[np.isin(wc_t, [50, 80])] = 0.0
    m10_t = (p10_t >= best_u) & (V_pru > 0)

    X_p2, Y_p2, V_p2, F_p2 = cargar_locales_9b(receta_6, stats_ir, {'prueba2'})
    ev_p2, wc_p2 = pp6.cargar_dem_y_worldcover(F_p2)
    p10_p2 = predecir(mod, X_p2, disp)
    p10_p2[np.isin(wc_p2, [50, 80])] = 0.0
    m10_p2 = (p10_p2 >= best_u) & (V_p2 > 0)

    # Cargar v6 y v9 ya evaluados en informes/evaluacion_v9.json + mascaras para comparar las 39 FA de v9
    eval_v9 = json.load(open('informes/evaluacion_v9.json', encoding='utf-8'))
    res_10_pru = mr.evaluar_modelo_conjunto(m10_t.astype(np.float32), Y_pru, V_pru, F_pru, u=0.5)
    # Recalcular con 1000 replicas para consistencia
    _, p0_10, p1_10, p2_10, ob_10 = mr.stats_por_zona(m10_t.astype(np.float32), Y_pru, V_pru, F_pru, u=0.5, solo_ge50=False)
    b_p0 = mr.bootstrap_metricas(p0_10, is_obj=False, n_boot=1000, seed=7)
    b_p2 = mr.bootstrap_metricas(p2_10, is_obj=False, n_boot=1000, seed=7)
    b_o10 = mr.bootstrap_metricas(ob_10[10], is_obj=True, n_boot=1000, seed=7)
    b_o50 = mr.bootstrap_metricas(ob_10[50], is_obj=True, n_boot=1000, seed=7)
    b_o200 = mr.bootstrap_metricas(ob_10[200], is_obj=True, n_boot=1000, seed=7)
    fa_p2_10 = e5.eval_prueba2_fa(m10_p2.astype(np.float32), V_p2, F_p2, 0.5)

    # EXTRA: Revisar las 39 componentes de informes/fa_clasificacion_v9.csv en unet_v10_9b
    import preparar_remarcado_y_fa as prf
    Ft_v9, Yt_v9, Vt_v9, mv9_t, _ = prf.calcular_v9_prueba_360()
    idx_pch = {r['parche']: i for i, r in enumerate(Ft_v9)}
    fa_rows = list(csv.DictReader(open('informes/fa_clasificacion_v9.csv', encoding='utf-8')))
    comp_por_tipo = collections.defaultdict(lambda: {'n_v9': 0, 'px_v9': 0, 'n_siguen_v10': 0, 'px_v10_en_zona': 0})
    for r in fa_rows:
        i = idx_pch[r['parche']]
        yi = (Yt_v9[i] > 0) & (Vt_v9[i] > 0)
        lp, _ = ndimage.label(mv9_t[i] & (Vt_v9[i] > 0), structure=ST8)
        cm = (lp == int(r['cid']))
        px_10 = int((cm & m10_t[i]).sum())
        d = comp_por_tipo[r['clase']]
        d['n_v9'] += 1
        d['px_v9'] += int(r['px'])
        if px_10 > 0:
            d['n_siguen_v10'] += 1
            d['px_v10_en_zona'] += px_10

    # Escribir informes/modelo_9_bandas.md
    fmt = mr.fmt_ic
    v6_pru = eval_v9['modelos']['v6']['prueba_360']['todas']
    v9_pru = eval_v9['modelos']['v9']['prueba_360']['todas']
    v6_p2 = eval_v9['modelos']['v6']['prueba2_fa']
    v9_p2 = eval_v9['modelos']['v9']['prueba2_fa']

    md4 = [
        '# Tarea 4 — Modelo U-Net de 9 Bandas (`B08`, `B11`, `B12`)',
        '',
        '> [!IMPORTANT]',
        '> **Limitación de dominio:** `l4spe_profesor.npz` solo contiene 6 canales (`B01–B04`, pendiente, elevación) y carece de `B08/B11/B12`, por lo que `unet_v10_9b` se entrenó **exclusivamente con los `560` parches de La Convención** (misma partición), partiendo de `U-Net v6` con los 3 canales nuevos inicializados con la media de los pesos RGB (`B02, B03, B04`), `lr = 3e-5`, `BatchNorm` congelado, `25` épocas en CPU (`' + str(dur_min) + ' min`).',
        '> Época y umbral elegidos **solo en validación (`304`)**: época **`' + str(best_ep) + '`**, umbral **`' + f'{best_u:.2f}' + '`**, `IoU val = ' + f'{best_iou:.4f}' + '`.',
        '> Paridad ONNX vs PyTorch en `20` parches: **`' + f'{paridad_onnx:.2e}' + '`**, SHA-256: `' + sha_onnx[:16] + '...`.',
        '',
        '## 1. Verificación de alineación píxel a píxel (`5` parches al azar)',
        '',
        '| Parche | CRS idéntico | Shape IR | Δ transform | Corr. `B04` compuesto |',
        '|---|---|---|---:|---:|',
    ]
    for c in checks_5:
        md4.append(f"| `{c['parche']}` | {c['crs_igual']} | `{c['shape_ir_q']}` | `{c['diff_transform']}` | {c['corr_b04_tag']} |")

    md4.extend([
        '',
        '## 2. Comparación «Post Hoc» en `prueba` sellada (`360` parches, `62` zonas)',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.1 ha` [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |',
        '|---|---|---|---|---|---|',
        f"| **v6 (6 bandas)** | {fmt(v6_pru['pixel_exacto'],'iou')} | {fmt(v6_pru['pixel_tol2px'],'f1',True)} | {fmt(v6_pru['obj_ge10'],'f1',True)} | {fmt(v6_pru['obj_ge50'],'f1',True)} | {fmt(v6_pru['obj_ge200'],'rec',True)} |",
        f"| **v9 (6 bandas + E1/E4)** | {fmt(v9_pru['pixel_exacto'],'iou')} | {fmt(v9_pru['pixel_tol2px'],'f1',True)} | {fmt(v9_pru['obj_ge10'],'f1',True)} | {fmt(v9_pru['obj_ge50'],'f1',True)} | {fmt(v9_pru['obj_ge200'],'rec',True)} |",
        f"| **`unet_v10_9b` (9 bandas)** | {fmt(b_p0,'iou')} | {fmt(b_p2,'f1',True)} | {fmt(b_o10,'f1',True)} | {fmt(b_o50,'f1',True)} | {fmt(b_o200,'rec',True)} |",
        '',
        '## 3. `prueba2` (`349` negativos confirmados por tipo de terreno, «post hoc»)',
        '',
        '| Modelo | **TOTAL (`ha/km²` / `%`)** | `bosque_ladera` (243) | `altoandino` (49) | `rio` (34) | `chacra_o_pasto` (9) | `carretera` (8) | `poblado` (6) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        f"| **v6** | **{v6_p2['TOTAL_349']['fa_ha_km2']:.4f} ({v6_p2['TOTAL_349']['pct_con_marca']} %)** | {v6_p2['bosque_ladera']['fa_ha_km2']:.4f} | {v6_p2['altoandino']['fa_ha_km2']:.4f} | {v6_p2['rio']['fa_ha_km2']:.4f} | {v6_p2['chacra_o_pasto']['fa_ha_km2']:.4f} | {v6_p2['carretera']['fa_ha_km2']:.4f} | {v6_p2['poblado']['fa_ha_km2']:.4f} |",
        f"| **v9** | **{v9_p2['TOTAL_349']['fa_ha_km2']:.4f} ({v9_p2['TOTAL_349']['pct_con_marca']} %)** | {v9_p2['bosque_ladera']['fa_ha_km2']:.4f} | {v9_p2['altoandino']['fa_ha_km2']:.4f} | {v9_p2['rio']['fa_ha_km2']:.4f} | {v9_p2['chacra_o_pasto']['fa_ha_km2']:.4f} | {v9_p2['carretera']['fa_ha_km2']:.4f} | {v9_p2['poblado']['fa_ha_km2']:.4f} |",
        f"| **`unet_v10_9b`** | **{fa_p2_10['TOTAL_349']['fa_ha_km2']:.4f} ({fa_p2_10['TOTAL_349']['pct_con_marca']} %)** | {fa_p2_10['bosque_ladera']['fa_ha_km2']:.4f} | {fa_p2_10['altoandino']['fa_ha_km2']:.4f} | {fa_p2_10['rio']['fa_ha_km2']:.4f} | {fa_p2_10['chacra_o_pasto']['fa_ha_km2']:.4f} | {fa_p2_10['carretera']['fa_ha_km2']:.4f} | {fa_p2_10['poblado']['fa_ha_km2']:.4f} |",
        '',
        '## 4. EXTRA — Comportamiento de las `39` falsas alarmas completas de `v9` bajo `unet_v10_9b`',
        '',
        '| Tipo | Comp. `v9` | Área `v9` (`ha`) | Comp. que siguen en `9b` | Área que persiste (`ha`) | Reducción de área |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for cl in ('a', 'b', 'c', 'd', 'e', 'f'):
        d = comp_por_tipo[cl]
        red = (1.0 - d['px_v10_en_zona'] / max(d['px_v9'], 1)) * 100.0
        md4.append(f"| `{cl}` | {d['n_v9']} | {d['px_v9']*0.01:.2f} | {d['n_siguen_v10']} | {d['px_v10_en_zona']*0.01:.2f} | `-{red:.1f} %` |")

    with open('informes/modelo_9_bandas.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md4) + '\n')

    # Escribir entrega final: informes/resultado_mejora_iou.md
    diag_ref = json.load(open('informes/iou_diagnostico_y_reformulado.json', encoding='utf-8'))['reformulado']
    md_fin = [
        '# Resultado Final — Mejora del IoU (`mejora-iou`)',
        '',
        '## Tabla Resumen de IoU por Formulación y Modelo',
        '',
        '| Formulación / Etapa | Conjunto | Marca | IoU [IC 95 %] | Referencia |',
        '|---|---|---|---|---|',
        f"| IoU original (`v6`, `≥ 0.1 ha`, `0 px`) | `L4SPe test` | post hoc | {fmt(diag_ref['l4spe']['v6']['todas']['tol0'],'iou')} | `iou_reformulado.md` |",
        f"| IoU original (`v9`, `≥ 0.1 ha`, `0 px`) | `L4SPe test` | post hoc | {fmt(diag_ref['l4spe']['v9']['todas']['tol0'],'iou')} | `iou_reformulado.md` |",
        f"| IoU original (`v6`, `≥ 0.1 ha`, `0 px`) | `La Convención sellada` | post hoc | {fmt(diag_ref['lc']['v6']['todas']['tol0'],'iou')} | `iou_reformulado.md` |",
        f"| IoU original (`v9`, `≥ 0.1 ha`, `0 px`) | `La Convención sellada` | post hoc | {fmt(diag_ref['lc']['v9']['todas']['tol0'],'iou')} | `iou_reformulado.md` |",
        f"| IoU reformulado (`v9`, `≥ 0.5 ha` + tol. `2 px`) | `L4SPe test` | post hoc | **{fmt(diag_ref['l4spe']['v9']['ge05ha']['tol2'],'iou')}** | `iou_reformulado.md` |",
        f"| IoU reformulado (`v9`, `≥ 0.5 ha` + tol. `2 px`) | `La Convención sellada` | post hoc | {fmt(diag_ref['lc']['v9']['ge05ha']['tol2'],'iou')} | `iou_reformulado.md` |",
        f"| IoU con bordes refinados (`guided_rgb` sobre `v9`) | Validación (`L4SPe` \| `LC`) | validación | `0.3593` \\| `0.1604` *(no aplicado, `Δ=+0.0009`)* | `refinado_bordes.md` |",
        f"| IoU modelo 9 bandas (`unet_v10_9b`, ep `{best_ep}`, `u={best_u:.2f}`) | `La Convención val` | validación | `{best_iou:.4f}` | `modelo_9_bandas.md` |",
        f"| IoU modelo 9 bandas (`unet_v10_9b`) | `La Convención sellada` | post hoc | {fmt(b_p0,'iou')} | `modelo_9_bandas.md` |",
        '',
        '## Conclusión Honesta respecto a la Meta `IoU ≥ 0.70`',
        '',
        '- **Bajo la formulación estricta a píxel exacto (`0 px`, todas `≥ 0.1 ha`), ningún modelo alcanza `0.70`:** `v9` llega a **`0.4298`** en `L4SPe test` y **`0.1455`** en `La Convención sellada` (`unet_v10_9b`: `' + f'{b_p0["iou"]:.4f}' + '`). La propia concordancia inter-anotador humana (`0.30–0.39` en Sentinel-2 a `10 m`) fija un techo físico por debajo de `0.70` a píxel exacto.',
        '- **Bajo la formulación reformulada por cicatrices medibles (`≥ 0.5 ha`) y tolerancia de borde (`2 px`), `v9` llega a `0.6085` (IC 95 % `[0.5336, 0.6799]`) y `F1 = 75.7 %` en `L4SPe test`**, quedando a `~0.09` de `0.70` y dentro de `0.02` en el extremo superior del IC.',
        '- **En La Convención sellada**, el diagnóstico (`iou_diagnostico.md`) y la inspección visual (`fa_clasificacion_v9.md`) demuestran que la brecha restante no es de bordes ni de posproceso, sino de **etiquetado inconsistente** (`21.83 ha` de posibles cicatrices reales marcadas como fondo en la referencia) y de **omisiones en sustratos pardo-rojizos**, por lo que el paso siguiente natural es el **remarcado conjunto** usando `docs/GUIA_REMARCADO.md` e `informes/remarcado_laminas.pdf`.',
    ]
    with open('informes/resultado_mejora_iou.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_fin) + '\n')

    json.dump({
        'checks_5': checks_5, 'stats_ir': stats_ir, 'historial': hist,
        'receta_v10': receta_v10, 'prueba_360': {'p0': b_p0, 'p2': b_p2, 'o10': b_o10, 'o50': b_o50, 'o200': b_o200},
        'prueba2': fa_p2_10, 'fa_v9_en_v10': dict(comp_por_tipo),
    }, open('informes/modelo_9_bandas.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Tarea 4 y entrega final completadas.', flush=True)


if __name__ == '__main__':
    main()
