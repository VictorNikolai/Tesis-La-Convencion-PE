"""Pipeline de Mejora v9 (herramientas/mejora_v9.py):
Corregir errores y mejorar aciertos combinando lo mejor de v6 (La Convencion) y v8 (L4SPe).

Regla metodologica:
  - Todos los hiperparametros de E1, E2, E3, E4 y E7 se eligen EXCLUSIVAMENTE en validacion
    (L4SPe val 127 + La Convencion val 304).
  - La evaluacion sobre prueba (360), prueba2 (349 negativos) y L4SPe test (124) se ejecuta
    UNA SOLA VEZ al final (--fase posthoc) y se rotula «post hoc».
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from pyproj import Transformer

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

import ajuste_fino as af
import evaluar_etapa5_v6 as e5
import filtro_dgrvi_v7 as fd7
import mejora_v8_l4spe as v8
import metricas_reformuladas as mr
import posproceso_v6 as pp6
import revisar_mascaras as rm
from nucleo import config as cfg_srv

ST8 = np.ones((3, 3), dtype=bool)


def cargar_contexto_espacial_locales(filas, carpeta_parches='parches_128',
                                     ruta_vias='herramientas/carreteras_laconvencion.geojson',
                                     ruta_edif='herramientas/edificaciones_laconvencion.geojson'):
    """Precalcula por parche local:
      - pend (N,128,128), elev (N,128,128), wc (N,128,128)
      - dgrvi (N,128,128), ok_ant (N,128,128)
      - tpi_cauce (N,128,128): True donde TPI < -1.5 m (red de drenaje)
      - cerca_via (N,128,128): True a < 2 px (20 m) de vias OSM
      - cerca_nube_sombra (N,128,128): True a < 5 px de nube o sombra espectral (sin SCL en TIF 6ch)
      - edif_mask (N,128,128): True sobre edificaciones/poligonos residenciales OSM
    """
    N = len(filas)
    pend = np.zeros((N, 128, 128), dtype=np.float32)
    elev = np.zeros((N, 128, 128), dtype=np.float32)
    wc = np.zeros((N, 128, 128), dtype=np.uint8)
    tpi_cauce = np.zeros((N, 128, 128), dtype=bool)
    cerca_via = np.zeros((N, 128, 128), dtype=bool)
    cerca_ns = np.zeros((N, 128, 128), dtype=bool)
    edif_mask = np.zeros((N, 128, 128), dtype=bool)

    dg, ok_ant, _ = fd7.cargar_dgrvi_conjunto(filas, carpeta_parches=carpeta_parches)

    # Cargar arboles espaciales una sola vez
    tree_v, vias_ll = rm._cargar_carreteras('.', ruta_vias)
    edifs_ll = []
    if os.path.exists(ruta_edif):
        gj_e = json.load(open(ruta_edif, encoding='utf-8'))
        for f in gj_e.get('features', []):
            g = f.get('geometry')
            if g:
                s = shape(g)
                if not s.is_empty:
                    edifs_ll.append(s)
    tree_e = STRtree(edifs_ll) if edifs_ll else None
    tr_cache = {}

    for i, r in enumerate(filas):
        p_img = os.path.join(carpeta_parches, r['archivo'])
        with rasterio.open(p_img) as ds:
            b1 = ds.read(1).astype(np.float32)
            b2 = ds.read(2).astype(np.float32)
            b3 = ds.read(3).astype(np.float32)
            b4 = ds.read(4).astype(np.float32)
            pend[i] = ds.read(5).astype(np.float32)
            elev[i] = ds.read(6).astype(np.float32)
            m_crs, m_tr, m_bounds = ds.crs, ds.transform, ds.bounds

        p_wc = os.path.join(carpeta_parches, 'worldcover', r['proyecto'], r['parche'] + '.tif')
        if os.path.exists(p_wc):
            with rasterio.open(p_wc) as dsw:
                wc[i] = dsw.read(1)

        # Red de drenaje: TPI < -1.5 m
        if float(elev[i].max() - elev[i].min()) > 1.0:
            tpi21 = elev[i] - ndimage.uniform_filter(elev[i], size=21)
            tpi_cauce[i] = tpi21 < -1.5

        # Mascara espectral de nube y sombra (6 canales sin banda SCL)
        nube = (b2 > 600.0) & (b4 < 1.25 * b2) & (b3 < 1.30 * b2)
        lab_n, nn = ndimage.label(nube, structure=ST8)
        nube_v = np.zeros_like(nube)
        for nid in range(1, nn + 1):
            nc = lab_n == nid
            if int(nc.sum()) >= 4 and float(b2[nc].max()) > 900.0:
                nube_v |= nc
        # Sombra adyacente a nube (oscura en B02 y B04)
        sombra = (b1 > 0) & (b2 < 220.0) & (b4 < 200.0)
        if nube_v.any():
            d_n = ndimage.distance_transform_edt(~nube_v)
            ns_mask = nube_v | (sombra & (d_n <= 15.0))
            cerca_ns[i] = ndimage.distance_transform_edt(~ns_mask) < 5.0

        # Vias OSM (< 2 px = 20 m) y edificaciones/residencial
        if m_crs is not None:
            kc = str(m_crs)
            if kc not in tr_cache:
                tr_cache[kc] = Transformer.from_crs('EPSG:4326', kc, always_xy=True).transform
            tr_utm = tr_cache[kc]
            b_ll = transform_bounds(m_crs, 'EPSG:4326', m_bounds.left, m_bounds.bottom, m_bounds.right, m_bounds.top)
            caja = box(b_ll[0] - 0.002, b_ll[1] - 0.002, b_ll[2] + 0.002, b_ll[3] + 0.002)

            if tree_v is not None:
                idxs = tree_v.query(caja)
                if len(idxs):
                    geoms_v = [shp_transform(tr_utm, vias_ll[idx]).buffer(20.0) for idx in idxs]
                    cerca_via[i] = rasterize([(g, 1) for g in geoms_v], out_shape=(128, 128), transform=m_tr, all_touched=True) > 0

            if tree_e is not None:
                idxs_e = tree_e.query(caja)
                if len(idxs_e):
                    geoms_e = []
                    for idx in idxs_e:
                        gu = shp_transform(tr_utm, edifs_ll[idx])
                        if gu.geom_type in ('Point', 'MultiPoint'):
                            gu = gu.buffer(15.0)
                        geoms_e.append(gu)
                    if geoms_e:
                        edif_mask[i] = rasterize([(g, 1) for g in geoms_e], out_shape=(128, 128), transform=m_tr, all_touched=True) > 0

    return {
        'pend': pend, 'elev': elev, 'wc': wc,
        'dgrvi': dg, 'ok_ant': ok_ant,
        'tpi_cauce': tpi_cauce, 'cerca_via': cerca_via,
        'cerca_ns': cerca_ns, 'edif_mask': edif_mask,
    }


def metrica_completa_val(mask_bool, Y, V, filas):
    """Metricas completas en validacion (objeto, pixel, tamano y FA)."""
    r = v8.metricas_rapidas_mascara(mask_bool, Y, V, filas)
    prob_bin = mask_bool.astype(np.float32)
    res_loc = af.resumen_local(prob_bin, Y, V, filas, 0.5)
    d30 = res_loc['deteccion_cicatriz_30pct']
    # Tamano 10-49, 50-199, >=200 px por componentes GT (>=30% solape)
    tp_10_49 = tot_10_49 = tp_50_199 = tot_50_199 = tp_200 = tot_200 = 0
    for i in range(len(Y)):
        yi = (Y[i] > 0) & (V[i] > 0)
        if not yi.any():
            continue
        pi = mask_bool[i] & (V[i] > 0)
        lab, nc = ndimage.label(yi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            if sz < 10:
                continue
            hit = int(float((cm & pi).sum()) / sz >= 0.30)
            if sz < 50:
                tot_10_49 += 1; tp_10_49 += hit
            elif sz < 200:
                tot_50_199 += 1; tp_50_199 += hit
            else:
                tot_200 += 1; tp_200 += hit
    return {
        'iou_0px': r['iou_0px'],
        'f1_2px': r['f1_2px'],
        'obj10_f1': r['obj10_f1'],
        'obj10_tp': int(round(d30['total_ge10']['detectadas'])),
        'obj10_tot': int(round(d30['total_ge10']['total'])),
        'det_10_49': f'{tp_10_49}/{tot_10_49}',
        'det_10_49_tp': tp_10_49,
        'det_50_199': f'{tp_50_199}/{tot_50_199}',
        'det_50_199_tp': tp_50_199,
        'det_ge200': f'{tp_200}/{tot_200}',
        'det_ge200_tp': tp_200,
        'det_ge200_rec': round(tp_200 / max(tot_200, 1), 4),
        'fa_ha_km2': r['fa_ha_km2'],
    }


def contar_ganadas_perdidas(mask_base, mask_new, Y, V):
    """Cuenta cicatrices (>=10 px) ganadas y perdidas frente a mask_base."""
    ganadas = perdidas = 0
    for i in range(len(Y)):
        yi = (Y[i] > 0) & (V[i] > 0)
        if not yi.any():
            continue
        pb = mask_base[i] & (V[i] > 0)
        pn = mask_new[i] & (V[i] > 0)
        lab, nc = ndimage.label(yi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            if sz < 10:
                continue
            hb = float((cm & pb).sum()) / sz >= 0.30
            hn = float((cm & pn).sum()) / sz >= 0.30
            if (not hb) and hn:
                ganadas += 1
            elif hb and (not hn):
                perdidas += 1
    return ganadas, perdidas


def cumple_criterio_v9(m_base_lc, m_new_lc, m_base_pe, m_new_pe, mask_base_lc, mask_new_lc, mask_base_pe, mask_new_pe):
    """Criterio de aceptacion v9:
    (a) sube det por objeto o F1 por objeto sin subir FA LC > +10% y sin perder >2% de cicatrices globales, o
    (b) baja FA LC sin perder > 2% de cicatrices detectadas."""
    fa_lim_10 = m_base_lc['fa_ha_km2'] * 1.10 + 1e-6
    det_b_tot = m_base_lc['obj10_tp'] + m_base_pe['obj10_tp']
    det_n_tot = m_new_lc['obj10_tp'] + m_new_pe['obj10_tp']
    no_pierde_2pct = (det_n_tot >= 0.98 * det_b_tot)

    sube_det_o_f1 = (
        (m_new_lc['obj10_tp'] > m_base_lc['obj10_tp'])
        or (m_new_lc['obj10_f1'] > m_base_lc['obj10_f1'])
        or (m_new_pe['obj10_tp'] > m_base_pe['obj10_tp'])
        or (m_new_pe['obj10_f1'] > m_base_pe['obj10_f1'])
    )
    crit_a = sube_det_o_f1 and (m_new_lc['fa_ha_km2'] <= fa_lim_10) and no_pierde_2pct
    crit_b = (m_new_lc['fa_ha_km2'] < m_base_lc['fa_ha_km2']) and no_pierde_2pct
    return bool(crit_a or crit_b)


# ==================== CORRECCIONES E1 - E4 ====================

def aplicar_e1_ensamble(p_v6, p_p3, elev, wc=None, corte_m=3200.0, cond_wc=False, w_v6=0.6, w_p3=0.4):
    """E1: ensamble v6 + P3, poniendo peso P3 = 0 donde elev > corte_m (o wc in {30,60} si cond_wc)."""
    supr = elev > float(corte_m)
    if cond_wc and wc is not None:
        supr = supr | np.isin(wc, [30, 60])
    p_ens = w_v6 * p_v6 + w_p3 * p_p3
    return np.where(supr, p_v6, p_ens).astype(np.float32)


def aplicar_e2_rescate_grande(mask_base, prob, ctx, u_rescate=0.20, t_rescate=-0.08):
    """E2: agrega componentes >= 50 px con prob >= u_rescate si dGRVI <= t_rescate, pend >= 20 deg,
    y no esta en WorldCover urbano/agua (50, 80)."""
    out = mask_base.copy()
    dg, ok_ant, pend, wc = ctx['dgrvi'], ctx['ok_ant'], ctx['pend'], ctx['wc']
    for i in range(len(prob)):
        cand = (prob[i] >= float(u_rescate)) & (~np.isin(wc[i], [50, 80]))
        if not cand.any():
            continue
        lab, nc = ndimage.label(cand, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            if sz < 50:
                continue
            # Si ya esta mayormente cubierta por mask_base, no cambia nada
            if float((cm & out[i]).sum()) / sz >= 0.80:
                continue
            ok_px = cm & ok_ant[i]
            if float(ok_px.sum()) / sz < 0.30:
                continue
            if float(dg[i][ok_px].mean()) > float(t_rescate):
                continue
            if float(pend[i][cm].mean()) < 20.0:
                continue
            out[i] |= cm
    return out


def aplicar_e3_rescate_pequeno(mask_base, prob, ctx, u_peq=0.30):
    """E3: agrega componentes de 10-49 px con prob >= u_peq si cumplen:
    dGRVI <= -0.10, pend >= 25 deg, eje_mayor/eje_menor >= 1.8, >=2 px de via y cauce (TPI<-1.5), >=5 px de nube/sombra."""
    out = mask_base.copy()
    dg, ok_ant, pend, wc = ctx['dgrvi'], ctx['ok_ant'], ctx['pend'], ctx['wc']
    cauce, via, ns = ctx['tpi_cauce'], ctx['cerca_via'], ctx['cerca_ns']
    for i in range(len(prob)):
        cand = (prob[i] >= float(u_peq)) & (~np.isin(wc[i], [50, 80]))
        if not cand.any():
            continue
        lab, nc = ndimage.label(cand, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            if sz < 10 or sz > 49:
                continue
            if (cm & out[i]).all():
                continue
            if (cm & via[i]).any() or (cm & cauce[i]).any() or (cm & ns[i]).any():
                continue
            ok_px = cm & ok_ant[i]
            if float(ok_px.sum()) / sz < 0.50:
                continue
            if float(dg[i][ok_px].mean()) > -0.10:
                continue
            if float(pend[i][cm].mean()) < 25.0:
                continue
            rr, cc = np.where(cm)
            elong, _ = rm._eje_principal(rr, cc)
            if elong < 1.8:
                continue
            out[i] |= cm
    return out


def _rectangularidad_minima(comp_mask):
    """Calcula area / area_rectangulo_minimo_rotado para una componente binaria."""
    rr, cc = np.where(comp_mask)
    if len(rr) < 4:
        return 0.0
    pts = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    # Caja minima rotada usando geometria de shapely sobre convex hull de centros + margen 0.5 px
    from shapely.geometry import MultiPoint
    hull = MultiPoint(pts).convex_hull.buffer(0.5)
    mrr = hull.minimum_rotated_rectangle
    if mrr.area <= 0:
        return 0.0
    return float(len(rr)) / float(mrr.area)


def aplicar_e4_filtros(mask_in, ctx, usar_forma=False, usar_edif=False, usar_rebrote=False):
    """E4: elimina componentes falsos por (a) forma rectangular/bordes rectos, (b) edificaciones >20%, (c) rebrote dGRVI > +0.05."""
    out = mask_in.copy()
    edif, dg, ok_ant = ctx['edif_mask'], ctx['dgrvi'], ctx['ok_ant']
    for i in range(len(out)):
        if not out[i].any():
            continue
        lab, nc = ndimage.label(out[i], structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            quitar = False
            if usar_forma and sz >= 10:
                rect = _rectangularidad_minima(cm)
                frac_r, perim = rm._fraccion_bordes_rectos(cm)
                if rect > 0.75 or (perim >= 25.0 and frac_r >= 0.55):
                    quitar = True
            if (not quitar) and usar_edif:
                if float((cm & edif[i]).sum()) / sz > 0.20:
                    quitar = True
            if (not quitar) and usar_rebrote:
                ok_px = cm & ok_ant[i]
                if float(ok_px.sum()) / sz >= 0.30 and float(dg[i][ok_px].mean()) > 0.05:
                    quitar = True
            if quitar:
                out[i][cm] = False
    return out


def fa_altoandina(mask_bool, V, filas, elev, wc):
    """Calcula FA (ha/km2) en teselas negativas con elev >= 3200 m o cobertura pajonal/desnudo (30,60)."""
    neg_tes = set()
    # Determinar teselas negativas en GT
    tes_pos = {}
    for i, r in enumerate(filas):
        k = (r['proyecto'], r['tesela'])
        tes_pos[k] = tes_pos.get(k, 0) + 0
    # Usar V y Y en el caller o evaluar pixeles altoandinos sobre toda tesela negativa
    return None


def _guardar_ejemplos_rescate(carpeta, nombre_prefix, X_raw, Y, V, filas, mask_base, mask_new, ctx, dg_thr):
    """Guarda hasta 3 ejemplos correctos y 3 incorrectos de rescates (E2/E3) en informes/v9_rescates/."""
    os.makedirs(carpeta, exist_ok=True)
    correctos, incorrectos = [], []
    for i in range(len(Y)):
        gan = (mask_new[i] & (~mask_base[i])) & (V[i] > 0)
        if not gan.any():
            continue
        yi = (Y[i] > 0) & (V[i] > 0)
        lab, nc = ndimage.label(gan, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            ov = float((cm & yi).sum()) / max(sz, 1)
            if ov >= 0.30:
                correctos.append((sz, ov, i, cm))
            elif ov == 0.0 and not yi.any():
                incorrectos.append((sz, ov, i, cm))
    correctos.sort(reverse=True)
    incorrectos.sort(reverse=True)

    for tag, lst in [('correcto', correctos[:3]), ('incorrecto', incorrectos[:3])]:
        for k, (sz, ov, i, cm) in enumerate(lst, 1):
            b = X_raw[i]
            rgb = np.stack([b[3], b[2], b[1]], axis=-1).astype(np.float32)
            v = rgb > 0
            if v.any():
                p2, p98 = np.percentile(rgb[v], [2, 98])
                rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=160, facecolor='white')
            axes[0].imshow(rgb); axes[0].set_title(f"{filas[i]['parche']}\nRGB (S2)", fontsize=9)
            axes[1].imshow(v8.overlay(rgb, (Y[i] > 0) & (V[i] > 0), (0.1, 0.85, 0.2), 0.55))
            axes[1].set_title(f"Referencia ({int(((Y[i]>0)&(V[i]>0)).sum())} px)", fontsize=9)
            im_r = v8.overlay(v8.overlay(rgb, mask_base[i], (0.2, 0.4, 1.0), 0.45), cm, (1.0, 0.15, 0.15), 0.65)
            axes[2].imshow(im_r); axes[2].set_title(f"{nombre_prefix} (+{sz} px, solape {ov*100:.0f}%)", fontsize=9)
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(carpeta, f'{nombre_prefix}_{tag}_{k}_{filas[i]["parche"]}.png'), bbox_inches='tight', facecolor='white')
            plt.close(fig)
    return len(correctos), len(incorrectos)


def _fa_altoandina_lc(mask_bool, Y, V, elev, wc, filas):
    tes_pos = {}
    for i, r in enumerate(filas):
        k = (r['proyecto'], r['tesela'])
        tes_pos[k] = tes_pos.get(k, 0) + int(((Y[i] > 0) & (V[i] > 0)).sum())
    px_tot = fa_tot = 0.0
    for i, r in enumerate(filas):
        if tes_pos[(r['proyecto'], r['tesela'])] > 0:
            continue
        alto = (V[i] > 0) & ((elev[i] >= 3200.0) | np.isin(wc[i], [30, 60]))
        px_tot += float(alto.sum())
        fa_tot += float((mask_bool[i] & alto).sum())
    return round(fa_tot / max(px_tot, 1.0) * 100.0, 4)


def _fa_por_cobertura_lc(mask_bool, Y, V, elev, wc, filas):
    tes_pos = {}
    for i, r in enumerate(filas):
        k = (r['proyecto'], r['tesela'])
        tes_pos[k] = tes_pos.get(k, 0) + int(((Y[i] > 0) & (V[i] > 0)).sum())
    grupos = {'altoandino': [0.0, 0.0], 'chacra_pasto_arbusto': [0.0, 0.0], 'urbano': [0.0, 0.0], 'bosque': [0.0, 0.0], 'otros': [0.0, 0.0]}
    for i, r in enumerate(filas):
        if tes_pos[(r['proyecto'], r['tesela'])] > 0:
            continue
        vi = V[i] > 0
        m_alto = vi & ((elev[i] >= 3200.0) | np.isin(wc[i], [30, 60]))
        m_urb = vi & (~m_alto) & (wc[i] == 50)
        m_chac = vi & (~m_alto) & (~m_urb) & np.isin(wc[i], [20, 40])
        m_bosq = vi & (~m_alto) & (~m_urb) & (~m_chac) & (wc[i] == 10)
        m_otr = vi & (~m_alto) & (~m_urb) & (~m_chac) & (~m_bosq)
        for g, mg in [('altoandino', m_alto), ('chacra_pasto_arbusto', m_chac), ('urbano', m_urb), ('bosque', m_bosq), ('otros', m_otr)]:
            grupos[g][0] += float(mg.sum())
            grupos[g][1] += float((mask_bool[i] & mg).sum())
    return {k: round(v[1] / max(v[0], 1.0) * 100.0, 4) for k, v in grupos.items()}


def _cargar_raw_locales(filas, carpeta_parches='parches_128'):
    arr = []
    for r in filas:
        with rasterio.open(os.path.join(carpeta_parches, r['archivo'])) as ds:
            arr.append(ds.read().astype(np.float32))
    return np.stack(arr)


def ejecutar_validacion(ckpt_p3, stats_p3):
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S

    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', disp)
    m_p3, _, _ = pp6.cargar_modelo_ckpt(ckpt_p3, stats_p3, disp)

    Xv_lc, Yv_lc, Vv_lc, Fv_lc = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'validacion'})
    Xv_pe, Yv_pe, Vv_pe, Fv_pe = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'validacion'})
    ctx_lc = cargar_contexto_espacial_locales(Fv_lc)
    Xraw_lc = _cargar_raw_locales(Fv_lc)
    elev_pe = Xv_pe[:, 5, :, :] * 1000.0  # canal 5 normalizado; carguemos elev cruda de l4spe_profesor.npz
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_pe = [i for i, c in enumerate(z['conjunto']) if str(c) == 'validacion']
        elev_pe = z['x'][sel_pe, 5, :, :].astype(np.float32)

    # Probabilidades base en validacion
    pv6_lc_no_tta = pp6.predecir_tta(m_v6, Xv_lc, disp, usar_tta=False)
    pv6_pe_no_tta = pp6.predecir_tta(m_v6, Xv_pe, disp, usar_tta=False)
    pv6_lc_tta = pp6.predecir_tta(m_v6, Xv_lc, disp, usar_tta=True)
    pv6_pe_tta = pp6.predecir_tta(m_v6, Xv_pe, disp, usar_tta=True)
    pp3_lc_tta = pp6.predecir_tta(m_p3, Xv_lc, disp, usar_tta=True)
    pp3_pe_tta = pp6.predecir_tta(m_p3, Xv_pe, disp, usar_tta=True)

    # Filtro wc_urb_agua (sin corte de altitud)
    wc_urb_agua = np.isin(ctx_lc['wc'], [50, 80])
    pv6_lc_base = pv6_lc_no_tta.copy()
    pv6_lc_base[wc_urb_agua] = 0.0

    # Punto de partida comun: U-Net v6 sin TTA + wc_urb_agua, u=0.45
    base_lc = (pv6_lc_base >= 0.45) & (Vv_lc > 0)
    base_pe = (pv6_pe_no_tta >= 0.45) & (Vv_pe > 0)
    mb_lc = metrica_completa_val(base_lc, Yv_lc, Vv_lc, Fv_lc)
    mb_pe = metrica_completa_val(base_pe, Yv_pe, Vv_pe, Fv_pe)
    fa_alto_base = _fa_altoandina_lc(base_lc, Yv_lc, Vv_lc, ctx_lc['elev'], ctx_lc['wc'], Fv_lc)
    print(f"[BASE v6] LC: IoU={mb_lc['iou_0px']} F1obj={mb_lc['obj10_f1']} det={mb_lc['obj10_tp']}/{mb_lc['obj10_tot']} FA={mb_lc['fa_ha_km2']} (alto={fa_alto_base}) | "
          f"L4SPe: IoU={mb_pe['iou_0px']} F1obj={mb_pe['obj10_f1']} det>=2ha={mb_pe['det_ge200']}", flush=True)

    # Referencia v8 en validacion
    pv8_lc = 0.6 * pv6_lc_tta + 0.4 * pp3_lc_tta
    pv8_lc = pp6.aplicar_filtro_altoandino(pv8_lc, ctx_lc['elev'], ctx_lc['wc'], 'wc_dem_3200_agua_urb')
    pv8_pe = 0.6 * pv6_pe_tta + 0.4 * pp3_pe_tta
    mv8_lc = v8.postproceso_objeto(v8.umbral_histeresis(pv8_lc, 0.70, 0.25) & (Vv_lc > 0), 20, False, True) & (Vv_lc > 0)
    mv8_pe = v8.postproceso_objeto(v8.umbral_histeresis(pv8_pe, 0.70, 0.25) & (Vv_pe > 0), 20, False, True) & (Vv_pe > 0)
    mv8_m_lc = metrica_completa_val(mv8_lc, Yv_lc, Vv_lc, Fv_lc)
    mv8_m_pe = metrica_completa_val(mv8_pe, Yv_pe, Vv_pe, Fv_pe)
    fa_alto_v8 = _fa_altoandina_lc(mv8_lc, Yv_lc, Vv_lc, ctx_lc['elev'], ctx_lc['wc'], Fv_lc)
    print(f"[REF  v8] LC: IoU={mv8_m_lc['iou_0px']} F1obj={mv8_m_lc['obj10_f1']} FA={mv8_m_lc['fa_ha_km2']} (alto={fa_alto_v8}) | "
          f"L4SPe: IoU={mv8_m_pe['iou_0px']} F1obj={mv8_m_pe['obj10_f1']} det>=2ha={mv8_m_pe['det_ge200']}", flush=True)

    resumen_corr = []

    def reg_corr(nombre, params_txt, param_sel, m_lc, m_pe, mk_lc, mk_pe, aceptada):
        g_lc, p_lc = contar_ganadas_perdidas(base_lc, mk_lc, Yv_lc, Vv_lc)
        g_pe, p_pe = contar_ganadas_perdidas(base_pe, mk_pe, Yv_pe, Vv_pe)
        resumen_corr.append({
            'nombre': nombre, 'parametros_probados': params_txt, 'parametro_elegido': param_sel,
            'lc_iou': m_lc['iou_0px'], 'pe_iou': m_pe['iou_0px'],
            'lc_f1_objeto': m_lc['obj10_f1'], 'pe_f1_objeto': m_pe['obj10_f1'],
            'lc_det_10_49': m_lc['det_10_49'], 'lc_det_50_199': m_lc['det_50_199'], 'lc_det_ge200': m_lc['det_ge200'],
            'pe_det_10_49': m_pe['det_10_49'], 'pe_det_50_199': m_pe['det_50_199'], 'pe_det_ge200': m_pe['det_ge200'],
            'fa_lc_val_ha_km2': m_lc['fa_ha_km2'],
            'cicatrices_ganadas_lc': g_lc, 'cicatrices_perdidas_lc': p_lc,
            'cicatrices_ganadas_pe': g_pe, 'cicatrices_perdidas_pe': p_pe,
            'decision': 'aceptada' if aceptada else 'descartada',
        })

    # ==================== E1: Quitar FA altoandinas de P3 en v8 ====================
    e1_rows = []
    best_e1 = None
    for corte in (2800, 3000, 3200, 3400, 3600):
        for cond_wc in (False, True):
            p_lc_e1 = aplicar_e1_ensamble(pv6_lc_tta, pp3_lc_tta, ctx_lc['elev'], ctx_lc['wc'], corte, cond_wc)
            p_lc_e1[wc_urb_agua] = 0.0
            p_pe_e1 = aplicar_e1_ensamble(pv6_pe_tta, pp3_pe_tta, elev_pe, None, corte, cond_wc)
            for ua, ub in ((0.70, 0.25), (0.60, 0.30)):
                mk_lc = v8.umbral_histeresis(p_lc_e1, ua, ub) & (Vv_lc > 0)
                mk_pe = v8.umbral_histeresis(p_pe_e1, ua, ub) & (Vv_pe > 0)
                m_lc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
                m_pe = metrica_completa_val(mk_pe, Yv_pe, Vv_pe, Fv_pe)
                fa_alto = _fa_altoandina_lc(mk_lc, Yv_lc, Vv_lc, ctx_lc['elev'], ctx_lc['wc'], Fv_lc)
                ok_c = cumple_criterio_v9(mb_lc, m_lc, mb_pe, m_pe, base_lc, mk_lc, base_pe, mk_pe)
                row = {
                    'corte_m': corte, 'cond_wc_30_60': int(cond_wc), 'hist': f'{ua:.2f}/{ub:.2f}',
                    'fa_altoandina_lc': fa_alto, 'fa_total_lc': m_lc['fa_ha_km2'], 'lc_iou': m_lc['iou_0px'],
                    'pe_f1_obj': m_pe['obj10_f1'], 'pe_det_ge200': m_pe['det_ge200'], 'pe_det_ge200_rec': m_pe['det_ge200_rec'],
                    'pe_iou': m_pe['iou_0px'], 'cumple_criterio': int(ok_c),
                }
                e1_rows.append(row)
                # Priorizar cumplimiento de criterio, det >=2ha L4SPe >= v8, menor FA altoandina y mayor F1 obj + puntaje
                key_e1 = (int(ok_c), m_pe['det_ge200_rec'] >= 0.85, -fa_alto, m_pe['obj10_f1'], 0.7 * m_pe['iou_0px'] + 0.3 * m_lc['iou_0px'])
                if best_e1 is None or key_e1 > best_e1[0]:
                    best_e1 = (key_e1, row, m_lc, m_pe, mk_lc, mk_pe, corte, cond_wc, ua, ub)
    with open('informes/v9_e1_detalle.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(e1_rows[0].keys())); w.writeheader(); w.writerows(e1_rows)
    e1_ok = bool(best_e1[1]['cumple_criterio'])
    reg_corr('E1_quitar_p3_altoandino', 'corte={2800..3600}, cond_wc={0,1}, hist={0.70/0.25,0.60/0.30}',
             f"corte={best_e1[6]}m, wc30_60={best_e1[7]}, hist={best_e1[8]}/{best_e1[9]}",
             best_e1[2], best_e1[3], best_e1[4], best_e1[5], e1_ok)
    print(f"[E1] {'ACEPTADA' if e1_ok else 'DESCARTADA'}: {best_e1[1]}", flush=True)

    # ==================== E2: Rescate por cambio (>= 50 px) ====================
    e2_rows = []
    best_e2 = None
    for u_r in (0.10, 0.15, 0.20, 0.25, 0.30):
        for t_r in (-0.05, -0.08, -0.10, -0.15):
            mk_lc = aplicar_e2_rescate_grande(base_lc, pv6_lc_base, ctx_lc, u_rescate=u_r, t_rescate=t_r) & (Vv_lc > 0)
            m_lc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
            g_lc, p_lc = contar_ganadas_perdidas(base_lc, mk_lc, Yv_lc, Vv_lc)
            ok_c = cumple_criterio_v9(mb_lc, m_lc, mb_pe, mb_pe, base_lc, mk_lc, base_pe, base_pe)
            row = {
                'u_rescate': u_r, 't_rescate': t_r,
                'lc_det_50_199': m_lc['det_50_199'], 'lc_det_ge200': m_lc['det_ge200'],
                'lc_f1_obj': m_lc['obj10_f1'], 'lc_iou': m_lc['iou_0px'],
                'fa_lc': m_lc['fa_ha_km2'], 'fa_agregada': round(m_lc['fa_ha_km2'] - mb_lc['fa_ha_km2'], 4),
                'ganadas_lc': g_lc, 'perdidas_lc': p_lc, 'cumple_criterio': int(ok_c),
            }
            e2_rows.append(row)
            key_e2 = (int(ok_c), g_lc - p_lc, m_lc['obj10_f1'], -row['fa_agregada'])
            if best_e2 is None or key_e2 > best_e2[0]:
                best_e2 = (key_e2, row, m_lc, mk_lc, u_r, t_r)
    with open('informes/v9_e2_detalle.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(e2_rows[0].keys())); w.writeheader(); w.writerows(e2_rows)
    e2_ok = bool(best_e2[1]['cumple_criterio'])
    _guardar_ejemplos_rescate('informes/v9_rescates', 'E2', Xraw_lc, Yv_lc, Vv_lc, Fv_lc, base_lc, best_e2[3], ctx_lc, best_e2[5])
    reg_corr('E2_rescate_cambio_grandes', 'u_rescate={0.10..0.30}, t_rescate={-0.05..-0.15}',
             f"u={best_e2[4]}, t={best_e2[5]}",
             best_e2[2], mb_pe, best_e2[3], base_pe, e2_ok)
    print(f"[E2] {'ACEPTADA' if e2_ok else 'DESCARTADA'}: {best_e2[1]}", flush=True)

    # ==================== E3: Recuperar cicatrices pequenas (10-49 px) ====================
    e3_rows = []
    best_e3 = None
    for u_p in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        mk_lc = aplicar_e3_rescate_pequeno(base_lc, pv6_lc_base, ctx_lc, u_peq=u_p) & (Vv_lc > 0)
        m_lc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
        g_lc, p_lc = contar_ganadas_perdidas(base_lc, mk_lc, Yv_lc, Vv_lc)
        ok_c = cumple_criterio_v9(mb_lc, m_lc, mb_pe, mb_pe, base_lc, mk_lc, base_pe, base_pe)
        row = {
            'u_peq': u_p, 'lc_det_10_49': m_lc['det_10_49'],
            'lc_f1_obj': m_lc['obj10_f1'], 'lc_iou': m_lc['iou_0px'],
            'fa_lc': m_lc['fa_ha_km2'], 'fa_agregada': round(m_lc['fa_ha_km2'] - mb_lc['fa_ha_km2'], 4),
            'ganadas_lc': g_lc, 'perdidas_lc': p_lc, 'cumple_criterio': int(ok_c),
        }
        e3_rows.append(row)
        key_e3 = (int(ok_c), g_lc - p_lc, m_lc['obj10_f1'], -row['fa_agregada'])
        if best_e3 is None or key_e3 > best_e3[0]:
            best_e3 = (key_e3, row, m_lc, mk_lc, u_p)
    with open('informes/v9_e3_detalle.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(e3_rows[0].keys())); w.writeheader(); w.writerows(e3_rows)
    e3_ok = bool(best_e3[1]['cumple_criterio'])
    _guardar_ejemplos_rescate('informes/v9_rescates', 'E3', Xraw_lc, Yv_lc, Vv_lc, Fv_lc, base_lc, best_e3[3], ctx_lc, -0.10)
    reg_corr('E3_rescate_pequenas', 'u_peq={0.20,0.25,0.30,0.35,0.40,0.45}',
             f"u_peq={best_e3[4]}",
             best_e3[2], mb_pe, best_e3[3], base_pe, e3_ok)
    print(f"[E3] {'ACEPTADA' if e3_ok else 'DESCARTADA'}: {best_e3[1]}", flush=True)

    # ==================== E4: Bajar FA en chacras, pastos y poblados ====================
    e4_specs = [
        ('E4a_forma', True, False, False),
        ('E4b_edificaciones', False, True, False),
        ('E4c_rebrote', False, False, True),
        ('E4_todos', True, True, True),
    ]
    e4_rows = []
    e4_aceptados = {}
    for nm4, uf, ue, ur in e4_specs:
        mk_lc = aplicar_e4_filtros(base_lc, ctx_lc, usar_forma=uf, usar_edif=ue, usar_rebrote=ur) & (Vv_lc > 0)
        mk_pe = aplicar_e4_filtros(base_pe, {'edif_mask': np.zeros_like(base_pe), 'dgrvi': np.zeros_like(base_pe, dtype=np.float32), 'ok_ant': np.zeros_like(base_pe)},
                                   usar_forma=uf, usar_edif=False, usar_rebrote=False) & (Vv_pe > 0)
        m_lc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
        m_pe = metrica_completa_val(mk_pe, Yv_pe, Vv_pe, Fv_pe)
        g_lc, p_lc = contar_ganadas_perdidas(base_lc, mk_lc, Yv_lc, Vv_lc)
        g_pe, p_pe = contar_ganadas_perdidas(base_pe, mk_pe, Yv_pe, Vv_pe)
        fa_cov = _fa_por_cobertura_lc(mk_lc, Yv_lc, Vv_lc, ctx_lc['elev'], ctx_lc['wc'], Fv_lc)
        ok_c = cumple_criterio_v9(mb_lc, m_lc, mb_pe, m_pe, base_lc, mk_lc, base_pe, mk_pe)
        row = {
            'filtro': nm4, 'fa_total_lc': m_lc['fa_ha_km2'], 'fa_chacra_pasto': fa_cov['chacra_pasto_arbusto'],
            'fa_urbano': fa_cov['urbano'], 'fa_altoandino': fa_cov['altoandino'], 'fa_bosque': fa_cov['bosque'],
            'lc_f1_obj': m_lc['obj10_f1'], 'pe_f1_obj': m_pe['obj10_f1'],
            'perdidas_lc': p_lc, 'perdidas_pe': p_pe, 'cumple_criterio': int(ok_c),
        }
        e4_rows.append(row)
        e4_aceptados[nm4] = ok_c
        reg_corr(nm4, nm4, nm4, m_lc, m_pe, mk_lc, mk_pe, ok_c)
        print(f"[{nm4}] {'ACEPTADA' if ok_c else 'DESCARTADA'}: {row}", flush=True)
    with open('informes/v9_e4_detalle.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(e4_rows[0].keys())); w.writeheader(); w.writerows(e4_rows)

    with open('informes/v9_correcciones_val.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(resumen_corr[0].keys())); w.writeheader(); w.writerows(resumen_corr)

    # ==================== E7: Combinacion final v9 y ablacion en validacion ====================
    def correr_combo(usar_e1=True, usar_e2=True, usar_e3=True, usar_e4a=True, usar_e4b=True, usar_e4c=True,
                     corte=best_e1[6], cond_wc=best_e1[7], ua=best_e1[8], ub=best_e1[9],
                     u_r=best_e2[4], t_r=best_e2[5], u_p=best_e3[4], min_px=20, closing=True):
        if usar_e1:
            p_lc = aplicar_e1_ensamble(pv6_lc_tta, pp3_lc_tta, ctx_lc['elev'], ctx_lc['wc'], corte, cond_wc)
            p_pe = aplicar_e1_ensamble(pv6_pe_tta, pp3_pe_tta, elev_pe, None, corte, cond_wc)
        else:
            p_lc = pv6_lc_no_tta.copy()
            p_pe = pv6_pe_no_tta.copy()
        p_lc[wc_urb_agua] = 0.0

        mk_lc = v8.umbral_histeresis(p_lc, ua, ub) & (Vv_lc > 0)
        mk_pe = v8.umbral_histeresis(p_pe, ua, ub) & (Vv_pe > 0)

        if usar_e2:
            mk_lc = aplicar_e2_rescate_grande(mk_lc, p_lc, ctx_lc, u_r, t_r) & (Vv_lc > 0)
        if usar_e3:
            mk_lc = aplicar_e3_rescate_pequeno(mk_lc, p_lc, ctx_lc, u_p) & (Vv_lc > 0)
        if usar_e4a or usar_e4b or usar_e4c:
            mk_lc = aplicar_e4_filtros(mk_lc, ctx_lc, usar_forma=usar_e4a, usar_edif=usar_e4b, usar_rebrote=usar_e4c) & (Vv_lc > 0)
            if usar_e4a:
                mk_pe = aplicar_e4_filtros(mk_pe, {'edif_mask': np.zeros_like(mk_pe), 'dgrvi': np.zeros_like(mk_pe, dtype=np.float32), 'ok_ant': np.zeros_like(mk_pe)},
                                           usar_forma=True, usar_edif=False, usar_rebrote=False) & (Vv_pe > 0)
        if min_px > 1 or closing:
            # Conservar rescates de E3 (10-49 px) si min_px=20? La instruccion pide:
            # orden: E1 -> histeresis -> rescates (E2, E3) -> filtros (E4) -> min_px y cierre 1 px de v8.
            # Probemos min_px in {10, 20} y veamos en validacion cual cumple <= FA_v6 y >= det_ge2ha_v8!
            mk_lc = v8.postproceso_objeto(mk_lc, min_px=min_px, fill_holes=False, closing_1px=closing) & (Vv_lc > 0)
            mk_pe = v8.postproceso_objeto(mk_pe, min_px=min_px, fill_holes=False, closing_1px=closing) & (Vv_pe > 0)
        return mk_lc, mk_pe

    # Buscar mejor combinacion entre correcciones aceptadas (respetando FA_LC <= FA_v6 y det>=2ha_L4SPe >= v8)
    usa_e1_def = e1_ok
    usa_e2_def = e2_ok
    usa_e3_def = e3_ok
    usa_e4a_def = bool(e4_aceptados['E4a_forma'])
    usa_e4b_def = bool(e4_aceptados['E4b_edificaciones'])
    usa_e4c_def = bool(e4_aceptados['E4c_rebrote'])

    cands_e7 = []
    for usar_e2_c in ([False, True] if usa_e2_def else [False]):
        for mpx in (10, 20):
            for ua_c, ub_c in ((best_e1[8], best_e1[9]), (0.70, 0.25), (0.60, 0.30)):
                for corte_c in (best_e1[6], 2800, 3000, 3200):
                    mk_lc, mk_pe = correr_combo(
                        usar_e1=usa_e1_def, usar_e2=usar_e2_c, usar_e3=usa_e3_def,
                        usar_e4a=usa_e4a_def, usar_e4b=usa_e4b_def, usar_e4c=usa_e4c_def,
                        corte=corte_c, cond_wc=best_e1[7], ua=ua_c, ub=ub_c,
                        u_r=best_e2[4], t_r=best_e2[5], u_p=best_e3[4], min_px=mpx, closing=True
                    )
                    mlc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
                    mpe = metrica_completa_val(mk_pe, Yv_pe, Vv_pe, Fv_pe)
                    punt = round(0.7 * mpe['iou_0px'] + 0.3 * mlc['iou_0px'], 4)
                    cumple_v9 = (mlc['fa_ha_km2'] <= mb_lc['fa_ha_km2'] + 1e-6) and (mpe['det_ge200_tp'] >= mv8_m_pe['det_ge200_tp'])
                    cands_e7.append((int(cumple_v9), punt, mpe['obj10_f1'], -mlc['fa_ha_km2'],
                                     {'usar_e2': usar_e2_c, 'corte': corte_c, 'cond_wc': best_e1[7], 'ua': ua_c, 'ub': ub_c, 'min_px': mpx,
                                      'puntaje_07_03': punt, 'cumple_cond_v9': cumple_v9, 'lc': mlc, 'pe': mpe}))
    cands_e7.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3]))
    gan_e7 = cands_e7[0][4]
    usa_e2_final = gan_e7['usar_e2']
    print(f"[E7 GANADOR]: {gan_e7}", flush=True)

    # Ablacion: quitar de a una cada correccion activa en v9
    ab_rows = []
    ab_specs = [('v9_completo', usa_e1_def, usa_e2_final, usa_e3_def, usa_e4a_def, usa_e4b_def, usa_e4c_def)]
    if usa_e1_def: ab_specs.append(('sin_E1', False, usa_e2_final, usa_e3_def, usa_e4a_def, usa_e4b_def, usa_e4c_def))
    if usa_e2_final: ab_specs.append(('sin_E2', usa_e1_def, False, usa_e3_def, usa_e4a_def, usa_e4b_def, usa_e4c_def))
    if usa_e3_def: ab_specs.append(('sin_E3', usa_e1_def, usa_e2_final, False, usa_e4a_def, usa_e4b_def, usa_e4c_def))
    if usa_e4a_def: ab_specs.append(('sin_E4a_forma', usa_e1_def, usa_e2_final, usa_e3_def, False, usa_e4b_def, usa_e4c_def))
    if usa_e4b_def: ab_specs.append(('sin_E4b_edif', usa_e1_def, usa_e2_final, usa_e3_def, usa_e4a_def, False, usa_e4c_def))
    if usa_e4c_def: ab_specs.append(('sin_E4c_rebrote', usa_e1_def, usa_e2_final, usa_e3_def, usa_e4a_def, usa_e4b_def, False))

    for nm_ab, u1, u2, u3, u4a, u4b, u4c in ab_specs:
        mk_lc, mk_pe = correr_combo(
            u1, u2, u3, u4a, u4b, u4c,
            corte=gan_e7['corte'], cond_wc=gan_e7['cond_wc'], ua=gan_e7['ua'], ub=gan_e7['ub'],
            u_r=best_e2[4], t_r=best_e2[5], u_p=best_e3[4], min_px=gan_e7['min_px'], closing=True
        )
        mlc = metrica_completa_val(mk_lc, Yv_lc, Vv_lc, Fv_lc)
        mpe = metrica_completa_val(mk_pe, Yv_pe, Vv_pe, Fv_pe)
        ab_rows.append({
            'variante': nm_ab,
            'puntaje_07_03': round(0.7 * mpe['iou_0px'] + 0.3 * mlc['iou_0px'], 4),
            'lc_iou': mlc['iou_0px'], 'lc_f1_obj': mlc['obj10_f1'], 'lc_fa_ha_km2': mlc['fa_ha_km2'],
            'lc_det': f"{mlc['obj10_tp']}/{mlc['obj10_tot']}",
            'pe_iou': mpe['iou_0px'], 'pe_f1_obj': mpe['obj10_f1'], 'pe_det_ge200': mpe['det_ge200'],
        })
    with open('informes/v9_e7_ablacion_val.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(ab_rows[0].keys())); w.writeheader(); w.writerows(ab_rows)

    res_val = {
        'base_v6': {'lc': mb_lc, 'pe': mb_pe, 'fa_alto_lc': fa_alto_base},
        'ref_v8': {'lc': mv8_m_lc, 'pe': mv8_m_pe, 'fa_alto_lc': fa_alto_v8},
        'aceptadas': {
            'E1': usa_e1_def, 'E2': usa_e2_def, 'E3': usa_e3_def,
            'E4a_forma': usa_e4a_def, 'E4b_edificaciones': usa_e4b_def, 'E4c_rebrote': usa_e4c_def,
        },
        'parametros_v9': {
            'modelos': {'principal': 'unet_v6', 'auxiliar': 'P3_solo_profesor'},
            'pesos_ensamble': {'unet_v6': 0.6, 'P3_solo_profesor': 0.4},
            'tta_4_espejos': True,
            'e1_corte_altitud_m': gan_e7['corte'],
            'e1_condicion_wc_30_60': gan_e7['cond_wc'],
            'filtro_wc_base': 'wc_urb_agua',
            'histeresis': {'u_alto': gan_e7['ua'], 'u_bajo': gan_e7['ub']},
            'e2_rescate_grande': {'activo': usa_e2_final, 'u_rescate': best_e2[4], 't_rescate': best_e2[5], 'pend_min': 20.0, 'area_min': 50},
            'e3_rescate_pequeno': {'activo': usa_e3_def, 'u_peq': best_e3[4], 'dgrvi_max': -0.10, 'pend_min': 25.0, 'elong_min': 1.8},
            'e4_filtros': {'forma_rect_075': usa_e4a_def, 'edificaciones_020': usa_e4b_def, 'rebrote_dgrvi_005': usa_e4c_def},
            'morfologia': {'min_px': gan_e7['min_px'], 'closing_1px': True, 'fill_holes': False},
        },
        'e7_ganador_val': gan_e7,
        'ablacion': ab_rows,
    }
    json.dump(res_val, open('informes/v9_validacion_resumen.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(res_val['parametros_v9'], open('modelos_ajustados/v9.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Validacion v9 completada -> informes/v9_validacion_resumen.json y modelos_ajustados/v9.json', flush=True)
    return res_val


def _bootstrap_pareado_diff(m_a, m_b, Y, V, filas, n_boot=2000, seed=42):
    """Calcula diferencias (m_a - m_b) con IC 95% por bootstrap de zonas (2000 replicas) para
    iou_0px, f1_2px, obj50_f1 y obj200_rec."""
    pa = m_a.astype(np.float32)
    pb = m_b.astype(np.float32)
    zonas_a, p0_a, _, p2_a, ob_a = mr.stats_por_zona(pa, Y, V, filas, u=0.5, solo_ge50=False)
    zonas_b, p0_b, _, p2_b, ob_b = mr.stats_por_zona(pb, Y, V, filas, u=0.5, solo_ge50=False)
    assert zonas_a == zonas_b
    nz = len(zonas_a)
    rng = np.random.default_rng(seed)

    def calc(p0, p2, ob):
        return {
            'iou_0px': mr.calc_metrics_from_counts(p0, is_obj=False)['iou'],
            'f1_2px': mr.calc_metrics_from_counts(p2, is_obj=False)['f1'],
            'obj50_f1': mr.calc_metrics_from_counts(ob[50], is_obj=True)['f1'],
            'obj200_rec': mr.calc_metrics_from_counts(ob[200], is_obj=True)['rec'],
        }

    pt_a = calc(p0_a.sum(axis=0), p2_a.sum(axis=0), {k: ob_a[k].sum(axis=0) for k in (50, 200)})
    pt_b = calc(p0_b.sum(axis=0), p2_b.sum(axis=0), {k: ob_b[k].sum(axis=0) for k in (50, 200)})
    diffs = {k: [] for k in pt_a}
    for _ in range(n_boot):
        idx = rng.integers(0, nz, size=nz)
        da = calc(p0_a[idx].sum(axis=0), p2_a[idx].sum(axis=0), {k: ob_a[k][idx].sum(axis=0) for k in (50, 200)})
        db = calc(p0_b[idx].sum(axis=0), p2_b[idx].sum(axis=0), {k: ob_b[k][idx].sum(axis=0) for k in (50, 200)})
        for k in diffs:
            diffs[k].append(da[k] - db[k])

    out = {}
    for k in pt_a:
        arr = np.sort(np.asarray(diffs[k], dtype=np.float64))
        out[k] = {
            'diff': round(float(pt_a[k] - pt_b[k]), 4),
            'ic95': [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
        }
    return out


def _det_tamano_4tramos(mask_bool, Y, V):
    """Deteccion por tamano (>=30% solape) en 4 tramos: 10-19, 20-49, 50-199, >=200 px."""
    tramos = {'10-19': [0, 0], '20-49': [0, 0], '50-199': [0, 0], '>=200': [0, 0]}
    for i in range(len(Y)):
        yi = (Y[i] > 0) & (V[i] > 0)
        if not yi.any():
            continue
        pi = mask_bool[i] & (V[i] > 0)
        lab, nc = ndimage.label(yi, structure=ST8)
        for cid in range(1, nc + 1):
            cm = lab == cid
            sz = int(cm.sum())
            if sz < 10:
                continue
            hit = int(float((cm & pi).sum()) / sz >= 0.30)
            k = '10-19' if sz < 20 else ('20-49' if sz < 50 else ('50-199' if sz < 200 else '>=200'))
            tramos[k][0] += hit
            tramos[k][1] += 1
    return {k: {'tp': v[0], 'tot': v[1], 'pct': round(v[0] / max(v[1], 1) * 100.0, 1)} for k, v in tramos.items()}


def _explicar_cambio_patch(idx, mask_v6, mask_v9, prob_v6, prob_v8_e1, ctx, es_local=True):
    """Indica que correccion v9 explica el cambio en un parche."""
    if not es_local:
        return 'E1 (ensamble v6+P3 con corte altoandino e histeresis 0.70/0.25 + min20/cierre1px)'
    # Revisar si actuo corte altoandino E1, edificaciones E4b, rebrote E4c, o histeresis+morfologia
    reasons = []
    if ((ctx['elev'][idx] > 3200.0) | np.isin(ctx['wc'][idx], [30, 60])).any():
        reasons.append('E1 (corte altoandino 3200m / WC pajonal-desnudo)')
    if (mask_v6[idx] & (~mask_v9[idx]) & ctx['edif_mask'][idx]).any():
        reasons.append('E4b (filtro edificaciones)')
    if (mask_v6[idx] & (~mask_v9[idx]) & ctx['ok_ant'][idx] & (ctx['dgrvi'][idx] > 0.05)).any():
        reasons.append('E4c (filtro rebrote dGRVI>+0.05)')
    if not reasons:
        reasons.append('E1+hist 0.70/0.25 + min_px=20/cierre 1px')
    return '; '.join(reasons)


def ejecutar_posthoc(ckpt_p3, stats_p3):
    """Evalua UNA SOLA VEZ (post hoc) v6, v8 y v9 en prueba (360), prueba2 (349 neg) y L4SPe test (124)."""
    import torch
    torch.set_num_threads(12)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    escala = cfg_srv.ESCALA_L4S

    cfg_v9 = json.load(open('modelos_ajustados/v9.json', encoding='utf-8'))
    m_v6, rec_u, _ = pp6.cargar_modelo_ckpt(
        'resultados_ajuste_v6/Control_B1_unet_s42/mejor_unet_b1b4b13b14_7c0b35452807.pt',
        'resultados_ajuste_v6/Control_B1_unet_s42/stats_norm_7c0b35452807.json', disp)
    m_p3, _, _ = pp6.cargar_modelo_ckpt(ckpt_p3, stats_p3, disp)

    # Cargar conjuntos de prueba UNA vez
    Xt, Yt, Vt, Ft = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba'})
    ctx_t = cargar_contexto_espacial_locales(Ft)
    Xraw_t = _cargar_raw_locales(Ft)

    Xp2, Yp2, Vp2, Fp2 = af.cargar_locales('parches_128', 'mascaras_nivel1/parches_128', rec_u, escala, {'prueba2'})
    ctx_p2 = cargar_contexto_espacial_locales(Fp2)
    Xraw_p2 = _cargar_raw_locales(Fp2)

    Xpt, Ypt, Vpt, Fpt = af.cargar_l4spe('l4spe_profesor.npz', rec_u, {'prueba'})
    with np.load('l4spe_profesor.npz', allow_pickle=False) as z:
        sel_pe = [i for i, c in enumerate(z['conjunto']) if str(c) == 'prueba']
        Xraw_pt = z['x'][sel_pe]
        elev_pt = Xraw_pt[:, 5, :, :].astype(np.float32)

    # Predicciones base
    pv6_t_notta = pp6.predecir_tta(m_v6, Xt, disp, False)
    pv6_p2_notta = pp6.predecir_tta(m_v6, Xp2, disp, False)
    pv6_pe_notta = pp6.predecir_tta(m_v6, Xpt, disp, False)

    pv6_t_tta = pp6.predecir_tta(m_v6, Xt, disp, True)
    pv6_p2_tta = pp6.predecir_tta(m_v6, Xp2, disp, True)
    pv6_pe_tta = pp6.predecir_tta(m_v6, Xpt, disp, True)

    pp3_t_tta = pp6.predecir_tta(m_p3, Xt, disp, True)
    pp3_p2_tta = pp6.predecir_tta(m_p3, Xp2, disp, True)
    pp3_pe_tta = pp6.predecir_tta(m_p3, Xpt, disp, True)

    # 1) v6 (sin TTA + wc_urb_agua, u=0.45)
    mv6_t = (np.where(np.isin(ctx_t['wc'], [50, 80]), 0.0, pv6_t_notta) >= 0.45) & (Vt > 0)
    mv6_p2 = (np.where(np.isin(ctx_p2['wc'], [50, 80]), 0.0, pv6_p2_notta) >= 0.45) & (Vp2 > 0)
    mv6_pe = (pv6_pe_notta >= 0.45) & (Vpt > 0)

    # 2) v8 (v6 0.6 + P3 0.4, TTA4, wc_dem_3200_agua_urb, hist 0.70/0.25, min20+cl1)
    pv8_t = pp6.aplicar_filtro_altoandino(0.6 * pv6_t_tta + 0.4 * pp3_t_tta, ctx_t['elev'], ctx_t['wc'], 'wc_dem_3200_agua_urb')
    pv8_p2 = pp6.aplicar_filtro_altoandino(0.6 * pv6_p2_tta + 0.4 * pp3_p2_tta, ctx_p2['elev'], ctx_p2['wc'], 'wc_dem_3200_agua_urb')
    pv8_pe = 0.6 * pv6_pe_tta + 0.4 * pp3_pe_tta
    mv8_t = v8.postproceso_objeto(v8.umbral_histeresis(pv8_t, 0.70, 0.25) & (Vt > 0), 20, False, True) & (Vt > 0)
    mv8_p2 = v8.postproceso_objeto(v8.umbral_histeresis(pv8_p2, 0.70, 0.25) & (Vp2 > 0), 20, False, True) & (Vp2 > 0)
    mv8_pe = v8.postproceso_objeto(v8.umbral_histeresis(pv8_pe, 0.70, 0.25) & (Vpt > 0), 20, False, True) & (Vpt > 0)

    # 3) v9 (E1 corte 3200 + WC 30/60 -> hist -> E2/E3 si activos -> E4b/E4c -> min_px + cl1)
    corte = float(cfg_v9['e1_corte_altitud_m'])
    cond_wc = bool(cfg_v9['e1_condicion_wc_30_60'])
    ua, ub = float(cfg_v9['histeresis']['u_alto']), float(cfg_v9['histeresis']['u_bajo'])
    mpx = int(cfg_v9['morfologia']['min_px'])
    cl1 = bool(cfg_v9['morfologia']['closing_1px'])

    pv9_t = aplicar_e1_ensamble(pv6_t_tta, pp3_t_tta, ctx_t['elev'], ctx_t['wc'], corte, cond_wc)
    pv9_t[np.isin(ctx_t['wc'], [50, 80])] = 0.0
    pv9_p2 = aplicar_e1_ensamble(pv6_p2_tta, pp3_p2_tta, ctx_p2['elev'], ctx_p2['wc'], corte, cond_wc)
    pv9_p2[np.isin(ctx_p2['wc'], [50, 80])] = 0.0
    pv9_pe = aplicar_e1_ensamble(pv6_pe_tta, pp3_pe_tta, elev_pt, None, corte, cond_wc)

    mv9_t = v8.umbral_histeresis(pv9_t, ua, ub) & (Vt > 0)
    mv9_p2 = v8.umbral_histeresis(pv9_p2, ua, ub) & (Vp2 > 0)
    mv9_pe = v8.umbral_histeresis(pv9_pe, ua, ub) & (Vpt > 0)

    if cfg_v9['e2_rescate_grande']['activo']:
        ur, tr = float(cfg_v9['e2_rescate_grande']['u_rescate']), float(cfg_v9['e2_rescate_grande']['t_rescate'])
        mv9_t = aplicar_e2_rescate_grande(mv9_t, pv9_t, ctx_t, ur, tr) & (Vt > 0)
        mv9_p2 = aplicar_e2_rescate_grande(mv9_p2, pv9_p2, ctx_p2, ur, tr) & (Vp2 > 0)
    if cfg_v9['e3_rescate_pequeno']['activo']:
        up = float(cfg_v9['e3_rescate_pequeno']['u_peq'])
        mv9_t = aplicar_e3_rescate_pequeno(mv9_t, pv9_t, ctx_t, up) & (Vt > 0)
        mv9_p2 = aplicar_e3_rescate_pequeno(mv9_p2, pv9_p2, ctx_p2, up) & (Vp2 > 0)
    if cfg_v9['e4_filtros']['edificaciones_020'] or cfg_v9['e4_filtros']['rebrote_dgrvi_005']:
        ue = bool(cfg_v9['e4_filtros']['edificaciones_020'])
        ur_f = bool(cfg_v9['e4_filtros']['rebrote_dgrvi_005'])
        mv9_t = aplicar_e4_filtros(mv9_t, ctx_t, usar_forma=False, usar_edif=ue, usar_rebrote=ur_f) & (Vt > 0)
        mv9_p2 = aplicar_e4_filtros(mv9_p2, ctx_p2, usar_forma=False, usar_edif=ue, usar_rebrote=ur_f) & (Vp2 > 0)

    mv9_t = v8.postproceso_objeto(mv9_t, mpx, False, cl1) & (Vt > 0)
    mv9_p2 = v8.postproceso_objeto(mv9_p2, mpx, False, cl1) & (Vp2 > 0)
    mv9_pe = v8.postproceso_objeto(mv9_pe, mpx, False, cl1) & (Vpt > 0)

    modelos = {
        'v6': (mv6_t, mv6_p2, mv6_pe),
        'v8': (mv8_t, mv8_p2, mv8_pe),
        'v9': (mv9_t, mv9_p2, mv9_pe),
    }

    res = {'config_v9': cfg_v9, 'modelos': {}, 'diferencias': {}, 'errores_corregidos_y_nuevos': {}}
    for nm, (mt, mp2, mpe) in modelos.items():
        print(f'[T2 POST HOC] Evaluando {nm}...', flush=True)
        res['modelos'][nm] = {
            'prueba_360': v8.evaluar_mascara_bootstrap_completo(mt, Yt, Vt, Ft),
            'prueba_360_tamano': _det_tamano_4tramos(mt, Yt, Vt),
            'prueba2_fa': e5.eval_prueba2_fa(mp2.astype(np.float32), Vp2, Fp2, 0.5),
            'l4spe_test': v8.evaluar_mascara_bootstrap_completo(mpe, Ypt, Vpt, Fpt),
            'l4spe_test_tamano': _det_tamano_4tramos(mpe, Ypt, Vpt),
        }

    # Diferencias v9 - v6 y v9 - v8 con IC
    res['diferencias'] = {
        'v9_menos_v6_prueba_360': _bootstrap_pareado_diff(mv9_t, mv6_t, Yt, Vt, Ft),
        'v9_menos_v8_prueba_360': _bootstrap_pareado_diff(mv9_t, mv8_t, Yt, Vt, Ft),
        'v9_menos_v6_l4spe_test': _bootstrap_pareado_diff(mv9_pe, mv6_pe, Ypt, Vpt, Fpt),
        'v9_menos_v8_l4spe_test': _bootstrap_pareado_diff(mv9_pe, mv8_pe, Ypt, Vpt, Fpt),
    }

    # Errores corregidos y nuevos (v9 vs v6)
    def analizar_cambios(Y, V, filas, m6, m9, pv6, pv9, ctx, es_local):
        corr_fa, corr_cic, nuevos_fn, nuevos_fp = [], [], [], []
        for i in range(len(Y)):
            yi = (Y[i] > 0) & (V[i] > 0)
            p6 = m6[i] & (V[i] > 0)
            p9 = m9[i] & (V[i] > 0)
            exp = _explicar_cambio_patch(i, m6, m9, pv6[i], pv9[i], ctx, es_local=es_local)
            if not yi.any():
                fp6, fp9 = int(p6.sum()), int(p9.sum())
                if fp6 >= 10 and fp9 == 0:
                    corr_fa.append({'parche': filas[i]['parche'], 'zona': filas[i]['zona'], 'px_fa_v6': fp6, 'px_fa_v9': fp9, 'explicacion': exp, '_idx': i})
                elif fp6 == 0 and fp9 >= 20:
                    nuevos_fp.append({'parche': filas[i]['parche'], 'zona': filas[i]['zona'], 'px_fa_v6': fp6, 'px_fa_v9': fp9, 'explicacion': exp, '_idx': i})
            else:
                lab, nc = ndimage.label(yi, structure=ST8)
                for cid in range(1, nc + 1):
                    cm = lab == cid
                    sz = int(cm.sum())
                    if sz < 10:
                        continue
                    h6 = float((cm & p6).sum()) / sz >= 0.30
                    h9 = float((cm & p9).sum()) / sz >= 0.30
                    if (not h6) and h9:
                        corr_cic.append({'parche': filas[i]['parche'], 'zona': filas[i]['zona'], 'px_gt': sz, 'explicacion': exp, '_idx': i})
                    elif h6 and (not h9):
                        nuevos_fn.append({'parche': filas[i]['parche'], 'zona': filas[i]['zona'], 'px_gt': sz, 'explicacion': exp, '_idx': i})
        corr_fa.sort(key=lambda d: -d['px_fa_v6'])
        corr_cic.sort(key=lambda d: -d['px_gt'])
        nuevos_fn.sort(key=lambda d: -d['px_gt'])
        nuevos_fp.sort(key=lambda d: -d['px_fa_v9'])
        return {'fa_eliminadas': corr_fa, 'cicatrices_recuperadas': corr_cic, 'cicatrices_perdidas': nuevos_fn, 'fa_nuevas': nuevos_fp}

    camb_t = analizar_cambios(Yt, Vt, Ft, mv6_t, mv9_t, pv6_t_notta, pv9_t, ctx_t, True)
    camb_pe = analizar_cambios(Ypt, Vpt, Fpt, mv6_pe, mv9_pe, pv6_pe_notta, pv9_pe, None, False)
    # En prueba2 tambien registrar FA eliminadas y nuevas sobre los 349 negativos
    p2_neg_elim, p2_neg_nuev = [], []
    for i, r in enumerate(Fp2):
        if r.get('decision') != 'negativo_confirmado':
            continue
        fp6, fp9 = int(mv6_p2[i].sum()), int(mv9_p2[i].sum())
        exp = _explicar_cambio_patch(i, mv6_p2, mv9_p2, pv6_p2_notta[i], pv9_p2[i], ctx_p2, True)
        if fp6 >= 10 and fp9 == 0:
            p2_neg_elim.append({'parche': r['parche'], 'zona': r['zona'], 'tipo': r.get('tipo_negativo', ''), 'px_fa_v6': fp6, 'px_fa_v9': fp9, 'explicacion': exp, '_idx': i})
        elif fp6 == 0 and fp9 >= 20:
            p2_neg_nuev.append({'parche': r['parche'], 'zona': r['zona'], 'tipo': r.get('tipo_negativo', ''), 'px_fa_v6': fp6, 'px_fa_v9': fp9, 'explicacion': exp, '_idx': i})
    p2_neg_elim.sort(key=lambda d: -d['px_fa_v6'])
    p2_neg_nuev.sort(key=lambda d: -d['px_fa_v9'])

    def strip_idx(lst):
        return [{k: v for k, v in d.items() if k != '_idx'} for d in lst]

    res['errores_corregidos_y_nuevos'] = {
        'prueba_360': {k: strip_idx(v) for k, v in camb_t.items()},
        'l4spe_test': {k: strip_idx(v) for k, v in camb_pe.items()},
        'prueba2_negativos': {'fa_eliminadas': strip_idx(p2_neg_elim), 'fa_nuevas': strip_idx(p2_neg_nuev)},
    }

    # Guardar informes/evaluacion_v9.json
    json.dump(res, open('informes/evaluacion_v9.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # Generar figuras finales en informes/figuras_finales/
    _generar_figuras_finales(res, Xraw_t, Yt, Vt, Ft, mv6_t, mv9_t,
                             Xraw_p2, Yp2, Vp2, Fp2, mv6_p2, mv9_p2,
                             Xraw_pt, Ypt, Vpt, Fpt, mv6_pe, mv9_pe,
                             camb_t, camb_pe, p2_neg_elim)

    # Escribir informes/resultado_v9.md y actualizar informes/metricas_reformuladas.md
    _escribir_resultado_v9(res)
    _actualizar_metricas_reformuladas(res)
    print('Evaluacion unica post hoc v9 completada -> informes/evaluacion_v9.json y resultado_v9.md', flush=True)


def _rgb_desde_raw(x_raw, is_l4spe=False):
    if is_l4spe:
        return v8.rgb_l4spe(x_raw)
    rgb = np.stack([x_raw[3], x_raw[2], x_raw[1]], axis=-1).astype(np.float32)
    v = np.isfinite(rgb) & (rgb > 0)
    if v.any():
        p2, p98 = np.percentile(rgb[v], [2, 98])
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return np.nan_to_num(rgb, nan=0.0)


def _generar_figuras_finales(res, Xraw_t, Yt, Vt, Ft, mv6_t, mv9_t,
                             Xraw_p2, Yp2, Vp2, Fp2, mv6_p2, mv9_p2,
                             Xraw_pt, Ypt, Vpt, Fpt, mv6_pe, mv9_pe,
                             camb_t, camb_pe, p2_neg_elim):
    fdir = 'informes/figuras_finales'
    os.makedirs(fdir, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': 'white', 'figure.facecolor': 'white'})

    def plot_4paneles(ruta, rgb, gt, p6, p9, titulo, linea_explicacion):
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), dpi=200, facecolor='white')
        paneles = [
            ('Imagen Sentinel-2', rgb),
            (f'Referencia ({int(gt.sum())} px)', v8.overlay(rgb, gt, (0.1, 0.85, 0.2), 0.55)),
            (f'Modelo v6 ({int(p6.sum())} px)', v8.overlay(rgb, p6, (1.0, 0.25, 0.15), 0.55)),
            (f'Modelo v9 ({int(p9.sum())} px)', v8.overlay(rgb, p9, (1.0, 0.15, 0.15), 0.55)),
        ]
        for ax, (tit, im) in zip(axes, paneles):
            ax.imshow(im)
            ax.set_title(tit, fontsize=9.5, fontweight='bold')
            ax.set_xlabel('Este (px = 10 m)', fontsize=8)
            ax.set_ylabel('Norte (px = 10 m)', fontsize=8)
            ax.set_xticks([0, 64, 127]); ax.set_yticks([0, 64, 127])
            ax.tick_params(labelsize=7)
        fig.suptitle(titulo, fontsize=11, fontweight='bold', y=1.02)
        fig.text(0.5, -0.02, linea_explicacion, ha='center', fontsize=9, style='italic')
        plt.tight_layout()
        fig.savefig(ruta, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # 2 FA eliminadas (priorizar prueba2 / prueba 360)
    fa_cands = []
    for d in p2_neg_elim:
        i = d['_idx']
        fa_cands.append((d['px_fa_v6'], d['parche'], _rgb_desde_raw(Xraw_p2[i]), (Vp2[i] < 0), mv6_p2[i], mv9_p2[i],
                         f"Falsa alarma eliminada ({d['tipo'] or 'prueba2'}): {d['parche']} — {d['px_fa_v6']} px -> 0 px",
                         f"Resuelto por: {d['explicacion']}"))
    for d in camb_t['fa_eliminadas']:
        i = d['_idx']
        fa_cands.append((d['px_fa_v6'], d['parche'], _rgb_desde_raw(Xraw_t[i]), (Yt[i] > 0) & (Vt[i] > 0), mv6_t[i], mv9_t[i],
                         f"Falsa alarma eliminada (prueba 360): {d['parche']} — {d['px_fa_v6']} px -> 0 px",
                         f"Resuelto por: {d['explicacion']}"))
    for k, item in enumerate(fa_cands[:2], 1):
        _, pch, rgb, gt, p6, p9, tit, exp = item
        plot_4paneles(os.path.join(fdir, f'0{k}_fa_eliminada_{pch}.png'), rgb, gt, p6, p9, tit, exp)

    # 2 cicatrices grandes (>= 50 px) y 2 pequenas (< 50 px) recuperadas (excluyendo parches con >=100 px FP)
    parches_fp_grande = {Fpt[i]['parche'] for i in range(len(Ypt)) if int((mv9_pe[i] & (~((Ypt[i] > 0) & (Vpt[i] > 0)))).sum()) >= 100}
    rec_grandes, rec_peq = [], []
    for d in camb_pe['cicatrices_recuperadas']:
        if d['parche'] in parches_fp_grande:
            continue
        i = d['_idx']
        item = (d['px_gt'], d['parche'], _rgb_desde_raw(Xraw_pt[i], True), (Ypt[i] > 0) & (Vpt[i] > 0), mv6_pe[i], mv9_pe[i],
                f"Cicatriz {'grande' if d['px_gt']>=50 else 'pequeña'} recuperada (L4SPe test): {d['parche']} ({d['px_gt']} px)",
                f"Resuelto por: {d['explicacion']}")
        if d['px_gt'] >= 50:
            rec_grandes.append(item)
        else:
            rec_peq.append(item)
    for k, item in enumerate(rec_grandes[:2], 3):
        _, pch, rgb, gt, p6, p9, tit, exp = item
        plot_4paneles(os.path.join(fdir, f'0{k}_grande_recuperada_{pch}.png'), rgb, gt, p6, p9, tit, exp)
    for k, item in enumerate(rec_peq[:2], 5):
        _, pch, rgb, gt, p6, p9, tit, exp = item
        plot_4paneles(os.path.join(fdir, f'0{k}_pequena_recuperada_{pch}.png'), rgb, gt, p6, p9, tit, exp)

    # 2 laminas de errores que siguen sin resolverse (1 FN grande en prueba 360 + 1 FP o FN en L4SPe)
    fn_lc, fn_pe = [], []
    for i in range(len(Yt)):
        yi = (Yt[i] > 0) & (Vt[i] > 0)
        if int(yi.sum()) >= 100 and int((yi & mv9_t[i]).sum()) == 0:
            fn_lc.append((int(yi.sum()), i))
    fn_lc.sort(reverse=True)
    if fn_lc:
        sz, i = fn_lc[0]
        plot_4paneles(os.path.join(fdir, f'07_error_sin_resolver_{Ft[i]["parche"]}.png'),
                      _rgb_desde_raw(Xraw_t[i]), (Yt[i] > 0) & (Vt[i] > 0), mv6_t[i], mv9_t[i],
                      f"Error sin resolver (prueba 360): {Ft[i]['parche']} ({sz} px no detectados)",
                      "Causa: cicatriz con baja probabilidad (<0.25) en sustrato pardo-rojizo o revegetado; sin rescate E2 activo por FA.")

    for i in range(len(Ypt)):
        yi = (Ypt[i] > 0) & (Vpt[i] > 0)
        fp_px = int((mv9_pe[i] & (~yi)).sum())
        if fp_px >= 100:
            fn_pe.append((fp_px, i))
    fn_pe.sort(reverse=True)
    if fn_pe:
        fp_px, i = fn_pe[0]
        plot_4paneles(os.path.join(fdir, f'08_error_sin_resolver_{Fpt[i]["parche"]}.png'),
                      _rgb_desde_raw(Xraw_pt[i], True), (Ypt[i] > 0) & (Vpt[i] > 0), mv6_pe[i], mv9_pe[i],
                      f"Error sin resolver (L4SPe test): {Fpt[i]['parche']} ({fp_px} px FP)",
                      "Causa: excavación/suelo desnudo extenso sin cobertura WorldCover ni imagen anterior en L4SPe.")

    # Grafico de barras v6 / v8 / v9 con IC y meta del proyecto
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200, facecolor='white')
    ms = ['v6', 'v8', 'v9']
    cols = ['#6c757d', '#2b7bba', '#1f9d55']

    # Panel 1: F1 objeto >= 0.5 ha (L4SPe test)
    vals1 = [res['modelos'][m]['l4spe_test']['todas']['obj_ge50']['f1'] * 100 for m in ms]
    lo1 = [vals1[idx] - res['modelos'][m]['l4spe_test']['todas']['obj_ge50']['f1_ic95'][0] * 100 for idx, m in enumerate(ms)]
    hi1 = [res['modelos'][m]['l4spe_test']['todas']['obj_ge50']['f1_ic95'][1] * 100 - vals1[idx] for idx, m in enumerate(ms)]
    axes[0].bar(ms, vals1, yerr=[lo1, hi1], capsize=5, color=cols, edgecolor='black', linewidth=0.7)
    axes[0].axhline(75.5, color='crimson', linestyle='--', linewidth=1.3, label='Meta (75.5 %)')
    axes[0].set_title('F1 por objeto ≥ 0.5 ha (L4SPe test)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Porcentaje (%)', fontsize=9); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=8)

    # Panel 2: Deteccion >= 2 ha (L4SPe test)
    vals2 = [res['modelos'][m]['l4spe_test']['todas']['obj_ge200']['rec'] * 100 for m in ms]
    lo2 = [vals2[idx] - res['modelos'][m]['l4spe_test']['todas']['obj_ge200']['rec_ic95'][0] * 100 for idx, m in enumerate(ms)]
    hi2 = [res['modelos'][m]['l4spe_test']['todas']['obj_ge200']['rec_ic95'][1] * 100 - vals2[idx] for idx, m in enumerate(ms)]
    axes[1].bar(ms, vals2, yerr=[lo2, hi2], capsize=5, color=cols, edgecolor='black', linewidth=0.7)
    axes[1].axhline(85.0, color='crimson', linestyle='--', linewidth=1.3, label='Meta (85.0 %)')
    axes[1].set_title('Detección ≥ 2 ha (L4SPe test)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Exhaustividad (%)', fontsize=9); axes[1].set_ylim(0, 105); axes[1].legend(fontsize=8)

    # Panel 3: Falsas alarmas en prueba2 (ha/km2)
    vals3 = [res['modelos'][m]['prueba2_fa']['TOTAL_349']['fa_ha_km2'] for m in ms]
    axes[2].bar(ms, vals3, color=cols, edgecolor='black', linewidth=0.7)
    axes[2].axhline(vals3[0], color='navy', linestyle=':', linewidth=1.3, label=f'Ref. v6 ({vals3[0]:.4f})')
    axes[2].set_title('Falsas alarmas en prueba2 (349 neg.)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Área falsa (ha/km²)', fontsize=9); axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Comparación v6 / v8 / v9 («post hoc») con IC 95 % por zonas', fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fdir, '09_comparativa_v6_v8_v9_ic.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _escribir_resultado_v9(res):
    fmt = mr.fmt_ic
    md = [
        '# Resultados Mejora v9 — Evaluación Única «Post Hoc»',
        '',
        '> [!IMPORTANT]',
        '> Todos los parámetros de `E1–E4` y `E7` fueron elegidos **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Toda cifra sobre `prueba` (360), `prueba2` (349 negativos) y `L4SPe test` (124) es **«post hoc»** con IC 95 % por bootstrap de zonas (`2 000` réplicas).',
        '> Nota técnica sobre `E3`: los TIF locales empaquetados tienen 6 canales (`B1, B2, B3, B4, pendiente, elevación`) sin banda `SCL`, por lo que la distancia a nube/sombra se evaluó mediante máscara espectral equivalente.',
        '',
        '## 1. Comparación `v6 / v8 / v9` en las 3 Pruebas («post hoc»)',
        '',
        '### 1a. `L4SPe` Test (`124` parches, `68` zonas)',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | F1 0 px [IC 95 %] | IoU 2 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |',
        '|---|---|---|---|---|---|---|',
    ]
    for nm in ('v6', 'v8', 'v9'):
        t = res['modelos'][nm]['l4spe_test']['todas']
        md.append(
            f"| **{nm}** | {fmt(t['pixel_exacto'],'iou')} | {fmt(t['pixel_exacto'],'f1',True)} | "
            f"{fmt(t['pixel_tol2px'],'iou')} | {fmt(t['pixel_tol2px'],'f1',True)} | "
            f"{fmt(t['obj_ge50'],'f1',True)} | {fmt(t['obj_ge200'],'rec',True)} |"
        )
    md.extend([
        '',
        '### 1b. `prueba` Sellada de La Convención (`360` parches, `62` zonas)',
        '',
        '| Modelo | IoU 0 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] | FA neg. (`ha/km²`) |',
        '|---|---|---|---|---|---:|',
    ])
    for nm in ('v6', 'v8', 'v9'):
        t = res['modelos'][nm]['prueba_360']['todas']
        fa = res['modelos'][nm]['prueba_360']['fa_ha_km2_negativos']
        md.append(
            f"| **{nm}** | {fmt(t['pixel_exacto'],'iou')} | {fmt(t['pixel_tol2px'],'f1',True)} | "
            f"{fmt(t['obj_ge50'],'f1',True)} | {fmt(t['obj_ge200'],'rec',True)} | {fa:.3f} |"
        )
    md.extend([
        '',
        '### 1c. `prueba2` Sellada (`349` negativos confirmados por `tipo_negativo`)',
        '',
        '| Modelo | **TOTAL (`ha/km²` / `%`)** | `altoandino` (49) | `chacra_o_pasto` (9) | `rio` (34) | `carretera` (8) | `poblado` (6) | `bosque_ladera` (243) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for nm in ('v6', 'v8', 'v9'):
        p2 = res['modelos'][nm]['prueba2_fa']
        md.append(
            f"| **{nm}** | **{p2['TOTAL_349']['fa_ha_km2']:.4f} ({p2['TOTAL_349']['pct_con_marca']} %)** | "
            f"{p2['altoandino']['fa_ha_km2']:.4f} | {p2['chacra_o_pasto']['fa_ha_km2']:.4f} | "
            f"{p2['rio']['fa_ha_km2']:.4f} | {p2['carretera']['fa_ha_km2']:.4f} | "
            f"{p2['poblado']['fa_ha_km2']:.4f} | {p2['bosque_ladera']['fa_ha_km2']:.4f} |"
        )
    with open('informes/resultado_v9.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')


def _actualizar_metricas_reformuladas(res):
    ruta = 'informes/metricas_reformuladas.md'
    if not os.path.exists(ruta):
        return
    txt = open(ruta, encoding='utf-8').read()
    fmt = mr.fmt_ic
    t_lc = res['modelos']['v9']['prueba_360']['todas']
    t_pe = res['modelos']['v9']['l4spe_test']['todas']
    fa_lc = res['modelos']['v9']['prueba_360']['fa_ha_km2_negativos']
    fa_p2 = res['modelos']['v9']['prueba2_fa']['TOTAL_349']
    bloque = (
        '\n\n## Actualización v9 («post hoc»)\n\n'
        '| Conjunto | IoU 0 px | IoU 2 px | F1 2 px | F1 obj `≥ 0.1 ha` | F1 obj `≥ 0.5 ha` | Det `≥ 2 ha` | FA neg. (`ha/km²`) |\n'
        '|---|---|---|---|---|---|---|---:|\n'
        f"| `prueba` (360) | {fmt(t_lc['pixel_exacto'],'iou')} | {fmt(t_lc['pixel_tol2px'],'iou')} | {fmt(t_lc['pixel_tol2px'],'f1',True)} | {fmt(t_lc['obj_ge10'],'f1',True)} | {fmt(t_lc['obj_ge50'],'f1',True)} | {fmt(t_lc['obj_ge200'],'rec',True)} | {fa_lc:.3f} |\n"
        f"| `L4SPe` test (124) | {fmt(t_pe['pixel_exacto'],'iou')} | {fmt(t_pe['pixel_tol2px'],'iou')} | {fmt(t_pe['pixel_tol2px'],'f1',True)} | {fmt(t_pe['obj_ge10'],'f1',True)} | {fmt(t_pe['obj_ge50'],'f1',True)} | {fmt(t_pe['obj_ge200'],'rec',True)} | {res['modelos']['v9']['l4spe_test']['fa_ha_km2_negativos']:.3f} |\n"
        f"| `prueba2` (349 neg.) | — | — | — | — | — | — | **{fa_p2['fa_ha_km2']:.4f}** ({fa_p2['pct_con_marca']} %) |\n"
    )
    if '## Actualización v9' in txt:
        txt = txt.split('## Actualización v9')[0].rstrip() + bloque
    else:
        txt = txt.rstrip() + bloque
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', choices=['val', 'posthoc'], required=True)
    ap.add_argument('--ckpt-p3', default='resultados_ajuste_v8/P3_solo_profesor/mejor_unet_b1b4b13b14_da008a8d0de3.pt')
    ap.add_argument('--stats-p3', default='resultados_ajuste_v8/P3_solo_profesor/stats_norm_da008a8d0de3.json')
    a = ap.parse_args()
    if a.fase == 'val':
        ejecutar_validacion(a.ckpt_p3, a.stats_p3)
    else:
        ejecutar_posthoc(a.ckpt_p3, a.stats_p3)


if __name__ == '__main__':
    main()
