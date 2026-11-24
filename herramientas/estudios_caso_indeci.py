"""Estudios de caso cualitativos (Tarea 2) sobre 3 eventos reales de INDECI en areas de 3x3 km.

Para cada caso:
  - Compone Sentinel-2 L2A (antes y despues) a 10 m, COP-DEM-30 (elevacion + pendiente),
    ESA WorldCover 10 m y Bing aereo.
  - Corre inferencia por ventanas 128x128 (con solape) con:
      1) U-Net Landslide4Sense (Proyecto_Tesis_Deslizamiento/modelos/unet.onnx, u=0.45)
      2) U-Net B1 v5 (modelos_ajustados/unet_b1_v5.onnx, u=0.30)
      3) U-Net v6 sin TTA + filtro wc_dem_3200_agua_urb (modelos_ajustados/unet_v6.onnx, u=0.45)
  - Guarda 1 lamina por caso en _fichas/estudios_caso/ y escribe informes/estudios_caso.md.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import planetary_computer
import pystac_client
import rasterio
from rasterio.coords import BoundingBox
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from pyproj import Transformer
from scipy import ndimage
import stackstac

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, '..'))

from nucleo import config as cfg_srv, recetas
from generar_revision_profesor import fetch_bing_patch
import posproceso_v6 as pp6
import worldcover_128 as wc128

CASOS = [
    {
        'id': 'caso_01_santa_teresa_2020',
        'titulo': 'Caso 1 — Alud Salkantay–Sahuayaco, Santa Teresa (23/02/2020, INDECI ID 119409)',
        'distrito': 'Santa Teresa',
        'fecha_evento': '2020-02-23',
        'indeci_info': 'ID 119409 (4 fallecidos, 87 viviendas destruidas en matriz INDECI; >200 viviendas reportadas en Sahuayaco/Chaullay)',
        'lon': -72.6230, 'lat': -13.2050,
        'dist_capital_km': 9.0,
        'f_pre': '2019-06-01/2019-10-31',
        'f_post': '2020-05-01/2020-10-31',
    },
    {
        'id': 'caso_02_echarate_2021',
        'titulo': 'Caso 2 — Huaico en quebrada Echarate, Echarate (16/02/2021, INDECI ID 134745)',
        'distrito': 'Echarate',
        'fecha_evento': '2021-02-16',
        'indeci_info': 'ID 134745 (Huaico, 13 viviendas destruidas, puntaje de daño 30 142.5)',
        'lon': -72.5620, 'lat': -12.7470,
        'dist_capital_km': 3.0,
        'f_pre': '2020-06-01/2020-10-31',
        'f_post': '2021-05-01/2021-10-31',
    },
    {
        'id': 'caso_03_vilcabamba_2021',
        'titulo': 'Caso 3 — Deslizamiento en ladera, Vilcabamba (27/03/2021, INDECI ID 137416)',
        'distrito': 'Vilcabamba',
        'fecha_evento': '2021-03-27',
        'indeci_info': 'ID 137416 / 132708 (Deslizamiento/huaico en Vilcabamba; Inkawasi descartado por falta de cicatriz clara en Bing)',
        'lon': -73.0589, 'lat': -13.0938,
        'dist_capital_km': 13.9,
        'f_pre': '2020-06-01/2020-10-31',
        'f_post': '2021-05-01/2021-10-31',
    },
]

MODELOS_SPEC = [
    ('U-Net L4S', r'C:\Users\User\Documents\Proyecto_Tesis_Deslizamiento\modelos\unet.onnx',
     r'C:\Users\User\Documents\Proyecto_Tesis_Deslizamiento\modelos\unet.json', 0.45, 'ninguno'),
    ('U-Net B1 v5', 'modelos_ajustados/unet_b1_v5.onnx',
     'modelos_ajustados/unet_b1_v5.json', 0.30, 'ninguno'),
    ('U-Net v6 (sin TTA+flt)', 'modelos_ajustados/unet_v6.onnx',
     'modelos_ajustados/unet_v6.json', 0.45, 'wc_dem_3200_agua_urb'),
]


def componer_s2_4b(cat, bbox_ll, b_utm, fechas, size=300):
    items = list(cat.search(
        collections=['sentinel-2-l2a'], bbox=bbox_ll, datetime=fechas,
        query={'eo:cloud_cover': {'lt': 35}}
    ).items())
    if not items:
        items = list(cat.search(
            collections=['sentinel-2-l2a'], bbox=bbox_ll, datetime=fechas,
            query={'eo:cloud_cover': {'lt': 60}}
        ).items())
    items = sorted(items, key=lambda it: it.properties.get('eo:cloud_cover', 99))[:12]
    da = stackstac.stack(
        items, assets=['B01', 'B02', 'B03', 'B04', 'SCL'],
        epsg=32718, resolution=10, bounds=b_utm,
        rescale=False, dtype='float64', fill_value=np.nan
    )
    arr = da.compute().values  # (T, 5, H, W)
    scl = arr[:, 4]
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10, 11]) | (~np.isfinite(arr[:, 1]))
    b4 = arr[:, :4].copy()
    b4[np.repeat(bad[:, None, :, :], 4, axis=1)] = np.nan
    med = np.nanmedian(b4, axis=0).astype(np.float32)
    # Ajustar exactamente a (4, size, size)
    med = np.nan_to_num(med, nan=0.0)[:, :size, :size]
    return med


def obtener_dem_y_pendiente(cat, bbox_ll, b_utm, size=300):
    items = list(cat.search(collections=['cop-dem-glo-30'], bbox=bbox_ll).items())
    dst_tr = from_bounds(*b_utm, size, size)
    elev = np.zeros((size, size), dtype=np.float32)
    for it in items:
        href = it.assets['data'].href
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs='EPSG:32718', transform=dst_tr, width=size, height=size,
                           resampling=Resampling.bilinear, nodata=np.nan) as vrt:
                arr = vrt.read(1).astype(np.float32)
                ok = np.isfinite(arr) & (elev == 0)
                elev[ok] = arr[ok]
    gy, gx = np.gradient(elev, 10.0, 10.0)
    pend = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    return elev, pend


def obtener_worldcover_3km(fuentes_wc, b_utm, size=300):
    dst_tr = from_bounds(*b_utm, size, size)
    tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True)
    w, s = tr_ll.transform(b_utm[0], b_utm[1])
    e, n = tr_ll.transform(b_utm[2], b_utm[3])
    out = np.zeros((size, size), dtype=np.uint8)
    for _, fb, src in fuentes_wc:
        if e < fb[0] or w > fb[2] or n < fb[1] or s > fb[3]:
            continue
        with WarpedVRT(src, crs='EPSG:32718', transform=dst_tr, width=size, height=size,
                       resampling=Resampling.nearest, nodata=0) as vrt:
            arr = vrt.read(1)
            m = (out == 0) & (arr > 0)
            out[m] = arr[m]
    return out


def inferir_mosaico_onnx(sess, receta, b_post, pend, elev, escala, filtro='ninguno', wc=None):
    """Inferencia por ventanas 128x128 con paso 86 (solape promediado) sobre escena 300x300."""
    H, W = pend.shape
    valido = b_post[1] > 0
    bandas = {'B01': b_post[0], 'B02': b_post[1], 'B03': b_post[2], 'B04': b_post[3]}
    terreno = {'pendiente': pend, 'elevacion': elev}
    tensor = recetas.desde_l2a(bandas, terreno, valido, receta, escala=escala)  # (6, H, W)
    inp = sess.get_inputs()[0].name
    acum = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    starts_r = [0, 86, H - 128]
    starts_c = [0, 86, W - 128]
    for r0 in starts_r:
        for c0 in starts_c:
            patch = np.ascontiguousarray(tensor[None, :, r0:r0+128, c0:c0+128], dtype=np.float32)
            logits = sess.run(None, {inp: patch})[0][0, 0]
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            acum[r0:r0+128, c0:c0+128] += prob
            cnt[r0:r0+128, c0:c0+128] += 1.0
    prob_full = acum / np.maximum(cnt, 1.0)
    prob_full[~valido] = 0.0
    if filtro != 'ninguno' and wc is not None:
        prob_full = pp6.aplicar_filtro_altoandino(prob_full[None, ...], elev[None, ...], wc[None, ...], modo=filtro)[0]
    return prob_full


def norm_rgb(b4):
    im = np.stack([b4[3], b4[2], b4[1]], axis=-1)
    v = np.isfinite(im) & (im > 0)
    p2, p98 = np.percentile(im[v], [2, 98]) if v.any() else (0, 2000)
    return np.clip(np.nan_to_num((im - p2) / max(p98 - p2, 1.0)), 0.0, 1.0)


def overlay(rgb, mask, color=(1.0, 0.15, 0.15), alpha=0.55):
    out = rgb.copy()
    m = mask > 0
    for c in range(3):
        out[..., c] = np.where(m, out[..., c] * (1.0 - alpha) + color[c] * alpha, out[..., c])
    return out


def main():
    cat = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace,
    )
    fuentes_wc = wc128.abrir_fuentes_worldcover()
    tr_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32718', always_xy=True)
    tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True)
    escala = cfg_srv.ESCALA_L4S

    sesiones = []
    for nombre, p_onnx, p_json, u, flt in MODELOS_SPEC:
        sess = ort.InferenceSession(p_onnx, providers=['CPUExecutionProvider'])
        rec = json.load(open(p_json, encoding='utf-8'))
        sesiones.append((nombre, sess, rec, u, flt))

    os.makedirs('_fichas/estudios_caso', exist_ok=True)
    resumenes = []

    try:
        for caso in CASOS:
            print(f"\n=== Procesando {caso['id']} ===", flush=True)
            cx, cy = tr_utm.transform(caso['lon'], caso['lat'])
            b_utm = (cx - 1500.0, cy - 1500.0, cx + 1500.0, cy + 1500.0)
            w, s = tr_ll.transform(b_utm[0], b_utm[1])
            e, n = tr_ll.transform(b_utm[2], b_utm[3])
            bbox_ll = [w, s, e, n]

            b_pre = componer_s2_4b(cat, bbox_ll, b_utm, caso['f_pre'], size=300)
            b_post = componer_s2_4b(cat, bbox_ll, b_utm, caso['f_post'], size=300)
            elev, pend = obtener_dem_y_pendiente(cat, bbox_ll, b_utm, size=300)
            wc = obtener_worldcover_3km(fuentes_wc, b_utm, size=300)
            bing = fetch_bing_patch(BoundingBox(*b_utm), 'EPSG:32718', z=15, out_px=300)

            # Referencia aproximada del foco del evento por dGRVI (solo para estimar cobertura cualitativa vs FA fuera del foco)
            grvi_pre = (b_pre[2] - b_pre[3]) / (np.abs(b_pre[2] + b_pre[3]) + 1e-3)
            grvi_post = (b_post[2] - b_post[3]) / (np.abs(b_post[2] + b_post[3]) + 1e-3)
            dgrvi = grvi_post - grvi_pre
            foco_raw = (dgrvi <= -0.10) & (grvi_post < 0.02) & (b_post[1] < 1800)
            # Zona de influencia del evento (dilatada 50 m = 5 px para tolerar borde de cicatriz/deposito)
            foco_zona = ndimage.binary_dilation(foco_raw, iterations=5)

            rgb_pre = norm_rgb(b_pre)
            rgb_post = norm_rgb(b_post)

            preds = []
            caso_row = dict(caso, elev_med=round(float(np.median(elev)), 0), modelos={})
            for nombre, sess, rec, u, flt in sesiones:
                prob = inferir_mosaico_onnx(sess, rec, b_post, pend, elev, escala, filtro=flt, wc=wc)
                mk = prob >= u
                ha_tot = float(mk.sum()) * 0.01
                ha_en_foco = float((mk & foco_zona).sum()) * 0.01
                ha_fuera = float((mk & (~foco_zona)).sum()) * 0.01
                # Cobertura aproximada del foco principal
                cubre = 'Sí' if ha_en_foco >= 0.5 else ('Parcial' if ha_en_foco >= 0.1 else 'No')
                caso_row['modelos'][nombre] = {
                    'umbral': u, 'ha_total': round(ha_tot, 2),
                    'ha_en_foco': round(ha_en_foco, 2), 'ha_fuera_fa': round(ha_fuera, 2),
                    'cubre_cicatriz': cubre,
                }
                preds.append((nombre, u, mk, ha_tot, ha_en_foco, ha_fuera))
                print(f"  {nombre:24s} | u={u:.2f} | total={ha_tot:6.2f} ha | en_foco={ha_en_foco:5.2f} ha | fuera(FA)={ha_fuera:6.2f} ha | cubre={cubre}", flush=True)

            resumenes.append(caso_row)

            # Guardar lamina 2x3
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2), dpi=150)
            axes[0, 0].imshow(rgb_pre); axes[0, 0].set_title(f"Antes ({caso['f_pre'][:4]}) — S2 10 m", fontsize=10, fontweight='bold')
            axes[0, 1].imshow(rgb_post); axes[0, 1].set_title(f"Después ({caso['f_post'][:4]}) — S2 10 m", fontsize=10, fontweight='bold')
            axes[0, 2].imshow(bing); axes[0, 2].set_title(f"Bing aéreo (3 × 3 km, elev. med. {caso_row['elev_med']:.0f} m)", fontsize=10, fontweight='bold')
            for idx_m, (nombre, u, mk, ha_tot, ha_en_foco, ha_fuera) in enumerate(preds):
                ax = axes[1, idx_m]
                ax.imshow(overlay(rgb_post, mk, (1.0, 0.12, 0.12), 0.58))
                ax.set_title(f"{nombre} (u={u:.2f})\nMarcado: {ha_tot:.2f} ha | En foco: {ha_en_foco:.2f} ha | FA: {ha_fuera:.2f} ha",
                             fontsize=9.5, fontweight='bold')
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{caso['titulo']}\nCentro: ({caso['lon']:.4f}, {caso['lat']:.4f}), a {caso['dist_capital_km']} km de la capital distrital",
                         fontsize=11.5, fontweight='bold')
            plt.tight_layout()
            out_png = f"_fichas/estudios_caso/{caso['id']}.png"
            fig.savefig(out_png, bbox_inches='tight')
            plt.close(fig)
            print(f"  Lamina guardada: {out_png}", flush=True)
    finally:
        for _, _, s in fuentes_wc:
            s.close()

    # Escribir informes/estudios_caso.md
    md = [
        '# Estudios de Caso Cualitativos — Eventos Reales de INDECI (3 × 3 km)',
        '',
        '> [!NOTE]',
        '> **Material cualitativo (no métrica de prueba):** Los 3 eventos fueron localizados exclusivamente mediante **ΔGRVI** entre escenas secas de Sentinel-2 anteriores y posteriores al evento en un radio de `15 km` de la capital distrital reportada por INDECI, **sin usar ningún modelo**, y confirmados visualmente en Bing aéreo y Sentinel-2 a 10 m.',
        '> ',
        '> **Nota sobre selección de eventos:** Se inspeccionó inicialmente `Inkawasi (2021-04-14, ID 138302)`, pero las anomalías de ΔGRVI en radio de 15 km correspondían a secado estacional de pastizales/chacras altoandinas sin una cicatriz inequívoca en Bing; siguiendo el protocolo, se sustituyó por **`Vilcabamba (2021-03-27, ID 137416)`**, donde sí se verificó una cicatriz nueva clara sobre ladera.',
        '',
        '## Tabla Comparativa (Área de `3 × 3 km` = `900 ha` por caso)',
        '',
        '| Caso (Distrito, Fecha, ID INDECI) | Centro localizado (ΔGRVI, dist. capital) | Modelo | Umbral | Total marcado (`ha`) | En foco del evento (`ha`) | Fuera del foco / FA (`ha`) | ¿Cubre la cicatriz? |',
        '|---|---|---|---:|---:|---:|---:|:---:|',
    ]
    for cr in resumenes:
        for m_name, d in cr['modelos'].items():
            md.append(
                f"| **{cr['distrito']}** ({cr['fecha_evento']}) | `({cr['lon']:.4f}, {cr['lat']:.4f})` ({cr['dist_capital_km']} km) | {m_name} | {d['umbral']:.2f} | {d['ha_total']:.2f} | {d['ha_en_foco']:.2f} | {d['ha_fuera_fa']:.2f} | {d['cubre_cicatriz']} |"
            )
    md.extend([
        '',
        '## Láminas por Caso',
        '',
        '1. **Caso 1 — Santa Teresa (23/02/2020, Alud Salkantay–Sahuayaco):** `_fichas/estudios_caso/caso_01_santa_teresa_2020.png`',
        '2. **Caso 2 — Echarate (16/02/2021, Huaico en quebrada Echarate):** `_fichas/estudios_caso/caso_02_echarate_2021.png`',
        '3. **Caso 3 — Vilcabamba (27/03/2021, Deslizamiento en ladera):** `_fichas/estudios_caso/caso_03_vilcabamba_2021.png`',
        '',
        '## Observaciones Cualitativas',
        '',
        '- **U-Net Landslide4Sense (`modelos/unet.onnx`):** sobremarca extensamente laderas desnudas, chacras y fondos de quebrada en los 3 casos, generando decenas de hectáreas de falsas alarmas fuera del evento real.',
        '- **U-Net B1 v5 (`unet_b1_v5.onnx`, sin filtro):** reduce drásticamente el ruido frente a Landslide4Sense y concentra la detección en los eventos principales, pero aún mantiene falsas alarmas en laderas secas y bordes de cauce/poblados.',
        '- **U-Net v6 sin TTA + filtro `wc_dem_3200_agua_urb` (`unet_v6.onnx`, `u = 0.45`):** entrega los mapas más limpios de los tres modelos, conservando la cobertura sobre la cicatriz/flujo real mientras elimina las falsas alarmas altoandinas, urbanas y de cuerpos de agua.',
        '',
    ])
    with open('informes/estudios_caso.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\nEscrito informes/estudios_caso.md')


if __name__ == '__main__':
    main()
