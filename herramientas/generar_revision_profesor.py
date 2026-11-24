"""Genera el PDF ciego de revision humana (informes/revision_profesor.pdf) y su clave aparte
(informes/revision_profesor_clave.csv) en formato A4 apaisado:
  - Pagina 1: instrucciones (5 lineas), Nombre/Fecha, y 2 ejemplos visuales (1 SI deslizamiento, 1 NO rio/falso).
  - Paginas 2..25: 2 parches por pagina, cada vista >= 6x6 cm, grilla tenue cada 10 px y barra de escala de 200 m.
  - Pagina final: resumen para anotar el total de Si / No / Dudoso.
"""
import concurrent.futures
import csv
import io
import math
import os
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from scipy import ndimage


SEED = 20261124


def latlon_to_tile(lat, lon, z):
    lat_r = math.radians(lat)
    n = 2.0 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return xt, yt


def tile_to_quadkey(tx, ty, z):
    qk = []
    for i in range(z, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if (tx & m) != 0:
            d += 1
        if (ty & m) != 0:
            d += 2
        qk.append(str(d))
    return ''.join(qk)


def fetch_bing_patch(bounds_utm, crs_utm='EPSG:32718', z=16, out_px=256):
    """Descarga mosaico Bing Aerial (VirtualEarth) para el bbox UTM exacto del parche."""
    tr = Transformer.from_crs(crs_utm, 'EPSG:4326', always_xy=True)
    lon_w, lat_s = tr.transform(bounds_utm.left, bounds_utm.bottom)
    lon_e, lat_n = tr.transform(bounds_utm.right, bounds_utm.top)
    x0, y0 = latlon_to_tile(lat_n, lon_w, z)
    x1, y1 = latlon_to_tile(lat_s, lon_e, z)
    tx0, tx1 = int(math.floor(x0)), int(math.floor(x1))
    ty0, ty1 = int(math.floor(y0)), int(math.floor(y1))
    canvas = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (220, 220, 220))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            qk = tile_to_quadkey(tx, ty, z)
            url = f'https://ecn.t{(tx + ty) % 4}.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    im = Image.open(io.BytesIO(resp.read())).convert('RGB')
                canvas.paste(im, ((tx - tx0) * 256, (ty - ty0) * 256))
            except Exception:
                pass
    px0 = int(round((x0 - tx0) * 256))
    py0 = int(round((y0 - ty0) * 256))
    px1 = max(px0 + 1, int(round((x1 - tx0) * 256)))
    py1 = max(py0 + 1, int(round((y1 - ty0) * 256)))
    crop = canvas.crop((px0, py0, px1, py1)).resize((out_px, out_px), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(crop, dtype=np.float32) / 255.0, 0.0, 1.0)


def s2_rgb(ruta_tif):
    with rasterio.open(ruta_tif) as ds:
        b2 = ds.read(2).astype(np.float32)
        b3 = ds.read(3).astype(np.float32)
        b4 = ds.read(4).astype(np.float32)
        bounds = ds.bounds
        crs = ds.crs
    rgb = np.stack([b4, b3, b2], axis=-1)
    valid = rgb > 0
    if valid.any():
        p2, p98 = np.percentile(rgb[valid], [2, 98])
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return rgb, bounds, crs


def s2_antes_rgb(r):
    """Lee la escena anterior (2018-2020) desde <proyecto>_antes/images/<tesela>/image.tif (o .tif directo)."""
    cands = [
        os.path.join(r['proyecto'] + '_antes', 'images', r['tesela'], 'image.tif'),
        os.path.join(r['proyecto'] + '_antes', r['tesela'] + '.tif'),
        os.path.join('parches_128', 'antes_2018_2020', r['proyecto'], r['parche'] + '.tif'),
    ]
    p_ant = next((p for p in cands if os.path.exists(p)), None)
    if p_ant is None:
        return None
    with rasterio.open(p_ant) as ds:
        if ds.height == 256 and ds.width == 256:
            q = r['parche'].rsplit('_', 1)[-1]
            r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
            a = ds.read()[:, r0:r0+128, c0:c0+128].astype(np.float32)
            rgb = np.stack([a[3], a[2], a[1]], axis=-1)
        else:
            a = ds.read().astype(np.float32)
            rgb = np.stack([a[3], a[2], a[1]], axis=-1) if a.shape[0] >= 4 else np.stack([a[2], a[1], a[0]], axis=-1)
    valid = np.isfinite(rgb) & (rgb > 0)
    if valid.mean() < 0.05:
        return None
    p2, p98 = np.percentile(rgb[valid], [2, 98])
    return np.clip(np.nan_to_num((rgb - p2) / max(p98 - p2, 1.0)), 0.0, 1.0)


def contorno_amarillo(rgb, mk):
    out = rgb.copy()
    m = (mk == 1)
    if m.any():
        st8 = np.ones((3, 3), dtype=bool)
        borde = ndimage.binary_dilation(m, structure=st8) & (~m)
        out[borde] = np.array([1.0, 0.92, 0.05], dtype=np.float32)
        out[m] = np.clip(out[m] * 0.82 + np.array([1.0, 0.92, 0.05], dtype=np.float32) * 0.18, 0.0, 1.0)
    return out


def decorar_vista(ax, img_shape):
    """Dibuja grilla tenue cada 10 px (100 m) y barra de escala de 200 m (20 px) en la vista."""
    H, W = img_shape[:2]
    # Escala: 128 px = 1280 m (si es Bing 256 px, 10 px S2 = 20 px Bing; 200 m = W * 20/128)
    step = W * (10.0 / 128.0)
    for pos in np.arange(step, W, step):
        ax.axvline(pos, color='white', alpha=0.25, lw=0.4)
        ax.axhline(pos, color='white', alpha=0.25, lw=0.4)
    # Barra de escala de 200 m en esquina inferior izquierda
    bar_len = W * (20.0 / 128.0)
    x0 = W * 0.05
    y0 = H * 0.92
    ax.plot([x0, x0 + bar_len], [y0, y0], color='black', lw=3.2, solid_capstyle='butt')
    ax.plot([x0, x0 + bar_len], [y0, y0], color='white', lw=1.6, solid_capstyle='butt')
    ax.text(x0 + bar_len / 2.0, y0 - H * 0.035, '200 m', color='white', fontsize=6.5, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.12', facecolor='black', alpha=0.55, edgecolor='none'))


def main():
    rng = np.random.default_rng(SEED)
    st8 = np.ones((3, 3), dtype=bool)
    dec = {r['parche']: r for r in csv.DictReader(open('informes/decisiones_agente.csv', encoding='utf-8'))}
    ind = [r for r in csv.DictReader(open('parches_128/indice_parches_ampliado.csv', encoding='utf-8'))
           if r['conjunto'] == 'prueba']

    pos_rows = []
    neg_rows = []
    for r in ind:
        m1 = os.path.join('mascaras_nivel1', 'parches_128', r['proyecto'], r['parche'] + '.tif')
        if not os.path.exists(m1):
            continue
        mk = rasterio.open(m1).read(1)
        if (mk == 1).any():
            l, nc = ndimage.label(mk == 1, structure=st8)
            max_sz = max(int((l == cid).sum()) for cid in range(1, nc + 1))
            bk = '10-19' if max_sz < 20 else ('20-50' if max_sz <= 50 else ('50-200' if max_sz <= 200 else '>200'))
            pos_rows.append(dict(r, _m1=m1, es_positivo=1, estrato=bk, px_pos=int((mk == 1).sum()), n_cic=nc))
        elif dec.get(r['parche'], {}).get('decision') == 'negativo_confirmado':
            neg_rows.append(dict(r, _m1=m1, es_positivo=0, estrato='negativo_control', px_pos=0, n_cic=0))

    sel_pos = pos_rows[:50] if len(pos_rows) <= 50 else list(rng.choice(pos_rows, 50, replace=False))
    idx_neg = rng.choice(len(neg_rows), min(10, len(neg_rows)), replace=False)
    sel_neg = [neg_rows[i] for i in idx_neg]
    muestra = list(sel_pos) + list(sel_neg)
    perm = rng.permutation(len(muestra))
    muestra = [muestra[i] for i in perm]

    print(f'Muestra ciega: {len(sel_pos)} positivos (100% de prueba 360) + {len(sel_neg)} negativos = {len(muestra)} parches', flush=True)

    # Cargar ejemplos de portada (fuera de prueba 360: 1 deslizamiento claro + 1 falso rio)
    ex_si_tif = 'parches_128/datos_nuevos_ent/ZN33_PICHARI_006_q1.tif'
    ex_si_m1 = 'mascaras_nivel1/parches_128/datos_nuevos_ent/ZN33_PICHARI_006_q1.tif'
    ex_no_tif = 'parches_128/datos_prueba2/ZP2_06_ECHARATE_001_q1.tif'
    ex_si_rgb, ex_si_b, ex_si_crs = s2_rgb(ex_si_tif)
    ex_si_msk = contorno_amarillo(ex_si_rgb, rasterio.open(ex_si_m1).read(1))
    ex_si_bing = fetch_bing_patch(ex_si_b, str(ex_si_crs), z=16, out_px=256)
    ex_no_rgb, ex_no_b, ex_no_crs = s2_rgb(ex_no_tif)
    ex_no_bing = fetch_bing_patch(ex_no_b, str(ex_no_crs), z=16, out_px=256)

    def load_item(item):
        rgb_post, bounds, crs = s2_rgb(os.path.join('parches_128', item['archivo']))
        rgb_ant = s2_antes_rgb(item)
        mk = rasterio.open(item['_m1']).read(1)
        rgb_mask = contorno_amarillo(rgb_post, mk)
        rgb_bing = fetch_bing_patch(bounds, str(crs), z=16, out_px=256)
        return (rgb_post, rgb_ant, rgb_bing, rgb_mask)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(load_item, muestra))

    con_ant = sum(1 for x in imgs if x[1] is not None)
    print(f'Parches con imagen anterior despejada: {con_ant}/{len(imgs)}', flush=True)

    # Guardar clave aparte
    clave_path = 'informes/revision_profesor_clave.csv'
    with open(clave_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['numero', 'proyecto', 'parche', 'tesela', 'zona', 'es_positivo', 'estrato_tamano', 'px_positivos', 'n_cicatrices'])
        w.writeheader()
        for k, it in enumerate(muestra, 1):
            w.writerow({
                'numero': f'{k:02d}', 'proyecto': it['proyecto'], 'parche': it['parche'],
                'tesela': it['tesela'], 'zona': it['zona'], 'es_positivo': it['es_positivo'],
                'estrato_tamano': it['estrato'], 'px_positivos': it['px_pos'], 'n_cicatrices': it['n_cic'],
            })

    # PDF A4 apaisado (29.7 x 21.0 cm = 11.69 x 8.27 in)
    # 2 parches por pagina, cada vista = 6.1 x 6.1 cm (2.40 x 2.40 in)
    pdf_path = 'informes/revision_profesor.pdf'
    FIG_W, FIG_H = 11.69, 8.27
    with PdfPages(pdf_path) as pdf:
        # ---------------- PAGINA 1: Portada, instrucciones 5 lineas, Nombre/Fecha y 2 ejemplos visuales ----------------
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
        fig.patch.set_facecolor('white')
        fig.text(0.05, 0.94, 'Revisión Ciega de Máscaras de Deslizamiento — Provincia de La Convención (Cusco)',
                 fontsize=15.5, fontweight='bold', color='#111111', va='top')
        fig.text(0.05, 0.885, f'Muestra ciega de {len(muestra)} parches (1,28 × 1,28 km; escala 200 m y grilla cada 100 m) — numerados #01 al #{len(muestra):02d}',
                 fontsize=10.5, color='#333333', va='top')

        # Caja de instrucciones (5 lineas exactas)
        ax_box = fig.add_axes([0.05, 0.56, 0.90, 0.28])
        ax_box.set_xlim(0, 1); ax_box.set_ylim(0, 1); ax_box.axis('off')
        ax_box.add_patch(patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.02',
                                                facecolor='#F4F7FA', edgecolor='#90A4AE', lw=1.0))
        instr_lines = [
            'INSTRUCCIONES (tiempo estimado: ~20 minutos):',
            '1. Qué SÍ es deslizamiento: cicatriz nueva de suelo desnudo (pardo, gris o claro) en ladera, alargada pendiente abajo, ausente en la imagen anterior.',
            '2. Qué NO es deslizamiento: carreteras o taludes viales, cauces u orillas de ríos, chacras/pastizales/quemas, techos/poblados ni nubes o sombras.',
            '3. Compare las 4 vistas por fila (Sentinel-2 2023–2025, Sentinel-2 anterior 2018–2020, Bing aéreo y contorno amarillo) y marque: Sí, No o Dudoso.',
            '4. La muestra mezcla parches marcados y controles negativos sin identificarlos; evalúe cada fila por separado y anote el resumen en la última página.',
        ]
        for idx_l, txt in enumerate(instr_lines):
            ax_box.text(0.02, 0.88 - idx_l * 0.19, txt, fontsize=9.5,
                        fontweight='bold' if idx_l == 0 else 'normal', color='#1A1A1A', va='top')

        # Campos Nombre y Fecha
        fig.text(0.05, 0.515, 'Nombre del revisor: ________________________________________________      Fecha: ________________________',
                 fontsize=10.5, fontweight='bold', color='#111111', va='top')

        # Dos ejemplos visuales (SI deslizamiento claro / NO falso rio)
        fig.text(0.05, 0.44, 'Ejemplo SÍ — Deslizamiento claro (cicatriz en ladera)',
                 fontsize=9.5, fontweight='bold', color='#1B5E20', va='top')
        fig.text(0.53, 0.44, 'Ejemplo NO — Falso (río / cauce, NO es deslizamiento)',
                 fontsize=9.5, fontweight='bold', color='#B71C1C', va='top')

        ex_panels = [
            (0.05, ex_si_msk, 'S2 + contorno amarillo (SÍ)'),
            (0.24, ex_si_bing, 'Bing aéreo (SÍ)'),
            (0.53, ex_no_rgb, 'Sentinel-2 río (NO)'),
            (0.72, ex_no_bing, 'Bing aéreo río (NO)'),
        ]
        for x0, im_e, tit_e in ex_panels:
            axe = fig.add_axes([x0, 0.08, 0.175, 0.30])
            axe.imshow(im_e)
            decorar_vista(axe, im_e.shape)
            axe.set_title(tit_e, fontsize=8.5, fontweight='bold', pad=4)
            axe.set_xticks([]); axe.set_yticks([])

        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- PAGINAS 2..25: 2 parches por pagina, cada vista = 6.1 x 6.1 cm ----------------
        per_page = 2
        n_pages = math.ceil(len(muestra) / per_page)
        # 6.1 cm = 2.40 in -> w_norm = 2.40 / 11.69 = 0.2053, h_norm = 2.40 / 8.27 = 0.2902
        W_IMG = 2.40 / FIG_W
        H_IMG = 2.40 / FIG_H
        XS = [0.025, 0.236, 0.447, 0.658]
        YS = [0.56, 0.12]
        col_titles = [
            'Sentinel-2 (2023–2025)',
            'Sentinel-2 anterior (2018–2020)',
            'Bing aéreo',
            'Contorno máscara (amarillo)',
        ]

        for pg in range(n_pages):
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
            fig.patch.set_facecolor('white')
            fig.text(0.5, 0.965, f'Revisión Ciega de Deslizamientos — Hoja {pg + 1} de {n_pages}  (grilla cada 10 px = 100 m · barra = 200 m)',
                     ha='center', va='top', fontsize=10, fontweight='bold', color='#222222')
            sub = muestra[pg * per_page:(pg + 1) * per_page]
            sub_imgs = imgs[pg * per_page:(pg + 1) * per_page]
            for r_i, (it, (r_post, r_ant, r_bing, r_msk)) in enumerate(zip(sub, sub_imgs)):
                num = pg * per_page + r_i + 1
                y_bot = YS[r_i]
                paneles = [r_post, r_ant, r_bing, r_msk]
                for c_i, (x0, p_img) in enumerate(zip(XS, paneles)):
                    ax = fig.add_axes([x0, y_bot, W_IMG, H_IMG])
                    if p_img is not None:
                        ax.imshow(p_img)
                        decorar_vista(ax, p_img.shape)
                    else:
                        ax.set_facecolor('#EEEEEE')
                        ax.text(0.5, 0.5, 'sin escena\ndespejada', ha='center', va='center', fontsize=9, color='#555555')
                    ax.set_title(col_titles[c_i], fontsize=8.5, fontweight='bold', pad=4)
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_edgecolor('#555555'); sp.set_linewidth(0.8)

                # Panel derecho de casillas
                ax_r = fig.add_axes([0.872, y_bot, 0.115, H_IMG])
                ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis('off')
                ax_r.add_patch(patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.02',
                                                      facecolor='#FAFAFA', edgecolor='#90A4AE', lw=0.9))
                ax_r.text(0.06, 0.94, f'Parche #{num:02d}', fontsize=10.5, fontweight='bold', va='top', color='#111111')
                for idx_opt, opt in enumerate(['Sí es\ndeslizamiento', 'No', 'Dudoso']):
                    yo = 0.74 - idx_opt * 0.17
                    ax_r.add_patch(patches.Rectangle((0.06, yo - 0.03), 0.12, 0.07, fill=False, edgecolor='#222222', lw=1.0))
                    ax_r.text(0.23, yo + 0.005, opt, fontsize=8.0, va='center', color='#111111')
                ax_r.text(0.06, 0.27, 'Comentario:', fontsize=7.5, fontweight='bold', color='#444444', va='top')
                for yl in (0.18, 0.11, 0.04):
                    ax_r.plot([0.06, 0.94], [yl, yl], color='#999999', lw=0.6)
            pdf.savefig(fig)
            plt.close(fig)

        # ---------------- PAGINA FINAL: Resumen para el revisor ----------------
        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
        fig.patch.set_facecolor('white')
        fig.text(0.06, 0.92, 'Página de Resumen — Totales de la Revisión Ciega',
                 fontsize=15, fontweight='bold', color='#111111', va='top')
        fig.text(0.06, 0.85, f'Al terminar los {len(muestra)} parches (#01 al #{len(muestra):02d}), anote el conteo total de sus marcas:',
                 fontsize=11, color='#333333', va='top')
        ax_t = fig.add_axes([0.06, 0.45, 0.55, 0.34])
        ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1); ax_t.axis('off')
        filas_res = [
            ('Categoría marcada', 'Cantidad de parches'),
            ('Sí es deslizamiento', ''),
            ('No es deslizamiento', ''),
            ('Dudoso', ''),
            (f'TOTAL (debe sumar {len(muestra)})', ''),
        ]
        for idx_f, (c1, c2) in enumerate(filas_res):
            y0 = 0.80 - idx_f * 0.19
            bg = '#E3EAF2' if idx_f == 0 else ('#F5F5F5' if idx_f == 4 else 'white')
            ax_t.add_patch(patches.Rectangle((0.0, y0), 0.65, 0.19, facecolor=bg, edgecolor='#666666', lw=0.9))
            ax_t.add_patch(patches.Rectangle((0.65, y0), 0.35, 0.19, facecolor=bg, edgecolor='#666666', lw=0.9))
            ax_t.text(0.03, y0 + 0.095, c1, fontsize=10.5, fontweight='bold' if idx_f in (0, 4) else 'normal', va='center')
            ax_t.text(0.825, y0 + 0.095, c2, fontsize=10.5, fontweight='bold', ha='center', va='center')

        fig.text(0.06, 0.38, 'Comentarios generales sobre criterios o patrones observados:', fontsize=11, fontweight='bold', va='top')
        ax_c = fig.add_axes([0.06, 0.10, 0.88, 0.25])
        ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis('off')
        for yl in (0.85, 0.65, 0.45, 0.25, 0.05):
            ax_c.plot([0.0, 1.0], [yl, yl], color='#888888', lw=0.8)
        pdf.savefig(fig)
        plt.close(fig)

    print(f'PDF generado: {pdf_path} ({os.path.getsize(pdf_path)/1024/1024:.2f} MB) | {n_pages + 2} paginas | Clave: {clave_path}')

    # Renderizar capturas de verificacion con pypdfium2 (pag 1, pag 2 y ultima)
    import pypdfium2 as pdfium
    cap_dir = '_fichas/revision_profesor_capturas'
    os.makedirs(cap_dir, exist_ok=True)
    doc = pdfium.PdfDocument(pdf_path)
    for pg_idx, name in [(0, 'pagina_01_portada.png'), (1, 'pagina_02_parches.png'), (len(doc) - 1, 'pagina_final_resumen.png')]:
        page = doc[pg_idx]
        bm = page.render(scale=2.0)
        bm.to_pil().save(os.path.join(cap_dir, name))
    print('Capturas PDF renderizadas en', cap_dir)


if __name__ == '__main__':
    main()
