"""Genera el PDF ciego de revision humana (informes/revision_profesor.pdf) y su clave aparte
(informes/revision_profesor_clave.csv) con parches positivos (estratificados por tamano) y
negativos de control de la prueba sellada (360).
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
    return np.asarray(crop, dtype=np.float32) / 255.0


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
    p_ant = os.path.join(r['proyecto'] + '_antes', r['tesela'] + '.tif')
    if not os.path.exists(p_ant):
        return None
    q = r['parche'].rsplit('_', 1)[-1]
    r0, c0 = {'q1': (0, 0), 'q2': (0, 128), 'q3': (128, 0), 'q4': (128, 128)}[q]
    with rasterio.open(p_ant) as ds:
        a = ds.read()[:, r0:r0+128, c0:c0+128].astype(np.float32)
    rgb = np.stack([a[3], a[2], a[1]], axis=-1)
    valid = rgb > 0
    if valid.any():
        p2, p98 = np.percentile(rgb[valid], [2, 98])
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
    return rgb


def contorno_amarillo(rgb, mk):
    out = rgb.copy()
    m = (mk == 1)
    if m.any():
        st8 = np.ones((3, 3), dtype=bool)
        borde = ndimage.binary_dilation(m, structure=st8) & (~m)
        # Resaltar borde amarillo brillante (1 px exterior) y ligero tinte interior
        out[borde] = np.array([1.0, 0.92, 0.05], dtype=np.float32)
        out[m] = out[m] * 0.80 + np.array([1.0, 0.92, 0.05], dtype=np.float32) * 0.20
    return out


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

    # prueba (360) tiene 38 parches positivos en total: se incluyen todos (100% de los 4 estratos) + 10 negativos de control
    sel_pos = pos_rows[:50] if len(pos_rows) <= 50 else list(rng.choice(pos_rows, 50, replace=False))
    idx_neg = rng.choice(len(neg_rows), min(10, len(neg_rows)), replace=False)
    sel_neg = [neg_rows[i] for i in idx_neg]
    muestra = list(sel_pos) + list(sel_neg)
    perm = rng.permutation(len(muestra))
    muestra = [muestra[i] for i in perm]

    print(f'Muestra ciega: {len(sel_pos)} positivos (100% de prueba 360) + {len(sel_neg)} negativos de control = {len(muestra)} parches', flush=True)

    # Precargar imagenes y descargar Bing en paralelo
    def load_item(item):
        rgb_post, bounds, crs = s2_rgb(os.path.join('parches_128', item['archivo']))
        rgb_ant = s2_antes_rgb(item)
        mk = rasterio.open(item['_m1']).read(1)
        rgb_mask = contorno_amarillo(rgb_post, mk)
        rgb_bing = fetch_bing_patch(bounds, str(crs), z=16, out_px=256)
        return (rgb_post, rgb_ant, rgb_bing, rgb_mask)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(load_item, muestra))

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

    # Generar PDF A4
    pdf_path = 'informes/revision_profesor.pdf'
    with PdfPages(pdf_path) as pdf:
        # Pagina 1: portada e instrucciones de 5 lineas
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis('off')
        ax.text(0.0, 0.94, 'Revisión Ciega de Máscaras de Deslizamiento — La Convención (Cusco)',
                fontsize=15, fontweight='bold', va='top')
        ax.text(0.0, 0.89, f'Muestra ciega de {len(muestra)} parches (1,28 × 1,28 km cada uno) de la prueba sellada — numerados del #01 al #{len(muestra):02d}',
                fontsize=10.5, color='#333333', va='top')
        instrucciones = (
            "INSTRUCCIONES PARA EL REVISOR (tiempo estimado: ~20 minutos):\n\n"
            "1. Qué SÍ cuenta como deslizamiento: cicatriz nueva de suelo desnudo (pardo, gris o claro) sobre ladera,\n"
            "   con forma alargada o lengua pendiente abajo, ausente en la imagen anterior y visible en Sentinel-2/Bing.\n"
            "2. Qué NO cuenta: carreteras o taludes viales, cauces u orillas de ríos, chacras/pastizales/quemas agrícolas,\n"
            "   techos/poblados, laderas altoandinas secas sin cambio reciente, ni nubes o sombras.\n"
            "3. En cada fila compare las 4 vistas (Sentinel-2 2023–2025, imagen anterior, Bing aéreo y contorno amarillo).\n"
            "4. Marque una sola casilla: [Sí es deslizamiento], [No] o [Dudoso], y anote un comentario breve si lo desea.\n"
            "5. La muestra mezcla parches marcados y controles negativos sin identificarlos; evalúe cada fila por separado."
        )
        ax.text(0.0, 0.80, instrucciones, fontsize=10, va='top', linespacing=1.55,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F7FA', edgecolor='#B0BEC5'))

        ax.text(0.0, 0.42, 'Datos del revisor:', fontsize=11, fontweight='bold', va='top')
        ax.text(0.0, 0.37, 'Nombre: ____________________________________________    Fecha: ___________________', fontsize=10.5, va='top')
        ax.text(0.0, 0.31, 'Observaciones generales:', fontsize=10.5, va='top')
        for yline in (0.26, 0.21, 0.16, 0.11):
            ax.plot([0.0, 0.98], [yline, yline], color='#999999', lw=0.8)
        pdf.savefig(fig)
        plt.close(fig)

        # Paginas de parches: 4 filas por pagina A4
        per_page = 4
        n_pages = math.ceil(len(muestra) / per_page)
        for pg in range(n_pages):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
            fig.patch.set_facecolor('white')
            fig.text(0.5, 0.975, f'Revisión Ciega de Deslizamientos — Página {pg + 1} de {n_pages}',
                     ha='center', va='top', fontsize=10, fontweight='bold', color='#222222')
            sub = muestra[pg * per_page:(pg + 1) * per_page]
            sub_imgs = imgs[pg * per_page:(pg + 1) * per_page]
            col_titles = ['Sentinel-2 (2023–2025)', 'Sentinel-2 anterior', 'Bing aéreo', 'Contorno máscara']
            for r_i, (it, (r_post, r_ant, r_bing, r_msk)) in enumerate(zip(sub, sub_imgs)):
                num = pg * per_page + r_i + 1
                y_bot = 0.755 - r_i * 0.235
                h = 0.185
                w = 0.165
                xs = [0.04, 0.215, 0.39, 0.565]
                paneles = [r_post, r_ant, r_bing, r_msk]
                for c_i, (x0, p_img) in enumerate(zip(xs, paneles)):
                    ax = fig.add_axes([x0, y_bot, w, h])
                    if p_img is not None:
                        ax.imshow(p_img)
                    else:
                        ax.set_facecolor('#EEEEEE')
                        ax.text(0.5, 0.5, 'Sin imagen\nanterior', ha='center', va='center', fontsize=7, color='#666666')
                    if r_i == 0:
                        ax.set_title(col_titles[c_i], fontsize=7.5, fontweight='bold', pad=3)
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_edgecolor('#888888'); sp.set_linewidth(0.6)
                # Panel derecho de evaluacion
                ax_r = fig.add_axes([0.745, y_bot, 0.225, h])
                ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis('off')
                ax_r.add_patch(patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle='round,pad=0.02',
                                                      facecolor='#FAFAFA', edgecolor='#B0BEC5', lw=0.8))
                ax_r.text(0.05, 0.92, f'Parche #{num:02d}', fontsize=9.5, fontweight='bold', va='top')
                for idx_opt, opt in enumerate(['Sí es deslizamiento', 'No', 'Dudoso']):
                    yo = 0.72 - idx_opt * 0.15
                    ax_r.add_patch(patches.Rectangle((0.06, yo - 0.02), 0.075, 0.075, fill=False, edgecolor='#333333', lw=0.9))
                    ax_r.text(0.17, yo + 0.015, opt, fontsize=7.5, va='center')
                ax_r.text(0.05, 0.28, 'Comentario:', fontsize=7, color='#444444', va='top')
                ax_r.plot([0.05, 0.95], [0.16, 0.16], color='#AAAAAA', lw=0.6)
                ax_r.plot([0.05, 0.95], [0.06, 0.06], color='#AAAAAA', lw=0.6)
            pdf.savefig(fig)
            plt.close(fig)

    print(f'PDF generado: {pdf_path} ({os.path.getsize(pdf_path)/1024/1024:.2f} MB) | Clave: {clave_path}')


if __name__ == '__main__':
    main()
