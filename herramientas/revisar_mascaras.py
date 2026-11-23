"""
Control de calidad automático sobre las máscaras Nivel 1 (`mascaras_nivel1/`).

Comprueba 8 criterios de integridad espacial, física y geométrica:
  1. Máscaras vacías marcadas como positivas en CSV (`decisiones_agente.csv` o `mascaras_nivel1.csv`).
  2. Georreferencia (CRS, transformada afín o dimensiones) distinta a la del parche fuente.
  3. Manchas positivas aisladas de menos de 3 píxeles (< 0,03 ha).
  4. Positivos cuya componente conectada tiene pendiente media < 8°.
  5. Positivos sobre agua probable (B03 > B04 y pendiente < 5°) o nube blanca (B02 muy alto sin suelo claro).
  6. Positivos que tocan el borde del parche formando una línea recta (>= 15 px seguidos en el borde).
  7. Zonas presentes en más de un conjunto (`entrenamiento`, `validacion`, `prueba`).
  8. Parches con máscaras contradictorias (discrepancia entre `mascaras_nivel1.csv`,
     `decisiones_agente.csv`, `mascaras_256/` o duplicados contradictorios).
  9. IoU > 0,80 entre la máscara (positiva o dudosa) y el polígono INGEMMET rasterizado.
 10. Fracción alta de borde geométrico recto (tramos rectos largos >= 8,5 px sobre perímetro interior).
 11. Positivos con GRVI medio > 0,05 dentro de la máscara (vegetación verde marcada como suelo).
 12. Positivos que coinciden en > 50 % con un cauce o fondo de valle (alargado, baja pendiente / TPI negativo).
 13. Positivos con blanco saturado de baja textura y nubes cercanas (<= 25 px) sin confirmación multifecha.
 14. Positivos a menos de 3 px (< 30 m) de una carretera OSM y paralelos a ella (probable corte de talud).

Uso:
  python herramientas/revisar_mascaras.py --raiz .
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np


def _borde_linea_recta(comp_mask, min_run=15):
    """True si un componente positivo forma una línea recta pegada al borde >= min_run px."""
    bordes = [
        comp_mask[0, :],
        comp_mask[-1, :],
        comp_mask[:, 0],
        comp_mask[:, -1],
    ]
    for b in bordes:
        racha = 0
        for val in b:
            if val:
                racha += 1
                if racha >= min_run:
                    return True
            else:
                racha = 0
    return False


def _fraccion_bordes_rectos(comp_mask, tol=1.1, min_seg=8.5):
    """Calcula la fracción del perímetro interior formada por tramos rectos largos."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    h, w = comp_mask.shape
    pad = np.pad(comp_mask.astype(np.uint8), 1)
    tot_int = 0.0
    rect_int = 0.0
    for geom, val in shapes(pad, mask=(pad == 1)):
        if val != 1:
            continue
        poly = shape(geom).simplify(tol, preserve_topology=True)
        rings = [poly.exterior] + list(poly.interiors) if poly.geom_type == 'Polygon' else []
        for ring in rings:
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                x0i, y0i = x0 - 1.0, y0 - 1.0
                x1i, y1i = x1 - 1.0, y1 - 1.0
                en_marco = (
                    (abs(x0i) < 0.2 and abs(x1i) < 0.2)
                    or (abs(x0i - w) < 0.2 and abs(x1i - w) < 0.2)
                    or (abs(y0i) < 0.2 and abs(y1i) < 0.2)
                    or (abs(y0i - h) < 0.2 and abs(y1i - h) < 0.2)
                )
                if en_marco:
                    continue
                seg = float(np.hypot(x1i - x0i, y1i - y0i))
                tot_int += seg
                if seg >= min_seg:
                    rect_int += seg
    return (rect_int / tot_int if tot_int > 0 else 0.0), tot_int


def _cargar_ingemmet_pols(raiz, ruta_ingemmet=None):
    ruta = ruta_ingemmet or os.path.join(
        raiz, 'ingemmet', 'ingemmet_cartografia_peligros_poligonos_bbox.geojson'
    )
    if not os.path.exists(ruta):
        return []
    from shapely.geometry import shape
    gj = json.load(open(ruta, encoding='utf-8'))
    pols = []
    for f in gj.get('features', []):
        g = f.get('geometry')
        if g:
            pols.append(shape(g))
    return pols


def _cargar_carreteras(raiz, ruta_carreteras=None):
    ruta = ruta_carreteras or os.path.join(
        raiz, 'herramientas', 'carreteras_laconvencion.geojson'
    )
    if not os.path.exists(ruta):
        return None, []
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    gj = json.load(open(ruta, encoding='utf-8'))
    vias = []
    for f in gj.get('features', []):
        g = f.get('geometry')
        if g:
            s = shape(g)
            if not s.is_empty and s.length > 0:
                vias.append(s)
    if not vias:
        return None, []
    return STRtree(vias), vias


def _eje_principal(rr, cc):
    """Devuelve (eig_ratio, v_mask_xy) en metros (x=col*10, y=-row*10)."""
    if len(rr) < 4:
        return 1.0, np.array([1.0, 0.0])
    xs_m = cc.astype(np.float64) * 10.0
    ys_m = -rr.astype(np.float64) * 10.0
    if np.allclose(xs_m, xs_m[0]) and np.allclose(ys_m, ys_m[0]):
        return 1.0, np.array([1.0, 0.0])
    cov = np.cov(np.vstack([xs_m - xs_m.mean(), ys_m - ys_m.mean()]))
    evals, evecs = np.linalg.eigh(cov)
    lmax = max(float(evals[1]), 1e-6)
    lmin = max(float(evals[0]), 0.25)
    return float(np.sqrt(lmax / lmin)), evecs[:, 1]


def revisar(raiz='.', nivel1=None, parches=None, decisiones=None, particion=None,
            ingemmet_geojson=None, carreteras_geojson=None):
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    from scipy import ndimage
    from shapely.geometry import box, Point
    from shapely.ops import transform as shp_transform, unary_union
    from pyproj import Transformer

    dir_n1 = nivel1 or os.path.join(raiz, 'mascaras_nivel1')
    dir_p128 = parches or os.path.join(raiz, 'parches_128')
    ruta_dec = decisiones or os.path.join(raiz, 'informes', 'decisiones_agente.csv')
    ruta_part = particion or os.path.join(raiz, 'particion.csv')
    ruta_n1_csv = os.path.join(dir_n1, 'mascaras_nivel1.csv')
    pols_ing = _cargar_ingemmet_pols(raiz, ingemmet_geojson)
    tree_vias, vias_ll = _cargar_carreteras(raiz, carreteras_geojson)
    transf_cache = {}
    transf_ll_cache = {}

    hallazgos = []

    def add(tipo, proy, parche, tesela, quien, detalle):
        hallazgos.append({
            'tipo': tipo, 'proyecto': proy, 'parche': parche,
            'tesela': tesela, 'quien': quien, 'detalle': detalle,
        })

    # Índices auxiliares
    n1_por_tes = {}
    if os.path.exists(ruta_n1_csv):
        for r in csv.DictReader(open(ruta_n1_csv, encoding='utf-8')):
            n1_por_tes[(r['proyecto'], r['tesela'])] = r

    dec_por_parche = {}
    if os.path.exists(ruta_dec):
        for r in csv.DictReader(open(ruta_dec, encoding='utf-8')):
            dec_por_parche[r['parche']] = r

    # Criterio 7: zonas en dos conjuntos (particion.csv y mascaras_nivel1.csv)
    zonas_conj = collections.defaultdict(set)
    for ruta_z in (ruta_part, ruta_n1_csv):
        if os.path.exists(ruta_z):
            for r in csv.DictReader(open(ruta_z, encoding='utf-8')):
                c = (r.get('conjunto') or '').strip()
                z = (r.get('zona') or '').strip()
                if z and c in ('entrenamiento', 'validacion', 'prueba'):
                    zonas_conj[z].add(c)
    for z, conjs in sorted(zonas_conj.items()):
        if len(conjs) > 1:
            add('zona_en_dos_conjuntos', '', '', '', '', f'zona {z} aparece en {sorted(conjs)}')

    # Inspección por parche Nivel 1
    patron = os.path.join(dir_n1, 'parches_128', '*', '*.tif')
    archivos_m = sorted(glob.glob(patron))
    for ruta_m in archivos_m:
        proy = os.path.basename(os.path.dirname(ruta_m))
        parche = os.path.splitext(os.path.basename(ruta_m))[0]
        tesela = '_'.join(parche.split('_')[:-1]) if parche.split('_')[-1].startswith('q') else parche
        info_tes = n1_por_tes.get((proy, tesela), {})
        quien = info_tes.get('quien', '')

        with rasterio.open(ruta_m) as dm:
            mask = dm.read(1)
            m_crs, m_tr, m_shape, m_bounds = dm.crs, dm.transform, dm.shape, dm.bounds
            if not quien:
                quien = dm.tags().get('quien', '')

        ruta_img = os.path.join(dir_p128, proy, f'{parche}.tif')
        # Criterio 2: georreferencia distinta a la del parche
        b2 = b3 = b4 = pend = elev = None
        if os.path.exists(ruta_img):
            with rasterio.open(ruta_img) as di:
                i_crs, i_tr, i_shape = di.crs, di.transform, di.shape
                if di.count >= 5:
                    b2 = di.read(2)
                    b3 = di.read(3)
                    b4 = di.read(4)
                    pend = di.read(5)
                if di.count >= 6:
                    elev = di.read(6)
            tr_diff = max(abs(a - b) for a, b in zip(tuple(m_tr)[:6], tuple(i_tr)[:6]))
            if m_shape != i_shape or str(m_crs) != str(i_crs) or tr_diff > 1e-3:
                add('georreferencia_distinta', proy, parche, tesela, quien,
                    f'shape {m_shape}!={i_shape} o CRS/transform distinto')

        pos = mask == 1
        valido_b2 = (b2 > 0) if b2 is not None else np.ones_like(mask, dtype=bool)
        dud = ((mask == 2) | (mask == 255)) & valido_b2
        n_pos = int(pos.sum())
        n_dud = int(dud.sum())

        # Criterio 1 y 8: contradicción / vacía marcada positiva
        dec = dec_por_parche.get(parche)
        if dec:
            d_val = (dec.get('decision') or '').strip().lower()
            if d_val == 'positivo' and n_pos == 0:
                add('vacia_marcada_positiva', proy, parche, tesela, quien or 'AGENTE',
                    'decisiones_agente.csv dice positivo pero el TIF tiene 0 píxeles=1')
            elif d_val == 'negativo_confirmado' and n_pos > 0:
                add('mascaras_contradictorias', proy, parche, tesela, quien or 'AGENTE',
                    f'decisiones_agente.csv dice negativo_confirmado pero TIF tiene {n_pos} px=1')

        # Criterio 9: IoU > 0.80 contra polígono INGEMMET rasterizado (aplica a positivo o dudoso)
        if pols_ing and (n_pos > 0 or n_dud > 0) and m_crs is not None:
            key_crs = str(m_crs)
            if key_crs not in transf_cache:
                transf_cache[key_crs] = Transformer.from_crs('EPSG:4326', key_crs, always_xy=True).transform
            tr_utm = transf_cache[key_crs]
            b_ll = transform_bounds(m_crs, 'EPSG:4326', m_bounds.left, m_bounds.bottom, m_bounds.right, m_bounds.top)
            caja_ll = box(*b_ll)
            geoms_ing = [shp_transform(tr_utm, p) for p in pols_ing if p.intersects(caja_ll)]
            if geoms_ing:
                union_ing = rasterize([(g, 1) for g in geoms_ing], out_shape=m_shape, transform=m_tr, all_touched=True) > 0
                cands_ing = [union_ing] + [
                    rasterize([(g, 1)], out_shape=m_shape, transform=m_tr, all_touched=True) > 0
                    for g in geoms_ing
                ]
                for etq_m, arr_m in (('positivo', pos), ('dudoso', dud), ('total', pos | dud)):
                    if not arr_m.any():
                        continue
                    max_iou = max(
                        float((arr_m & g_m).sum()) / float((arr_m | g_m).sum())
                        for g_m in cands_ing if (arr_m | g_m).any()
                    )
                    if max_iou > 0.80:
                        add('iou_ingemmet_alto', proy, parche, tesela, quien,
                            f'IoU {max_iou:.3f} (> 0.80) entre mascara ({etq_m}) y poligono INGEMMET')
                        break

        # Criterio 10 sobre componentes dudosos (tramos rectos largos)
        if n_dud > 0:
            lbl_d, nd_comp = ndimage.label(dud)
            for cid in range(1, nd_comp + 1):
                comp = lbl_d == cid
                frac_r, perim_int = _fraccion_bordes_rectos(comp)
                if perim_int >= 25.0 and frac_r >= 0.55:
                    add('bordes_geometricos', proy, parche, tesela, quien,
                        f'dudoso comp {cid}: {frac_r:.0%} del perimetro interior ({perim_int:.0f} px) son tramos rectos >= 8.5 px')

        if n_pos == 0:
            continue

        # Preparar mapas auxiliares v3 por parche cuando hay positivos
        fondo_valle = None
        if pend is not None:
            if elev is not None and float(elev.max() - elev.min()) > 1.0:
                tpi21 = elev.astype(np.float32) - ndimage.uniform_filter(elev.astype(np.float32), size=21)
                fondo_valle = ((tpi21 < -1.5) & (pend <= 32.0)) | (pend < 12.0)
            else:
                fondo_valle = pend < 14.0

        dist_nube = None
        std_b2 = None
        if b2 is not None and b3 is not None and b4 is not None:
            b2f = b2.astype(np.float32)
            b3f = b3.astype(np.float32)
            b4f = b4.astype(np.float32)
            m1_f = ndimage.uniform_filter(b2f, size=3)
            m2_f = ndimage.uniform_filter(b2f * b2f, size=3)
            std_b2 = np.sqrt(np.maximum(0.0, m2_f - m1_f * m1_f))
            nube_raw = (~pos) & (b2f > 600.0) & (b4f < 1.25 * b2f) & (b3f < 1.30 * b2f)
            lbl_n, nn_comp = ndimage.label(nube_raw)
            nube_valida = np.zeros_like(nube_raw)
            for nid in range(1, nn_comp + 1):
                nc = lbl_n == nid
                if int(nc.sum()) >= 5 and float(b2f[nc].max()) > 950.0:
                    nube_valida |= nc
            if nube_valida.any():
                dist_nube = ndimage.distance_transform_edt(~nube_valida)

        vias_utm = []
        u_via = None
        if tree_vias is not None and m_crs is not None:
            key_crs = str(m_crs)
            if key_crs not in transf_cache:
                transf_cache[key_crs] = Transformer.from_crs('EPSG:4326', key_crs, always_xy=True).transform
            tr_utm = transf_cache[key_crs]
            b_ll = transform_bounds(m_crs, 'EPSG:4326', m_bounds.left, m_bounds.bottom, m_bounds.right, m_bounds.top)
            idxs_v = tree_vias.query(box(b_ll[0] - 0.002, b_ll[1] - 0.002, b_ll[2] + 0.002, b_ll[3] + 0.002))
            if len(idxs_v):
                vias_utm = [shp_transform(tr_utm, vias_ll[i]) for i in idxs_v]
                u_via = unary_union(vias_utm)

        just_txt = ((dec or {}).get('justificacion') or '').lower()
        verif_multifecha = any(k in just_txt for k in ('multifecha', 'otra fecha', 'temporada seca', 'temporadas secas'))

        # Criterio 3, 4, 6, 10, 12, 13 y 14 por componente positivo conectado
        lbl, n_comp = ndimage.label(pos)
        for cid in range(1, n_comp + 1):
            comp = lbl == cid
            sz = int(comp.sum())
            if sz < 3:
                add('mancha_menor_3px', proy, parche, tesela, quien,
                    f'componente {cid} tiene solo {sz} px (< 3)')
            elif sz < 10:
                add('mancha_menor_10px', proy, parche, tesela, quien,
                    f'componente {cid} tiene {sz} px (< 10 px minimo v4)')
            if pend is not None and sz >= 3:
                p_med = float(pend[comp].mean())
                if p_med < 8.0:
                    add('pendiente_menor_8', proy, parche, tesela, quien,
                        f'componente {cid} ({sz} px) con pendiente media {p_med:.1f}° (< 8°)')
            if _borde_linea_recta(comp):
                add('borde_linea_recta', proy, parche, tesela, quien,
                    f'componente {cid} toca el borde formando línea recta >= 15 px')
            frac_r, perim_int = _fraccion_bordes_rectos(comp)
            if perim_int >= 25.0 and frac_r >= 0.55:
                add('bordes_geometricos', proy, parche, tesela, quien,
                    f'positivo comp {cid}: {frac_r:.0%} del perimetro interior ({perim_int:.0f} px) son tramos rectos >= 8.5 px')

            rr_c, cc_c = np.where(comp)
            eig_r, v_mask = _eje_principal(rr_c, cc_c)

            # Criterio 12: >50% sobre cauce/fondo de valle alargado siguiendo su eje
            if sz >= 12 and fondo_valle is not None and eig_r >= 2.3:
                frac_v = float(fondo_valle[comp].mean())
                if frac_v > 0.50:
                    vecindad = ndimage.binary_dilation(comp, iterations=12) & fondo_valle
                    vr, vc = np.where(vecindad)
                    alineado = True
                    if len(vr) >= 15:
                        _, v_valle = _eje_principal(vr, vc)
                        alineado = abs(float(np.dot(v_mask, v_valle))) >= 0.65
                    if alineado:
                        add('positivo_en_cauce', proy, parche, tesela, quien,
                            f'positivo comp {cid} ({sz} px): {frac_v:.0%} en fondo de valle/cauce (eje={eig_r:.1f}x)')

            # Criterio 13: blanco saturado de baja textura con nubes cercanas sin verificacion multifecha
            if sz >= 10 and dist_nube is not None and not verif_multifecha:
                d_min_n = float(dist_nube[comp].min())
                if d_min_n <= 25.0:
                    px_sat = comp & (b2.astype(np.float32) > 900.0) & (b4.astype(np.float32) < 1.55 * b2.astype(np.float32)) & (std_b2 < 240.0)
                    if int(px_sat.sum()) >= 5:
                        add('blanco_saturado_cerca_nube', proy, parche, tesela, quien,
                            f'positivo comp {cid}: {int(px_sat.sum())} px blanco saturado y nube a {d_min_n:.1f} px (revisar otra fecha seca)')

            # Criterio 14: >50% a < 3 px (30 m) de carretera y paralelo a ella (< 25°)
            if sz >= 10 and u_via is not None and eig_r >= 2.2:
                paso = max(1, len(rr_c) // 150)
                pts_m = [Point(*(m_tr * (float(c0) + 0.5, float(r0) + 0.5))) for r0, c0 in zip(rr_c[::paso], cc_c[::paso])]
                frac_3px = float(np.mean([pt.distance(u_via) < 30.0 for pt in pts_m]))
                if frac_3px > 0.50:
                    cx_u, cy_u = m_tr * (float(cc_c.mean()) + 0.5, float(rr_c.mean()) + 0.5)
                    pt_c = Point(cx_u, cy_u)
                    near_v = min(vias_utm, key=lambda v: v.distance(pt_c))
                    proj = near_v.project(pt_c)
                    p1 = near_v.interpolate(max(0.0, proj - 15.0))
                    p2 = near_v.interpolate(min(near_v.length, proj + 15.0))
                    dv = np.array([p2.x - p1.x, p2.y - p1.y], dtype=np.float64)
                    nrm = float(np.linalg.norm(dv))
                    if nrm > 1e-3:
                        dv /= nrm
                        ang_deg = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(v_mask, dv))), 0.0, 1.0))))
                        if ang_deg < 25.0:
                            add('positivo_paralelo_carretera', proy, parche, tesela, quien,
                                f'positivo comp {cid}: {frac_3px:.0%} a <30m de via OSM y paralelo ({ang_deg:.1f}°)')

        # Criterio 11: GRVI medio > 0.05 dentro de la mascara positiva
        if b3 is not None and b4 is not None:
            den = b3.astype(np.float32) + b4.astype(np.float32)
            grvi = np.where(den > 0, (b3.astype(np.float32) - b4.astype(np.float32)) / np.where(den > 0, den, 1.0), 0.0)
            grvi_med = float(grvi[pos].mean())
            if grvi_med > 0.05:
                add('grvi_verde_positivo', proy, parche, tesela, quien,
                    f'GRVI medio dentro de mascara positiva es {grvi_med:.3f} (> 0.05, probable vegetacion verde)')

        # Criterio 5: positivos sobre agua o nube blanca (B02 muy alto sin suelo claro)
        if b2 is not None:
            suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
            nube = pos & (b2 > 1000) & ~suelo_claro
            agua = pos & (b3 > b4) & (b2 > 0) & (pend < 5.0)
            if int(nube.sum()) > 0:
                add('positivo_sobre_nube', proy, parche, tesela, quien,
                    f'{int(nube.sum())} px positivos con B2>1000 y sin firma de suelo claro')
            if int(agua.sum()) >= 3:
                add('positivo_sobre_agua', proy, parche, tesela, quien,
                    f'{int(agua.sum())} px positivos con B3>B4 y pendiente<5° (posible agua/cauce)')

    # Verificar coherencia global entre mascaras_nivel1.csv y parches_128/
    for (proy, tes), r in sorted(n1_por_tes.items()):
        cics_csv = int(float(r.get('cicatrices', 0) or 0))
        q_files = glob.glob(os.path.join(dir_n1, 'parches_128', proy, f'{tes}_q*.tif'))
        if q_files:
            tot_pos = 0
            for qf in q_files:
                with rasterio.open(qf) as dq:
                    tot_pos += int((dq.read(1) == 1).sum())
            if cics_csv > 0 and tot_pos == 0:
                add('vacia_marcada_positiva', proy, '', tes, r.get('quien', ''),
                    f'mascaras_nivel1.csv reporta {cics_csv} cicatrices pero parches tienen 0 px=1')
            elif cics_csv == 0 and tot_pos > 0:
                add('mascaras_contradictorias', proy, '', tes, r.get('quien', ''),
                    f'mascaras_nivel1.csv reporta 0 cicatrices pero parches tienen {tot_pos} px=1')

    return hallazgos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--salida-csv', default=None)
    a = ap.parse_args()
    hallazgos = revisar(a.raiz)
    por_tipo = collections.Counter(h['tipo'] for h in hallazgos)
    print(f'Revisión Nivel 1 completada: {len(hallazgos)} hallazgos.')
    for k, v in sorted(por_tipo.items()):
        print(f'  {k}: {v}')
    if a.salida_csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.salida_csv)), exist_ok=True)
        with open(a.salida_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['tipo', 'proyecto', 'parche', 'tesela', 'quien', 'detalle'])
            w.writeheader()
            w.writerows(hallazgos)


if __name__ == '__main__':
    main()
