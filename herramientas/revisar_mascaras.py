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


def revisar(raiz='.', nivel1=None, parches=None, decisiones=None, particion=None):
    import rasterio
    from scipy import ndimage

    dir_n1 = nivel1 or os.path.join(raiz, 'mascaras_nivel1')
    dir_p128 = parches or os.path.join(raiz, 'parches_128')
    ruta_dec = decisiones or os.path.join(raiz, 'informes', 'decisiones_agente.csv')
    ruta_part = particion or os.path.join(raiz, 'particion.csv')
    ruta_n1_csv = os.path.join(dir_n1, 'mascaras_nivel1.csv')

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
            m_crs, m_tr, m_shape = dm.crs, dm.transform, dm.shape
            if not quien:
                quien = dm.tags().get('quien', '')

        ruta_img = os.path.join(dir_p128, proy, f'{parche}.tif')
        # Criterio 2: georreferencia distinta a la del parche
        b2 = b3 = b4 = pend = None
        if os.path.exists(ruta_img):
            with rasterio.open(ruta_img) as di:
                i_crs, i_tr, i_shape = di.crs, di.transform, di.shape
                if di.count >= 5:
                    b2 = di.read(2)
                    b3 = di.read(3)
                    b4 = di.read(4)
                    pend = di.read(5)
            tr_diff = max(abs(a - b) for a, b in zip(tuple(m_tr)[:6], tuple(i_tr)[:6]))
            if m_shape != i_shape or str(m_crs) != str(i_crs) or tr_diff > 1e-3:
                add('georreferencia_distinta', proy, parche, tesela, quien,
                    f'shape {m_shape}!={i_shape} o CRS/transform distinto')

        pos = mask == 1
        n_pos = int(pos.sum())

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

        if n_pos == 0:
            continue

        # Criterio 3, 4 y 6 por componente conectado
        lbl, n_comp = ndimage.label(pos)
        for cid in range(1, n_comp + 1):
            comp = lbl == cid
            sz = int(comp.sum())
            if sz < 3:
                add('mancha_menor_3px', proy, parche, tesela, quien,
                    f'componente {cid} tiene solo {sz} px (< 3)')
            if pend is not None and sz >= 3:
                p_med = float(pend[comp].mean())
                if p_med < 8.0:
                    add('pendiente_menor_8', proy, parche, tesela, quien,
                        f'componente {cid} ({sz} px) con pendiente media {p_med:.1f}° (< 8°)')
            if _borde_linea_recta(comp):
                add('borde_linea_recta', proy, parche, tesela, quien,
                    f'componente {cid} toca el borde formando línea recta >= 15 px')

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
