"""
Arma datos_locales_ajuste.zip para el cuaderno Ajuste_Fino_LaConvencion (Colab).

Mete solo lo que el ajuste fino usa: los parches de imagen de 128 px que tienen máscara, sus
máscaras de nivel 1 y, aparte, las de nivel 2 (propuestas multitemporales y geológicas
clasificadas en Tiers Oro, Plata y Bronce) para entrenamiento débil ponderado. Las de zonas de
validación también entran: ajuste_fino.py --val-provisional las usa SOLO para medir (nunca para entrenar).

  python herramientas/mascaras_desde_marcas.py     (opcional si hay marcas manuales)
  python herramientas/clasificar_tiers_y_expandir.py (extrae parches centrados y clasifica tiers)
  python herramientas/empaquetar_para_colab.py

Después se sube datos_locales_ajuste.zip a «Mi unidad» de Google Drive.
"""
import argparse
import csv
import os
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--nivel1', default=None, help='por defecto <raiz>/mascaras_nivel1')
    ap.add_argument('--salida', default=None, help='por defecto <raiz>/datos_locales_ajuste.zip')
    a = ap.parse_args()
    n1 = a.nivel1 or os.path.join(a.raiz, 'mascaras_nivel1')
    n2 = os.path.join(a.raiz, 'mascaras_provisionales', 'nivel2')
    par = os.path.join(a.raiz, 'parches_128')

    ruta_ind_amp = os.path.join(par, 'indice_parches_ampliado.csv')
    ruta_ind = ruta_ind_amp if os.path.exists(ruta_ind_amp) else os.path.join(par, 'indice_parches.csv')
    if not os.path.exists(ruta_ind):
        raise SystemExit(f'falta {ruta_ind}: corre antes cortar_parches_128.py')

    hay_csv_n1 = os.path.exists(os.path.join(n1, 'mascaras_nivel1.csv'))
    with open(ruta_ind, encoding='utf-8') as f_ind:
        indice = list(csv.DictReader(f_ind))

    con_n2 = set()
    ruta_tiers = os.path.join(n2, 'catalogo_parches_tiers.csv')
    ruta_n2_comp = os.path.join(n2, 'mascaras_nivel2_completo.csv')
    ruta_n2 = os.path.join(n2, 'mascaras_nivel2.csv')

    parches_tiers_permitidos = set()
    if os.path.exists(ruta_tiers):
        with open(ruta_tiers, encoding='utf-8') as ft:
            for r in csv.DictReader(ft):
                if float(r.get('peso_perdida', 0) or 0) > 0:
                    parches_tiers_permitidos.add((r['proyecto'], r['parche']))
                    con_n2.add((r['proyecto'], r['tesela']))
    elif os.path.exists(ruta_n2_comp):
        with open(ruta_n2_comp, encoding='utf-8') as fn2:
            con_n2 = {(r['proyecto'], r['tesela']) for r in csv.DictReader(fn2) if r.get('incluir_debil') == 'si'}
    elif os.path.exists(ruta_n2):
        with open(ruta_n2, encoding='utf-8') as fn2:
            con_n2 = {('datos_ingemmet', r['tesela']) for r in csv.DictReader(fn2) if int(r['px_1']) > 0}

    if not hay_csv_n1 and not con_n2:
        raise SystemExit(f'falta {os.path.join(n1, "mascaras_nivel1.csv")}: corre antes nutrir_mascaras.py')

    salida = a.salida or os.path.join(a.raiz, 'datos_locales_ajuste.zip')
    n = {'imagen': 0, 'nivel1': 0, 'nivel2': 0}
    faltan = []

    with zipfile.ZipFile(salida, 'w', zipfile.ZIP_STORED) as z:
        z.write(os.path.join(par, 'indice_parches.csv'), 'parches_128/indice_parches.csv')
        if os.path.exists(ruta_ind_amp):
            z.write(ruta_ind_amp, 'parches_128/indice_parches_ampliado.csv')
        if hay_csv_n1:
            z.write(os.path.join(n1, 'mascaras_nivel1.csv'), 'mascaras_nivel1/mascaras_nivel1.csv')
        if os.path.exists(ruta_n2):
            z.write(ruta_n2, 'mascaras_nivel2/mascaras_nivel2.csv')
        if os.path.exists(ruta_n2_comp):
            z.write(ruta_n2_comp, 'mascaras_nivel2/mascaras_nivel2_completo.csv')
        if os.path.exists(ruta_tiers):
            z.write(ruta_tiers, 'mascaras_nivel2/catalogo_parches_tiers.csv')

        for r in indice:
            m1 = os.path.join(n1, 'parches_128', r['proyecto'], r['parche'] + '.tif')
            m2 = os.path.join(n2, 'parches_128', r['proyecto'], r['parche'] + '.tif')
            if not os.path.exists(m2):
                m2 = os.path.join(n2, 'parches_128', r['parche'] + '.tif')

            usa1 = os.path.exists(m1)
            if parches_tiers_permitidos:
                usa2 = (
                    (r['proyecto'], r['parche']) in parches_tiers_permitidos
                    and os.path.exists(m2)
                    and r['conjunto'] in ('entrenamiento', 'validacion')
                )
            else:
                usa2 = (
                    (r['proyecto'], r['tesela']) in con_n2
                    and os.path.exists(m2)
                    and r['conjunto'] in ('entrenamiento', 'validacion')
                )

            if not (usa1 or usa2):
                continue

            img = os.path.join(par, r['archivo'])
            if not os.path.exists(img):
                faltan.append(r['archivo'])
                continue

            z.write(img, 'parches_128/' + r['archivo'])
            n['imagen'] += 1
            if usa1:
                z.write(m1, f"mascaras_nivel1/parches_128/{r['proyecto']}/{r['parche']}.tif")
                n['nivel1'] += 1
            if usa2:
                z.write(m2, f"mascaras_nivel2/parches_128/{r['proyecto']}/{r['parche']}.tif")
                n['nivel2'] += 1
                if r['proyecto'] == 'datos_ingemmet':
                    z.write(m2, f"mascaras_nivel2/parches_128/{r['parche']}.tif")

    if faltan:
        print(f'AVISO: {len(faltan)} parches de imagen no están: {faltan[:5]}')

    print(f"{salida}: {n['imagen']} parches de imagen, {n['nivel1']} máscaras de nivel 1, "
          f"{n['nivel2']} de nivel 2 (débiles ponderadas por Tiers); {os.path.getsize(salida) / 1e6:.1f} MB")
    print('Súbelo a «Mi unidad» de Google Drive y abre Ajuste_Fino_LaConvencion.ipynb en Colab.')


if __name__ == '__main__':
    main()
