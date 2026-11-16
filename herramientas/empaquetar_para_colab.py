"""
Arma datos_locales_ajuste.zip para el cuaderno Ajuste_Fino_LaConvencion (Colab).

Mete solo lo que el ajuste fino usa: los parches de imagen de 128 px que tienen máscara, sus
máscaras de nivel 1 (marcado humano) y, aparte, las de nivel 2 (propuestas de INGEMMET) de
las teselas que tienen alguna pieza aceptada, por si se activan las etiquetas débiles.

  python herramientas/mascaras_desde_marcas.py     (antes: genera mascaras_nivel1)
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
    for ruta in (os.path.join(n1, 'mascaras_nivel1.csv'), os.path.join(par, 'indice_parches.csv')):
        if not os.path.exists(ruta):
            raise SystemExit(f'falta {ruta}: corre antes mascaras_desde_marcas.py y cortar_parches_128.py')
    indice = list(csv.DictReader(open(os.path.join(par, 'indice_parches.csv'), encoding='utf-8')))
    con_n2 = set()
    ruta_n2 = os.path.join(n2, 'mascaras_nivel2.csv')
    if os.path.exists(ruta_n2):
        con_n2 = {r['tesela'] for r in csv.DictReader(open(ruta_n2, encoding='utf-8')) if int(r['px_1']) > 0}
    salida = a.salida or os.path.join(a.raiz, 'datos_locales_ajuste.zip')
    n = {'imagen': 0, 'nivel1': 0, 'nivel2': 0}
    faltan = []
    with zipfile.ZipFile(salida, 'w', zipfile.ZIP_STORED) as z:
        z.write(os.path.join(par, 'indice_parches.csv'), 'parches_128/indice_parches.csv')
        z.write(os.path.join(n1, 'mascaras_nivel1.csv'), 'mascaras_nivel1/mascaras_nivel1.csv')
        if con_n2:
            z.write(ruta_n2, 'mascaras_nivel2/mascaras_nivel2.csv')
        for r in indice:
            m1 = os.path.join(n1, 'parches_128', r['proyecto'], r['parche'] + '.tif')
            m2 = os.path.join(n2, 'parches_128', r['parche'] + '.tif')
            usa1 = os.path.exists(m1)
            usa2 = (r['proyecto'] == 'datos_ingemmet' and r['tesela'] in con_n2 and os.path.exists(m2)
                    and r['conjunto'] == 'entrenamiento')
            if not (usa1 or usa2):
                continue
            img = os.path.join(par, r['archivo'])
            if not os.path.exists(img):
                faltan.append(r['archivo'])
                continue
            z.write(img, 'parches_128/' + r['archivo']); n['imagen'] += 1
            if usa1:
                z.write(m1, f"mascaras_nivel1/parches_128/{r['proyecto']}/{r['parche']}.tif"); n['nivel1'] += 1
            if usa2:
                z.write(m2, f"mascaras_nivel2/parches_128/{r['parche']}.tif"); n['nivel2'] += 1
    if faltan:
        print(f'AVISO: {len(faltan)} parches de imagen no están (vuelve a correr cortar_parches_128.py): {faltan[:5]}')
    print(f"{salida}: {n['imagen']} parches de imagen, {n['nivel1']} máscaras de nivel 1, "
          f"{n['nivel2']} de nivel 2 (débiles, opcionales); {os.path.getsize(salida) / 1e6:.1f} MB")
    print('Súbelo a «Mi unidad» de Google Drive y abre Ajuste_Fino_LaConvencion.ipynb en Colab.')


if __name__ == '__main__':
    main()
