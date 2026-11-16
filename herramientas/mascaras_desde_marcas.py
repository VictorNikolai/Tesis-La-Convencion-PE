"""
Máscaras de NIVEL 1: las que ustedes marcaron y confirmaron en QGIS. Son la verdad de terreno.

Lee la capa «marcas» exportada como GeoJSON y los revisadas.csv de cada carpeta. Solo se
genera máscara de las teselas que tienen «si» en la columna revisada: una tesela revisada
sin polígonos es un NEGATIVO confirmado (máscara toda en cero) y vale tanto como una con
cicatrices.

Valores (uint8, misma malla que la tesela):
    0    fondo confirmado
    1    deslizamiento
    255  ignorar: polígonos «dudoso», nube blanca o sin dato (no cuentan ni para entrenar
         ni para medir). La regla de nube (B2 > 1000) no se aplica dentro de un polígono
         de deslizamiento: el suelo desnudo muy claro también supera ese valor

Si un polígono de deslizamiento lleva «2025» en el campo nota (la cicatriz solo se ve en
la imagen de 2025), en la imagen base queda como 255 y se escribe además una máscara para
la tesela de datos_laconvencion_2025, donde ese polígono vale 1 y los demás 255.

  1. En QGIS: clic derecho en «marcas» > Exportar > GeoJSON, EPSG:4326,
     guardado como datos_laconvencion/marcas.geojson
  2. python herramientas/mascaras_desde_marcas.py
  3. python herramientas/cortar_parches_128.py     (si aún no están los parches de imagen)

Escribe en mascaras_nivel1/:
  mascaras_256/<proyecto>/<tesela>.tif    parches_128/<proyecto>/<tesela>_qN.tif
  mascaras_nivel1.csv                     resumen por tesela, con zona y conjunto
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cortar_parches_128 import escribir_parches  # noqa: E402

BASES = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']
OTRA_2025 = {'datos_laconvencion': 'datos_laconvencion_2025'}
SI = {'si', 'sí', 's', 'x', '1', 'yes'}


def crs_de(gj):
    nombre = ((gj.get('crs') or {}).get('properties') or {}).get('name', '')
    if 'EPSG' in nombre:
        cod = nombre.replace('::', ':').split(':')[-1]
        return 'EPSG:4326' if cod in ('CRS84', '4326') else f'EPSG:{cod}'
    return 'EPSG:4326'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--marcas', default=None, help='por defecto <raiz>/datos_laconvencion/marcas.geojson')
    ap.add_argument('--salida', default=None, help='por defecto <raiz>/mascaras_nivel1')
    a = ap.parse_args()
    import rasterio
    from rasterio.features import rasterize
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_geom
    from scipy import ndimage

    ruta_marcas = a.marcas or os.path.join(a.raiz, 'datos_laconvencion', 'marcas.geojson')
    if not os.path.exists(ruta_marcas):
        raise SystemExit(f'no está {ruta_marcas}: exporta la capa «marcas» de QGIS como GeoJSON')
    gj = json.load(open(ruta_marcas, encoding='utf-8'))
    crs_marcas = crs_de(gj)
    marcas, raras = [], collections.Counter()
    for f in gj.get('features', []):
        if not f.get('geometry'):
            continue
        p = f.get('properties') or {}
        clase = (p.get('clase') or '').strip().lower()
        if clase not in ('deslizamiento', 'dudoso'):
            raras[clase or '(vacía)'] += 1
            continue
        marcas.append({'geom': f['geometry'], 'clase': clase, 'nota': str(p.get('nota') or ''), 'usada': False})
    print(f'{len(marcas)} polígonos: {sum(m["clase"] == "deslizamiento" for m in marcas)} deslizamiento, '
          f'{sum(m["clase"] == "dudoso" for m in marcas)} dudoso')
    if raras:
        print(f'  AVISO: {sum(raras.values())} polígonos con clase desconocida, no se usan: {dict(raras)}')

    part = {}
    ruta_part = os.path.join(a.raiz, 'particion.csv')
    if os.path.exists(ruta_part):
        part = {(r['proyecto'], r['tesela']): r for r in csv.DictReader(open(ruta_part, encoding='utf-8'))}
    sal = a.salida or os.path.join(a.raiz, 'mascaras_nivel1')
    filas, cajas = [], []

    def escribir(proy, tes, mask, perfil, tr, crs, forma):
        os.makedirs(os.path.join(sal, 'mascaras_256', proy), exist_ok=True)
        p8 = dict(driver='GTiff', height=forma[0], width=forma[1], count=1, dtype='uint8', crs=crs,
                  transform=tr, compress='deflate')
        with MemoryFile() as mem:
            with mem.open(**p8) as d:
                d.write(mask, 1)
                d.update_tags(nivel='1', valores='0 fondo, 1 deslizamiento, 255 ignorar',
                              origen='marcado humano en QGIS')
            open(os.path.join(sal, 'mascaras_256', proy, tes + '.tif'), 'wb').write(mem.read())
        escribir_parches(mask[None], perfil, os.path.join(sal, 'parches_128', proy), tes, ['mascara nivel 1'])

    for proy in BASES:
        rev = os.path.join(a.raiz, proy, 'revisadas.csv')
        if not os.path.exists(rev):
            continue
        for r in csv.DictReader(open(rev, encoding='utf-8-sig')):
            if (r.get('revisada') or '').strip().lower() not in SI:
                continue
            tes = r['tesela'].strip()
            ruta = os.path.join(a.raiz, proy, 'images', tes, 'image.tif')
            if not os.path.exists(ruta):
                print(f'  AVISO: {proy}/{tes} está como revisada pero no tiene imagen')
                continue
            with rasterio.open(ruta) as ds:
                x = ds.read()
                perfil, tr, crs, forma, b = ds.profile, ds.transform, ds.crs, ds.shape, ds.bounds
            capas = {'des': [], 'des2025': [], 'dud': []}
            for m in marcas:
                g = transform_geom(crs_marcas, crs, m['geom'])
                z = rasterize([(g, 1)], out_shape=forma, transform=tr, fill=0, dtype='uint8') == 1
                if not z.any():
                    continue
                m['usada'] = True
                clave = 'dud' if m['clase'] == 'dudoso' else 'des2025' if '2025' in m['nota'] else 'des'
                capas[clave].append(z)
            une = lambda zs: np.any(zs, axis=0) if zs else np.zeros(forma, bool)
            des, des25, dud = une(capas['des']), une(capas['des2025']), une(capas['dud'])
            # nube blanca: B2 > 1000. Dentro de un polígono de deslizamiento NO se aplica, porque
            # el suelo desnudo muy claro también pasa de 1000 y ahí ya decidió una persona
            sin_dato = (x[1] <= 0) | ((x[1] > 1000) & ~des)
            mask = np.zeros(forma, 'uint8')
            mask[des] = 1
            mask[des25 | dud | sin_dato] = 255
            escribir(proy, tes, mask, perfil, tr, crs, forma)
            info = part.get((proy, tes), {})
            filas.append({'proyecto': proy, 'tesela': tes, 'quien': (r.get('quien') or '').strip(),
                          'poligonos_deslizamiento': len(capas['des']), 'poligonos_dudoso': len(capas['dud']),
                          'cicatrices': int(ndimage.label(mask == 1)[1]),
                          'ha_deslizamiento': round(float((mask == 1).sum()) * 0.01, 2),
                          'ha_ignorar': round(float((mask == 255).sum()) * 0.01, 2),
                          'zona': info.get('zona', ''), 'conjunto': info.get('conjunto', ''),
                          'nota': (r.get('nota') or '').strip()})
            cajas.append((proy, tes, b))
            otra = OTRA_2025.get(proy)
            ruta25 = os.path.join(a.raiz, otra, 'images', tes, 'image.tif') if otra else ''
            if capas['des2025'] and ruta25 and os.path.exists(ruta25):
                with rasterio.open(ruta25) as ds:
                    x25 = ds.read(2)
                    perfil25, tr25, forma25 = ds.profile, ds.transform, ds.shape
                if tr25 == tr and forma25 == forma:
                    m25 = np.zeros(forma, 'uint8')
                    m25[des25] = 1
                    m25[des | dud | (x25 <= 0) | ((x25 > 1000) & ~des25)] = 255
                    escribir(otra, tes, m25, perfil25, tr25, crs, forma)
                    filas.append({'proyecto': otra, 'tesela': tes, 'quien': (r.get('quien') or '').strip(),
                                  'poligonos_deslizamiento': len(capas['des2025']), 'poligonos_dudoso': len(capas['dud']),
                                  'cicatrices': int(ndimage.label(m25 == 1)[1]),
                                  'ha_deslizamiento': round(float((m25 == 1).sum()) * 0.01, 2),
                                  'ha_ignorar': round(float((m25 == 255).sum()) * 0.01, 2),
                                  'zona': info.get('zona', ''), 'conjunto': info.get('conjunto', ''),
                                  'nota': 'cicatriz marcada sobre la imagen 2025'})
    if not filas:
        raise SystemExit('ninguna tesela tiene «si» en revisadas.csv: no hay nada que generar')
    os.makedirs(sal, exist_ok=True)
    with open(os.path.join(sal, 'mascaras_nivel1.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    sueltas = [m for m in marcas if not m['usada']]
    if sueltas:
        print(f'  AVISO: {len(sueltas)} polígonos caen en teselas que NO están como revisadas («si»): no se usaron. '
              'Si esas teselas ya están completas, pon «si» en su revisadas.csv y vuelve a correr.')
    solapes = [(p1, t1, p2, t2) for i, (p1, t1, b1) in enumerate(cajas) for p2, t2, b2 in cajas[i + 1:]
               if b1.left < b2.right and b2.left < b1.right and b1.bottom < b2.top and b2.bottom < b1.top]
    if solapes:
        print(f'  OJO: {len(solapes)} pares de teselas revisadas se solapan y sus imágenes pueden ser de fechas distintas; '
              'cada polígono se aplicó a las dos. Comprueben que la cicatriz se ve en ambas:')
        for p1, t1, p2, t2 in solapes[:20]:
            print(f'     {t1}  y  {t2}')
    c = collections.Counter(r['conjunto'] or 'sin zona' for r in filas)
    print(f'{len(filas)} máscaras de nivel 1 en {sal}: ' + ', '.join(f'{k} {v}' for k, v in c.items()))
    print(f'  con deslizamiento: {sum(r["ha_deslizamiento"] > 0 for r in filas)} teselas, '
          f'{sum(r["cicatrices"] for r in filas)} cicatrices, {sum(r["ha_deslizamiento"] for r in filas):.1f} ha; '
          f'negativos confirmados: {sum(r["ha_deslizamiento"] == 0 for r in filas)} teselas')


if __name__ == '__main__':
    main()
