"""
Máscaras PROVISIONALES de nivel 2: propuestas de INGEMMET visibles en la imagen.

NO son etiquetas humanas. Salen de cruzar el polígono levantado en campo por INGEMMET con
lo que la imagen Sentinel-2 muestra como suelo desnudo, y de una revisión visual asistida
(archivo decisiones.csv). Sirven para probar el código y, con peso bajo, como entrenamiento
débil DESPUÉS de que una persona acepte o rechace cada pieza. Nunca se usan para medir.

Valores de la máscara (uint8, misma malla que la tesela):
    0    fondo
    1    propuesta aceptada en la revisión visual
    255  ignorar: pieza dudosa o descartada, resto del cuerpo del polígono de INGEMMET
         (la parte vegetada o latente: no se afirma que sea fondo), nube blanca o sin dato

Decisión de cada pieza, por prioridad:
  1. fila de decisiones.csv con ámbito «pid» (una pieza)
  2. fila de decisiones.csv con ámbito «tesela» (todas las piezas de la tesela)
  3. tipo de INGEMMET: flujo de detritos, aluvión, inundación y erosión quedan como dudosa
  4. tamaño: dudosa si mide menos de 0,10 ha; aceptada en otro caso

  python herramientas/mascaras_provisionales.py            genera todo
  python herramientas/mascaras_provisionales.py --hojas     además dibuja las hojas de revisión

Escribe en mascaras_provisionales/nivel2/:
  propuestas_nivel2.geojson     las piezas con pid, decisión y motivo (para verlas en QGIS)
  mascaras_256/<tesela>.tif     máscara de la tesela completa
  parches_128/<tesela>_qN.tif   la misma máscara cortada como los parches de imagen
  mascaras_nivel2.csv           resumen por tesela
  _revision/<tesela>.png        antes | después | después con contornos (rojo = aceptada,
                                naranja = dudosa, gris = descartada, amarillo = INGEMMET)
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

MIN_HA = 0.10
# tipos de INGEMMET que son cauce o erosión, no una cicatriz que arranca en la ladera
TIPOS_CAUCE = ('flujo de detritos', 'aluvión', 'inundación', 'erosión')


def leer_decisiones(ruta):
    por_pid, por_tesela = {}, {}
    if os.path.exists(ruta):
        for r in csv.DictReader(open(ruta, encoding='utf-8')):
            destino = por_pid if r['ambito'] == 'pid' else por_tesela
            clave = int(r['id']) if r['ambito'] == 'pid' else r['id']
            destino[clave] = (r['decision'].strip(), r['motivo'].strip())
    return por_pid, por_tesela


def color_natural(x, lo=None, hi=None):
    rgb = np.stack([x[3], x[2], x[1]], -1).astype('float32')
    ok = x[1] > 0
    if lo is None:
        lo, hi = (np.percentile(rgb[ok], 1), np.percentile(rgb[ok], 99)) if ok.any() else (0, 1)
    y = np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) ** 0.8
    return (y * 255).astype('uint8'), lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--proyecto', default='datos_ingemmet')
    ap.add_argument('--hojas', action='store_true')
    ap.add_argument('--solo', help='solo estas teselas (separadas por coma), para rehacer hojas')
    a = ap.parse_args()
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    from rasterio.io import MemoryFile

    sal = os.path.join(a.raiz, 'mascaras_provisionales', 'nivel2')
    for d in ('mascaras_256', 'parches_128', '_revision'):
        os.makedirs(os.path.join(sal, d), exist_ok=True)
    piezas = json.load(open(os.path.join(a.raiz, 'ingemmet', 'propuestas_ingemmet_visibles.geojson'), encoding='utf-8'))['features']
    cuerpos = json.load(open(os.path.join(a.raiz, 'ingemmet', 'propuestas_ingemmet.geojson'), encoding='utf-8'))['features']
    por_pid, por_tesela = leer_decisiones(os.path.join(sal, 'decisiones.csv'))
    tipo = {f['properties']['objectid']: f['properties'].get('peligro_es', '') for f in cuerpos}
    por_tes = collections.defaultdict(list)
    for pid, f in enumerate(piezas):
        f['properties']['pid'] = pid
        por_tes[f['properties']['tesela']].append(f)
    solo = set(a.solo.split(',')) if a.solo else None
    teselas = sorted(os.listdir(os.path.join(a.raiz, a.proyecto, 'images')))
    filas = []
    for tes in teselas:
        ruta = os.path.join(a.raiz, a.proyecto, 'images', tes, 'image.tif')
        if not os.path.exists(ruta) or (solo and tes not in solo):
            continue
        with rasterio.open(ruta) as ds:
            x = ds.read()
            perfil = ds.profile
            tr, crs, forma = ds.transform, ds.crs, ds.shape
            caja = ds.bounds
        geo_cuerpos = []
        for f in cuerpos:
            g = transform_geom('EPSG:4326', crs, f['geometry'])
            geo_cuerpos.append(g)
        cuerpo = rasterize([(g, 1) for g in geo_cuerpos], out_shape=forma, transform=tr, fill=0, dtype='uint8')
        mask = np.zeros(forma, 'uint8')
        mask[cuerpo == 1] = 255
        n = collections.Counter()
        ha = collections.Counter()
        capas = {'aceptada': [], 'dudosa': [], 'descartada': []}
        for f in por_tes.get(tes, []):
            p = f['properties']
            g = transform_geom('EPSG:4326', crs, f['geometry'])
            z = rasterize([(g, 1)], out_shape=forma, transform=tr, fill=0, dtype='uint8') == 1
            p['dem_medio'] = round(float(x[5][z].mean())) if z.any() else None
            if p['pid'] in por_pid:
                dec, mot = por_pid[p['pid']]
            elif tes in por_tesela:
                dec, mot = por_tesela[tes]
            elif any(t in tipo.get(p['objectid'], '').lower() for t in TIPOS_CAUCE):
                dec, mot = 'dudosa', 'tipo INGEMMET de cauce o erosión (' + tipo[p['objectid']] + '): la guía lo deja como dudoso salvo que se vea el arranque en la ladera'
            elif p['area_ha'] < MIN_HA:
                dec, mot = 'dudosa', 'automática: menos de 0,10 ha'
            else:
                dec, mot = 'aceptada', 'automática: sin observación'
            p['tipo_ingemmet'] = tipo.get(p['objectid'], '')
            p['decision'], p['motivo'], p['nivel'] = dec, mot, 2
            p['fuente'] = 'INGEMMET ∩ suelo desnudo visible; revisión visual asistida; NO es etiqueta humana'
            n[dec] += 1
            ha[dec] += p['area_ha']
            capas[dec].append(g)
            mask[z] = 1 if dec == 'aceptada' else 255
        # la regla de nube (B2 > 1000) no se aplica sobre una pieza aceptada ni sobre suelo desnudo claro
        suelo_claro = (x[1] <= 2000) & (x[3] > x[2]) & (x[3] > 1.2 * x[1])
        sin_dato = (x[1] <= 0) | ((x[1] > 1000) & ~suelo_claro & (mask != 1))
        mask[sin_dato] = 255
        p8 = dict(driver='GTiff', height=forma[0], width=forma[1], count=1, dtype='uint8', crs=crs,
                  transform=tr, compress='deflate', nodata=None)
        with MemoryFile() as mem:
            with mem.open(**p8) as d:
                d.write(mask, 1)
                d.update_tags(nivel='2', valores='0 fondo, 1 propuesta aceptada, 255 ignorar',
                              aviso='provisional: no es etiqueta humana')
            open(os.path.join(sal, 'mascaras_256', tes + '.tif'), 'wb').write(mem.read())
        escribir_parches(mask[None], perfil, os.path.join(sal, 'parches_128'), tes, ['mascara nivel 2'])
        filas.append({'tesela': tes, 'piezas': sum(n.values()), 'aceptadas': n['aceptada'], 'dudosas': n['dudosa'],
                      'descartadas': n['descartada'], 'ha_aceptadas': round(ha['aceptada'], 2),
                      'ha_dudosas': round(ha['dudosa'], 2), 'ha_descartadas': round(ha['descartada'], 2),
                      'px_1': int((mask == 1).sum()), 'px_255': int((mask == 255).sum()),
                      'px_cuerpo_ingemmet': int(cuerpo.sum()),
                      'observacion': por_tesela.get(tes, ('', ''))[1] if tes in por_tesela else ''})
        if a.hojas:
            from PIL import Image, ImageDraw
            E = 3
            tr3 = tr * tr.scale(1 / E, 1 / E)
            f3 = (forma[0] * E, forma[1] * E)

            def borde(geos, grosor=1):
                if not geos:
                    return np.zeros(f3, bool)
                from scipy import ndimage
                m = rasterize([(g, 1) for g in geos], out_shape=f3, transform=tr3, fill=0, dtype='uint8') == 1
                return m & ~ndimage.binary_erosion(m, iterations=grosor)

            rgb, lo, hi = color_natural(x)
            grande = np.asarray(Image.fromarray(rgb).resize((f3[1], f3[0]), Image.NEAREST)).copy()
            antes = np.zeros_like(grande)
            ruta_a = os.path.join(a.raiz, a.proyecto + '_antes', 'images', tes, 'image.tif')
            if os.path.exists(ruta_a):
                with rasterio.open(ruta_a) as da:
                    xa = da.read()
                antes = np.asarray(Image.fromarray(color_natural(xa)[0]).resize((f3[1], f3[0]), Image.NEAREST)).copy()
            marc = grande.copy()
            marc[borde(geo_cuerpos)] = (255, 230, 0)
            marc[borde(capas['descartada'], 2)] = (170, 170, 170)
            marc[borde(capas['dudosa'], 2)] = (255, 140, 0)
            marc[borde(capas['aceptada'], 2)] = (255, 0, 0)
            hoja = Image.fromarray(np.concatenate([antes, grande, marc], 1))
            lienzo = Image.new('RGB', (hoja.width, hoja.height + 22), (20, 20, 20))
            lienzo.paste(hoja, (0, 22))
            dib = ImageDraw.Draw(lienzo)
            dib.text((6, 5), f'{tes}   antes | después | después + contornos   rojo aceptada {n["aceptada"]} ({ha["aceptada"]:.1f} ha)   '
                             f'naranja dudosa {n["dudosa"]}   gris descartada {n["descartada"]}   amarillo INGEMMET', fill=(255, 255, 255))
            for f in por_tes.get(tes, []):
                p = f['properties']
                if p['area_ha'] >= 0.3:              # número de pieza, sobre un píxel de la pieza
                    g = transform_geom('EPSG:4326', crs, f['geometry'])
                    fs, cs = np.nonzero(rasterize([(g, 1)], out_shape=f3, transform=tr3, fill=0, dtype='uint8'))
                    if len(fs):
                        k = np.argmin((fs - fs.mean()) ** 2 + (cs - cs.mean()) ** 2)
                        dib.text((2 * f3[1] + cs[k] + 3, 22 + fs[k] - 5), str(p['pid']), fill=(0, 255, 255))
            lienzo.save(os.path.join(sal, '_revision', tes + '.png'))
    if not solo:
        json.dump({'type': 'FeatureCollection', 'features': piezas},
                  open(os.path.join(sal, 'propuestas_nivel2.geojson'), 'w', encoding='utf-8'), ensure_ascii=False)
        with open(os.path.join(sal, 'mascaras_nivel2.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    tot = collections.Counter()
    for r in filas:
        for k in ('piezas', 'aceptadas', 'dudosas', 'descartadas', 'ha_aceptadas', 'ha_dudosas', 'ha_descartadas'):
            tot[k] += r[k]
    print(f'{len(filas)} teselas; ' + ', '.join(f'{k} {round(v, 1)}' for k, v in tot.items()))


if __name__ == '__main__':
    main()
