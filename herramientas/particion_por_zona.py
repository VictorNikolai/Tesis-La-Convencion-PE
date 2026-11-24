"""
Reparte las teselas de La Convención en entrenamiento y validación POR ZONA.

Dos teselas van a la misma zona si se solapan o están a menos de 1 km una de otra
(distancia entre centros menor que 3,56 km en x y en y; las teselas miden 2,56 km).
Las zonas se encadenan: si A toca a B y B toca a C, las tres son una sola zona.
Una zona entera va a un solo conjunto, así ningún píxel de validación (ni su vecindad)
se ve en el entrenamiento. Las otras fechas del mismo sitio (2025, antes del evento)
heredan la zona y el conjunto de su tesela base.

Reglas, en este orden:
  1. zonas ancla de validación (lista ANCLAS_VALIDACION): se eligieron a mano para que la
     validación tenga cicatrices y no solo negativos
  2. zona con teselas de ronda (para marcar y entrenar) -> entrenamiento
  3. zona con teselas marcadas «validacion» en revision_teselas.csv y sin rondas -> validación
  4. el resto (reserva): 1 de cada 4 zonas a validación, por un hash fijo del nombre
Las teselas «descartar» quedan como excluida y las «rebajar» como pendiente.

  python herramientas/particion_por_zona.py            (desde Tesis-Proyecto)

Escribe particion.csv (una fila por tesela base) y particion.md (resumen legible).
"""
import argparse
import collections
import csv
import glob
import hashlib
import os

BASES = ['datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar']
# zonas que van a validación aunque tengan teselas de ronda (se nombra una tesela de la zona)
ANCLAS_VALIDACION = {
    'ingemmet_002_001': 'Sillapata (Inkawasi): cicatrices claras de INGEMMET en un distrito aparte',
    'sinpad_21603_001': 'Quellouno/Echarate: dos teselas de ronda junto a dos negativos de validación',
    'sinpad_37418_001': 'Santa Ana (Quillabamba): taludes, Paraíso Alto y el negativo urbano 61214',
}
UMBRAL_M = 3560.0
MIN_FRACCION_PROVINCIA = 0.50


def comprobar_teselas_en_provincia(filas, raiz='.', ruta_limite=None, min_fraccion=MIN_FRACCION_PROVINCIA,
                                   conjuntos=('entrenamiento', 'validacion', 'prueba', 'prueba2')):
    """Devuelve lista de errores para teselas activas con < min_fraccion de su área dentro de La Convención."""
    import glob
    import json
    import rasterio
    from shapely.geometry import shape, box
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    ruta_lim = ruta_limite or os.path.join(raiz, 'herramientas', 'limite_laconvencion.geojson')
    if not os.path.exists(ruta_lim):
        return [f'No existe el límite provincial: {ruta_lim}']
    lim = shape(json.load(open(ruta_lim, encoding='utf-8'))['geometry'])
    tr_ll = Transformer.from_crs('EPSG:32718', 'EPSG:4326', always_xy=True).transform

    errores = []
    for r in filas:
        conj = (r.get('conjunto') or '').strip()
        if conj not in conjuntos:
            continue
        t = (r.get('tesela') or '').strip()
        bx = None
        if t:
            q_tifs = sorted(glob.glob(os.path.join(raiz, 'parches_128', '*', f'{t}_q*.tif')))
            if q_tifs:
                xs, ys = [], []
                for qp in q_tifs:
                    with rasterio.open(qp) as ds:
                        b = ds.bounds
                    xs.extend([b.left, b.right])
                    ys.extend([b.bottom, b.top])
                bx = box(min(xs), min(ys), max(xs), max(ys))
        if bx is None:
            try:
                x, y = float(r['x']), float(r['y'])
                bx = box(x - 1280.0, y - 1280.0, x + 1280.0, y + 1280.0)
            except (KeyError, TypeError, ValueError):
                errores.append(f'{t or "?"} ({conj}): sin geometría ni coordenadas UTM válidas')
                continue
        bx_ll = shp_transform(tr_ll, bx)
        area = bx_ll.area
        frac = (bx_ll.intersection(lim).area / area) if area > 0 else 0.0
        if frac < min_fraccion:
            errores.append(
                f'{t} ({conj}, zona {r.get("zona", "?")}): solo {frac:.1%} dentro de La Convención (< {min_fraccion:.0%})'
            )
    return errores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    a = ap.parse_args()
    import rasterio
    from rasterio.warp import transform as tr_coords

    centro, proyecto, uso, distrito = {}, {}, {}, {}
    for d in BASES:
        rev = os.path.join(a.raiz, d, 'revision_teselas.csv')
        if os.path.exists(rev):
            for r in csv.DictReader(open(rev, encoding='utf-8')):
                uso[r['tesela']] = (r.get('uso_propuesto') or 'reserva').strip()
                distrito[r['tesela']] = (r.get('distrito') or '').strip().upper().replace('ECHARATI', 'ECHARATE')
        for p in sorted(glob.glob(os.path.join(a.raiz, d, 'images', '*', 'image.tif'))):
            t = os.path.basename(os.path.dirname(p))
            with rasterio.open(p) as ds:
                b = ds.bounds
            centro[t] = ((b.left + b.right) / 2, (b.bottom + b.top) / 2)
            proyecto[t] = d
    claves = sorted(centro)
    padre = {k: k for k in claves}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for i, p in enumerate(claves):
        for q in claves[i + 1:]:
            if abs(centro[p][0] - centro[q][0]) < UMBRAL_M and abs(centro[p][1] - centro[q][1]) < UMBRAL_M:
                padre[raiz(p)] = raiz(q)
    grupos = collections.defaultdict(list)
    for k in claves:
        grupos[raiz(k)].append(k)
    orden = sorted(grupos.values(), key=lambda ts: (-len(ts), ts[0]))

    filas, resumen = [], []
    for n, ts in enumerate(orden, 1):
        dist = collections.Counter(distrito.get(t) or '?' for t in ts).most_common(1)[0][0]
        nombre = f'Z{n:02d}_{dist.replace(" ", "_")}'
        usos = [uso.get(t, 'reserva') for t in ts]
        rondas = [t for t in ts if uso.get(t, '').startswith('ronda')]
        vals = [t for t in ts if uso.get(t) == 'validacion']
        ancla = [t for t in ts if t in ANCLAS_VALIDACION]
        if ancla:
            conj, motivo = 'validacion', 'ancla: ' + ANCLAS_VALIDACION[ancla[0]]
        elif rondas:
            conj, motivo = 'entrenamiento', 'tiene teselas de ronda'
        elif vals:
            conj, motivo = 'validacion', 'negativos de validación de la guía'
        elif all(u == 'azar' for u in usos):
            h = int(hashlib.md5(ts[0].encode()).hexdigest(), 16) % 2
            conj, motivo = ('validacion', 'tesela al azar (mitad a validación)') if h == 0 else ('entrenamiento', 'tesela al azar (mitad a entrenamiento)')
        else:
            h = int(hashlib.md5(ts[0].encode()).hexdigest(), 16) % 4
            conj, motivo = ('validacion', 'reserva, por sorteo fijo (1 de 4)') if h == 0 else ('entrenamiento', 'reserva, por sorteo fijo')
        conflicto = [t for t in vals if conj == 'entrenamiento']
        xs, ys = zip(*[centro[t] for t in ts])
        lon, lat = tr_coords('EPSG:32718', 'EPSG:4326', [sum(xs) / len(xs)], [sum(ys) / len(ys)])
        for t in ts:
            u = uso.get(t, 'reserva')
            c = 'excluida' if u == 'descartar' else 'pendiente' if u == 'rebajar' else conj
            nota = 'era de validación en la guía; su zona entrena' if t in conflicto else ''
            filas.append({'tesela': t, 'proyecto': proyecto[t], 'zona': nombre, 'conjunto': c, 'uso': u,
                          'distrito': distrito.get(t, ''), 'x': round(centro[t][0]), 'y': round(centro[t][1]), 'nota': nota})
        resumen.append({'zona': nombre, 'teselas': len(ts), 'conjunto': conj, 'motivo': motivo,
                        'rondas': len(rondas), 'validacion_guia': len(vals), 'conflictos': len(conflicto),
                        'lon': round(lon[0], 4), 'lat': round(lat[0], 4),
                        'marcables': ' '.join(t.replace('_001', '') for t in rondas + vals)})
    with open(os.path.join(a.raiz, 'particion.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(sorted(filas, key=lambda r: (r['zona'], r['tesela'])))

    cuenta = collections.Counter((r['conjunto'], r['uso'].startswith('ronda')) for r in filas)
    L = ['# Partición por zona (propuesta)', '',
         'Generado por `herramientas/particion_por_zona.py`. La regla y el porqué están al inicio de ese archivo.',
         'Una zona = teselas que se solapan o están a menos de 1 km, encadenadas. Toda la zona va a un solo conjunto.', '',
         f'- Teselas base: {len(filas)} en {len(orden)} zonas.',
         f'- Entrenamiento: {sum(1 for r in filas if r["conjunto"] == "entrenamiento")} teselas '
         f'({cuenta[("entrenamiento", True)]} de ronda).',
         f'- Validación: {sum(1 for r in filas if r["conjunto"] == "validacion")} teselas '
         f'({cuenta[("validacion", True)]} de ronda, {sum(1 for r in filas if r["conjunto"] == "validacion" and r["uso"] == "validacion")} negativos de la guía).',
         f'- Excluidas: {sum(1 for r in filas if r["conjunto"] == "excluida")}; pendientes de rebajar: '
         f'{sum(1 for r in filas if r["conjunto"] == "pendiente")}.', '',
         'Solo cuentan las teselas que se lleguen a marcar; las de reserva sin marcar no entran a ningún cálculo.',
         'La prueba final sigue siendo aparte: el test de Landslide4Sense y, si se arma, un conjunto local nunca usado para decidir nada.', '',
         '| Zona | Teselas | Conjunto | Motivo | Para marcar (rondas y validación de la guía) |', '|---|---|---|---|---|']
    for r in resumen:
        if r['rondas'] or r['validacion_guia']:
            L.append(f"| {r['zona']} | {r['teselas']} | {r['conjunto']} | {r['motivo']} | {r['marcables']} |")
    otras = [r for r in resumen if not (r['rondas'] or r['validacion_guia'])]
    L += ['', f'Las otras {len(otras)} zonas solo tienen teselas de reserva o al azar: '
          f'{sum(1 for r in otras if r["conjunto"] == "validacion")} a validación y '
          f'{sum(1 for r in otras if r["conjunto"] == "entrenamiento")} a entrenamiento (detalle en particion.csv).']
    conf = [r for r in filas if r['nota']]
    if conf:
        L += ['', '## Cambios respecto de la guía', '']
        L += [f"- `{r['tesela']}` figuraba como validación, pero está en la zona {r['zona']}, que entrena. "
              'Se marca igual; cuenta como entrenamiento.' for r in conf]
    open(os.path.join(a.raiz, 'particion.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L[4:10]))
    print(f'{len(conf)} cambios respecto de la guía; detalle en particion.md')


if __name__ == '__main__':
    main()
