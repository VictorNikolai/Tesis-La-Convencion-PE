"""
Elige teselas AL AZAR dentro de la provincia de La Convención.

Todas las demás teselas nacen de un punto donde se registró un evento, así que están
sesgadas hacia lugares donde «algo pasó». Estas sirven para lo contrario: medir cuánto
pinta el modelo donde no hay ningún registro (falsos positivos reales) y dar negativos
variados al entrenamiento. Igual hay que mirarlas: en una tesela al azar puede haber una
cicatriz que nadie registró.

Reglas del sorteo (semilla fija, se puede repetir):
  - la tesela entera (2,56 km) cae dentro de la provincia
  - está a más de 6 km de cualquier tesela existente y de las otras sorteadas (zona propia)
  - dos estratos: «cerca» (a menos de 20 km de alguna tesela con evento, donde está la
    población y las laderas intervenidas) y «lejos» (el resto de la provincia)
  - años de imagen repartidos: 2019, 2021 y 2023

  python herramientas/elegir_teselas_azar.py --cerca 40 --lejos 20 --reserva 12

Escribe azar/puntos_azar.csv (id, lon, lat, fecha, estrato, orden), listo para
descargar_desde_puntos.py con --prefijo azar.
"""
import argparse
import csv
import json
import math
import os
import random

AQUI = os.path.dirname(os.path.abspath(__file__))


def dentro(x, y, anillo):
    n, c = len(anillo), False
    for i in range(n):
        x1, y1 = anillo[i]
        x2, y2 = anillo[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--limite', default=os.path.join(AQUI, 'limite_laconvencion.geojson'))
    ap.add_argument('--cerca', type=int, default=40)
    ap.add_argument('--lejos', type=int, default=20)
    ap.add_argument('--reserva', type=int, default=12, help='puntos de repuesto por si el servidor rechaza alguno')
    ap.add_argument('--semilla', type=int, default=20261115)
    a = ap.parse_args()
    from rasterio.warp import transform

    g = json.load(open(a.limite, encoding='utf-8'))
    geo = (g['features'][0] if 'features' in g else g).get('geometry', g)
    anillo = geo['coordinates'][0] if geo['type'] == 'Polygon' else max((p[0] for p in geo['coordinates']), key=len)
    xs, ys = transform('EPSG:4326', 'EPSG:32718', [p[0] for p in anillo], [p[1] for p in anillo])
    poli = list(zip(xs, ys))
    existentes = [(float(r['x']), float(r['y'])) for r in csv.DictReader(open(os.path.join(a.raiz, 'particion.csv'), encoding='utf-8'))
                  if r['proyecto'] != 'datos_azar']
    rnd = random.Random(a.semilla)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    medio = 1280 + 300                      # media tesela más un margen
    elegidos = {'cerca': [], 'lejos': []}
    meta = {'cerca': a.cerca + (2 * a.reserva) // 3, 'lejos': a.lejos + a.reserva // 3}
    intentos = 0
    while any(len(elegidos[k]) < meta[k] for k in meta) and intentos < 400000:
        intentos += 1
        x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        if not all(dentro(x + dx, y + dy, poli) for dx in (-medio, medio) for dy in (-medio, medio)):
            continue
        d = min(math.hypot(x - ex, y - ey) for ex, ey in existentes)
        if d < 6000:
            continue
        k = 'cerca' if d < 20000 else 'lejos'
        if len(elegidos[k]) >= meta[k]:
            continue
        if any(math.hypot(x - px, y - py) < 6000 for v in elegidos.values() for px, py in v):
            continue
        elegidos[k].append((x, y))
    filas, n = [], 0
    anios = [2019, 2021, 2023]
    for tanda in ('principal', 'reserva'):
        for k, cuantos in (('cerca', a.cerca), ('lejos', a.lejos)):
            pts = elegidos[k][:cuantos] if tanda == 'principal' else elegidos[k][cuantos:]
            for x, y in pts:
                n += 1
                lon, lat = transform('EPSG:32718', 'EPSG:4326', [x], [y])
                filas.append({'id': f'{n:03d}', 'lon': round(lon[0], 5), 'lat': round(lat[0], 5),
                              'fecha': f'15/01/{anios[n % 3]}', 'estrato': k, 'orden': tanda})
    os.makedirs(os.path.join(a.raiz, 'azar'), exist_ok=True)
    with open(os.path.join(a.raiz, 'azar', 'puntos_azar.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f'{len(filas)} puntos ({sum(1 for r in filas if r["orden"] == "principal")} principales) en azar/puntos_azar.csv; '
          f'cerca {len(elegidos["cerca"])}, lejos {len(elegidos["lejos"])}; {intentos} sorteos')


if __name__ == '__main__':
    main()
