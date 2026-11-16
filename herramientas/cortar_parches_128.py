"""
Corta las teselas de 256×256 px en 4 parches de 128×128 px, el tamaño del modelo y de L4S.

Se CORTA, no se encoge: cada píxel sigue midiendo 10 m. Los cuadrantes no se solapan:
    q1 = arriba izquierda (noroeste)   q2 = arriba derecha (noreste)
    q3 = abajo izquierda  (suroeste)   q4 = abajo derecha  (sureste)
De las 8 bandas de la tesela solo se guardan las 6 que usa el modelo:
B1, B2, B3, B4, pendiente (grados) y DEM (m). La probabilidad del modelo y las cajas
NO se guardan en el parche: son salidas de otro modelo y contaminarían el entrenamiento.

No toca las teselas originales. Escribe:
  parches_128/<proyecto>/<tesela>_q1.tif ... _q4.tif    GeoTIFF con su georreferencia
  parches_128/indice_parches.csv                        una fila por parche

El índice trae la zona y el conjunto (entrenamiento o validación) de particion.csv; los 4
parches de una tesela, y todas las fechas del mismo sitio, comparten zona y conjunto.

  python herramientas/particion_por_zona.py
  python herramientas/cortar_parches_128.py

También lo usan los guiones de máscaras (funciones cuadrantes y escribir_parches).
"""
import argparse
import csv
import glob
import os

import numpy as np

LADO = 128
PROYECTOS = {                      # carpeta -> (carpeta base de la que hereda zona, papel de la fecha)
    'datos_laconvencion': ('datos_laconvencion', 'base'),
    'datos_indeci': ('datos_indeci', 'base'),
    'datos_ingemmet': ('datos_ingemmet', 'base'),
    'datos_azar': ('datos_azar', 'base'),
    'datos_laconvencion_2025': ('datos_laconvencion', 'otra_fecha_2025'),
    'datos_indeci_antes': ('datos_indeci', 'otra_fecha_antes'),
    'datos_ingemmet_antes': ('datos_ingemmet', 'otra_fecha_antes'),
}
BANDAS = ['B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)']


def cuadrantes(alto, ancho, lado=LADO):
    """Devuelve [(nombre, fila0, col0)] de los cuadrantes sin solape."""
    if alto % lado or ancho % lado:
        raise ValueError(f'la tesela mide {alto}×{ancho}; no es múltiplo de {lado}')
    out, k = [], 1
    for f in range(0, alto, lado):
        for c in range(0, ancho, lado):
            out.append((f'q{k}', f, c))
            k += 1
    return out


def escribir_parches(arr, perfil, carpeta, nombre, descripciones=None, nodata=None):
    """Corta arr (bandas, alto, ancho) en cuadrantes y los escribe como GeoTIFF. Devuelve
    [(ruta, nombre_cuadrante, fila0, col0, transform)]."""
    from rasterio.io import MemoryFile
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform
    os.makedirs(carpeta, exist_ok=True)
    out = []
    for q, f0, c0 in cuadrantes(arr.shape[1], arr.shape[2]):
        tr = win_transform(Window(c0, f0, LADO, LADO), perfil['transform'])
        p = dict(driver='GTiff', height=LADO, width=LADO, count=arr.shape[0], dtype=str(arr.dtype),
                 crs=perfil['crs'], transform=tr, compress='deflate')
        if nodata is not None:
            p['nodata'] = nodata
        ruta = os.path.join(carpeta, f'{nombre}_{q}.tif')
        # se arma en memoria y se escribe de un golpe: así se puede volver a correr encima
        # (sobrescribe sin borrar, que en carpetas protegidas no está permitido)
        with MemoryFile() as mem:
            with mem.open(**p) as d:
                d.write(arr[:, f0:f0 + LADO, c0:c0 + LADO])
                if descripciones:
                    d.descriptions = tuple(descripciones)
            with open(ruta, 'wb') as f:
                f.write(mem.read())
        out.append((ruta, q, f0, c0, tr))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raiz', default='.')
    ap.add_argument('--salida', default='parches_128')
    a = ap.parse_args()
    import rasterio
    from rasterio.warp import transform as tr_coords

    part = {}
    ruta_part = os.path.join(a.raiz, 'particion.csv')
    if not os.path.exists(ruta_part):
        raise SystemExit('falta particion.csv: corre antes herramientas/particion_por_zona.py')
    for r in csv.DictReader(open(ruta_part, encoding='utf-8')):
        part[(r['proyecto'], r['tesela'])] = r
    filas, sin_zona = [], []
    for proy, (base, papel) in PROYECTOS.items():
        for ruta in sorted(glob.glob(os.path.join(a.raiz, proy, 'images', '*', 'image.tif'))):
            tes = os.path.basename(os.path.dirname(ruta))
            with rasterio.open(ruta) as ds:
                x = ds.read().astype('float32')
                perfil = ds.profile
            info = part.get((base, tes))
            if info is None:
                sin_zona.append(f'{proy}/{tes}')
                info = {'zona': '', 'conjunto': 'sin_zona', 'uso': ''}
            for rp, q, f0, c0, tr in escribir_parches(x[:6], perfil, os.path.join(a.raiz, a.salida, proy), tes, BANDAS):
                v = x[:, f0:f0 + LADO, c0:c0 + LADO]
                cx, cy = tr * (LADO / 2, LADO / 2)
                lon, lat = tr_coords(perfil['crs'], 'EPSG:4326', [cx], [cy])
                valido = (v[1] > 0) & (v[1] <= 1000)          # con dato y sin nube blanca
                filas.append({'parche': f'{tes}_{q}', 'proyecto': proy, 'tesela': tes, 'cuadrante': q,
                              'fila0': f0, 'col0': c0, 'fecha': papel, 'zona': info['zona'],
                              'conjunto': info['conjunto'], 'uso': info['uso'],
                              'lon': round(lon[0], 5), 'lat': round(lat[0], 5),
                              'valido_pct': round(float(valido.mean()) * 100, 1),
                              'pend_media': round(float(v[4].mean()), 1),
                              'dem_medio': round(float(v[5].mean())),
                              'archivo': os.path.relpath(rp, os.path.join(a.raiz, a.salida)).replace(os.sep, '/')})
    os.makedirs(os.path.join(a.raiz, a.salida), exist_ok=True)
    with open(os.path.join(a.raiz, a.salida, 'indice_parches.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f'{len(filas)} parches de {len(filas) // 4} imágenes en {os.path.join(a.raiz, a.salida)}')
    if sin_zona:
        print(f'{len(sin_zona)} imágenes sin tesela base en particion.csv (quedan como sin_zona): ' + ', '.join(sin_zona))


if __name__ == '__main__':
    main()
