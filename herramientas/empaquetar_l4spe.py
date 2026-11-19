"""
Empaqueta el conjunto L4SPe del profesor (parches de 128 x 128 con máscara manual, formato
Landslide4Sense) en un solo archivo `l4spe_profesor.npz` para el ajuste fino en Colab.

Entrada (la carpeta que bajó de Drive):
    l4s/image_ls4pe/point_XXXX.h5   imagen (128, 128, 14) en unidades originales
    l4s/mask_ls4pe/point_XXXX.h5    máscara (128, 128) con 0 = no, 1 = deslizamiento
    (opcional) landslide_l4spe_iris/landslide_ok/point_XXXX/metadata.json   con «location»: [lat, lon]

Qué hace:
  1. Se queda con los 6 canales que usa el modelo: B01, B02, B03, B04, pendiente (en radianes,
     como viene) y elevación (m). Las bandas vienen en niveles digitales, NO normalizadas por
     escena como Landslide4Sense; la normalización se hace en ajuste_fino.py.
  2. Agrupa los puntos por zona (encadena los que están a menos de --km km) y reparte las ZONAS
     enteras en entrenamiento / validación / prueba (70 / 15 / 15 % de los parches, semilla
     fija). Un mismo sitio nunca queda en dos conjuntos. Sin coordenadas, cada punto es su zona.
  3. Calcula la escala de bandas (media por banda sobre píxeles válidos) SOLO con el
     entrenamiento y la guarda dentro del archivo.
  4. Escribe también particion_l4spe.csv, para que la partición quede registrada.

Uso:
  python herramientas/empaquetar_l4spe.py --carpeta "D:/INDECI/Documentos de Referencia/l4s-.../l4s" \
      --iris "C:/.../Profesor/landslide_l4spe_iris/landslide_ok" --salida l4spe_profesor.npz
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import h5py
import numpy as np

CANALES = [0, 1, 2, 3, 12, 13]            # B01 B02 B03 B04 pendiente DEM, posiciones en el h5
BANDAS = ['B01', 'B02', 'B03', 'B04']


def zonas_por_cercania(lat, lon, km):
    """Une los puntos a menos de `km` km (cadena). Devuelve un entero de zona por punto."""
    from scipy.spatial import cKDTree
    n = len(lat)
    xy = np.column_stack([np.asarray(lon) * 111.32 * np.cos(np.radians(np.mean(lat))), np.asarray(lat) * 110.57])
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for i, j in cKDTree(xy).query_pairs(km):
        padre[raiz(i)] = raiz(j)
    return np.array([raiz(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--carpeta', required=True, help='carpeta con image_ls4pe/ y mask_ls4pe/')
    ap.add_argument('--iris', help='carpeta landslide_ok del proyecto IRIS (para las coordenadas)')
    ap.add_argument('--salida', default='l4spe_profesor.npz')
    ap.add_argument('--km', type=float, default=3.0, help='distancia que encadena puntos en una misma zona')
    ap.add_argument('--semilla', type=int, default=42)
    ap.add_argument('--prop', type=float, nargs=3, default=(0.70, 0.15, 0.15), help='entrenamiento validación prueba')
    a = ap.parse_args()

    imgs = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(a.carpeta, 'image_ls4pe', 'point_*.h5'))}
    msks = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(a.carpeta, 'mask_ls4pe', 'point_*.h5'))}
    puntos = sorted(set(imgs) & set(msks))
    if not puntos:
        raise SystemExit(f'no hay pares image_ls4pe / mask_ls4pe en {a.carpeta}')
    print(f'{len(puntos)} puntos con imagen y máscara ({len(set(imgs) ^ set(msks))} sin pareja)')

    X = np.zeros((len(puntos), 6, 128, 128), np.float32)
    Y = np.zeros((len(puntos), 128, 128), np.uint8)
    for k, p in enumerate(puntos):
        with h5py.File(imgs[p], 'r') as f:
            im = np.asarray(f[list(f.keys())[0]], np.float32)
        with h5py.File(msks[p], 'r') as f:
            m = np.asarray(f[list(f.keys())[0]]).squeeze()
        if im.shape != (128, 128, 14) or m.shape != (128, 128):
            raise SystemExit(f'{p}: forma inesperada {im.shape} / {m.shape}')
        X[k] = im[..., CANALES].transpose(2, 0, 1)
        Y[k] = (m > 0).astype(np.uint8)
    V = np.isfinite(X).all(axis=1)                 # (N, 128, 128): píxel con todos los canales

    lat = np.full(len(puntos), np.nan); lon = np.full(len(puntos), np.nan)
    if a.iris:
        for k, p in enumerate(puntos):
            ruta = os.path.join(a.iris, p, 'metadata.json')
            if os.path.exists(ruta):
                with open(ruta, encoding='utf-8') as f:
                    la, lo = json.load(f)['location']
                lat[k], lon[k] = la, lo
    con_coord = np.isfinite(lat)
    zona = np.arange(len(puntos))
    if con_coord.all():
        zona = zonas_por_cercania(lat, lon, a.km)
    else:
        print(f'AVISO: {int((~con_coord).sum())} puntos sin coordenadas: cada uno cuenta como su propia zona')
        if con_coord.any():
            zona[con_coord] = zonas_por_cercania(lat[con_coord], lon[con_coord], a.km) + len(puntos)
    ids = sorted(set(zona.tolist()), key=lambda z: hashlib.sha1(f'{a.semilla}-{z}'.encode()).hexdigest())
    n_por_zona = {z: int((zona == z).sum()) for z in ids}
    corte = np.cumsum(a.prop) / sum(a.prop) * len(puntos)
    conj_de_zona, acum = {}, 0
    for z in ids:
        conj_de_zona[z] = 'entrenamiento' if acum < corte[0] else ('validacion' if acum < corte[1] else 'prueba')
        acum += n_por_zona[z]
    conjunto = np.array([conj_de_zona[z] for z in zona])

    tr = (conjunto == 'entrenamiento')[:, None, None] & V
    escala = {}
    for k, b in enumerate(BANDAS):
        escala[b] = round(float(X[:, k][tr].mean()), 3)
    print('escala de bandas (media sobre píxeles válidos de entrenamiento):', escala)

    np.savez_compressed(a.salida, x=X, y=Y, punto=np.array(puntos), zona=np.array([f'Z{z}' for z in zona]),
                        conjunto=conjunto, lat=lat, lon=lon, escala_json=np.array(json.dumps(escala)),
                        canales=np.array(['B01', 'B02', 'B03', 'B04', 'pendiente_rad', 'elevacion_m']))
    ruta_csv = os.path.splitext(a.salida)[0] + '_particion.csv'
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['punto', 'zona', 'conjunto', 'lat', 'lon', 'px_deslizamiento', 'px_validos'])
        for k, p in enumerate(puntos):
            w.writerow([p, f'Z{zona[k]}', conjunto[k], lat[k], lon[k], int(Y[k].sum()), int(V[k].sum())])
    for c in ('entrenamiento', 'validacion', 'prueba'):
        s = conjunto == c
        print(f'{c:14s}: {int(s.sum()):4d} parches, {len({zona[i] for i in np.where(s)[0]}):4d} zonas, '
              f'{int((Y[s].sum(axis=(1, 2)) > 0).sum()):4d} con deslizamiento, '
              f'{Y[s].sum() / max(V[s].sum(), 1) * 100:.2f} % de píxeles')
    print('escrito', a.salida, f'({os.path.getsize(a.salida) / 1e6:.0f} MB) y', ruta_csv)


if __name__ == '__main__':
    main()
