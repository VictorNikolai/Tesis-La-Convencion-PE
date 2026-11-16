"""
Vía 1: Aumento Geofísico Invariante del Grupo Diedral D4 (Rotaciones y Simetrías)
sobre los parches de entrenamiento de máxima certeza (Tier ORO y Tier PLATA).

En imágenes satelitales cenitales (Sentinel-2 + ALOS PALSAR), la firma espectral y la pendiente
local son invariantes ante transformaciones ortogonales rígidas del grupo D4 (7 simetrías no triviales:
rot90, rot180, rot270, flip horizontal, flip vertical, transpuesta y anti-transpuesta).

Reglas de rigor científico:
  1. Solo se aumentan parches con `conjunto == 'entrenamiento'` (CERO fuga hacia validación o evaluación ciega).
  2. Solo se aplican sobre Tier ORO (peso 1.0) y Tier PLATA (peso 0.5) para equilibrar la clase minoritaria
     sin amplificar ruido ni falsas alarmas.
  3. Idempotente: ignora parches que ya contienen `_d4_` en su nombre.
"""
import csv
import os
import sys

import numpy as np
import rasterio
from rasterio.io import MemoryFile

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)

LADO = 128
BANDAS_6 = ['B1', 'B2', 'B3', 'B4', 'B13 pendiente (grados)', 'B14 DEM (m)']

TRANSFORMACIONES_D4 = [
    ('r90', lambda a: np.rot90(a, k=1, axes=(-2, -1))),
    ('r180', lambda a: np.rot90(a, k=2, axes=(-2, -1))),
    ('r270', lambda a: np.rot90(a, k=3, axes=(-2, -1))),
    ('fh', lambda a: np.flip(a, axis=-1)),
    ('fv', lambda a: np.flip(a, axis=-2)),
    ('tr', lambda a: np.swapaxes(a, -2, -1)),
    ('atr', lambda a: np.rot90(np.swapaxes(a, -2, -1), k=2, axes=(-2, -1))),
]


def ejecutar(raiz='.'):
    dir_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2')
    dir_p128 = os.path.join(raiz, 'parches_128')

    ruta_tiers = os.path.join(dir_n2, 'catalogo_parches_tiers.csv')
    ruta_ind_amp = os.path.join(dir_p128, 'indice_parches_ampliado.csv')
    if not os.path.exists(ruta_tiers) or not os.path.exists(ruta_ind_amp):
        raise SystemExit("Faltan catálogos base. Ejecuta primero clasificar_tiers_y_expandir.py")

    with open(ruta_tiers, encoding='utf-8') as f:
        tiers_rows = [r for r in csv.DictReader(f) if '_d4_' not in r['parche']]

    with open(ruta_ind_amp, encoding='utf-8') as f:
        ind_rows = [r for r in csv.DictReader(f) if '_d4_' not in r['parche']]

    meta_ind = {(r['proyecto'], r['parche']): r for r in ind_rows}

    elegibles = [
        r for r in tiers_rows
        if r['conjunto'] == 'entrenamiento' and r['tier'] in ('oro', 'plata')
    ]
    print(f"Parches de entrenamiento elegibles para aumento D4 (Tier Oro/Plata): {len(elegibles)}")

    nuevos_tiers = []
    nuevos_ind = []

    for r in elegibles:
        proy = r['proyecto']
        par = r['parche']
        ruta_im = os.path.join(dir_p128, proy, f"{par}.tif")
        ruta_mk = os.path.join(dir_n2, 'parches_128', proy, f"{par}.tif")
        if not os.path.exists(ruta_mk):
            ruta_mk = os.path.join(dir_n2, 'parches_128', f"{par}.tif")
        if not os.path.exists(ruta_im) or not os.path.exists(ruta_mk):
            continue

        with rasterio.open(ruta_im) as di:
            img = di.read()
            crs_orig = di.crs
            tr_orig = di.transform
        with rasterio.open(ruta_mk) as dm:
            mk = dm.read(1)

        base_ind = meta_ind.get((proy, par), {})

        for sufijo, fn in TRANSFORMACIONES_D4:
            nuevo_par = f"{par}_d4_{sufijo}"
            img_t = np.ascontiguousarray(fn(img), dtype=np.float32)
            mk_t = np.ascontiguousarray(fn(mk), dtype=np.uint8)

            out_im = os.path.join(dir_p128, proy, f"{nuevo_par}.tif")
            out_mk = os.path.join(dir_n2, 'parches_128', proy, f"{nuevo_par}.tif")

            p6 = dict(driver='GTiff', height=LADO, width=LADO, count=6, dtype='float32',
                      crs=crs_orig, transform=tr_orig, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p6) as d:
                    d.write(img_t)
                    d.descriptions = tuple(BANDAS_6)
                with open(out_im, 'wb') as fo:
                    fo.write(mem.read())

            p_mk = dict(driver='GTiff', height=LADO, width=LADO, count=1, dtype='uint8',
                        crs=crs_orig, transform=tr_orig, compress='deflate')
            with MemoryFile() as mem:
                with mem.open(**p_mk) as d:
                    d.write(mk_t, 1)
                with open(out_mk, 'wb') as fo:
                    fo.write(mem.read())

            nt = dict(r)
            nt['parche'] = nuevo_par
            nt['cuadrante'] = f"{r['cuadrante']}_d4_{sufijo}"
            nt['archivo_imagen'] = f"{proy}/{nuevo_par}.tif"
            nt['archivo_mascara'] = f"mascaras_provisionales/nivel2/parches_128/{proy}/{nuevo_par}.tif"
            nuevos_tiers.append(nt)

            if base_ind:
                ni = dict(base_ind)
                ni['parche'] = nuevo_par
                ni['cuadrante'] = f"{base_ind.get('cuadrante', '')}_d4_{sufijo}"
                ni['uso'] = 'aumento_d4'
                ni['archivo'] = f"{proy}/{nuevo_par}.tif"
                nuevos_ind.append(ni)

    todos_tiers = tiers_rows + nuevos_tiers
    todos_ind = ind_rows + nuevos_ind

    with open(ruta_tiers, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(todos_tiers[0].keys()))
        w.writeheader()
        w.writerows(todos_tiers)

    with open(ruta_ind_amp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(todos_ind[0].keys()))
        w.writeheader()
        w.writerows(todos_ind)

    ent_pos = sum(1 for x in todos_tiers if x['conjunto'] == 'entrenamiento' and int(x['px_deslizamiento']) > 0)
    ent_neg = sum(1 for x in todos_tiers if x['conjunto'] == 'entrenamiento' and x['tier'] == 'bronce_negativo')

    print(f"-> Generados {len(nuevos_tiers)} parches positivos mediante simetría D4.")
    print(f"Balance final en conjunto 'entrenamiento':")
    print(f"  • Positivos con deslizamiento: {ent_pos} parches")
    print(f"  • Negativos de control:        {ent_neg} parches")
    return len(nuevos_tiers)


if __name__ == '__main__':
    ejecutar()

