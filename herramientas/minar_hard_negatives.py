"""
Vía 4: Minería Activa de Negativos Difíciles (Hard Negative Mining).

Examina todos los parches negativos confirmados del conjunto de entrenamiento (`bronce_negativo`),
cruza su área contra la banda 7 (`probabilidad del modelo` base) de las teselas originales,
y detecta dónde el modelo base dispara falsas alarmas (probabilidad >= 0.40 en trochas,
playas de río, pastos secos o techos urbanos).

A esos parches críticos se les asigna el tier `hard_negative` con un peso de pérdida elevado (0.8),
forzando al ajuste fino a inmunizarse contra falsos positivos sin afectar la sensibilidad en cicatrices reales.
"""
import csv
import os
import sys

import numpy as np
import rasterio

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)

LADO = 128
CUADRANTES_COORDS = {
    'q1': (0, 0),
    'q2': (0, 128),
    'q3': (128, 0),
    'q4': (128, 128),
}


def ejecutar(raiz='.'):
    dir_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2')
    ruta_tiers = os.path.join(dir_n2, 'catalogo_parches_tiers.csv')
    if not os.path.exists(ruta_tiers):
        raise SystemExit(f"Falta {ruta_tiers}. Ejecuta primero clasificar_tiers_y_expandir.py")

    with open(ruta_tiers, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Cachear banda 7 (probabilidad del modelo) por (proyecto, tesela)
    cache_prob = {}
    hard_negatives_count = 0
    total_neg_entrenamiento = 0

    for r in rows:
        if r['conjunto'] != 'entrenamiento':
            continue
        if r['tier'] not in ('bronce_negativo', 'hard_negative'):
            continue

        total_neg_entrenamiento += 1
        proy = r['proyecto']
        tes = r['tesela']
        cuad = r['cuadrante']

        if cuad not in CUADRANTES_COORDS:
            continue

        # Ignorar teselas con banda 7 corrupta conocida
        if tes in ('azar_012_001', 'azar_015_001'):
            continue

        key = (proy, tes)
        if key not in cache_prob:
            ruta_img = os.path.join(raiz, proy, 'images', tes, 'image.tif')
            if os.path.exists(ruta_img):
                with rasterio.open(ruta_img) as ds:
                    if ds.count >= 7:
                        cache_prob[key] = ds.read(7)
                    else:
                        cache_prob[key] = None
            else:
                cache_prob[key] = None

        prob_256 = cache_prob[key]
        if prob_256 is None:
            continue

        f0, c0 = CUADRANTES_COORDS[cuad]
        sub_prob = prob_256[f0:f0 + LADO, c0:c0 + LADO]

        prob_max = float(np.nanmax(sub_prob))
        px_fa = int((sub_prob >= 0.40).sum())

        # Si el modelo base marcó falsa alarma (>=0.40 en al menos 5 px o pico >=0.50)
        if prob_max >= 0.50 or px_fa >= 5:
            r['tier'] = 'hard_negative'
            r['peso_perdida'] = '0.8'
            hard_negatives_count += 1
        else:
            r['tier'] = 'bronce_negativo'
            r['peso_perdida'] = '0.2'

    with open(ruta_tiers, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Minería Activa de Negativos Difíciles (Vía 4) completada:")
    print(f"  • Negativos de entrenamiento evaluados: {total_neg_entrenamiento}")
    print(f"  • 🔥 Hard Negatives detectados (falsas alarmas del modelo base, peso 0.8): {hard_negatives_count} parches")
    print(f"  • Negativos estándar restantes (peso 0.2): {total_neg_entrenamiento - hard_negatives_count} parches")
    return hard_negatives_count


if __name__ == '__main__':
    ejecutar()

