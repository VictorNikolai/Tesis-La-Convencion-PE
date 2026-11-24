# Métricas Reformuladas — Evaluación «Post Hoc» (Tarea 5)

> [!IMPORTANT]
> **Nota metodológica:** Todas las cifras de este informe son **«post hoc»** (calculadas sobre conjuntos ya vistos en v5/v6, sin reentrenar y con los umbrales fijados exclusivamente en validación). Todos los intervalos son **IC 95 % por bootstrap de zonas (`2 000` remuestreos)**.
> 
> **Definición de métricas por objeto (8-conectado):** acierto (`TP`) = cicatriz GT con solape `≥ 30 %`; falsa alarma (`FP`) = componente predicho `≥ 10 px` que no toca ninguna cicatriz GT; tasa de falsos positivos por objeto = `FP / (TP + FP) = 1 − Precisión`.

## 1. Prueba Sellada de La Convención (`360` parches, `62` zonas) — «post hoc»

### 1a. Métricas por píxel (exacto, tol. 1 px y tol. 2 px) — Todas las cicatrices vs. Solo `≥ 50 px`

| Modelo | Evaluación | IoU exacto [IC 95 %] | IoU tol. 1 px [IC 95 %] | IoU tol. 2 px [IC 95 %] | F1 píxel exacto [IC 95 %] |
|---|---|---|---|---|---|
| U-Net B1 v5 | Todas (`≥ 10 px`) | 0.1311 [0.0530, 0.2355] | 0.1506 [0.0614, 0.2677] | 0.1572 [0.0646, 0.2762] | 23.2 % [10.1, 38.1] |
| U-Net B1 v5 | Solo `≥ 50 px` (`<50→255`) | 0.1314 [0.0469, 0.2466] | 0.1499 [0.0540, 0.2793] | 0.1563 [0.0558, 0.2872] | 23.2 % [9.0, 39.6] |
| U-Net v6 (sin TTA+flt) | Todas (`≥ 10 px`) | 0.1387 [0.0483, 0.2754] | 0.1598 [0.0554, 0.3189] | 0.1663 [0.0577, 0.3284] | 24.4 % [9.2, 43.2] |
| U-Net v6 (sin TTA+flt) | Solo `≥ 50 px` (`<50→255`) | 0.1429 [0.0452, 0.2985] | 0.1640 [0.0520, 0.3460] | 0.1703 [0.0536, 0.3584] | 25.0 % [8.6, 46.0] |
| DeepLabV3+ v6 (sin TTA+flt) | Todas (`≥ 10 px`) | 0.0184 [0.0044, 0.0949] | 0.0228 [0.0057, 0.1185] | 0.0253 [0.0065, 0.1328] | 3.6 % [0.9, 17.3] |
| DeepLabV3+ v6 (sin TTA+flt) | Solo `≥ 50 px` (`<50→255`) | 0.0185 [0.0043, 0.1023] | 0.0226 [0.0054, 0.1255] | 0.0248 [0.0060, 0.1394] | 3.6 % [0.9, 18.6] |

### 1b. Métricas por objeto (acierto `≥ 30 %`, FA componente `≥ 10 px` sin tocar GT)

| Modelo | Tramo GT | TP / FN / FP | Precisión obj. [IC 95 %] | Exhaustividad (Recall) obj. [IC 95 %] | F1 objeto [IC 95 %] | Tasa FP obj. [IC 95 %] |
|---|---|---|---:|---:|---:|---:|
| U-Net B1 v5 | Total (`≥ 10 px`) | `18 / 62 / 116` | 13.4 % [6.5, 26.3] | 22.5 % [14.3, 35.2] | 16.8 % [9.4, 24.9] | 86.6 % [73.7, 93.5] |
| U-Net B1 v5 | `≥ 50 px` (`0.5 ha`) | `13 / 13 / 116` | 10.1 % [3.8, 20.4] | 50.0 % [28.6, 66.7] | 16.8 % [6.8, 29.4] | 89.9 % [79.6, 96.2] |
| U-Net B1 v5 | `≥ 200 px` (`2 ha`) | `7 / 4 / 116` | 5.7 % [1.4, 10.8] | 63.6 % [27.3, 88.9] | 10.4 % [2.6, 18.8] | 94.3 % [89.2, 98.6] |
| U-Net v6 (sin TTA+flt) | Total (`≥ 10 px`) | `15 / 65 / 62` | 19.5 % [8.1, 36.1] | 18.8 % [11.5, 28.1] | 19.1 % [10.1, 27.8] | 80.5 % [63.9, 92.0] |
| U-Net v6 (sin TTA+flt) | `≥ 50 px` (`0.5 ha`) | `12 / 14 / 62` | 16.2 % [5.7, 31.5] | 46.2 % [26.9, 61.5] | 24.0 % [9.5, 39.4] | 83.8 % [68.5, 94.3] |
| U-Net v6 (sin TTA+flt) | `≥ 200 px` (`2 ha`) | `6 / 5 / 62` | 8.8 % [2.0, 19.7] | 54.5 % [25.0, 85.7] | 15.2 % [3.8, 29.5] | 91.2 % [80.3, 98.0] |
| DeepLabV3+ v6 (sin TTA+flt) | Total (`≥ 10 px`) | `9 / 71 / 124` | 6.8 % [1.3, 17.9] | 11.2 % [2.9, 20.9] | 8.5 % [1.8, 16.6] | 93.2 % [82.0, 98.7] |
| DeepLabV3+ v6 (sin TTA+flt) | `≥ 50 px` (`0.5 ha`) | `8 / 18 / 124` | 6.1 % [1.2, 16.1] | 30.8 % [10.7, 47.1] | 10.1 % [2.1, 22.2] | 93.9 % [83.9, 98.8] |
| DeepLabV3+ v6 (sin TTA+flt) | `≥ 200 px` (`2 ha`) | `5 / 6 / 124` | 3.9 % [0.6, 10.9] | 45.5 % [14.3, 75.0] | 7.1 % [1.2, 18.2] | 96.1 % [89.1, 99.4] |

## 2. Prueba L4SPe del Profesor (`124` parches, `68` zonas) — «post hoc»

### 1a. Métricas por píxel (exacto, tol. 1 px y tol. 2 px) — Todas las cicatrices vs. Solo `≥ 50 px`

| Modelo | Evaluación | IoU exacto [IC 95 %] | IoU tol. 1 px [IC 95 %] | IoU tol. 2 px [IC 95 %] | F1 píxel exacto [IC 95 %] |
|---|---|---|---|---|---|
| U-Net B1 v5 | Todas (`≥ 10 px`) | 0.4297 [0.3739, 0.4872] | 0.5470 [0.4809, 0.6137] | 0.6087 [0.5396, 0.6780] | 60.1 % [54.4, 65.5] |
| U-Net B1 v5 | Solo `≥ 50 px` (`<50→255`) | 0.4391 [0.3825, 0.4974] | 0.5520 [0.4855, 0.6207] | 0.6099 [0.5400, 0.6801] | 61.0 % [55.3, 66.4] |
| U-Net v6 (sin TTA+flt) | Todas (`≥ 10 px`) | 0.4229 [0.3687, 0.4803] | 0.5404 [0.4749, 0.6090] | 0.6016 [0.5332, 0.6738] | 59.4 % [53.9, 64.9] |
| U-Net v6 (sin TTA+flt) | Solo `≥ 50 px` (`<50→255`) | 0.4337 [0.3785, 0.4923] | 0.5470 [0.4809, 0.6157] | 0.6049 [0.5340, 0.6774] | 60.5 % [54.9, 66.0] |
| DeepLabV3+ v6 (sin TTA+flt) | Todas (`≥ 10 px`) | 0.3844 [0.3219, 0.4499] | 0.4978 [0.4235, 0.5791] | 0.5656 [0.4864, 0.6515] | 55.5 % [48.7, 62.1] |
| DeepLabV3+ v6 (sin TTA+flt) | Solo `≥ 50 px` (`<50→255`) | 0.3918 [0.3284, 0.4578] | 0.4999 [0.4247, 0.5790] | 0.5623 [0.4820, 0.6465] | 56.3 % [49.4, 62.8] |

### 1b. Métricas por objeto (acierto `≥ 30 %`, FA componente `≥ 10 px` sin tocar GT)

| Modelo | Tramo GT | TP / FN / FP | Precisión obj. [IC 95 %] | Exhaustividad (Recall) obj. [IC 95 %] | F1 objeto [IC 95 %] | Tasa FP obj. [IC 95 %] |
|---|---|---|---:|---:|---:|---:|
| U-Net B1 v5 | Total (`≥ 10 px`) | `220 / 186 / 106` | 67.5 % [61.5, 74.5] | 54.2 % [47.7, 61.8] | 60.1 % [54.8, 66.0] | 32.5 % [25.5, 38.5] |
| U-Net B1 v5 | `≥ 50 px` (`0.5 ha`) | `164 / 93 / 106` | 60.7 % [54.3, 68.9] | 63.8 % [57.1, 71.4] | 62.2 % [57.0, 68.2] | 39.3 % [31.1, 45.7] |
| U-Net B1 v5 | `≥ 200 px` (`2 ha`) | `60 / 13 / 106` | 36.1 % [28.7, 45.7] | 82.2 % [74.2, 89.6] | 50.2 % [41.9, 59.4] | 63.9 % [54.3, 71.3] |
| U-Net v6 (sin TTA+flt) | Total (`≥ 10 px`) | `217 / 189 / 71` | 75.3 % [69.6, 80.9] | 53.4 % [46.7, 61.2] | 62.5 % [56.6, 68.8] | 24.6 % [19.1, 30.4] |
| U-Net v6 (sin TTA+flt) | `≥ 50 px` (`0.5 ha`) | `164 / 93 / 71` | 69.8 % [63.9, 76.3] | 63.8 % [57.5, 71.5] | 66.7 % [61.6, 72.2] | 30.2 % [23.7, 36.0] |
| U-Net v6 (sin TTA+flt) | `≥ 200 px` (`2 ha`) | `58 / 15 / 71` | 45.0 % [37.5, 54.5] | 79.5 % [70.7, 89.4] | 57.4 % [49.8, 66.3] | 55.0 % [45.5, 62.5] |
| DeepLabV3+ v6 (sin TTA+flt) | Total (`≥ 10 px`) | `195 / 211 / 99` | 66.3 % [59.9, 73.0] | 48.0 % [40.0, 57.9] | 55.7 % [49.2, 62.7] | 33.7 % [27.0, 40.2] |
| DeepLabV3+ v6 (sin TTA+flt) | `≥ 50 px` (`0.5 ha`) | `139 / 118 / 99` | 58.4 % [51.6, 66.0] | 54.1 % [45.6, 64.3] | 56.2 % [49.8, 62.8] | 41.6 % [34.0, 48.4] |
| DeepLabV3+ v6 (sin TTA+flt) | `≥ 200 px` (`2 ha`) | `50 / 23 / 99` | 33.6 % [27.1, 42.2] | 68.5 % [52.7, 87.9] | 45.1 % [36.8, 54.8] | 66.4 % [57.8, 72.9] |

## 3. Contraste frente a las Metas del Proyecto (Todas las Formulaciones, «post hoc»)

Metas originales del proyecto: **`IoU ≥ 0.70`**, **`F1 > 75.5 %`**, **`Recall > 85 %`**, **`Falsos positivos ≤ 15 %`**.

| Meta del proyecto | Formulación evaluada | Mejor valor en `prueba` (360) *(modelo)* | Mejor valor en `L4SPe` test *(modelo)* | ¿Cumple la meta? |
|---|---|---|---|---|
| **IoU ≥ 0.70** | Píxel exacto (todas) | 0.1387 [0.0483, 0.2754] *(U-Net v6 (sin TTA+flt))* | 0.4297 [0.3739, 0.4872] *(U-Net B1 v5)* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 1 px (todas) | 0.1598 [0.0554, 0.3189] *(U-Net v6 (sin TTA+flt))* | 0.5470 [0.4809, 0.6137] *(U-Net B1 v5)* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 2 px (todas) | 0.1663 [0.0577, 0.3284] *(U-Net v6 (sin TTA+flt))* | 0.6087 [0.5396, 0.6780] *(U-Net B1 v5)* | **No** |
| **IoU ≥ 0.70** | Píxel exacto (solo `≥ 50 px`) | 0.1429 [0.0452, 0.2985] *(U-Net v6 (sin TTA+flt))* | 0.4391 [0.3825, 0.4974] *(U-Net B1 v5)* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 2 px (solo `≥ 50 px`) | 0.1703 [0.0536, 0.3584] *(U-Net v6 (sin TTA+flt))* | 0.6099 [0.5400, 0.6801] *(U-Net B1 v5)* | **No** |
| **F1 > 75.5 %** | Píxel exacto (todas) | 24.4 % [9.2, 43.2] *(U-Net v6 (sin TTA+flt))* | 60.1 % [54.4, 65.5] *(U-Net B1 v5)* | **No** |
| **F1 > 75.5 %** | Píxel tol. 2 px (todas) | 28.5 % [10.9, 49.4] *(U-Net v6 (sin TTA+flt))* | 75.7 % [70.1, 80.8] *(U-Net B1 v5)* | **Sí (L4SPe)** |
| **F1 > 75.5 %** | Objeto total (`≥ 10 px`) | 19.1 % [10.1, 27.8] *(U-Net v6 (sin TTA+flt))* | 62.5 % [56.6, 68.8] *(U-Net v6 (sin TTA+flt))* | **No** |
| **F1 > 75.5 %** | Objeto `≥ 50 px` (`0.5 ha`) | 24.0 % [9.5, 39.4] *(U-Net v6 (sin TTA+flt))* | 66.7 % [61.6, 72.2] *(U-Net v6 (sin TTA+flt))* | **No** |
| **F1 > 75.5 %** | Objeto `≥ 200 px` (`2 ha`) | 15.2 % [3.8, 29.5] *(U-Net v6 (sin TTA+flt))* | 57.4 % [49.8, 66.3] *(U-Net v6 (sin TTA+flt))* | **No** |
| **Recall > 85 %** | Píxel exacto (todas) | 34.2 % [14.4, 60.0] *(U-Net B1 v5)* | 58.5 % [51.5, 65.5] *(U-Net B1 v5)* | **No** |
| **Recall > 85 %** | Objeto total (`≥ 10 px`) | 22.5 % [14.3, 35.2] *(U-Net B1 v5)* | 54.2 % [47.7, 61.8] *(U-Net B1 v5)* | **No** |
| **Recall > 85 %** | Objeto `≥ 50 px` (`0.5 ha`) | 50.0 % [28.6, 66.7] *(U-Net B1 v5)* | 63.8 % [57.1, 71.4] *(U-Net B1 v5)* | **No** |
| **Recall > 85 %** | Objeto `≥ 200 px` (`2 ha`) | 63.6 % [27.3, 88.9] *(U-Net B1 v5)* | 82.2 % [74.2, 89.6] *(U-Net B1 v5)* | **No** |
| **FP ≤ 15 %** | Tasa FP píxel (`1 − Prec`, todas) | 78.7 % [52.8, 93.1] *(U-Net v6 (sin TTA+flt))* | 36.2 % [29.6, 42.8] *(DeepLabV3+ v6 (sin TTA+flt))* | **No** |
| **FP ≤ 15 %** | Tasa FP objeto (total `≥ 10 px`) | 80.5 % [63.9, 92.0] *(U-Net v6 (sin TTA+flt))* | 24.6 % [19.1, 30.4] *(U-Net v6 (sin TTA+flt))* | **No** |
| **FP ≤ 15 %** | Tasa FP objeto (`≥ 50 px`) | 83.8 % [68.5, 94.3] *(U-Net v6 (sin TTA+flt))* | 30.2 % [23.7, 36.0] *(U-Net v6 (sin TTA+flt))* | **No** |
