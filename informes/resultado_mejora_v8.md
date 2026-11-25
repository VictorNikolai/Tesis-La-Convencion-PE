# Resultados Mejora v8 — Evaluación Única «Post Hoc» (Tarea 5)

> [!IMPORTANT]
> **Regla metodológica:** Todos los hiperparámetros de histéresis, post-proceso morfológico y pesos de ensamble fueron elegidos **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Todas las cifras sobre `L4SPe test`, `prueba` (360) y `prueba2` son **«post hoc»**, con **IC 95 % por bootstrap de zonas (`2 000` réplicas)**.

## 1. L4SPe Test (`124` parches, `68` zonas) — «post hoc»

### 1a. Por píxel (tolerancia `0`, `1` y `2 px`)

| Modelo | IoU 0 px [IC 95 %] | F1 0 px [IC 95 %] | IoU 1 px [IC 95 %] | F1 1 px [IC 95 %] | IoU 2 px [IC 95 %] | **F1 2 px [IC 95 %]** | FA neg. (`ha/km²`) |
|---|---|---|---|---|---|---|---:|
| U-Net B1 v5 (ref) | 0.4297 [0.3739, 0.4872] | 60.1 % [54.4, 65.5] | 0.5470 [0.4809, 0.6137] | 70.7 % [65.0, 76.1] | 0.6087 [0.5396, 0.6780] | **75.7 % [70.1, 80.8]** | 0.481 |
| U-Net v6 (sin TTA+flt) | 0.4229 [0.3687, 0.4803] | 59.4 % [53.9, 64.9] | 0.5404 [0.4749, 0.6090] | 70.2 % [64.4, 75.7] | 0.6016 [0.5332, 0.6738] | **75.1 % [69.6, 80.5]** | 0.612 |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | 0.4303 [0.3777, 0.4850] | 60.2 % [54.8, 65.3] | 0.5450 [0.4795, 0.6118] | 70.5 % [64.8, 75.9] | 0.6053 [0.5360, 0.6763] | **75.4 % [69.8, 80.7]** | 0.823 |

### 1b. Por objeto en L4SPe Test (`total ≥ 10 px`, `≥ 0.5 ha`, `≥ 2 ha`)

| Modelo | Tramo | TP / FN / FP | Precisión [IC 95 %] | **Exhaustividad [IC 95 %]** | **F1 objeto [IC 95 %]** |
|---|---|---|---:|---:|---:|
| U-Net B1 v5 (ref) | Total (`≥ 0.1 ha`) | `220 / 186 / 106` | 67.5 % [61.5, 74.5] | **54.2 % [47.7, 61.8]** | **60.1 % [54.8, 66.0]** |
| U-Net B1 v5 (ref) | `≥ 0.5 ha` (`50 px`) | `164 / 93 / 106` | 60.7 % [54.3, 68.9] | **63.8 % [57.1, 71.4]** | **62.2 % [57.0, 68.2]** |
| U-Net B1 v5 (ref) | `≥ 2 ha` (`200 px`) | `60 / 13 / 106` | 36.1 % [28.7, 45.7] | **82.2 % [74.2, 89.6]** | **50.2 % [41.9, 59.4]** |
| U-Net v6 (sin TTA+flt) | Total (`≥ 0.1 ha`) | `217 / 189 / 71` | 75.3 % [69.6, 80.9] | **53.4 % [46.7, 61.2]** | **62.5 % [56.6, 68.8]** |
| U-Net v6 (sin TTA+flt) | `≥ 0.5 ha` (`50 px`) | `164 / 93 / 71` | 69.8 % [63.9, 76.3] | **63.8 % [57.5, 71.5]** | **66.7 % [61.6, 72.2]** |
| U-Net v6 (sin TTA+flt) | `≥ 2 ha` (`200 px`) | `58 / 15 / 71` | 45.0 % [37.5, 54.5] | **79.5 % [70.7, 89.4]** | **57.4 % [49.8, 66.3]** |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | Total (`≥ 0.1 ha`) | `226 / 180 / 36` | 86.3 % [81.7, 90.6] | **55.7 % [49.5, 62.9]** | **67.7 % [62.4, 73.3]** |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | `≥ 0.5 ha` (`50 px`) | `174 / 83 / 36` | 82.9 % [77.5, 88.1] | **67.7 % [60.9, 76.1]** | **74.5 % [69.3, 80.2]** |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | `≥ 2 ha` (`200 px`) | `61 / 12 / 36` | 62.9 % [53.6, 72.9] | **83.6 % [75.0, 94.1]** | **71.8 % [64.2, 79.7]** |

## 2. Prueba Sellada de La Convención (`360` parches, `62` zonas) — «post hoc»

| Modelo | IoU 0 px [IC 95 %] | IoU 1 px [IC 95 %] | IoU 2 px [IC 95 %] | F1 2 px [IC 95 %] | Recall obj `≥ 0.5 ha` | Recall obj `≥ 2 ha` | FA neg. (`ha/km²`) |
|---|---|---|---|---|---|---|---:|
| U-Net B1 v5 (ref) | 0.1311 [0.0530, 0.2355] | 0.1506 [0.0614, 0.2677] | 0.1572 [0.0646, 0.2762] | 27.2 % [12.1, 43.3] | 50.0 % [28.6, 66.7] | 63.6 % [27.3, 88.9] | 0.220 |
| U-Net v6 (sin TTA+flt) | 0.1387 [0.0483, 0.2754] | 0.1598 [0.0554, 0.3189] | 0.1663 [0.0577, 0.3284] | 28.5 % [10.9, 49.4] | 46.2 % [26.9, 61.5] | 54.5 % [25.0, 85.7] | 0.161 |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | 0.1138 [0.0336, 0.2625] | 0.1311 [0.0392, 0.3094] | 0.1348 [0.0409, 0.3202] | 23.8 % [7.8, 48.5] | 42.3 % [21.4, 60.0] | 54.5 % [25.0, 85.7] | 0.298 |

## 3. `prueba2` Sellada (`349` negativos confirmados por `tipo_negativo`) — Verificación de Falsas Alarmas

| Modelo | **TOTAL (`ha/km²` / `%`)** | `altoandino` (49) | `chacra_o_pasto` (9) | `rio` (34) | `carretera` (8) | `poblado` (6) | `bosque_ladera` (243) |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net B1 v5 (ref) | **0.0948 (14.9 %)** | 0.3483 | 0.5876 | 0.0823 | 0.0000 | 0.0845 | 0.0306 |
| U-Net v6 (sin TTA+flt) | **0.0212 (5.4 %)** | 0.0273 | 0.3196 | 0.0086 | 0.0000 | 0.0947 | 0.0096 |
| Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20) | **0.0499 (3.2 %)** | 0.2301 | 0.2700 | 0.0278 | 0.0000 | 0.1181 | 0.0085 |

## 4. Meta del Proyecto frente al Mejor Valor (Todas las Formulaciones, «post hoc»)

| Meta del proyecto | Formulación evaluada | Mejor valor en `prueba` (360) *(modelo)* | Mejor valor en `L4SPe` test *(modelo)* | ¿Cumple la meta? |
|---|---|---|---|---|
| **IoU ≥ 0.70** | Píxel exacto (0 px) | 0.1387 [0.0483, 0.2754] *(U-Net v6 (sin TTA+flt))* | 0.4303 [0.3777, 0.4850] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 1 px | 0.1598 [0.0554, 0.3189] *(U-Net v6 (sin TTA+flt))* | 0.5470 [0.4809, 0.6137] *(U-Net B1 v5 (ref))* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 2 px | 0.1663 [0.0577, 0.3284] *(U-Net v6 (sin TTA+flt))* | 0.6087 [0.5396, 0.6780] *(U-Net B1 v5 (ref))* | **No** |
| **IoU ≥ 0.70** | Píxel tol. 2 px (solo `≥ 0.5 ha`) | 0.1703 [0.0536, 0.3584] *(U-Net v6 (sin TTA+flt))* | 0.6099 [0.5400, 0.6801] *(U-Net B1 v5 (ref))* | **No** |
| **F1 > 75.5 %** | Píxel exacto (0 px) | 24.4 % [9.2, 43.2] *(U-Net v6 (sin TTA+flt))* | 60.2 % [54.8, 65.3] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **F1 > 75.5 %** | Píxel tol. 1 px | 27.6 % [10.5, 48.4] *(U-Net v6 (sin TTA+flt))* | 70.7 % [65.0, 76.1] *(U-Net B1 v5 (ref))* | **No** |
| **F1 > 75.5 %** | Píxel tol. 2 px | 28.5 % [10.9, 49.4] *(U-Net v6 (sin TTA+flt))* | 75.7 % [70.1, 80.8] *(U-Net B1 v5 (ref))* | **Sí (L4SPe)** |
| **F1 > 75.5 %** | Objeto total (`≥ 0.1 ha`) | 25.2 % [13.8, 34.9] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | 67.7 % [62.4, 73.3] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **F1 > 75.5 %** | Objeto `≥ 0.5 ha` (`50 px`) | 36.1 % [16.1, 52.3] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | 74.5 % [69.3, 80.2] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **F1 > 75.5 %** | Objeto `≥ 2 ha` (`200 px`) | 29.3 % [8.7, 47.8] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | 71.8 % [64.2, 79.7] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **Recall > 85 %** | Píxel exacto (0 px) | 34.2 % [14.4, 60.0] *(U-Net B1 v5 (ref))* | 62.3 % [55.0, 69.6] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **Recall > 85 %** | Objeto total (`≥ 0.1 ha`) | 22.5 % [14.3, 35.2] *(U-Net B1 v5 (ref))* | 55.7 % [49.5, 62.9] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **Recall > 85 %** | Objeto `≥ 0.5 ha` (`50 px`) | 50.0 % [28.6, 66.7] *(U-Net B1 v5 (ref))* | 67.7 % [60.9, 76.1] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **Recall > 85 %** | Objeto `≥ 2 ha` (`200 px`) | 63.6 % [27.3, 88.9] *(U-Net B1 v5 (ref))* | 83.6 % [75.0, 94.1] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
| **FP ≤ 15 %** | Tasa FP píxel (`1 − Prec`) | 78.7 % [52.8, 93.1] *(U-Net v6 (sin TTA+flt))* | 38.2 % [32.0, 44.2] *(U-Net B1 v5 (ref))* | **No** |
| **FP ≤ 15 %** | Tasa FP objeto (`≥ 0.1 ha`) | 61.5 % [39.5, 82.9] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | 13.7 % [9.4, 18.3] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **Sí (L4SPe)** |
| **FP ≤ 15 %** | Tasa FP objeto (`≥ 0.5 ha`) | 68.6 % [46.4, 88.1] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | 17.1 % [11.9, 22.5] *(Ganador v8 (U-Net_v6 0.6 + P3 0.4, TTA4, hist 0.70/0.25, min20))* | **No** |
