# Resultados Mejora v9 — Evaluación Única «Post Hoc»

> [!IMPORTANT]
> Todos los parámetros de `E1–E4` y `E7` fueron elegidos **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Toda cifra sobre `prueba` (360), `prueba2` (349 negativos) y `L4SPe test` (124) es **«post hoc»** con IC 95 % por bootstrap de zonas (`2 000` réplicas).
> Nota técnica sobre `E3`: los TIF locales empaquetados tienen 6 canales (`B1, B2, B3, B4, pendiente, elevación`) sin banda `SCL`, por lo que la distancia a nube/sombra se evaluó mediante máscara espectral equivalente.

## 1. Comparación `v6 / v8 / v9` en las 3 Pruebas («post hoc»)

### 1a. `L4SPe` Test (`124` parches, `68` zonas)

| Modelo | IoU 0 px [IC 95 %] | F1 0 px [IC 95 %] | IoU 2 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |
|---|---|---|---|---|---|---|
| **v6** | 0.4229 [0.3687, 0.4803] | 59.4 % [53.9, 64.9] | 0.6016 [0.5332, 0.6738] | 75.1 % [69.6, 80.5] | 66.7 % [61.6, 72.2] | 79.5 % [70.7, 89.4] |
| **v8** | 0.4303 [0.3777, 0.4850] | 60.2 % [54.8, 65.3] | 0.6053 [0.5360, 0.6763] | 75.4 % [69.8, 80.7] | 74.5 % [69.3, 80.2] | 83.6 % [75.0, 94.1] |
| **v9** | 0.4298 [0.3771, 0.4847] | 60.1 % [54.8, 65.3] | 0.6055 [0.5351, 0.6771] | 75.4 % [69.7, 80.8] | 74.5 % [69.4, 80.0] | 83.6 % [75.0, 94.1] |

### 1b. `prueba` Sellada de La Convención (`360` parches, `62` zonas)

| Modelo | IoU 0 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] | FA neg. (`ha/km²`) |
|---|---|---|---|---|---:|
| **v6** | 0.1425 [0.0494, 0.2691] | 29.2 % [10.8, 48.7] | 25.0 % [9.5, 39.7] | 72.7 % [27.3, 100.0] | 0.181 |
| **v8** | 0.1138 [0.0336, 0.2625] | 23.8 % [7.8, 48.5] | 36.1 % [16.1, 52.3] | 54.5 % [25.0, 85.7] | 0.298 |
| **v9** | 0.1455 [0.0455, 0.2838] | 29.7 % [10.5, 50.9] | 34.7 % [13.6, 51.0] | 72.7 % [27.3, 100.0] | 0.193 |

### 1c. `prueba2` Sellada (`349` negativos confirmados por `tipo_negativo`)

| Modelo | **TOTAL (`ha/km²` / `%`)** | `altoandino` (49) | `chacra_o_pasto` (9) | `rio` (34) | `carretera` (8) | `poblado` (6) | `bosque_ladera` (243) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v6** | **0.0305 (6.6 %)** | 0.0910 | 0.3325 | 0.0086 | 0.0000 | 0.0947 | 0.0096 |
| **v8** | **0.0499 (3.2 %)** | 0.2301 | 0.2700 | 0.0278 | 0.0000 | 0.1181 | 0.0085 |
| **v9** | **0.0157 (2.3 %)** | 0.0614 | 0.0000 | 0.0000 | 0.0000 | 0.0692 | 0.0084 |

## 2. Verificación Independiente (`herramientas/verificar_metricas.py`)

- **Recálculo desde cero:** `iou_0px`, `f1_0px`, `obj10_f1`, `obj50_f1`, `obj200_rec` y `prueba2` (`0.0305` / `0.0499` / `0.0157 ha/km²`) coinciden con `informes/evaluacion_v9.json` con diferencia `< 1e-4`. Las únicas diferencias `> 1e-3` (`0.007–0.019`) ocurren en `iou_2px / f1_2px` porque `metricas_reformuladas.py` define `F1_tol` a partir de `prec_tol = tp_p / (tp_p + fp)` y `rec_tol = tp_g / (tp_g + fn)` (y `IoU_tol = F1_tol / (2 - F1_tol)`), mientras el script independiente evaluó `tp = min(tp_p, tp_g)` sobre `tp + fp + fn`.
- **Auditoría de fuga e integridad:** conteos exactos `560 / 304 / 360 / 564` (`conteos_ok = True`), `0` teselas compartidas, `0` zonas compartidas, `0` parches de prueba en entrenamiento, y distancia espacial mínima `>= 3 km` verificada por `test_herramientas.py`.
- **ONNX y paridad:** `modelos_ajustados/unet_v6.onnx` (`SHA-256: 5419f7e38bb5bd814ae04a29ffa3f01748190cd42004920f9b6c5f6a1a6dcaa3`), diferencia máxima absoluta de logit frente a PyTorch en `20` parches = `1.53e-05`.
- **Pruebas unitarias:** `pytest herramientas/test_herramientas.py -q` → `21 passed`.
