# Tarea 2 — IoU por Cicatrices Medibles («post hoc»)

> [!IMPORTANT]
> Todas las métricas corresponden a una evaluación **«post hoc»** sobre `L4SPe test` (`124` parches, `68` zonas) y `La Convención sellada` (`360` parches, `62` zonas), con **IC 95 % por bootstrap de zonas (`1 000` réplicas)**.

## ¿Por qué se reformula la métrica?
- **Resolución espacial de `10 m` (Sentinel-2):** en cicatrices pequeñas o estrechas (p. ej. `10–49 px`, ancho de `1–3 px`), un desplazamiento de apenas `1–2 px` (`10–20 m`) por corregistro o suavizado de bordes destruye el IoU exacto aunque el deslizamiento esté correctamente localizado.
- **Concordancia de la referencia humana (`0.30–0.39`):** en la auditoría de acuerdo entre anotadores (profesor vs rondas independientes en `acuerdo_anotadores.py`), el IoU inter-anotador sobre la misma imagen Sentinel-2 a `10 m` se sitúa entre `0.30` y `0.39`; exigir `IoU ≥ 0.70` a píxel exacto sobre todas las cicatrices `≥ 10 px` supera el techo de reproducibilidad de la propia etiqueta humana.
- Por ello se evalúan explícitamente: **a)** solo cicatrices medibles `≥ 0.5 ha` (`≥ 50 px`, pasando las menores a `255` dudoso/ignorado), **b)** tolerancia de borde de `1 px` y `2 px` (`8`-conectada), y **c)** la combinación de **a) y b)**.

## L4SPe Test (`124` parches, `68` zonas) — «post hoc»

| Modelo | Formulación | IoU [IC 95 %] | F1 [IC 95 %] | Recall [IC 95 %] |
|---|---|---|---|---|
| **v6** | Original (`≥ 0.1 ha`, `0 px`) | 0.4229 [0.3684, 0.4794] | 59.4 % [53.8, 64.8] | 57.3 % [50.3, 64.2] |
| **v6** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.4337 [0.3773, 0.4919] | 60.5 % [54.8, 65.9] | 60.1 % [53.0, 67.2] |
| **v6** | b1) Todas + tol. `1 px` | 0.5404 [0.4738, 0.6088] | 70.2 % [64.3, 75.7] | 66.8 % [59.5, 73.8] |
| **v6** | b2) Todas + tol. `2 px` | 0.6016 [0.5313, 0.6745] | 75.1 % [69.4, 80.6] | 71.3 % [64.1, 78.0] |
| **v6** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.5470 [0.4784, 0.6156] | 70.7 % [64.7, 76.2] | 69.4 % [62.0, 76.4] |
| **v6** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.6049 [0.5306, 0.6768] | 75.4 % [69.3, 80.7] | 73.9 % [66.6, 80.5] |
| **v8** | Original (`≥ 0.1 ha`, `0 px`) | 0.4303 [0.3760, 0.4840] | 60.2 % [54.6, 65.2] | 62.3 % [54.9, 69.5] |
| **v8** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.4400 [0.3841, 0.4945] | 61.1 % [55.5, 66.2] | 65.4 % [57.9, 72.7] |
| **v8** | b1) Todas + tol. `1 px` | 0.5450 [0.4780, 0.6113] | 70.5 % [64.7, 75.9] | 70.0 % [62.6, 77.2] |
| **v8** | b2) Todas + tol. `2 px` | 0.6053 [0.5346, 0.6756] | 75.4 % [69.7, 80.6] | 73.0 % [65.9, 80.1] |
| **v8** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.5517 [0.4831, 0.6187] | 71.1 % [65.1, 76.4] | 73.0 % [65.5, 80.3] |
| **v8** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.6095 [0.5351, 0.6807] | 75.7 % [69.7, 81.0] | 76.0 % [68.7, 83.1] |
| **v9** | Original (`≥ 0.1 ha`, `0 px`) | 0.4298 [0.3753, 0.4837] | 60.1 % [54.6, 65.2] | 62.4 % [55.0, 69.7] |
| **v9** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.4392 [0.3825, 0.4935] | 61.0 % [55.3, 66.1] | 65.5 % [58.0, 72.8] |
| **v9** | b1) Todas + tol. `1 px` | 0.5448 [0.4777, 0.6108] | 70.5 % [64.6, 75.8] | 70.2 % [62.8, 77.3] |
| **v9** | b2) Todas + tol. `2 px` | 0.6055 [0.5350, 0.6766] | 75.4 % [69.7, 80.7] | 73.2 % [66.1, 80.2] |
| **v9** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.5509 [0.4815, 0.6188] | 71.0 % [65.0, 76.4] | 73.1 % [65.7, 80.2] |
| **v9** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.6085 [0.5336, 0.6799] | 75.7 % [69.6, 81.0] | 76.1 % [68.9, 83.2] |

## La Convención Sellada (`360` parches, `62` zonas) — «post hoc»

| Modelo | Formulación | IoU [IC 95 %] | F1 [IC 95 %] | Recall [IC 95 %] |
|---|---|---|---|---|
| **v6** | Original (`≥ 0.1 ha`, `0 px`) | 0.1425 [0.0496, 0.2761] | 24.9 % [9.5, 43.3] | 31.7 % [11.9, 53.8] |
| **v6** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.1466 [0.0461, 0.2936] | 25.6 % [8.8, 45.4] | 35.7 % [12.3, 63.7] |
| **v6** | b1) Todas + tol. `1 px` | 0.1641 [0.0574, 0.3138] | 28.2 % [10.9, 47.8] | 36.5 % [13.7, 61.8] |
| **v6** | b2) Todas + tol. `2 px` | 0.1712 [0.0587, 0.3276] | 29.2 % [11.1, 49.4] | 38.1 % [14.4, 65.3] |
| **v6** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.1683 [0.0518, 0.3365] | 28.8 % [9.8, 50.4] | 40.9 % [14.2, 73.5] |
| **v6** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.1753 [0.0540, 0.3463] | 29.8 % [10.2, 51.4] | 42.7 % [14.6, 76.7] |
| **v8** | Original (`≥ 0.1 ha`, `0 px`) | 0.1138 [0.0343, 0.2626] | 20.4 % [6.6, 41.6] | 31.8 % [11.8, 57.5] |
| **v8** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.1147 [0.0314, 0.2764] | 20.6 % [6.1, 43.3] | 35.6 % [12.1, 68.0] |
| **v8** | b1) Todas + tol. `1 px` | 0.1311 [0.0402, 0.3163] | 23.2 % [7.7, 48.1] | 35.5 % [13.6, 62.1] |
| **v8** | b2) Todas + tol. `2 px` | 0.1348 [0.0409, 0.3216] | 23.8 % [7.9, 48.7] | 36.5 % [13.8, 63.1] |
| **v8** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.1309 [0.0363, 0.3247] | 23.2 % [7.0, 49.0] | 39.5 % [13.9, 73.4] |
| **v8** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.1347 [0.0369, 0.3340] | 23.8 % [7.1, 50.1] | 40.6 % [14.1, 74.4] |
| **v9** | Original (`≥ 0.1 ha`, `0 px`) | 0.1455 [0.0456, 0.2860] | 25.4 % [8.7, 44.5] | 34.3 % [12.6, 59.1] |
| **v9** | a) Solo `≥ 0.5 ha` (`0 px`) | 0.1487 [0.0437, 0.3014] | 25.9 % [8.4, 46.3] | 38.5 % [12.7, 69.7] |
| **v9** | b1) Todas + tol. `1 px` | 0.1686 [0.0538, 0.3292] | 28.8 % [10.2, 49.5] | 38.4 % [14.2, 64.8] |
| **v9** | b2) Todas + tol. `2 px` | 0.1744 [0.0551, 0.3379] | 29.7 % [10.4, 50.5] | 39.6 % [14.6, 67.3] |
| **v9** | c1) `≥ 0.5 ha` + tol. `1 px` | 0.1708 [0.0501, 0.3456] | 29.2 % [9.5, 51.4] | 42.9 % [14.3, 75.6] |
| **v9** | c2) `≥ 0.5 ha` + tol. `2 px` | 0.1768 [0.0510, 0.3565] | 30.0 % [9.7, 52.6] | 44.3 % [14.7, 78.4] |

