# Tarea 4 — Modelo U-Net de 9 Bandas (`B08`, `B11`, `B12`)

> [!IMPORTANT]
> **Limitación de dominio:** `l4spe_profesor.npz` solo contiene 6 canales (`B01–B04`, pendiente, elevación) y carece de `B08/B11/B12`, por lo que `unet_v10_9b` se entrenó **exclusivamente con los `560` parches de La Convención** (misma partición), partiendo de `U-Net v6` con los 3 canales nuevos inicializados con la media de los pesos RGB (`B02, B03, B04`), `lr = 3e-5`, `BatchNorm` congelado, `25` épocas en CPU (`17.5 min`).
> Época y umbral elegidos **solo en validación (`304`)**: época **`0`**, umbral **`0.40`**, `IoU val = 0.1366`.
> Paridad ONNX vs PyTorch en `20` parches: **`3.24e-05`**, SHA-256: `04ad2b4e7dce99a0...`.

## 1. Verificación de alineación píxel a píxel (`5` parches al azar)

| Parche | CRS idéntico | Shape IR | Δ transform | Corr. `B04` compuesto |
|---|---|---|---:|---:|
| `ZN25_ECHARATE_001_q4` | True | `[3, 128, 128]` | `0.0` | 0.4382 |
| `ingemmet_043_001_q2` | True | `[3, 128, 128]` | `0.0` | 0.8363 |
| `azar_048_001_q2` | True | `[3, 128, 128]` | `0.0` | 0.6661 |
| `sinpad_47364_001_q2` | True | `[3, 128, 128]` | `0.0` | 0.9988 |
| `ingemmet_041_001_q3` | True | `[3, 128, 128]` | `0.0` | 0.7451 |

## 2. Comparación «Post Hoc» en `prueba` sellada (`360` parches, `62` zonas)

| Modelo | IoU 0 px [IC 95 %] | F1 2 px [IC 95 %] | F1 obj `≥ 0.1 ha` [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |
|---|---|---|---|---|---|
| **v6 (6 bandas)** | 0.1425 [0.0494, 0.2691] | 29.2 % [10.8, 48.7] | 20.1 % [10.2, 29.2] | 25.0 % [9.5, 39.7] | 72.7 % [27.3, 100.0] |
| **v9 (6 bandas + E1/E4)** | 0.1455 [0.0455, 0.2838] | 29.7 % [10.5, 50.9] | 25.6 % [13.2, 36.0] | 34.7 % [13.6, 51.0] | 72.7 % [27.3, 100.0] |
| **`unet_v10_9b` (9 bandas)** | 0.1211 [0.0292, 0.2416] | 26.1 % [7.3, 45.7] | 12.4 % [4.2, 21.9] | 16.4 % [4.3, 30.6] | 63.6 % [14.3, 92.3] |

## 3. `prueba2` (`349` negativos confirmados por tipo de terreno, «post hoc»)

| Modelo | **TOTAL (`ha/km²` / `%`)** | `bosque_ladera` (243) | `altoandino` (49) | `rio` (34) | `chacra_o_pasto` (9) | `carretera` (8) | `poblado` (6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v6** | **0.0305 (6.6 %)** | 0.0096 | 0.0910 | 0.0086 | 0.3325 | 0.0000 | 0.0947 |
| **v9** | **0.0157 (2.3 %)** | 0.0084 | 0.0614 | 0.0000 | 0.0000 | 0.0000 | 0.0692 |
| **`unet_v10_9b`** | **0.0203 (6.6 %)** | 0.0121 | 0.0516 | 0.0020 | 0.1371 | 0.0000 | 0.0550 |

## 4. EXTRA — Comportamiento de las `39` falsas alarmas completas de `v9` bajo `unet_v10_9b`

| Tipo | Comp. `v9` | Área `v9` (`ha`) | Comp. que siguen en `9b` | Área que persiste (`ha`) | Reducción de área |
|---|---:|---:|---:|---:|---:|
| `a` | 9 | 21.83 | 2 | 7.88 | `-63.9 %` |
| `b` | 16 | 57.00 | 16 | 30.06 | `-47.3 %` |
| `c` | 4 | 8.61 | 4 | 5.25 | `-39.0 %` |
| `d` | 5 | 16.25 | 3 | 8.18 | `-49.7 %` |
| `e` | 3 | 5.71 | 3 | 5.34 | `-6.5 %` |
| `f` | 2 | 0.51 | 2 | 0.08 | `-84.3 %` |
