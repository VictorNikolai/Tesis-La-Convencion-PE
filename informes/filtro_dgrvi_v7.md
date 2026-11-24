# Filtro de Cambio entre Fechas ΔGRVI — Tarea 6 (Validación y Evaluación «Post Hoc»)

> [!IMPORTANT]
> **Regla aplicada:** El umbral `t` de `ΔGRVI = GRVI_actual − GRVI_anterior` (se suprimen píxeles con escena anterior despejada `B02 < 1000` y `ΔGRVI > t`) se barrió en `[-0.10, +0.05]` **exclusivamente en validación** (`304` parches). La evaluación en `prueba` (`360`) y `prueba2` (`349` negativos) se realizó **una sola vez («post hoc»)**.
> 
> **Cobertura de escenas anteriores (`2018–2020`):** `validacion`: `304/304`, `prueba`: `358/360`, `prueba2`: `564/564`. **L4SPe (`l4spe_profesor.npz`) NO dispone de fecha ni imagen anterior**, por lo que este filtro no aplica sobre L4SPe.

## 1. Barrido en Validación (`U-Net v6 sin TTA`, `u = 0.45`) — Efecto de `t` y WorldCover (con y sin corte de altitud)

| Filtro WorldCover | `t` (`ΔGRVI > t` se borra) | IoU val LC | F1 val LC | FA val (`ha/km²`) | Cicatrices tocadas (borradas) | Det. `≥ 30 %` (borradas) |
|---|---|---:|---:|---:|---|---|
| `ninguno` | `sin_dgrvi` | 0.1543 | 0.2674 | 0.036 | 6/63 (−0) | 4/59 (−0) |
| `ninguno` | `-0.10` | 0.0317 | 0.0614 | 0.008 | 4/63 (−2) | 1/59 (−3) |
| `ninguno` | `-0.08` | 0.0337 | 0.0653 | 0.009 | 5/63 (−1) | 1/59 (−3) |
| `ninguno` | `-0.06` | 0.0385 | 0.0741 | 0.011 | 5/63 (−1) | 1/59 (−3) |
| `ninguno` | `-0.04` | 0.0466 | 0.0891 | 0.013 | 5/63 (−1) | 2/59 (−2) |
| `ninguno` | `-0.02` | 0.0611 | 0.1151 | 0.017 | 5/63 (−1) | 2/59 (−2) |
| `ninguno` | `+0.00` | 0.0939 | 0.1716 | 0.026 | 6/63 (−0) | 2/59 (−2) |
| `ninguno` | `+0.02` | 0.1374 | 0.2416 | 0.033 | 6/63 (−0) | 4/59 (−0) |
| `ninguno` | `+0.05` | 0.1525 | 0.2647 | 0.035 | 6/63 (−0) | 4/59 (−0) |
| `wc_urb_agua` | `sin_dgrvi` | 0.1542 | 0.2672 | 0.036 | 6/63 (−0) | 4/59 (−0) |
| `wc_urb_agua` | `-0.10` | 0.0317 | 0.0614 | 0.008 | 4/63 (−2) | 1/59 (−3) |
| `wc_urb_agua` | `-0.08` | 0.0338 | 0.0653 | 0.009 | 5/63 (−1) | 1/59 (−3) |
| `wc_urb_agua` | `-0.06` | 0.0385 | 0.0741 | 0.010 | 5/63 (−1) | 1/59 (−3) |
| `wc_urb_agua` | `-0.04` | 0.0467 | 0.0892 | 0.013 | 5/63 (−1) | 2/59 (−2) |
| `wc_urb_agua` | `-0.02` | 0.0611 | 0.1151 | 0.017 | 5/63 (−1) | 2/59 (−2) |
| `wc_urb_agua` | `+0.00` | 0.0939 | 0.1717 | 0.026 | 6/63 (−0) | 2/59 (−2) |
| `wc_urb_agua` | `+0.02` | 0.1374 | 0.2415 | 0.032 | 6/63 (−0) | 4/59 (−0) |
| `wc_urb_agua` | `+0.05` | 0.1525 | 0.2646 | 0.035 | 6/63 (−0) | 4/59 (−0) |
| `wc_dem_3200_agua_urb` | `sin_dgrvi` | 0.1575 | 0.2722 | 0.027 | 6/63 (−0) | 4/59 (−0) |
| `wc_dem_3200_agua_urb` | `-0.10` | 0.0317 | 0.0614 | 0.008 | 4/63 (−2) | 1/59 (−3) |
| `wc_dem_3200_agua_urb` | `-0.08` | 0.0338 | 0.0654 | 0.009 | 5/63 (−1) | 1/59 (−3) |
| `wc_dem_3200_agua_urb` | `-0.06` | 0.0385 | 0.0742 | 0.010 | 5/63 (−1) | 1/59 (−3) |
| `wc_dem_3200_agua_urb` | `-0.04` | 0.0468 | 0.0893 | 0.012 | 5/63 (−1) | 2/59 (−2) |
| `wc_dem_3200_agua_urb` | `-0.02` | 0.0613 | 0.1155 | 0.015 | 5/63 (−1) | 2/59 (−2) |
| `wc_dem_3200_agua_urb` | `+0.00` | 0.0952 | 0.1739 | 0.020 | 6/63 (−0) | 2/59 (−2) |
| `wc_dem_3200_agua_urb` | `+0.02` | 0.1401 | 0.2458 | 0.025 | 6/63 (−0) | 4/59 (−0) |
| `wc_dem_3200_agua_urb` | `+0.05` | 0.1557 | 0.2694 | 0.027 | 6/63 (−0) | 4/59 (−0) |

## 2. Evaluación Única «Post Hoc» en `prueba` (360) y `prueba2` (349 negativos por tipo)

| Modelo | WorldCover | Filtro ΔGRVI (elegido en val.) | `prueba` IoU (tol 1 px) | `prueba` Det. `≥ 30 %` | `prueba` FA (`ha/km²`) | **`prueba2` TOTAL (`ha/km²` / `%`)** | `altoandino` | `chacra` | `rio` | `poblado` |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| U-Net v6 (sin TTA) | `ninguno` | `sin_dgrvi` | 0.1436 (0.1647) | 17/80 | 0.181 | **0.0307 (6.6 %)** | 0.0911 | 0.3325 | 0.0086 | 0.1049 |
| U-Net v6 (sin TTA) | `ninguno` | `dgrvi_opt(+0.05)` | 0.1464 (0.1691) | 17/80 | 0.167 | **0.0214 (6.3 %)** | 0.0623 | 0.1791 | 0.0040 | 0.0845 |
| U-Net v6 (sin TTA) | `wc_urb_agua` | `sin_dgrvi` | 0.1425 (0.1641) | 17/80 | 0.181 | **0.0305 (6.6 %)** | 0.0910 | 0.3325 | 0.0086 | 0.0947 |
| U-Net v6 (sin TTA) | `wc_urb_agua` | `dgrvi_opt(+0.05)` | 0.1455 (0.1685) | 17/80 | 0.167 | **0.0212 (6.3 %)** | 0.0622 | 0.1791 | 0.0040 | 0.0754 |
| U-Net v6 (sin TTA) | `wc_dem_3200_agua_urb` | `sin_dgrvi` | 0.1387 (0.1598) | 15/80 | 0.161 | **0.0212 (5.4 %)** | 0.0273 | 0.3196 | 0.0086 | 0.0947 |
| U-Net v6 (sin TTA) | `wc_dem_3200_agua_urb` | `dgrvi_opt(+0.05)` | 0.1403 (0.1625) | 15/80 | 0.151 | **0.0147 (5.2 %)** | 0.0182 | 0.1662 | 0.0040 | 0.0754 |
| U-Net B1 v5 | `ninguno` | `sin_dgrvi` | 0.1311 (0.1506) | 18/80 | 0.220 | **0.0948 (14.9 %)** | 0.3483 | 0.5876 | 0.0823 | 0.0845 |
| U-Net B1 v5 | `ninguno` | `dgrvi_opt(+0.05)` | 0.1368 (0.1579) | 18/80 | 0.197 | **0.0619 (14.9 %)** | 0.2137 | 0.2510 | 0.0384 | 0.0672 |
| U-Net B1 v5 | `wc_urb_agua` | `sin_dgrvi` | 0.1307 (0.1507) | 18/80 | 0.219 | **0.0946 (14.9 %)** | 0.3482 | 0.5876 | 0.0823 | 0.0764 |
| U-Net B1 v5 | `wc_urb_agua` | `dgrvi_opt(+0.05)` | 0.1365 (0.1579) | 18/80 | 0.197 | **0.0618 (14.9 %)** | 0.2136 | 0.2510 | 0.0384 | 0.0601 |
| U-Net B1 v5 | `wc_dem_3200_agua_urb` | `sin_dgrvi` | 0.1331 (0.1538) | 16/80 | 0.176 | **0.0529 (9.7 %)** | 0.0567 | 0.5550 | 0.0823 | 0.0764 |
| U-Net B1 v5 | `wc_dem_3200_agua_urb` | `dgrvi_opt(+0.05)` | 0.1374 (0.1592) | 16/80 | 0.162 | **0.0355 (9.7 %)** | 0.0322 | 0.2198 | 0.0384 | 0.0601 |
| DeepLabV3+ v6 (sin TTA) | `ninguno` | `sin_dgrvi` | 0.0214 (0.0262) | 11/80 | 1.480 | **0.0914 (13.5 %)** | 0.5065 | 0.3657 | 0.0251 | 0.0234 |
| DeepLabV3+ v6 (sin TTA) | `ninguno` | `dgrvi_opt(+0.05)` | 0.0231 (0.0284) | 11/80 | 1.326 | **0.0557 (13.2 %)** | 0.3108 | 0.2544 | 0.0075 | 0.0204 |
| DeepLabV3+ v6 (sin TTA) | `wc_urb_agua` | `sin_dgrvi` | 0.0214 (0.0262) | 11/80 | 1.476 | **0.0914 (13.5 %)** | 0.5062 | 0.3657 | 0.0251 | 0.0224 |
| DeepLabV3+ v6 (sin TTA) | `wc_urb_agua` | `dgrvi_opt(+0.05)` | 0.0232 (0.0284) | 11/80 | 1.322 | **0.0556 (13.2 %)** | 0.3106 | 0.2544 | 0.0075 | 0.0193 |
| DeepLabV3+ v6 (sin TTA) | `wc_dem_3200_agua_urb` | `sin_dgrvi` | 0.0184 (0.0228) | 9/80 | 1.443 | **0.0259 (8.0 %)** | 0.0472 | 0.3243 | 0.0251 | 0.0224 |
| DeepLabV3+ v6 (sin TTA) | `wc_dem_3200_agua_urb` | `dgrvi_opt(+0.05)` | 0.0199 (0.0246) | 9/80 | 1.292 | **0.0154 (7.2 %)** | 0.0305 | 0.2171 | 0.0075 | 0.0193 |
