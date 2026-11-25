# Modelos Ajustados para La Convención y Perú (`modelos_ajustados/`)

> [!IMPORTANT]
> **Advertencia — Fuera del sistema desplegado:**
> Todos los modelos contenidos en `modelos_ajustados/` se mantienen **fuera del sistema desplegado** (`modelos/` no se modifica). Su integración en el servidor de producción o en Render requiere una decisión independiente posterior a la defensa y cierre experimental.

---

## Recomendación v9 (Combinación de `unet_v6` + `P3` con posproceso selectivo — `v9.json`)

- **¿Qué cambió frente a `v6` y `v8`?**
  1. Frente a **`v6`** (que era fuerte en La Convención pero perdía cicatrices frente al especialista L4SPe), **`v9`** incorpora el ensamble `0.6 × U-Net_v6 + 0.4 × P3_solo_profesor` con TTA de 4 espejos, umbral por histéresis (`0.70 / 0.25`) y morfología (`min_px = 20`, cierre de `1 px`), heredados de `v8`.
  2. Frente a **`v8`** (donde `P3` disparaba falsas alarmas en laderas altoandinas de roca y pajonal), **`v9` (`E1`)** anula el peso de `P3` (`peso P3 = 0`, quedando solo `U-Net_v6`) por encima de **`3 200 m`** o sobre cobertura **WorldCover `30` (pajonal) y `60` (suelo desnudo/escaso)**, manteniendo el filtro base `wc_urb_agua` (urbano `50` y agua `80`) sin corte fijo de altitud sobre `U-Net_v6`.
  3. **`v9` añade filtros selectivos sin costo en cicatrices reales (`E4b` y `E4c`):** elimina componentes que se solapan `> 20 %` con edificaciones/zonas residenciales de OSM (`E4b`) y componentes que ganaron verdor entre 2018–2020 y 2023–2025 (`ΔGRVI > +0.05`, rebrote de chacra o pasto, `E4c`).
  4. **Qué se probó y se descartó en validación:** el rescate de pequeñas `10–49 px` (`E3`) no recuperó cicatrices válidas (`0/15`), el filtro de rectangularidad (`E4a`) destruyó `62` cicatrices reales en `L4SPe val`, y el rescate de grandes `E2` elevó las falsas alarmas sobre `E1` por encima del tope de `v6`.
- **Efecto en validación:** en `La Convención val (304)` reduce las falsas alarmas a **`0.0263 ha/km²`** (`−26 %` frente a `v6 = 0.0355` y `−25 %` frente a `v8 = 0.0350`) y sube el `IoU` a **`0.1633`** (`v6 = 0.1542`), mientras en `L4SPe val (127)` conserva **`41/48` (`85.4 %`)** en cicatrices `≥ 2 ha` y `F1` por objeto de **`65.6 %`**. Receta completa en [v9.json](v9.json).

---

## Recomendación de Uso Operativo (Protocolo v6 — 24/11/2026)

- **Modelo principal recomendado para despliegue en Render: `unet_v6` (`U-Net · resnet34` con `GRVI` y `ExG` internos, umbral $u^* = 0.45$, SIN TTA + filtro `wc_dem_3200_agua_urb`).**
  - En **`prueba2` sellada (principal, 349 negativos confirmados por tipo)** reduce las falsas alarmas a **`0.0212 ha/km²`** (`5.4 %` de parches negativos con marca; **`−78 %`** frente a `unet_b1_v5` sin posproceso [`0.0948 ha/km²`, `14.9 %`] y **`−60 %`** frente a `unet_b1_v5` con el mismo filtro [`0.0529 ha/km²`, `9.7 %`]). En altoandino cae de `0.3483` (`53.1 %`) a **`0.0273 ha/km²`** (`10.2 %`).
  - En **`prueba` sellada antigua (360, «post hoc»)** alcanza **`IoU = 0.1387`** (`0.1598` tol `1 px` 8-con, `FA = 0.161 ha/km²`), y con **TTA 4 espejos** sube a **`IoU = 0.1453`** (`0.1667` tol `1 px` 8-con, `FA = 0.134 ha/km²`), superando en ambos casos a `unet_b1_v5` (`0.1311` / `0.220` sin posproceso; `0.1419` / `0.155` con TTA+filtro). En **`L4SPe` test («post hoc»)** mantiene **`IoU = 0.4229`** sin TTA y **`0.4249`** con TTA.
  - **Costo en CPU (ONNX Runtime):** **`18.6 ms/parche`** sin TTA vs **`60.9 ms/parche`** con TTA 4 (`3.28×`). Para servicio web en Render se recomienda **sin TTA + filtro `wc_dem_3200_agua_urb`**; para análisis fuera de línea (informes/mapas de tesis) puede activarse **TTA 4** (`+0.0066` IoU post hoc y `−0.027 ha/km²` FA en prueba 360).
- **Modelo alternativo: `deeplabv3plus_v6` (`DeepLabV3+ · resnet34`, umbral $u^* = 0.70$ sin TTA / $u^* = 0.60$ con TTA + filtro `wc_dem_3200_agua_urb`).**
  - Reduce las falsas alarmas en `prueba2` a **`0.0249 ha/km²`** con TTA (`0.0259` sin TTA) frente a `0.1056 ha/km²` de `deeplabv3plus_b2_v5` (`−76 %`), pero en `prueba` 360 («post hoc») conserva falsas alarmas altas (`1.408 ha/km²`, `IoU = 0.0193`) en laderas secas por debajo de 3 200 m, lo que también penaliza el ensamble `0.5/0.5` en `prueba` 360 (`IoU = 0.0477`, `FA = 0.537 ha/km²`).

---

## 0. Modelos Protocolo v6 (`unet_v6` y `deeplabv3plus_v6` — 24/11/2026)

### 0.1 `unet_v6` — U-Net Ajustada (Modelo Principal v6)
- **Archivos:** `unet_v6.onnx` (`97.8 MB`, SHA-256: `5419f7e38bb5bd814ae04a29ffa3f01748190cd42004920f9b6c5f6a1a6dcaa3`) + receta `unet_v6.json`.
- **Estado:** **Fuera del sistema desplegado** (`modelos/` intacto).
- **Firma / Época / Semilla:** `7c0b35452807` (`Control_B1_unet_s42`), época **`24`** (`alfa = 1.0`), entrenada sobre datos de Tarea 4 (fragmentos `10–19 px` de entrenamiento devueltos de fondo `0` a `DUDOSO 255`, sin `10a/10b`).
- **Paridad PyTorch / ONNX:** diferencia máxima de logit **`4.29e-06`** (`opset 17`, `46` capas convolucionales, `0` sin entrenar).
- **Filtro posproceso (`wc_dem_3200_agua_urb`):** suprime predicciones en `DEM ≥ 3 200 m` sobre clases ESA WorldCover 10 m `{20, 30, 60, 70, 100}` y en clases `{50: urbano, 80: agua}` a cualquier altitud (fuente: colección `esa-worldcover` 2021 v200 de Planetary Computer).
- **Umbral operativo:** **`0.45`** (con y sin TTA). **Latencia ORT CPU:** `18.6 ms/parche` sin TTA | `60.9 ms/parche` con TTA 4.

### 0.2 `deeplabv3plus_v6` — DeepLabV3+ Ajustada (Modelo Alternativo v6)
- **Archivos:** `deeplabv3plus_v6.onnx` (`89.8 MB`, SHA-256: `94e56a68ff89f6bc5fa476506020f27fec6e7e2dc67ce4901c3f3beea04e36ca`) + receta `deeplabv3plus_v6.json`.
- **Estado:** **Fuera del sistema desplegado** (`modelos/` intacto).
- **Firma / Época / Semilla:** `ac82bc42d339` (`C2_deeplab_p2_cp_s42`), época **`28`** (`alfa = 1.0`), entrenada sobre datos de Tarea 4 + `peso_pequenas = 2.0` + `copy-paste = 0.5` (solo `manual_v4`).
- **Paridad PyTorch / ONNX:** diferencia máxima de logit **`8.58e-06`** (`opset 17`, `51` capas convolucionales, `0` sin entrenar).
- **Umbral operativo:** **`0.70`** sin TTA / **`0.60`** con TTA 4 + filtro `wc_dem_3200_agua_urb`. **Latencia ORT CPU:** `16.0 ms/parche` sin TTA | `55.4 ms/parche` con TTA 4.

---

## Recomendación Histórica Protocolo v5 (23/11/2026)

- **Modelo principal recomendado: `unet_b1_v5` (`U-Net · resnet34` con índices internos `GRVI` y `ExG`, umbral $u^* = 0.30$).**
  Obtiene el mejor desempeño tanto en la **`prueba` sellada de La Convención (`360` parches, 100 % `manual_v4`)** con **`IoU = 0.1311`** (`0.1506` con tolerancia de borde de `1 px` 8-con, `F1 = 0.2319`, `22.5 %` de cicatrices $\ge 10\text{ px}$ detectadas, `40.0 %` en `50–200 px` y `63.6 %` en `> 200 px`) y la menor tasa de falsas alarmas en teselas negativas (**`0.220 ha/km²`**), como en la **`prueba` del experto humano L4SPe (`124` parches)** con **`IoU = 0.4297`** (`0.5470` con tolerancia `1 px` 8-con, `F1 = 0.6011`, `54.2 %` de cicatrices detectadas y `82.2 %` en `> 200 px`).
- **Modelo alternativo: `deeplabv3plus_b2_v5` (`DeepLabV3+ · resnet34`, umbral $u^* = 0.40$).**
  Alcanza un desempeño sólido en la **`prueba` de L4SPe (`124` parches)** con **`IoU = 0.4029`** (`0.5152` tol `1 px` 8-con, `F1 = 0.5744`, falsas alarmas `0.353 ha/km²`) y detecta `63.6 %` (`7/11`) de los grandes deslizamientos (`> 200 px`) en la `prueba` sellada de La Convención (o `72.7 %` [`8/11`] cuando se combina en el **Ensamble `B1 + B2`** con $u^* = 0.35$).
  **Aviso importante:** `deeplabv3plus_b2_v5` es propenso a **falsas alarmas extensas en laderas altoandinas naturalmente desnudas** (pajonal seco y afloramientos rocosos con $\text{GRVI} < 0$ y pendiente $> 25^\circ$, como en Vilcabamba altoandino `ZN41_VILCABAMBA_001_q3` o Huarocondo `ZN47_HUAROCONDO_001`), elevando las falsas alarmas en La Convención a `1.426 ha/km²`.

---

## 1. `unet_b1_v5` — U-Net Ajustada desde Base v4 (Modelo Principal, 23/11/2026)

- **Archivos:** `unet_b1_v5.onnx` (`97.8 MB`, SHA-256: `b459854db20f67f2b2ec8fac3a2f5be5c343cdbbddff8a9bbd55d2c5d7419af7`) + receta `unet_b1_v5.json`.
- **Estado:** **Fuera del sistema desplegado** (`modelos/` intacto).
- **Arquitectura y entrada:** `U-Net` con codificador `resnet34`, entrada de `6` canales (`['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion']` a `128×128 px`) y cálculo interno dentro del grafo ONNX de los índices de verdor `GRVI` y `ExG` (`8` canales efectivos al codificador). Normalización `media_desv` de Landslide4Sense.
- **Firma del entrenamiento:** `dfbbe6695037` (punto de partida: checkpoint base v4 `mejor_unet_b1b4b13b14_b2809be440d6.pt`, MD5 `abad6e808c27a43ef766e6ca2a169b23`, sin datos previos de La Convención).
- **Configuración de ajuste fino:** `--sin-l4s --debiles --frac-l4spe 0.5 --peso-l4spe 1.0 --peso-clasificador-v5 0.5 --peso-val-l4spe 0.7`, red completa con `BatchNorm` congelado, `lr = 3e-5`, `15` épocas de `100` lotes sobre `560` parches Nivel 1 de entrenamiento de La Convención (`152` `manual_v4` peso `1.0` + `408` `clasificador_v5` limpio peso `0.5`), `1,148` parches adicionales Nivel 2 sin fuga espacial ($\ge 3\text{ km}$) y `587` parches L4SPe del profesor (peso `1.0`).
- **Época elegida y umbral operativo:** **Época `5`**, seleccionada por el promedio ponderado `0.70` validación L4SPe (`0.3712` a $u=0.50$; `0.3741` a $u=0.30$) + `0.30` validación Nivel 1 La Convención (`0.1260` a $u=0.50$; `0.1638` a $u=0.30$), IoU ponderado `0.2977` (`0.3110` en el **umbral operativo $u^* = 0.30$**).
- **Paridad PyTorch / ONNX:** diferencia máxima de logit **`1.72e-05`** (`opset 17`, `46` capas convolucionales verificadas, `0` capas en inicialización).

### Métricas de `unet_b1_v5` ($u^* = 0.30$) en los 3 Conjuntos de Prueba

| Conjunto de Prueba | Base v4 (`unet_base_v4`, $u=0.25$) | `unet_b1_v5` ($u^* = 0.30$) | Tol. `1 px` (4c / 8c) | F1 píxel (`Prec.` / `Exhaus.`) | Detección de cicatrices ($\ge 30\%$ solape): `10–19` / `20–50` / `50–200` / `> 200 px` (Total $\ge 10\text{ px}$) | Falsas alarmas (`ha/km²`) | IC 95 % $\Delta\text{IoU}$ vs. base v4 (bootstrap por zona) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `prueba` sellada La Convención** (`360` parches, `62` zonas, 100 % `manual_v4`) | `IoU = 0.0113` (`FA = 12.470`) | **`IoU = 0.1311`** | **`0.1476` / `0.1506`** | **`0.2319`** (`0.1755` / `0.3417`) | `1/27` (`3.7%`) / `4/27` (`14.8%`) / **`6/15` (`40.0%`)** / **`7/11` (`63.6%`)** (**Total: `18/80` [`22.5%`]**) | **`0.220`** | **`[+0.0413, +0.2206]`** (gana en `100 %` de réplicas) |
| **2. `prueba` L4SPe Profesor** (`124` parches, `68` zonas, máscara experto) | `IoU = 0.3558` (`FA = 0.072`) | **`IoU = 0.4297`** | **`0.5224` / `0.5470`** | **`0.6011`** (`0.6178` / `0.5853`) | `10/26` (`38.5%`) / `46/124` (`37.1%`) / **`104/183` (`56.8%`)** / **`60/73` (`82.2%`)** (**Total: `220/406` [`54.2%`]**) | **`0.481`** | **`[+0.0452, +0.1010]`** (gana en `100 %` de réplicas) |
| **3. `prueba_externa`** (`16` parches, `4` teselas fuera de La Convención, máscara heurística L2A)* | `IoU = 0.6335` | **`IoU = 0.0145`** | `0.0212` / `0.0234` | `0.0286` (`0.6425` / `0.0146`) | `1/21` (`4.8%`) / `4/44` (`9.1%`) / `0/25` (`0.0%`) / `1/25` (`4.0%`) (Total: `6/115` [`5.2%`]) | `0.000` | `[-0.7309, -0.0322]` |

*\*Véase `informes/prueba_externa_nota.md`: la máscara de `prueba_externa` es una heurística de suelo desnudo en pendiente ($\text{GRVI} \le -0.03 \land \text{pend} \ge 20^\circ$) dominada en un `97.6 %` por vertientes altoandinas semiáridas de Ollantaytambo y Huarocondo; `unet_b1_v5` rechaza correctamente esas laderas desnudas sin deslizamiento.*

---

## 2. `deeplabv3plus_b2_v5` — DeepLabV3+ Ajustada desde Base v4 (Modelo Alternativo, 23/11/2026)

- **Archivos:** `deeplabv3plus_b2_v5.onnx` (`89.8 MB`, SHA-256: `09f6a89cd8728bbf1cefeef1ca888b77a4e1f733fa7fa55a967c5877a1724397`) + receta `deeplabv3plus_b2_v5.json`.
- **Estado:** **Fuera del sistema desplegado** (`modelos/` intacto).
- **Arquitectura y entrada:** `DeepLabV3+` con codificador `resnet34`, entrada de `6` canales (`['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion']` a `128×128 px`, sin índices internos). Normalización `media_desv` de Landslide4Sense.
- **Firma del entrenamiento:** `3539764d8289` (punto de partida: checkpoint base v4 `mejor_deeplabv3plus_b1b4b13b14_e838f4a37bf6.pt`, MD5 `ea1d3fb06830d0d2448f6db4285e3495`, sin datos previos de La Convención).
- **Configuración de ajuste fino:** Misma partición y pesos que `B1` (`--sin-l4s --debiles --frac-l4spe 0.5 --peso-l4spe 1.0 --peso-clasificador-v5 0.5 --peso-val-l4spe 0.7`), red completa con `BatchNorm` congelado, `lr = 3e-5`, **`30` épocas** de `100` lotes.
- **Época elegida y umbral operativo:** **Época `18`**, seleccionada por el promedio ponderado `0.70` validación L4SPe (`0.3863` a $u=0.50$; `0.3858` a $u=0.40$) + `0.30` validación Nivel 1 La Convención (`0.0372` a $u=0.50$; `0.0398` a $u=0.40$), IoU ponderado `0.2816` (`0.2820` en el **umbral operativo $u^* = 0.40$**).
- **Paridad PyTorch / ONNX:** diferencia máxima de logit **`7.63e-06`** (`opset 17`, `51` capas convolucionales verificadas, `0` capas en inicialización).
- **Advertencia operativa:** Presenta falsas alarmas elevadas (`1.426 ha/km²`) cuando se aplica sobre vertientes altoandinas secas o afloramientos rocosos naturales sin cobertura forestal (`ZN41_VILCABAMBA_001_q3`, `ZN47_HUAROCONDO_001`).

### Métricas de `deeplabv3plus_b2_v5` ($u^* = 0.40$) en los 3 Conjuntos de Prueba

| Conjunto de Prueba | Base v4 (`deeplabv3plus_base_v4`, $u=0.20$) | `deeplabv3plus_b2_v5` ($u^* = 0.40$) | Tol. `1 px` (4c / 8c) | F1 píxel (`Prec.` / `Exhaus.`) | Detección de cicatrices ($\ge 30\%$ solape): `10–19` / `20–50` / `50–200` / `> 200 px` (Total $\ge 10\text{ px}$) | Falsas alarmas (`ha/km²`) | IC 95 % $\Delta\text{IoU}$ vs. base v4 (bootstrap por zona) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `prueba` sellada La Convención** (`360` parches, `62` zonas, 100 % `manual_v4`) | `IoU = 0.0030` (`FA = 2.367`) | **`IoU = 0.0187`** | **`0.0218` / `0.0226`** | **`0.0367`** (`0.0209` / `0.1506`) | `0/27` (`0.0%`) / `0/27` (`0.0%`) / `3/15` (`20.0%`) / **`7/11` (`63.6%`)** (Total: `10/80` [`12.5%`]) | `1.426` | **`[+0.0024, +0.0776]`** (gana en `99.5 %` de réplicas) |
| **2. `prueba` L4SPe Profesor** (`124` parches, `68` zonas, máscara experto) | `IoU = 0.2188` (`FA = 0.000`) | **`IoU = 0.4029`** | **`0.4897` / `0.5152`** | **`0.5744`** (`0.6337` / `0.5252`) | `9/26` (`34.6%`) / `38/124` (`30.6%`) / **`88/183` (`48.1%`)** / **`49/73` (`67.1%`)** (**Total: `184/406` [`45.3%`]**) | **`0.353`** | **`[+0.0894, +0.2627]`** (gana en `100 %` de réplicas) |
| **3. `prueba_externa`** (`16` parches, `4` teselas fuera de La Convención, máscara heurística L2A) | `IoU = 0.6757` | **`IoU = 0.4169`** | `0.4756` / `0.4912` | `0.5885` (`0.8545` / `0.4488`) | `1/21` (`4.8%`) / `4/44` (`9.1%`) / `0/25` (`0.0%`) / `9/25` (`36.0%`) (Total: `14/115` [`12.2%`]) | `0.000` | `[-0.4508, -0.1543]` |

---

## 3. Modelos Anteriores de Referencia (`unet_peru` y `deeplabv3plus_peru` — 18–19/11/2026)

> [!WARNING]
> `unet_peru.onnx` (firma `809315dc820e`, umbral `0.40`) y `deeplabv3plus_peru.onnx` (firma `2551b2eefe0c`, umbral `0.40`) se entrenaron únicamente con máscaras provisionales de Nivel 2 y con la partición previa a la auditoría de solape ($\le 10\%$) y distancia mínima ($\ge 3\text{ km}$). Se conservan en esta carpeta únicamente para reproducibilidad histórica; para todo uso y reporte de la tesis deben emplearse **`unet_b1_v5`** y **`deeplabv3plus_b2_v5`**.
