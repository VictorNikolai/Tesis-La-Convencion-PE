# Modelos Ajustados para La Convención y Perú (`modelos_ajustados/`)

> [!IMPORTANT]
> **Advertencia — Fuera del sistema desplegado:**
> Todos los modelos contenidos en `modelos_ajustados/` se mantienen **fuera del sistema desplegado** (`modelos/` no se modifica). Su integración en el servidor de producción o en Render requiere una decisión independiente posterior a la defensa y cierre experimental.

---

## Recomendación de Uso Operativo

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
