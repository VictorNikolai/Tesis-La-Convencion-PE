# Resultado Final — PARADA (i): Cierre de Condiciones A–C, Corridas B1/B2/A1/A2, Evaluación Sellada (E) y Exportación ONNX

---

## A. Corrección del Índice de Parches (`parches_128/`) y Verificación Estricta `560 / 304 / 360`

1. **Respaldo previo en `_respaldos/`:**
   - [_respaldos/indice_parches_antes_parada_i.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/_respaldos/indice_parches_antes_parada_i.csv) (`1,588` filas previas).
   - [_respaldos/indice_parches_ampliado_antes_parada_i.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/_respaldos/indice_parches_ampliado_antes_parada_i.csv) (`3,480` filas previas).
2. **Regeneración desde `particion.csv` por `tesela` (nunca por nombre de proyecto):**
   - Actualizado [herramientas/cortar_parches_128.py](file:///c:/Users/User/Documents/Tesis-Proyecto/herramientas/cortar_parches_128.py#L25-L37) para incluir `datos_prueba`, `datos_nuevos_ent` y `datos_val_dgrvi`.
   - Regenerados [parches_128/indice_parches.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/parches_128/indice_parches.csv) (`3,072` filas) y [parches_128/indice_parches_ampliado.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/parches_128/indice_parches_ampliado.csv) (`4,964` filas).
   - Las filas con `conjunto` en `{"excluida", "pendiente", "sin_zona"}` no se cargan nunca (`CONJUNTOS_EXCLUIDOS` en [herramientas/ajuste_fino.py](file:///c:/Users/User/Documents/Tesis-Proyecto/herramientas/ajuste_fino.py#L61-L86)).
3. **Verificación bloqueante en `ajuste_fino.py` (`verificar_conteos_nivel1`):**
   - Si los parches de Nivel 1 cargados no son exactamente **`560 / 304 / 360`**, `ajuste_fino.py` aborta con `SystemExit`.
   - Conteos exactos impresos al inicio de cada corrida y verificados por prueba unitaria:

| Conjunto Nivel 1 (La Convención) | Parches cargados | `manual_v4` (peso `1.0`) | `clasificador_v5` (peso `0.5`) | Parches positivos | Cicatrices $\ge 10\text{ px}$ (8-con) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`entrenamiento`** | **`560 / 560`** | `152` | `408` | `138` (`6` v4 + `132` v5) | `269` (`1` de `10–19 px`, `162` de `20–50 px`, `81` de `50–200 px`, `25` `> 200 px`) |
| **`validacion`** | **`304 / 304`** | `236` | `68` | `39` (`8` v4 + `31` v5) | `59` (`0` de `10–19 px`, `16` de `20–50 px`, `26` de `50–200 px`, `17` `> 200 px`) |
| **`prueba` (sellada)** | **`360 / 360`** | `360` (`100 %`) | `0` (`0 %`) | `38` (`38` v4 + `0` v5) | `80` (`27` de `10–19 px`, `27` de `20–50 px`, `15` de `50–200 px`, `11` `> 200 px`) |
| **Total Nivel 1 activo** | **`1,224`** | **`748`** | **`476`** | **`215`** | **`408`** |

4. **Prueba unitaria y `pytest`:**
   - Añadida `test_indice_parches_nivel1_560_304_360_y_nivel2_sin_fuga` en [herramientas/test_herramientas.py](file:///c:/Users/User/Documents/Tesis-Proyecto/herramientas/test_herramientas.py#L430-L473).
   - Resultado de `python -m pytest herramientas/test_herramientas.py -q`: **`17 passed`**.

---

## B. Auditoría y Eliminación de Fuga Espacial en Nivel 2 (`--debiles`)

- **Regla aplicada:** Un parche de Nivel 2 entra a entrenamiento **solo si** su `tesela` base es `"entrenamiento"` en `particion.csv` **y** está a **$\ge 3{,}000\text{ m}$ borde a borde** de toda tesela de `validacion`, `prueba` y `prueba_externa`.
- **Respaldos previos:** [_respaldos/mascaras_nivel2_completo_antes_parada_i.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/_respaldos/mascaras_nivel2_completo_antes_parada_i.csv) y [_respaldos/catalogo_parches_tiers_antes_parada_i.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/_respaldos/catalogo_parches_tiers_antes_parada_i.csv).
- **Lista completa de teselas que salieron:** guardada en [informes/nivel2_teselas_excluidas.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/informes/nivel2_teselas_excluidas.csv) (**`75` entradas de tesela = `44` teselas base únicas + `31` variantes multitemporales `*_antes`/`*_2025`, equivalentes a `863` parches**):
  - **`1` tesela pasó a `validacion`:** `datos_azar/azar_039_001` (`4` parches).
  - **`74` entradas (`43` teselas base únicas, `859` parches) pasaron a `excluida`:** incluye las 6 teselas a $< 3\text{ km}$ de `validacion` (`sinpad_91376_001` a `2,580.0 m`, `sinpad_63076_001` a `2,160.0 m`, `ingemmet_021_001` a `1,180.0 m`, `sinpad_92057_001` a `2,279.7 m`, `sinpad_43368_001` a `2,156.7 m`, `ingemmet_009_001` a `2,906.9 m`) y las 37 teselas duplicadas (`> 10 %` de solape) que estaban `excluida` en `particion.csv`.
- **Conteo de Nivel 2 que queda en `entrenamiento`:**
  - **`149` entradas de tesela** (`90` teselas base únicas + `59` variantes multitemporales), todas con distancia mínima borde a borde **$\ge 3{,}054.1\text{ m}$** respecto a `validacion`, `prueba` y `prueba_externa`.
  - **`1,528` parches totales de Nivel 2** en `entrenamiento` dentro de `catalogo_parches_tiers.csv` (`tiers = {'oro', 'plata', 'hard_negative'}`), de los cuales **`1,148` parches son adicionales a Nivel 1** y son los que carga `--debiles` durante el entrenamiento.

---

## C. Corrección de `ZN43_VILCABAMBA_001_q3` y Auditoría de Quebradas en `validacion`

- **`datos_prueba__ZN43_VILCABAMBA_001_q3` (`validacion`):** el positivo seguía una quebrada lineal y se veía igual en `"antes"`; reclasificado íntegramente a **`dudoso`** (`0` píxeles positivos).
- **Verificación de todos los demás positivos de `validacion` frente a cauces/quebradas (`TPI_21 < -1.5 m`):**
  - Se revisaron los `41` parches positivos de `validacion`; se detectaron y pasaron también a `dudoso` `datos_prueba__ZN43_VILCABAMBA_001_q2` (`TPI = -23.15 m`, continuación de la misma quebrada de `ZN43_001`) y el componente cóncavo de `datos_val_dgrvi__znv_ingc6_11573_001_q2` (`TPI = -2.22 m`).
  - Confirmado: quedan **`0` positivos alineados con quebradas o cauces** en `validacion` (`39` parches positivos y `59` cicatrices $\ge 10\text{ px}$ en ladera).

---

## D. Corridas de Ajuste Fino en CPU (12 Hilos) y Archivos JSON de Resultados

- **Verificación MD5 de `ckpt_base/` (base v4 sin La Convención):**
  - `ckpt_base/mejor_unet_b1b4b13b14_b2809be440d6.pt`: `abad6e808c27a43ef766e6ca2a169b23` ✅
  - `ckpt_base/mejor_deeplabv3plus_b1b4b13b14_e838f4a37bf6.pt`: `ea1d3fb06830d0d2448f6db4285e3495` ✅
- **Parámetros comunes:** `--sin-l4s --debiles --frac-l4spe 0.5 --peso-l4spe 1.0 --peso-clasificador-v5 0.5 --peso-val-l4spe 0.7 --hilos-cpu 12`.
- **Nota metodológica sobre `A1` y `A2`:** `unet_peru.onnx` y `deeplabv3plus_peru.onnx` (`ckpt_peru/`) se entrenaron previamente con la partición antigua, por lo que `A1` y `A2` se reportan únicamente como **corridas secundarias**, siendo **`B1` y `B2` (desde `ckpt_base/` v4) las corridas principales**.

| Corrida | Rol | Arquitectura | Punto de partida | Épocas (`lr`) | Mejor época | IoU Val Ponderado (`0.7 L4SPe + 0.3 LC`) | IoU Val L4SPe (`127`) | IoU Val LC Nivel 1 (`304`) | Umbral operativo $u^*$ | Archivo JSON de resultados |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B1`** | **PRINCIPAL** | U-Net (`resnet34` + GRVI/ExG) | `ckpt_base/` (base v4) | `15` (`3e-5`) | **Época 5** | **`0.2977`** (`u=0.30`: **`0.3110`**) | `0.3712` (`u=0.30`: **`0.3741`**) | `0.1260` (`u=0.30`: **`0.1638`**) | **`0.30`** | [resultados_unet_b1b4b13b14_dfbbe6695037.json](file:///c:/Users/User/Documents/Tesis-Proyecto/resultados_ajuste/B1_unet_base/resultados_unet_b1b4b13b14_dfbbe6695037.json) |
| **`B2`** | **PRINCIPAL** | DeepLabV3+ (`resnet34`) | `ckpt_base/` (base v4) | `30` (`3e-5`) | **Época 18** | **`0.2816`** (`u=0.40`: **`0.2820`**) | `0.3863` (`u=0.40`: **`0.3858`**) | `0.0372` (`u=0.40`: **`0.0398`**) | **`0.40`** | [resultados_deeplabv3plus_b1b4b13b14_3539764d8289.json](file:///c:/Users/User/Documents/Tesis-Proyecto/resultados_ajuste/B2_deeplabv3plus_base/resultados_deeplabv3plus_b1b4b13b14_3539764d8289.json) |
| **`A1`** | Secundaria | U-Net (`resnet34` + GRVI/ExG) | `ckpt_peru/` (`unet_peru`) | `15` (`3e-5`) | Época 3 | `0.3037` (`u=0.35`: `0.3095`) | `0.3693` (`u=0.35`: `0.3700`) | `0.1506` (`u=0.35`: `0.1682`) | `0.35` | [resultados_unet_b1b4b13b14_996abb3d1469.json](file:///c:/Users/User/Documents/Tesis-Proyecto/resultados_ajuste/A1_unet_desde_peru/resultados_unet_b1b4b13b14_996abb3d1469.json) |
| **`A2`** | Secundaria | DeepLabV3+ (`resnet34`) | `ckpt_peru/` (`deeplabv3plus_peru`) | `15` (`3e-5`) | Época 7 | `0.2788` (`u=0.35`: `0.2815`) | `0.3831` (`u=0.35`: `0.3833`) | `0.0355` (`u=0.35`: `0.0442`) | `0.35` | [resultados_deeplabv3plus_b1b4b13b14_e949f9ff21d7.json](file:///c:/Users/User/Documents/Tesis-Proyecto/resultados_ajuste/A2_deeplabv3plus_desde_peru/resultados_deeplabv3plus_b1b4b13b14_e949f9ff21d7.json) |

### Modelos Ganadores Exportados a ONNX en `modelos_ajustados/` (sin tocar `modelos/`)
- **[modelos_ajustados/unet_b1_v5.onnx](file:///c:/Users/User/Documents/Tesis-Proyecto/modelos_ajustados/unet_b1_v5.onnx)** (`97.8 MB`, paridad PyTorch/ONNX `2.00e-05`, `0/46` capas en inicialización) + receta **[modelos_ajustados/unet_b1_v5.json](file:///c:/Users/User/Documents/Tesis-Proyecto/modelos_ajustados/unet_b1_v5.json)**.
- **[modelos_ajustados/deeplabv3plus_b2_v5.onnx](file:///c:/Users/User/Documents/Tesis-Proyecto/modelos_ajustados/deeplabv3plus_b2_v5.onnx)** (`89.8 MB`, paridad PyTorch/ONNX `7.63e-06`, `0/51` capas en inicialización) + receta **[modelos_ajustados/deeplabv3plus_b2_v5.json](file:///c:/Users/User/Documents/Tesis-Proyecto/modelos_ajustados/deeplabv3plus_b2_v5.json)**.

---

## E. Evaluación Final Única sobre Conjuntos de Prueba Sellados ([informes/evaluacion_final_parada_i.json](file:///c:/Users/User/Documents/Tesis-Proyecto/informes/evaluacion_final_parada_i.json))

> [!IMPORTANT]
> **Explicación sobre las cicatrices de `10–19 px` en `prueba`:**
> En el conjunto sellado de **`prueba` de La Convención (`360` parches, 100 % `manual_v4`)** existen **`80` cicatrices $\ge 10\text{ px}$** (8-con), de las cuales **`27` (`33.8 %`) miden `10–19 px`** (`0.10–0.19 ha`) y otras `27` miden `20–50 px`. En cambio, en **`entrenamiento` (`560` parches)** —tras eliminar en `clasificador_v5` todo componente aislado `< 20 px` según la Regla 2a de la Parada (h)— quedó **solo `1` cicatriz de `10–19 px`** (proveniente de `manual_v4`) frente a `268` cicatrices $\ge 20\text{ px}$. Por ello, los modelos ajustados con el filtro $\ge 20\text{ px}$ suprimen manchas de `10–19 px` para abatir las falsas alarmas (`12.47` $\rightarrow$ `0.22 ha/km²` en `B1`), concentrando su detección en cicatrices $\ge 20\text{ px}$ (`40.0 %` en `50–200 px` y `63.6 %–72.7 %` en `> 200 px`).

### E.1. Tabla Comparativa Integrada en los 3 Conjuntos de Prueba

| Conjunto de Prueba | Modelo / Método | $u^*$ | IoU píxel | IoU tol 1px (4c / 8c) | F1 píxel | Prec. / Exhaus. | Cicatrices `10–19 px` ($\ge 30\%$) | Cicatrices `20–50 px` ($\ge 30\%$) | Cicatrices `50–200 px` ($\ge 30\%$) | Cicatrices `> 200 px` ($\ge 30\%$) | Total Cicatrices $\ge 10\text{ px}$ | Falsas alarmas (`ha/km²`) | IC 95 % $\Delta\text{IoU}$ vs. base v4 (bootstrap por zona) | Coincidencia Polígonos INGEMMET Capa 6 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `prueba` sellada La Convención** (`360` parches, `90` teselas, `62` zonas, `80` cicatrices) | `regla_sin_aprendizaje` ($\text{GRVI}<0 \land \text{pend}\ge 15^\circ$) | `0.50` | `0.0091` | `0.0103` / `0.0105` | `0.0180` | `0.0091` / `0.7702` | `27/27` (`100.0%`) | `25/27` (`92.6%`) | `15/15` (`100.0%`) | `10/11` (`90.9%`) | `77/80` (`96.2%`) | `15.163` | `[-0.0062, +0.0003]` | `2/2` parches (`31.8%` área) |
| | `unet_base_v4` (sin La Convención) | `0.25` | `0.0113` | `0.0130` / `0.0133` | `0.0223` | `0.0113` / `0.8278` | `24/27` (`88.9%`) | `24/27` (`88.9%`) | `13/15` (`86.7%`) | `10/11` (`90.9%`) | `71/80` (`88.8%`) | `12.470` | Referencia U-Net | `0/2` (`0.0%`) |
| | `deeplabv3plus_base_v4` (sin La Convención) | `0.20` | `0.0030` | `0.0037` / `0.0039` | `0.0060` | `0.0033` / `0.0411` | `0/27` (`0.0%`) | `1/27` (`3.7%`) | `0/15` (`0.0%`) | `2/11` (`18.2%`) | `3/80` (`3.8%`) | `2.367` | Referencia DeepLab | `0/2` (`0.0%`) |
| | `unet_peru_anterior` (partición vieja) | `0.40` | `0.0391` | `0.0435` / `0.0447` | `0.0753` | `0.0459` / `0.2087` | `0/27` (`0.0%`) | `0/27` (`0.0%`) | `2/15` (`13.3%`) | `6/11` (`54.5%`) | `8/80` (`10.0%`) | `0.849` | `[-0.0014, +0.1501]` | `0/2` (`0.0%`) |
| | `deeplabv3plus_peru_anterior` (partición vieja) | `0.40` | `0.0103` | `0.0124` / `0.0132` | `0.0204` | `0.0115` / `0.0901` | `0/27` (`0.0%`) | `0/27` (`0.0%`) | `2/15` (`13.3%`) | `4/11` (`36.4%`) | `6/80` (`7.5%`) | `1.607` | `[+0.0004, +0.0603]` | `0/2` (`0.0%`) |
| | **`B1` (U-Net desde base v4 — PRINCIPAL)** | **`0.30`** | **`0.1311`** | **`0.1476` / `0.1506`** | **`0.2319`** | **`0.1755` / `0.3417`** | **`1/27` (`3.7%`)** | **`4/27` (`14.8%`)** | **`6/15` (`40.0%`)** | **`7/11` (`63.6%`)** | **`18/80` (`22.5%`)** | **`0.220`** | **`[+0.0413, +0.2206]` (gana `100%`)** | `0/2` (`0.0%`)* |
| | **`B2` (DeepLabV3+ desde base v4 — PRINCIPAL)** | **`0.40`** | **`0.0187`** | **`0.0218` / `0.0226`** | **`0.0367`** | **`0.0209` / `0.1506`** | `0/27` (`0.0%`) | `0/27` (`0.0%`) | `3/15` (`20.0%`) | **`7/11` (`63.6%`)** | `10/80` (`12.5%`) | `1.426` | **`[+0.0024, +0.0776]` (gana `99.5%`)** | `0/2` (`0.0%`) |
| | `A1` (U-Net desde `unet_peru` — secundaria) | `0.35` | `0.1033` | `0.1170` / `0.1205` | `0.1873` | `0.1508` / `0.2470` | `0/27` (`0.0%`) | `1/27` (`3.7%`) | `4/15` (`26.7%`) | `6/11` (`54.5%`) | `11/80` (`13.8%`) | `0.233` | `[+0.0289, +0.1840]` (gana `100%`) | `0/2` (`0.0%`) |
| | `A2` (DeepLabV3+ desde `dl_peru` — secundaria) | `0.35` | `0.0162` | `0.0197` / `0.0208` | `0.0320` | `0.0180` / `0.1392` | `0/27` (`0.0%`) | `0/27` (`0.0%`) | `2/15` (`13.3%`) | `4/11` (`36.4%`) | `6/80` (`7.5%`) | `1.522` | `[+0.0022, +0.0788]` (gana `99.6%`) | `0/2` (`0.0%`) |
| | **`Ensamble B1 + B2`** (promedio prob.) | **`0.35`** | **`0.0410`** | **`0.0461` / `0.0472`** | **`0.0788`** | **`0.0451` / `0.3102`** | `0/27` (`0.0%`) | **`4/27` (`14.8%`)** | **`6/15` (`40.0%`)** | **`8/11` (`72.7%`)** | **`18/80` (`22.5%`)** | `1.286` | **`[+0.0020, +0.1382]` (gana `98.5%`)** | `0/2` (`0.0%`) |
| **2. `prueba_externa`** (`16` parches, `4` teselas fuera de La Convención: Machupicchu, Samugari, Ollantaytambo, Huarocondo; `115` cicatrices) | `regla_sin_aprendizaje` | `0.50` | `0.6405` | `0.7605` / `0.7856` | `0.7809` | `0.6959` / `0.8894` | `15/21` (`71.4%`) | `39/44` (`88.6%`) | `25/25` (`100.0%`) | `25/25` (`100.0%`) | `104/115` (`90.4%`) | `0.000` | `[-0.0589, +0.1520]` | s/p (`0` parches) |
| | `unet_base_v4` | `0.25` | `0.6335` | `0.7290` / `0.7497` | `0.7757` | `0.6958` / `0.8763` | `12/21` (`57.1%`) | `20/44` (`45.5%`) | `16/25` (`64.0%`) | `23/25` (`92.0%`) | `71/115` (`61.7%`) | `0.000` | Referencia U-Net | s/p |
| | `deeplabv3plus_base_v4` | `0.20` | `0.6757` | `0.7639` / `0.7838` | `0.8065` | `0.7951` / `0.8182` | `6/21` (`28.6%`) | `6/44` (`13.6%`) | `6/25` (`24.0%`) | `15/25` (`60.0%`) | `33/115` (`28.7%`) | `0.000` | Referencia DeepLab | s/p |
| | `unet_peru_anterior` | `0.40` | `0.0118` | `0.0154` / `0.0162` | `0.0233` | `0.6116` / `0.0119` | `1/21` (`4.8%`) | `6/44` (`13.6%`) | `0/25` (`0.0%`) | `1/25` (`4.0%`) | `8/115` (`7.0%`) | `0.000` | `[-0.7396, -0.0348]` | s/p |
| | `deeplabv3plus_peru_anterior` | `0.40` | `0.5360` | `0.5957` / `0.6081` | `0.6979` | `0.7673` / `0.6400` | `3/21` (`14.3%`) | `7/44` (`15.9%`) | `3/25` (`12.0%`) | `11/25` (`44.0%`) | `24/115` (`20.9%`) | `0.000` | `[-0.4744, +0.0092]` | s/p |
| | **`B1` (U-Net desde base v4 — PRINCIPAL)** | `0.30` | `0.0145` | `0.0212` / `0.0234` | `0.0286` | `0.6425` / `0.0146` | `1/21` (`4.8%`) | `4/44` (`9.1%`) | `0/25` (`0.0%`) | `1/25` (`4.0%`) | `6/115` (`5.2%`) | `0.000` | `[-0.7309, -0.0322]` | s/p |
| | **`B2` (DeepLabV3+ desde base v4 — PRINCIPAL)** | **`0.40`** | **`0.4169`** | **`0.4756` / `0.4912`** | **`0.5885`** | **`0.8545` / `0.4488`** | `1/21` (`4.8%`) | `4/44` (`9.1%`) | `0/25` (`0.0%`) | **`9/25` (`36.0%`)** | `14/115` (`12.2%`) | `0.000` | `[-0.4508, -0.1543]` | s/p |
| | `A1` (U-Net desde `unet_peru` — secundaria) | `0.35` | `0.0104` | `0.0135` / `0.0140` | `0.0206` | `0.6253` / `0.0105` | `1/21` (`4.8%`) | `4/44` (`9.1%`) | `0/25` (`0.0%`) | `1/25` (`4.0%`) | `6/115` (`5.2%`) | `0.000` | `[-0.7424, -0.0365]` | s/p |
| | `A2` (DeepLabV3+ desde `dl_peru` — secundaria) | `0.35` | `0.5130` | `0.5708` / `0.5834` | `0.6782` | `0.7725` / `0.6043` | `3/21` (`14.3%`) | `7/44` (`15.9%`) | `1/25` (`4.0%`) | `10/25` (`40.0%`) | `21/115` (`18.3%`) | `0.000` | `[-0.4916, -0.0182]` | s/p |
| | **`Ensamble B1 + B2`** | **`0.35`** | **`0.3825`** | **`0.4352` / `0.4491`** | **`0.5533`** | **`0.8596` / `0.4079`** | `1/21` (`4.8%`) | `5/44` (`11.4%`) | `0/25` (`0.0%`) | `8/25` (`32.0%`) | `14/115` (`12.2%`) | `0.000` | `[-0.5089, -0.0255]` | s/p |
| **3. `prueba` L4SPe Profesor** (`124` parches, `68` zonas, `406` cicatrices del experto humano) | `regla_sin_aprendizaje` | `0.50` | `0.1178` | `0.1455` / `0.1534` | `0.2108` | `0.1256` / `0.6545` | `21/26` (`80.8%`) | `99/124` (`79.8%`) | `127/183` (`69.4%`) | `66/73` (`90.4%`) | `313/406` (`77.1%`) | `0.481` | `[-0.2987, -0.1763]` | s/p |
| | `unet_base_v4` | `0.25` | `0.3558` | `0.4350` / `0.4567` | `0.5249` | `0.5386` / `0.5118` | `11/26` (`42.3%`) | `55/124` (`44.4%`) | `77/183` (`42.1%`) | `56/73` (`76.7%`) | `199/406` (`49.0%`) | `0.072` | Referencia U-Net | s/p |
| | `deeplabv3plus_base_v4` | `0.20` | `0.2188` | `0.2567` / `0.2677` | `0.3591` | `0.2856` / `0.4833` | `5/26` (`19.2%`) | `29/124` (`23.4%`) | `66/183` (`36.1%`) | `49/73` (`67.1%`) | `149/406` (`36.7%`) | `0.000` | Referencia DeepLab | s/p |
| | `unet_peru_anterior` | `0.40` | `0.4021` | `0.4891` / `0.5124` | `0.5736` | `0.6109` / `0.5406` | `10/26` (`38.5%`) | `48/124` (`38.7%`) | `102/183` (`55.7%`) | `57/73` (`78.1%`) | `217/406` (`53.4%`) | `0.731` | `[+0.0013, +0.0827]` | s/p |
| | `deeplabv3plus_peru_anterior` | `0.40` | `0.4011` | `0.4915` / `0.5183` | `0.5726` | `0.6175` / `0.5338` | `7/26` (`26.9%`) | `46/124` (`37.1%`) | `96/183` (`52.5%`) | `52/73` (`71.2%`) | `201/406` (`49.5%`) | `0.425` | `[+0.0968, +0.2462]` | s/p |
| | **`B1` (U-Net desde base v4 — PRINCIPAL)** | **`0.30`** | **`0.4297`** | **`0.5224` / `0.5470`** | **`0.6011`** | **`0.6178` / `0.5853`** | **`10/26` (`38.5%`)** | **`46/124` (`37.1%`)** | **`104/183` (`56.8%`)** | **`60/73` (`82.2%`)** | **`220/406` (`54.2%`)** | **`0.481`** | **`[+0.0452, +0.1010]` (gana `100%`)** | s/p |
| | **`B2` (DeepLabV3+ desde base v4 — PRINCIPAL)** | **`0.40`** | **`0.4029`** | **`0.4897` / `0.5152`** | **`0.5744`** | **`0.6337` / `0.5252`** | `9/26` (`34.6%`) | `38/124` (`30.6%`) | `88/183` (`48.1%`) | `49/73` (`67.1%`) | `184/406` (`45.3%`) | **`0.353`** | **`[+0.0894, +0.2627]` (gana `100%`)** | s/p |
| | `A1` (U-Net desde `unet_peru` — secundaria) | `0.35` | `0.4136` | `0.5020` / `0.5262` | `0.5852` | `0.6160` / `0.5573` | `9/26` (`34.6%`) | `46/124` (`37.1%`) | `102/183` (`55.7%`) | `59/73` (`80.8%`) | `216/406` (`53.2%`) | `0.712` | `[+0.0197, +0.0919]` (gana `99.7%`) | s/p |
| | `A2` (DeepLabV3+ desde `dl_peru` — secundaria) | `0.35` | `0.4002` | `0.4892` / `0.5158` | `0.5716` | `0.6143` / `0.5345` | `8/26` (`30.8%`) | `46/124` (`37.1%`) | `95/183` (`51.9%`) | `53/73` (`72.6%`) | `202/406` (`49.8%`) | `0.467` | `[+0.0967, +0.2440]` (gana `100%`) | s/p |
| | **`Ensamble B1 + B2`** | **`0.35`** | **`0.4303`** | **`0.5241` / `0.5507`** | **`0.6017`** | **`0.6311` / `0.5748`** | **`10/26` (`38.5%`)** | **`47/124` (`37.9%`)** | **`103/183` (`56.3%`)** | **`58/73` (`79.5%`)** | **`218/406` (`53.7%`)** | **`0.364`** | **`[+0.0486, +0.1003]` (gana `100%`)** | s/p |

*\*Nota sobre Polígonos INGEMMET Capa 6 en `prueba` (`360` parches):* Solo `2` parches del conjunto `prueba` (`datos_prueba__ZN01_OCOBAMBA_001_q2` y `q4`) intersectan polígonos históricos de la Capa 6 de INGEMMET, los cuales corresponden a paleo-deslizamientos/unidades geomorfológicas antiguas densamente revegetadas (`GRVI > 0.08`, marcados como negativos/dudosos en Nivel 1); por ello los modelos entrenados para detectar cicatrices activas de suelo desnudo predicen `0 px` sobre esos dos polígonos revegetados (`0/2`), mientras que la regla ingenua ($\text{GRVI} < 0$) toca pequeños claros agrícolas dentro del polígono (`2/2`).

---

## F. Seis Láminas del Conjunto `prueba` Sellado (`3` Aciertos y `3` Errores)

Listado en [informes/parada_i_prueba6.csv](file:///c:/Users/User/Documents/Tesis-Proyecto/informes/parada_i_prueba6.csv) y láminas de 4 paneles (`Sentinel-2 L2A Color Natural 4×`, `Bing Aéreo`, `Máscara Nivel 1 manual_v4 Sellada`, `Predicción Ensamble B1+B2: Verde=TP, Rojo=FP, Amarillo=FN`) en `_fichas/parada_i_prueba6/`:

### 1. Tres Aciertos en `prueba`
1. **[01_acierto_ZN59_INKAWASI_001_q1.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/01_acierto_ZN59_INKAWASI_001_q1.png)** (`datos_prueba__ZN59_INKAWASI_001_q1`, `GT = 317 px`, `Pred = 276 px`, **`IoU = 0.8079`**): lengua de deslizamiento bien definida en ladera montañosa de Inkawasi; el modelo captura con alta fidelidad la corona y el cuerpo principal sin activar falsas alarmas en el bosque circundante.
2. **[02_acierto_ZN57_VILCABAMBA_001_q4.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/02_acierto_ZN57_VILCABAMBA_001_q4.png)** (`datos_prueba__ZN57_VILCABAMBA_001_q4`, `GT = 151 px`, `Pred = 121 px`, **`IoU = 0.7895`**): cicatriz activa en el flanco derecho de Vilcabamba; el modelo segmenta la cicatriz real e **ignora por completo la carretera serpenteante** que cruza toda la mitad izquierda del parche.
3. **[03_acierto_ZN87_QUELLOUNO_001_q1.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/03_acierto_ZN87_QUELLOUNO_001_q1.png)** (`datos_prueba__ZN87_QUELLOUNO_001_q1`, `GT = 997 px`, `Pred = 954 px`, **`IoU = 0.7672`**): deslizamiento grande alargado en Quellouno; coincidencia espacial de extremo a extremo desde la corona hasta la lengua de depósito.

### 2. Tres Errores Representativos en `prueba`
4. **[04_error_FN_pequena_ZN59_INKAWASI_001_q4.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/04_error_FN_pequena_ZN59_INKAWASI_001_q4.png)** (`datos_prueba__ZN59_INKAWASI_001_q4`, `GT = 24 px`, `Pred = 0 px`, **`IoU = 0.0000` — Falso Negativo en cicatriz pequeña**): cicatriz estrecha de `24 px` (`0.24 ha`, `2–3 px` de ancho) en Inkawasi; ejemplifica la pérdida de sensibilidad en el rango `10–25 px` derivada de entrenar con máscaras filtradas a $\ge 20\text{ px}$.
5. **[05_error_FN_ladera_ZN63_QUELLOUNO_001_q1.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/05_error_FN_ladera_ZN63_QUELLOUNO_001_q1.png)** (`datos_prueba__ZN63_QUELLOUNO_001_q1`, `GT = 2,450 px`, `Pred = 0 px`, **`IoU = 0.0000` — Falso Negativo en gran cicatriz arcillosa rojiza**): gran deslizamiento en Quellouno cuyo sustrato expuesto presenta tonalidad pardo-rojiza oscura (baja reflectancia en `B02/B03`), que el modelo confunde con pastizal seco/chacra para evitar falsas alarmas agrícolas.
6. **[06_error_FP_falsa_alarma_ZN41_VILCABAMBA_001_q3.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/parada_i_prueba6/06_error_FP_falsa_alarma_ZN41_VILCABAMBA_001_q3.png)** (`datos_prueba__ZN41_VILCABAMBA_001_q3`, `GT = 0 px`, `Pred = 13,940 px`, **`IoU = 0.0000` — Falso Positivo en ladera altoandina de roca/pajonal desnudo**): vertiente altoandina de Vilcabamba naturalmente desprovista de vegetación arbórea (`GRVI < 0` y pendiente $> 30^\circ$ en casi todo el parche), donde `B2` (DeepLabV3+) activa una falsa alarma masiva sobre afloramientos rocosos y suelo mineral natural.
