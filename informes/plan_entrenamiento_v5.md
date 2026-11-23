# Plan de Entrenamiento — Parada (h) (Preparado, aún sin ejecutar)

## 1. Datos de Entrenamiento y Pesos en la Pérdida (Sin Landslide4Sense: `--sin-l4s`)

No se utiliza ningún parche de Landslide4Sense (`--sin-l4s`) ni para entrenar, ni como piso de admisibilidad, ni para elegir época o umbral. El entrenamiento emplea exclusivamente datos del Perú:

| Fuente de datos | Conjunto | Método / Tier | Parches (`128×128`) | Cicatrices $\ge 10\text{ px}$ | Peso en la función de pérdida (`BCE + Dice + OHEM`) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **L4SPe del Profesor** (`l4spe_profesor.npz`) | `entrenamiento` (`181` zonas) | Humano experto (manual) | **587** | 1,680+ | **`1.00`** (`--peso-l4spe 1.0`, `--frac-l4spe 0.5`) |
| **La Convención Nivel 1** (`mascaras_nivel1/`) | `entrenamiento` (`38` teselas) | `manual_v4` (visual por parche) | **152** (`6` pos, `2` dud, `144` neg) | **16** | **`1.00`** (peso unitario en `Wl`) |
| **La Convención Nivel 1** (`mascaras_nivel1/`) | `entrenamiento` (`102` teselas) | `clasificador_v5` limpio ($\ge 20\text{ px}$, sin pueblos/pastizales) | **408** (`132` pos, `162` dud, `114` neg) | **253** (`1.9` cic/pos) | **`0.50`** (`--peso-clasificador-v5 0.5`) |
| **La Convención Nivel 2** (`mascaras_provisionales/nivel2/`) | `entrenamiento` (sin solape con Nivel 1) | `oro`, `hard_negative`, `plata` (`catalogo_parches_tiers.csv`) | Parches débiles de zonas de entrenamiento | Propuestas multitemporales / INGEMMET | **Peso actual de su Tier** (`oro=1.0`, `hard_negative=0.8`, `plata=0.5`, respaldo `--peso-debil 0.3`) |

---

## 2. Selección de Época y Umbral: Promedio Ponderado `0.70 / 0.30`

Implementado en `herramientas/ajuste_fino.py` (`--peso-val-l4spe 0.7`):

1. **Criterio principal (`70 %` del peso):** Validación de **L4SPe del profesor** (`127` parches, `31` zonas con máscara manual humana).
2. **Criterio secundario (`30 %` del peso):** Validación de **La Convención Nivel 1** (`304` parches, `76` teselas, `51` zonas; `41` parches positivos revisados visualmente uno por uno con Bing: `8` `manual_v4` + `33` `clasificador_v5` limpios, con **`63` cicatrices $\ge 10\text{ px}$** y media de **`1.5` cicatrices por parche positivo**).
3. **Fórmula de puntaje por época $e$:**
   $$\text{Puntaje}(e) = 0.70 \cdot \text{IoU}_{\text{val, L4SPe}}(e) + 0.30 \cdot \text{IoU}_{\text{val, La Convención}}(e)$$
4. **Elección del umbral de operación $u^*$:**
   Sobre el barrido de umbrales $u \in \{0.05, 0.10, \dots, 0.95\}$ de la mejor época (incluyendo las interpolaciones WiSE-FT $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$), se elige el umbral $u^*$ que maximiza:
   $$u^* = \arg\max_{u} \left[ 0.70 \cdot \text{IoU}_{\text{val, L4SPe}}(u) + 0.30 \cdot \text{IoU}_{\text{val, La Convención}}(u) \right]$$

---

## 3. Arquitecturas y Puntos de Partida en CPU de 12 Núcleos (`--hilos-cpu 12`)

Se entrenarán y compararán las **dos arquitecturas** (`U-Net ResNet-34` y `DeepLabV3+ ResNet-34`) en la CPU local de 12 núcleos (`torch.set_num_threads(12)`), tanto desde los modelos ya ajustados (`unet_peru` / `deeplabv3plus_peru`) como desde los modelos base v4:

1. **Puntos de control reconstruidos y verificados en disco (`ckpt_peru/`):**
   - `ckpt_peru/mejor_unet_b1b4b13b14_809315dc820e.pt` + `ckpt_peru/stats_norm_809315dc820e.json` (reconstruido desde `modelos_ajustados/unet_peru.onnx` con `herramientas/onnx_a_checkpoint.py`, paridad PyTorch vs. ONNX = **`5.72e-06`**).
   - `ckpt_peru/mejor_deeplabv3plus_b1b4b13b14_2551b2eefe0c.pt` + `ckpt_peru/stats_norm_2551b2eefe0c.json` (reconstruido desde `modelos_ajustados/deeplabv3plus_peru.onnx` con `herramientas/onnx_a_checkpoint.py`, paridad PyTorch vs. ONNX = **`2.48e-05`**).
2. **Corridas programadas (listas para ejecutarse tras aprobar la Parada h):**
   - **Corrida A1 (`U-Net` desde `unet_peru`):**
     ```powershell
     python herramientas/ajuste_fino.py --sin-l4s ^
         --desde ckpt_peru/mejor_unet_b1b4b13b14_809315dc820e.pt ^
         --stats ckpt_peru/stats_norm_809315dc820e.json ^
         --parches parches_128 --mascaras mascaras_nivel1/parches_128 ^
         --debiles mascaras_provisionales/nivel2/parches_128 --tiers oro,plata,hard_negative ^
         --l4spe l4spe_profesor.npz --frac-l4spe 0.5 --peso-l4spe 1.0 ^
         --peso-clasificador-v5 0.5 --peso-val-l4spe 0.7 ^
         --epocas 15 --pasos 60 --lote 16 --lr 1.5e-5 --interpolar --hilos-cpu 12 ^
         --salida resultados_ajuste/unet_desde_peru
     ```
   - **Corrida A2 (`DeepLabV3+` desde `deeplabv3plus_peru`):**
     ```powershell
     python herramientas/ajuste_fino.py --sin-l4s ^
         --desde ckpt_peru/mejor_deeplabv3plus_b1b4b13b14_2551b2eefe0c.pt ^
         --stats ckpt_peru/stats_norm_2551b2eefe0c.json ^
         --parches parches_128 --mascaras mascaras_nivel1/parches_128 ^
         --debiles mascaras_provisionales/nivel2/parches_128 --tiers oro,plata,hard_negative ^
         --l4spe l4spe_profesor.npz --frac-l4spe 0.5 --peso-l4spe 1.0 ^
         --peso-clasificador-v5 0.5 --peso-val-l4spe 0.7 ^
         --epocas 15 --pasos 60 --lote 16 --lr 1.5e-5 --interpolar --hilos-cpu 12 ^
         --salida resultados_ajuste/deeplabv3plus_desde_peru
     ```
   - **Corridas B1 y B2 (`U-Net` y `DeepLabV3+` desde los modelos base v4 `b2809be440d6` y `e838f4a37bf6`):**
     - Si se colocan `unet.onnx` / `deeplabv3plus.onnx` (o sus `.pt` base de `entrenamiento_v4`) en `ckpt_base/`, se reconstruyen con `onnx_a_checkpoint.py` y se corren con los mismos hiperparámetros (`--lr 3e-5`, `--epocas 15` para U-Net y `--epocas 25` para DeepLabV3+, `--hilos-cpu 12`).

---

## 4. Evaluación Final en Prueba (Una Sola Vez)

Al terminar la selección de época y umbral en validación, `ajuste_fino.py` evalúa **una sola vez** sobre los dos conjuntos de prueba sellados:
1. **Prueba de La Convención (`360` parches, `90` teselas, `62` zonas, 100 % `manual_v4` sellada, `80` cicatrices $\ge 10\text{ px}$).**
2. **Prueba de L4SPe del Profesor (`124` parches, `68` zonas, 100 % manual del experto).**

Para cada conjunto de prueba y cada modelo (`regla sin aprendizaje`, `base`, `ajustado` y `ensamble`), `resumen_local` reporta:
- **Por píxel exacto:** `IoU`, `F1 (Dice)`, `precisión`, `exhaustividad (recall)`.
- **Con tolerancia de borde de 1 px:** `iou_tol1px_4c` (4-conectado) e `iou_tol1px_8c` (8-conectado).
- **Por cicatriz:** cicatrices tocadas (`>= 1 px`), detección con solape $\ge 30\%$ desglosada por tamaño (`10–50 px`, `50–200 px`, `> 200 px` y `total >= 10 px`), falsas alarmas en teselas negativas ($\text{ha/km}^2$) e intervalo de confianza del $95\%$ por bootstrap de zonas (`2,000` remuestreos).
