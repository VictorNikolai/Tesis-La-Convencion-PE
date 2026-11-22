# Línea Base de Modelos (Fase A — Diagnóstico)

Fecha de registro: 2026-11-22  
Rama de origen de los pesos: `main` / `ajuste-fino-variantes` (`modelos_ajustados/`)

> **Advertencia metodológica:** A la fecha de inicio de esta sesión, **`mascaras_nivel1/` no existe (0 máscaras Nivel 1 en La Convención)**. Las cifras locales de esta tabla provienen de máscaras automáticas provisionales (Nivel 2), y ese conjunto de validación se usó para elegir las épocas (`12` en U-Net, `21` en DeepLabV3+). Por ello el conjunto de prueba definitivo de la tesis se construirá con **zonas nuevas**.

## 1. Desempeño actual registrado

| Modelo | Base de partida | Prueba L4SPe (124 parches, 68 zonas, manual) | Validación La Convención (128 teselas / 504 parches, Nivel 2 provisional) |
| :--- | :--- | :--- | :--- |
| **`unet_peru.onnx`** (época 12, umbral 0.40) | U-Net v4 (`b2809be440d6`) | IoU **0.4021** · F1 **0.5736** · Precisión **61.1%** · Recall **54.1%** · FP **38.9%** | IoU **0.4085** · F1 **0.5801** · Precisión **58.2%** · Recall **57.8%** · Cicatrices: **53 / 228** · Falsas alarmas neg.: **0.051 ha/km²** |
| **`deeplabv3plus_peru.onnx`** (época 21, umbral 0.40) | DeepLabV3+ (`e838f4a37bf6`) | IoU **0.4011** · F1 **0.5726** · Precisión **61.8%** · Recall **53.4%** · FP **38.2%** | IoU **0.3228** · F1 **0.4881** · Precisión **49.3%** · Recall **48.4%** · Cicatrices: **74 / 228** · Falsas alarmas neg.: **0.071 ha/km²** |
| **Ensamble promedio** (umbral 0.40, según `LEEME.md`) | `unet_peru` + `deeplabv3plus_peru` | IoU **0.414** | IoU **0.392** · Cicatrices: **71 / 228** · Falsas alarmas neg.: **0.06 ha/km²** |

## 2. Metas de la tesis (RQ-07) y brecha actual

| Métrica | Meta mínima | Meta objetivo | Exigente | Estado actual (L4SPe / La Convención Nivel 2) |
| :--- | :---: | :---: | :---: | :--- |
| **IoU** | $\ge 0{,}61$ | **$\ge 0{,}70$** | $\ge 0{,}70$ | 0.402 – 0.414 / 0.323 – 0.409 (**brecha: ~0.29 puntos**) |
| **F1-Score** | — | **$> 75{,}5\%$** | — | 57.3% / 48.8% – 58.0% (**brecha: ~17.5 pp**) |
| **Exhaustividad (Recall)** | — | **$> 85{,}0\%$** | $\ge 85{,}0\%$ | 53.4% – 54.1% / 48.4% – 57.8% (**brecha: ~27 pp**) |
| **Falsos positivos ($1 - \text{Precisión}$)** | — | **$< 15{,}0\%$** | $\le 15{,}0\%$ | 38.2% – 38.9% / 41.8% – 50.7% (**brecha: ~27 pp**) |
