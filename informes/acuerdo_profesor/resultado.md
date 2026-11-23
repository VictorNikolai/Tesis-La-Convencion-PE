# Resultado del Acuerdo Ciego con el Experto (`l4spe_profesor.npz`)

## 1. Resumen Comparativo de las Tres Rondas Ciegas (Sin Prior de Centro y Sin Repetición de Parches)

Todas las evaluaciones se realizaron sobre parches reales de `l4spe_profesor.npz` **nunca vistos previamente por el agente**, sin repetir ningún parche entre rondas ni con las calibraciones, sin usar negativos sintéticos y **sin utilizar ninguna regla ni atributo de distancia al centro del parche (`dist_c`)**:

| Métrica | Ronda 1 (`60` reales: `40` pos + `20` neg, Protocolo v4) | Ronda 2 (`43` reales: `40` pos + `3` neg, Protocolo v5 **sin centro**) | Ronda 2 (`60` reales: `57` pos + `3` neg, Protocolo v5 **sin centro**) | Ronda 3 (`42` reales: `40` pos + `2` neg, Protocolo v5 **sin centro**) | Ronda 3 (`60` reales: `58` pos + `2` neg, Protocolo v5 **sin centro**) | Meta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Píxeles positivos Profesor / Agente / Intersección** | 21,956 / 16,660 / 8,042 | 15,065 / 14,960 / 7,937 | 24,782 / 29,277 / 15,209 | 17,823 / 20,741 / 10,079 | 22,248 / 29,796 / 12,079 | — |
| **IoU por píxel agregado (exacto)** | 0.264 | **0.359** ✅ | **0.391** ✅ | **0.354** ✅ | 0.302 | $\ge 0.35$ |
| **IoU con tolerancia de borde 1 px (4-con / 8-con)** | — | **0.444 / 0.467** | **0.477 / 0.502** | **0.448 / 0.474** | **0.381 / 0.403** | — |
| **Precisión por píxel** | 0.483 | **0.531** | **0.519** | **0.486** | 0.405 | — |
| **Exhaustividad (Recall) por píxel** | 0.366 | **0.527** | **0.614** | **0.566** | **0.543** | — |
| **F1 (Dice) por píxel** | 0.416 | **0.529** | **0.563** | **0.523** | 0.464 | — |
| **Detección global por cicatriz ($\ge 30\%$ solape exacto)** | 28.9 % (50/173) | **53.7 % (66/123)** ✅ | **58.0 % (98/169)** ✅ | **52.3 % (91/174)** ✅ | **52.3 % (113/216)** ✅ | $\ge 50.0\%$ |
| **Detección global por cicatriz (tolerancia 1 px)** | — | **56.1 % (69/123)** | **60.4 % (102/169)** | **58.0 % (101/174)** | **57.4 % (124/216)** | — |
| **Tasa de falsas alarmas por componente (exacto / tol 1 px)** | 64.1 % (84/131) | **46.2 % / 42.5 %** | **43.9 % / 40.3 %** | **44.4 % / 37.9 %** | **48.4 % / 42.2 %** | — |
| **Acierto Presencia / Ausencia** | 60.0 % (36/60) | **83.7 % (36/43)** | **85.0 % (51/60)** | **83.3 % (35/42)** | **86.7 % (52/60)** | — |
| **Matriz Presencia / Ausencia (`TP, TN, FP, FN`)** | `29, 7, 13, 11` | `35, 1, 2, 5` | `50, 1, 2, 7` | `35, 0, 2, 5` | `52, 0, 2, 6` | — |

---

## 2. Detección por Tamaño de Cicatriz del Profesor ($\ge 30\%$ de área cubierta)

| Rango de tamaño (píxeles a 10 m) | Ronda 1 (exacto) | Ronda 2 (`43` parches, exacto / tol 1 px) | Ronda 2 (`60` parches, exacto / tol 1 px) | Ronda 3 (`42` parches, exacto / tol 1 px) | Ronda 3 (`60` parches, exacto / tol 1 px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`< 10 px` ($< 0.1\text{ ha}$)** | 0.0 % (0/13) | 100.0 % (1/1) | 100.0 % (1/1) | 0.0 % / 30.0 % (0/10 / 3/10) | 0.0 % / 21.4 % (0/14 / 3/14) |
| **`10–50 px` ($0.1\text{--}0.5\text{ ha}$)** | 19.8 % (16/81) | **43.4 % / 47.2 %** (23/53 / 25/53) | **38.3 % / 41.7 %** (23/60 / 25/60) | **39.5 % / 46.9 %** (32/81 / 38/81) | **39.4 % / 45.7 %** (37/94 / 43/94) |
| **`50–200 px` ($0.5\text{--}2.0\text{ ha}$)** | 38.6 % (22/57) | **56.0 % / 58.0 %** (28/50 / 29/50) | **62.7 % / 65.3 %** (47/75 / 49/75) | **66.7 % / 71.7 %** (40/60 / 43/60) | **67.9 % / 72.8 %** (55/81 / 59/81) |
| **`> 200 px` ($> 2.0\text{ ha}$)** | 54.5 % (12/22) | **73.7 % / 73.7 %** (14/19 / 14/19) | **81.8 % / 84.8 %** (27/33 / 28/33) | **82.6 % / 87.0 %** (19/23 / 20/23) | **77.8 % / 81.5 %** (21/27 / 22/27) |
| **Total cicatrices** | **28.9 % (50/173)** | **53.7 % / 56.1 % (66/123 / 69/123)** | **58.0 % / 60.4 % (98/169 / 102/169)** | **52.3 % / 58.0 % (91/174 / 101/174)** | **52.3 % / 57.4 % (113/216 / 124/216)** |

---

## 3. Trazabilidad y Aislamiento
- Estas máscaras sobre `l4spe_profesor.npz` se utilizan **únicamente como control de calidad externo del agente frente al experto humano** y nunca se mezclan con las máscaras de Nivel 1 de La Convención (`mascaras_nivel1/`).
- El detalle completo de los 30 parches negativos reales (`y == 0`) y la separación entre reglas L1C (solo L4SPe) y reglas L2A (La Convención) se documenta en `informes/acuerdo_profesor/calibracion.md`.
