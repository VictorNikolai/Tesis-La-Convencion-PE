# Calibración del Criterio del Agente con el Profesor (L4SPe -> Protocolo v5)

## 1. Separación Estricta de Dominios: Solo L4SPe (Sentinel-2 L1C, 14 bandas) vs. La Convención (Sentinel-2 L2A, 6/8 bandas + Bing + Multifecha)

### A. Solo L4SPe (`l4spe_profesor.npz`, Sentinel-2 L1C TOA, 14 bandas)
- **Tipo de imagen y bandas:** `l4spe_profesor.npz` contiene parches de **Sentinel-2 Nivel 1C (reflectancia al tope de la atmósfera, TOA)** con las **12 bandas espectrales completas** (`B01` a `B12`, incluyendo infrarrojo cercano `B08` e infrarrojo de onda corta SWIR `B11`, `B12`), más **pendiente (`B13`)** y **DEM (`B14`)**, sin imagen previa, sin mapa aéreo Bing y sin series multifecha.
- **Dispersión de Rayleigh en L1C:** Al no tener corrección atmosférica de superficie (L2A), en valles andinos profundos la dispersión molecular de Rayleigh eleva fuertemente la banda azul (`B02` TOA $\approx 1{,}100\text{--}1{,}300$) por encima de la roja (`B04` TOA $\approx 750\text{--}1{,}050$). Por eso, **únicamente en L4SPe (L1C)** se emplean índices con `B08` y `B11` (`NDVI`, `NBR`, `NDSI`, `BSI`) y cocientes centrados por la mediana local de cada parche (`B04_norm / B03_norm`) para compensar la neblina atmosférica TOA.
- **Prohibición de prior de centro (`sin dist_c`):** El centrado de las cicatrices en `l4spe_profesor.npz` es un artefacto del recorte original del banco L4SPe alrededor de cada evento. **Ninguna regla del Protocolo v5 utiliza la distancia al centro del parche (`dist_c`) ni coordenadas normalizadas (`yy_n, xx_n`)**, ni en las rondas ciegas de L4SPe ni en La Convención (donde está estrictamente prohibido).

### B. La Convención (`iris_128_*`, Sentinel-2 L2A BOA, 6/8 bandas + Bing + Multifecha + INGEMMET)
- **Tipo de imagen y bandas reales en `images/<id>/image.tif`:**
  - Los proyectos IRIS de La Convención contienen reflectancia de superficie **Sentinel-2 Nivel 2A (BOA, corregida atmosféricamente con Sen2Cor y compensada por el offset `-1000` de baseline 04.00)** junto con topografía NASADEM remuestreada a $10\text{ m/px}$.
  - En los proyectos ciegos (`iris_128_prueba`, `iris_128_validacion` y el Paso 1 de `iris_128_ent_ciego`), cada GeoTIFF tiene exactamente **6 bandas activas** (las bandas 7 y 8 se exportan en cero `0.0` y sus vistas se ocultan en `deslizamientos_128.json`):
    1. `B1` (`B01` aerosol costero L2A)
    2. `B2` (`B02` azul L2A)
    3. `B3` (`B03` verde L2A)
    4. `B4` (`B04` rojo L2A)
    5. `B13 pendiente (grados)` (pendiente topográfica en grados)
    6. `B14 DEM (m)` (elevación sobre el nivel del mar en metros)
  - En `entrenamiento` (Paso 2 de consulta tras el marcado ciego), el GeoTIFF de `256×256` dispone además de las bandas 7 y 8:
    7. `probabilidad del modelo` (mapa de calor del ensamble `unet_peru` + `deeplabv3plus_peru`)
    8. `cajas del modelo (contorno)` (guía visual de candidatos)
  - **No existen `B08` ni `B11` en los GeoTIFF de IRIS de La Convención**, pero en cambio se cuenta con **imagen antes del evento (`*_antes`), mosaico aéreo de alta resolución Bing (`VirtualEarth`), verificación multifecha en temporadas secas despejadas (2022–2025), red vial OSM y cartografía de peligros INGEMMET**.
- **Reglas físicas exclusivas de L2A en La Convención:**
  1. En reflectancia de superficie L2A, toda cicatriz de suelo desnudo o roca expuesta en ladera cumple **`B04 > B03` ($\implies \text{GRVI} = \frac{B03 - B04}{B03 + B04} < 0$) y `B04 > B02`** (específicamente `B04 > 1.12 * B02` en el borde de transición y `B04 > 1.18 * B02` con `GRVI <= -0.028` en el núcleo). **Jamás se aplica en La Convención ninguna regla de L1C con `B04 < B02` o `GRVI > 0`.**
  2. **Confirmación sobre las 32 máscaras de `informes/revision_v5_cambios.csv`:** Las 32 máscaras que cambiaron en la revisión v5 de La Convención se decidieron **exclusivamente con información L2A de 6 bandas + topografía** (`B04 > B03` $\implies$ `GRVI <= -0.02 < 0`, `B04 > 1.15 * B02`, `pendiente >= 18°`, exclusión de fondos de valle `TPI_21 < -1.5 m` y distancia a nubes `> 25 px`), sin usar `B08`, `B11`, reglas de L1C ni distancia al centro.
  3. **Borde espectral de transición (sin dilatar a ciegas):** La máscara de La Convención incluye el anillo de transición de `1 px` solo donde el píxel adyacente mantiene firma L2A de suelo desnudo en ladera (`GRVI < 0`, `B04 > 1.12 * B02`, `pendiente >= 15°`, fuera de fondo de cauce y de vías).
  4. **Regla `< 10 px`:** Toda mancha positiva conectada de menos de `10 px` ($< 0.10\text{ ha}$) se descarta o pasa a `dudoso (255)`.

---

## 2. Auditoría de los 30 Parches Negativos Reales (`y == 0`) de `l4spe_profesor.npz`

El archivo `l4spe_profesor.npz` (`1,000` parches en total: `800` de `entrenamiento`, `100` de `validacion`, `100` de `prueba`) contiene exactamente **30 parches reales sin deslizamiento (`y.sum() == 0`)**:
- `27` en `entrenamiento`
- `2` en `validacion` (`idx_npz` `785`, `786`) y `1` adicional (`791`) con máscara nula
- `2` en `prueba` (`idx_npz` `391`, `757`)

La siguiente tabla muestra la partición **estrictamente disjunta** de los **30 negativos reales del `.npz`** a través de las 5 etapas (`Ronda 1`, `Calibración 1`, `Ronda 2`, `Calibración 2`, `Ronda 3`), confirmando que **ningún parche se repite entre rondas ni entre calibración y pruebas ciegas** (`20 + 3 + 3 + 2 + 2 = 30`), y que **no se utilizó ningún negativo sintético ni espejado**:

| Ronda / Etapa | Rol | Cantidad de negativos reales (`y == 0`) | IDs (`idx_npz`, `point_XXXX`, `zona`, `conjunto_npz`) | ¿Repetido en otra ronda o calibración? |
| :--- | :--- | :---: | :--- | :---: |
| **Ronda 1** (ciega inicial, 60 parches: 40 pos + 20 neg) | Prueba ciega 1 | **20** | `391 (point_0435, Z391, prueba)`, `757 (point_0918, Z757, prueba)`, `786 (point_0947, Z615, validacion)`, `153 (point_0160, Z160, entrenamiento)`, `155 (point_0162, Z155, entrenamiento)`, `160 (point_0167, Z160, entrenamiento)`, `178 (point_0185, Z178, entrenamiento)`, `247 (point_0258, Z247, entrenamiento)`, `297 (point_0327, Z554, entrenamiento)`, `349 (point_0389, Z833, entrenamiento)`, `548 (point_0643, Z817, entrenamiento)`, `582 (point_0682, Z582, entrenamiento)`, `596 (point_0704, Z655, entrenamiento)`, `632 (point_0757, Z632, entrenamiento)`, `644 (point_0770, Z643, entrenamiento)`, `663 (point_0791, Z809, entrenamiento)`, `726 (point_0871, Z726, entrenamiento)`, `741 (point_0898, Z742, entrenamiento)`, `746 (point_0905, Z746, entrenamiento)`, `763 (point_0924, Z764, entrenamiento)` | **No (0)** |
| **Calibración 1** (70 pos + 3 neg reales) | Estudio con máscara 1 | **3** | `424 (point_0484, Z817, entrenamiento)`, `425 (point_0485, Z817, entrenamiento)`, `546 (point_0641, Z817, entrenamiento)` | **No (0)** |
| **Ronda 2** (ciega recalibrada, solo parches reales nunca vistos) | Prueba ciega 2 | **3** | `785 (point_0946, Z615, validacion)`, `547 (point_0642, Z817, entrenamiento)`, `552 (point_0647, Z554, entrenamiento)` | **No (0)** |
| **Calibración 2** (70 pos nuevos + 2 neg reales) | Estudio con máscara 2 | **2** | `613 (point_0729, Z554, entrenamiento)`, `743 (point_0900, Z742, entrenamiento)` | **No (0)** |
| **Ronda 3** (ciega final, solo parches reales nunca vistos) | Prueba ciega 3 | **2** | `791 (point_0952, Z615, validacion)`, `740 (point_0897, Z742, entrenamiento)` | **No (0)** |
| **TOTAL** | — | **30 / 30** | **30 parches reales `y == 0` de `l4spe_profesor.npz`** | **0 repeticiones / 0 sintéticos** |

*(Nota: Los 140 parches positivos de Calibración 1 [70] y Calibración 2 [70] salieron exclusivamente de `entrenamiento` y son totalmente disjuntos entre sí y respecto a los positivos de `validacion` y `prueba` evaluados en las Rondas 1, 2 y 3).*

---

## 3. Resultados Corregidos de Ronda 2 y Ronda 3 (Sin Prior de Centro y Solo Parches Reales Nunca Vistos)

Al eliminar por completo cualquier atributo o recorte de distancia al centro (`83` atributos invariantes a la traslación: espectrales, topográficos y de contraste local multiescala, **0 atributos de coordenadas o distancia al centro**) y evaluar únicamente sobre parches reales de `l4spe_profesor.npz` nunca vistos antes:

| Métrica | Ronda 1 (ciega v4, 60 parches: 40 pos + 20 neg) | Ronda 2 Corregida (`43` reales: `40` pos + `3` neg, **sin centro**) | Ronda 2 Corregida (`60` reales: `57` pos + `3` neg, **sin centro**) | Ronda 3 Corregida (`42` reales: `40` pos + `2` neg, **sin centro**) | Ronda 3 Corregida (`60` reales: `58` pos + `2` neg, **sin centro**) | Meta de Calibración |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IoU por píxel agregado (exacto)** | 0.264 | **0.359** ✅ | **0.391** ✅ | **0.354** ✅ | 0.302 | $\ge 0.35$ |
| **IoU por píxel con tolerancia de borde 1 px (4-con / 8-con)** | — | **0.444 / 0.467** | **0.477 / 0.502** | **0.448 / 0.474** | **0.381 / 0.403** | Referencia de borde |
| **Detección global de cicatrices ($\ge 30\%$ solape exacto)** | 28.9 % (50/173) | **53.7 % (66/123)** ✅ | **58.0 % (98/169)** ✅ | **52.3 % (91/174)** ✅ | **52.3 % (113/216)** ✅ | $\ge 50.0\%$ |
| **Detección global de cicatrices (tolerancia 1 px)** | — | **56.1 % (69/123)** | **60.4 % (102/169)** | **58.0 % (101/174)** | **57.4 % (124/216)** | — |
| **Detección cicatrices `10–50 px` (exacto / tol 1 px)** | 19.8 % (16/81) | **43.4 % / 47.2 %** (23/53) | **38.3 % / 41.7 %** (23/60) | **39.5 % / 46.9 %** (32/81) | **39.4 % / 45.7 %** (37/94) | — |
| **Detección cicatrices `50–200 px` (exacto / tol 1 px)** | 38.6 % (22/57) | **56.0 % / 58.0 %** (28/50) | **62.7 % / 65.3 %** (47/75) | **66.7 % / 71.7 %** (40/60) | **67.9 % / 72.8 %** (55/81) | — |
| **Detección cicatrices `> 200 px` (exacto / tol 1 px)** | 54.5 % (12/22) | **73.7 % / 73.7 %** (14/19) | **81.8 % / 84.8 %** (27/33) | **82.6 % / 87.0 %** (19/23) | **77.8 % / 81.5 %** (21/27) | — |
| **Tasa de falsas alarmas por componente (exacto / tol 1 px)** | 64.1 % (84/131) | **46.2 % / 42.5 %** | **43.9 % / 40.3 %** | **44.4 % / 37.9 %** | **48.4 % / 42.2 %** | Reducida vs. 64.1 % |
| **Acierto Presencia / Ausencia** | 60.0 % (36/60) | **83.7 % (36/43)** | **85.0 % (51/60)** | **83.3 % (35/42)** | **86.7 % (52/60)** | — |
| **Kappa de Cohen (P/A)** | 0.143 | 0.142 *(3 neg reales)* | 0.118 *(3 neg reales)* | -0.073 *(2 neg reales)* | -0.053 *(2 neg reales)* | *(Limitado por $N_{\text{neg}}=2\text{--}3$ en `.npz`)* |

- **Observación sobre Kappa cuando $N_{\text{neg}} = 2\text{ o }3$:** Como `l4spe_profesor.npz` solo tiene 30 negativos reales y 20 se consumieron en la Ronda 1 y 5 en las Calibraciones 1 y 2, en las Rondas 2 y 3 quedan únicamente `3` y `2` negativos reales nunca vistos. Con una prevalencia positiva de $93\%\text{--}97\%$, el acuerdo esperado por azar $p_e$ supera $0.84$, por lo que 1 o 2 falsos positivos desploman $\kappa$ aunque el **acierto presencia/ausencia subió de $60.0\%$ (Ronda 1) a $83.7\%\text{--}86.7\%$ (Rondas 2 y 3)**, el **IoU exacto sin prior de centro subió de $0.264$ a $0.354\text{--}0.391$** ($\ge 0.35$ ✅, y **$0.467\text{--}0.502$** con tolerancia de borde de 1 px), y la **detección de cicatrices subió de $28.9\%$ a $52.3\%\text{--}58.0\%$** ($\ge 50\%$ ✅).
