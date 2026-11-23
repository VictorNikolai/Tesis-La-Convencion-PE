# Nota Técnica sobre el Conjunto `prueba_externa` (4 Teselas / 16 Parches fuera de La Convención)

## 1. Origen, Ubicación y Exclusión del Conjunto `prueba` de La Convención

Durante la expansión inicial de zonas con eventos reportados por INGEMMET e INDECI (Protocolo v2), se descargaron cuatro teselas de `256×256 px` (`16` parches de `128×128 px` en `datos_prueba/`) ubicadas en distritos vecinos pero **fuera del límite provincial de La Convención** (`herramientas/limite_laconvencion.geojson`, entre `5 km` y `21 km` al exterior del borde provincial):

1. **`ZN44_MACHUPICCHU_001`** (`4` parches: `q1..q4`, provincia de Urubamba, cañón del Vilcanota).
2. **`ZN45_SAMUGARI_001`** (`4` parches: `q1..q4`, provincia de La Mar, Ayacucho, valle del Apurímac).
3. **`ZN46_OLLANTAYTAMBO_001`** (`4` parches: `q1..q4`, provincia de Urubamba, vertiente interandina semiárida).
4. **`ZN47_HUAROCONDO_001`** (`4` parches: `q1..q4`, provincia de Anta, puna y vertiente altoandina seca).

En el **Protocolo v3 (Punto 1)** estas cuatro teselas fueron retiradas inmediatamente del conjunto `prueba` de La Convención —antes de iniciar el marcado manual de Nivel 1— y aisladas en `particion.csv` bajo la etiqueta `conjunto = "prueba_externa"` (`iris_128_prueba_externa/`). Nunca se mezclaron con la `prueba` sellada de La Convención (`360` parches).

---

## 2. De Dónde Salieron sus Máscaras (Método y Fuente) y Por Qué la Regla sin Aprendizaje Logra `IoU = 0.6405`

Al haber sido separadas antes del etiquetado manual de Nivel 1, las 16 imágenes de `iris_128_prueba_externa/masks/` no tuvieron máscaras PNG manuales (`0` archivos en `iris_128_prueba_externa/masks/`). En consecuencia, el cargador de evaluación (`cargar_prueba_externa`) generó sus máscaras de referencia de manera **automática (heurística física de Nivel 2 sobre Sentinel-2 L2A y NASADEM)** mediante la regla:

$$\text{Máscara de referencia (`prueba\_externa`)}:\quad 0 < \text{B02} \le 900 \;\land\; \text{pendiente} \ge 20^\circ \;\land\; \text{GRVI} \le -0.03 \;\land\; \text{B04} > 1.18\,\text{B02} \;\land\; \text{área conectada (8-con)} \ge 15\text{ px}$$

marcando como `dudoso (255)` los píxeles sin datos (`B02 <= 0`) o saturados por nube/nieve (`B02 > 1000`).

### Por qué la `regla_sin_aprendizaje` alcanza `IoU = 0.6405` en `prueba_externa`

1. **Coincidencia de formulación algebraica:** La `regla_sin_aprendizaje` evaluada en `ajuste_fino.py` (`regla_suelo_pendiente`) define positivo todo píxel con $\text{GRVI} < 0 \;\land\; \text{pendiente} \ge 15^\circ$. Como la máscara de referencia de `prueba_externa` no es una máscara manual de cicatrices sino un umbral físico ($\text{GRVI} \le -0.03 \;\land\; \text{pendiente} \ge 20^\circ \;\land\; \text{B04} > 1.18\,\text{B02}$), ambas expresiones miden prácticamente el mismo predicado espectral-topográfico de **suelo mineral o pajonal seco en ladera inclinada**.
2. **Dominio de vertientes altoandinas semiáridas (`Ollantaytambo` y `Huarocondo`):** De los `78,238 px` positivos en la máscara de referencia de `prueba_externa`, **`76,346 px` (`97.6 %`)** pertenecen a las dos teselas interandinas secas (`ZN46_OLLANTAYTAMBO_001` con `27,572 px` y `ZN47_HUAROCONDO_001` con `48,774 px`), cuyas laderas enteras carecen de cobertura arbórea y presentan $\text{GRVI} < -0.03$ y pendiente $> 20^\circ$ de forma natural, sin ser deslizamientos activos.
3. **Comportamiento contrastante de U-Net (`B1`) frente a DeepLabV3+ (`B2`):**
   - **U-Net `B1` (`unet_b1_v5`, $u = 0.30$)** aprendió en La Convención a rechazar vertientes altoandinas desnudas, pastizales secos y parcelas agrícolas extensas, exigiendo contraste morfológico local de cicatriz; por eso predice solo `1,010 px` en Ollantaytambo y `705 px` en Huarocondo (`IoU = 0.0145` frente a la máscara heurística de suelo desnudo, evitando falsas alarmas masivas).
   - **DeepLabV3+ `B2` (`deeplabv3plus_b2_v5`, $u = 0.40$)** conserva alta respuesta sobre suelo mineral extenso en pendiente (`37,646 px` en Huarocondo y `3,406 px` en Ollantaytambo), por lo que coincide con la máscara heurística en `IoU = 0.4169`, pero a costa de activar falsas alarmas sobre laderas altoandinas naturalmente desnudas (el mismo fenómeno observado en la lámina de error 06 de Vilcabamba `ZN41_VILCABAMBA_001_q3`).

> [!NOTE]
> Este conjunto `prueba_externa` no se modifica ni se vuelve a evaluar; sus cifras ilustran la diferencia entre un detector espectral de suelo desnudo en pendiente y un segmentador morfológico de cicatrices de deslizamiento en selva alta/ceja de selva.

---

## 3. Cuatro Láminas Comparativas por Tesela (`U-Net B1` vs. `DeepLabV3+ B2` lado a lado)

| # | Tesela (`256×256 px`) | Píxeles Ref. Heurística | Predicción `U-Net B1` ($u=0.30$) | Predicción `DeepLabV3+ B2` ($u=0.40$) | Ruta de la lámina |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `ZN44_MACHUPICCHU_001` | `375 px` | `64 px` (`IoU = 0.1086`) | `36 px` (`IoU = 0.0353`) | [_fichas/prueba_externa_4/01_ZN44_MACHUPICCHU_001.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/prueba_externa_4/01_ZN44_MACHUPICCHU_001.png) |
| 2 | `ZN45_SAMUGARI_001` | `1,517 px` | `0 px` (`IoU = 0.0000`) | `0 px` (`IoU = 0.0000`) | [_fichas/prueba_externa_4/02_ZN45_SAMUGARI_001.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/prueba_externa_4/02_ZN45_SAMUGARI_001.png) |
| 3 | `ZN46_OLLANTAYTAMBO_001` | `27,572 px` | `1,010 px` (`IoU = 0.0193`) | `3,406 px` (`IoU = 0.0840`) | [_fichas/prueba_externa_4/03_ZN46_OLLANTAYTAMBO_001.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/prueba_externa_4/03_ZN46_OLLANTAYTAMBO_001.png) |
| 4 | `ZN47_HUAROCONDO_001` | `48,774 px` | `705 px` (`IoU = 0.0114`) | `37,646 px` (`IoU = 0.6086`) | [_fichas/prueba_externa_4/04_ZN47_HUAROCONDO_001.png](file:///c:/Users/User/Documents/Tesis-Proyecto/_fichas/prueba_externa_4/04_ZN47_HUAROCONDO_001.png) |
