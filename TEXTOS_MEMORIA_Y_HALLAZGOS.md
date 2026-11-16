# Diagnóstico técnico de problemas abiertos y textos corregidos para la Memoria Descriptiva (OE2 / OE4)

Este documento reúne los resultados de las verificaciones técnicas pedidas en las secciones 6, 7, 8 y 9 del *Informe de avance de la tesis (15/11/2026)*, junto con los bloques de texto redactados y listos para incorporar en la **Memoria Descriptiva del Artefacto (OE2)** y en el capítulo de **Validación y Discusión (OE4)**.

---

## Parte I. Resolución de los hallazgos técnicos abiertos (Secciones 6 y 7)

### 1. Causa raíz del fallo en las teselas `azar_012_001` y `azar_015_001` ($\text{prob} \approx 0{,}73$)

* **Síntoma reportado:** En `datos_azar`, las teselas `azar_012_001` y `azar_015_001` (bosque y chacras normales de 2019) salieron con probabilidad mediana de $0{,}732$ y $0{,}729$ y máximo de $0{,}751$ y $0{,}752$ en casi todos los píxeles ($97{,}7\%$ y $94{,}4\%$ del área marcada a umbral $0{,}45$).
* **Evidencia forense en los datos:**
  1. De las **397 imágenes** del proyecto (`datos_laconvencion`, `datos_indeci`, `datos_ingemmet`, `datos_azar` y pares temporales), **únicamente** `azar_012_001` y `azar_015_001` presentan este comportamiento (logits comprimidos en $[+0{,}99; +1{,}11]$ en el interior y correlación *negativa* con las bandas espectrales: $r = -0{,}44\text{ a }-0{,}62$).
  2. Al buscar los vecinos más cercanos en el espacio de 29 características espectrales y topográficas (`B1`–`B4`, `GRVI`, `ExG`, pendiente y DEM), las teselas `sinpad_63076_001` ($\text{dist} = 0{,}85$), `sinpad_63073_001` ($\text{dist} = 1{,}01$) y `azar_042_001` ($\text{dist} = 1{,}32$) tienen **exactamente las mismas distribuciones de bandas y exactamente las mismas 6 escenas Sentinel-2 de 2019** (`2019-07-22` a `2019-09-25`), pero en todas ellas la mediana de probabilidad es normal ($0{,}000\text{ a }0{,}003$).
  3. Al inspeccionar los tiempos de escritura (`mtime`), `azar_015_001` (`18:53:40`) y `azar_012_001` (`18:53:50`) se ejecutaron **simultáneamente en el mismo lote paralelo** (`011`–`015` con `--paralelo 5`). Justo cuando `011` terminó (`18:53:29`), el siguiente subproceso invocó `modelos.ficha('unet')` $\to$ `_guardar_pesos_aparte()`, que borraba y reescribía sin bloqueo el archivo compartido `%TEMP%/modelos_pesos_aparte/unet.onnx.data` en el mismo instante en que `012` y `015` abrían su sesión de inferencia ONNX.
* **Conclusión e impacto:**
  * **No es un fallo del modelo U-Net v4 ante ese tipo de bosque ni ocurre en el servidor web de Render** (donde corre un único proceso con candado `_LOCK`). Fue una condición de carrera de disco entre subprocesos locales sobre `unet.onnx.data`.
  * **Solución aplicada:** Se corrigió `_guardar_pesos_aparte()` en [`nucleo/modelos.py`](nucleo/modelos.py) para que, si `unet.onnx` y `unet.onnx.data` ya existen con el tamaño y fecha esperados, se reutilicen sin borrarlos ni truncarlos. Además, como los parches de entrenamiento (`parches_128/`) solo guardan los 6 canales de entrada (`B1`–`B4`, pendiente y DEM) y descartan la banda 7 (`prob`), **los parches de `azar_012_001` y `azar_015_001` en `parches_128/` están 100 % limpios y usables**.

### 2. Verificación de la pendiente «a cuadros» y borde del mosaico

* **Diagnóstico cuantitativo sobre las teselas actuales:**
  * Se midió la pendiente media y la tasa de activación de la U-Net ($\text{prob} \ge 0{,}45$) en el borde exterior de $2\text{ px}$ frente al interior ($252 \times 252\text{ px}$) sobre 20 teselas:
    * Pendiente media en el borde de $2\text{ px}$: **$26{,}18^\circ$** frente a **$26{,}23^\circ$** en el interior (sin inflado de borde).
    * Fracción marcada ($\text{prob} \ge 0{,}45$) en el borde: **$4{,}00\%$** frente a **$4{,}73\%$** en el interior (la red no se enciende en el borde).
  * La primera diferencia espacial de la pendiente ($|\Delta \text{pendiente} / \Delta \text{fila}|$) es uniforme en los tres pasos del remuestreo de $30\text{ m}$ a $10\text{ m}$ ($\text{fila} \bmod 3 = 0: 1{,}7349^\circ$; $\bmod 3 = 1: 1{,}7349^\circ$; $\bmod 3 = 2: 1{,}7359^\circ$).
* **Conclusión:** El problema diagnosticado el 22/10 ya quedó resuelto en las teselas actuales gracias al margen de planificación (`plan['ventana']`) previo al recorte (`datos.recortar`) y a la interpolación bilineal de la pendiente.

### 3. Regla de nubes frente a suelo desnudo muy claro (Sillapata y Mesacancha)

* **Medición:** En las cicatrices aceptadas de nivel 2 en Sillapata (`ingemmet_003_001` y `036_001`), el **$32{,}5\%$ y $28{,}6\%$** de los píxeles de la cicatriz tienen $\text{B2} > 1000$, pero presentan una razón rojo/azul $\text{B4}/\text{B2}$ mediana de **$1{,}56\text{ a }1{,}71$** ($\text{GRVI} < 0$), mientras que las nubes reales tienen $\text{B4}/\text{B2} \approx 0{,}92\text{ a }1{,}04$ ($\text{GRVI} > 0$).
* **Corrección aplicada:** Se actualizó la detección de nube blanca en [`herramientas/candidatos_para_qgis.py`](herramientas/candidatos_para_qgis.py), [`herramientas/descargar_desde_puntos.py`](herramientas/descargar_desde_puntos.py) y [`herramientas/cambios_para_qgis.py`](herramientas/cambios_para_qgis.py) para no clasificar como nube el suelo desnudo rojizo/ocre ($\text{B2} \le 2000 \land \text{B4} > \text{B3} \land \text{B4} > 1{,}2\,\text{B2}$). Con ello, los falsos positivos de nube dentro de las cicatrices de Sillapata cayeron del **$32{,}5\%$ al $0{,}0\%$** manteniendo intacta la detección de nubes reales. Se regeneraron los `candidatos.geojson`, `cambios.geojson` y las **242 máscaras de nivel 3** (`mascaras_provisionales/nivel3/`).

### 4. Comparador sin aprendizaje (línea base espectral-topográfica)

* Siguiendo la recomendación de la sección 9 del informe, se incorporó en [`herramientas/ajuste_fino.py`](herramientas/ajuste_fino.py) (y en [`herramientas/Ajuste_Fino_LaConvencion.ipynb`](herramientas/Ajuste_Fino_LaConvencion.ipynb)) la función `regla_suelo_pendiente`, que evalúa una regla determinista sin aprendizaje profundo ($\text{GRVI} \le 0 \land \text{B4} \ge 1{,}29\text{ L4S} \approx 600\text{ DN} \land \text{pendiente} \ge 10^\circ$) sobre los mismos conjuntos de prueba de Landslide4Sense y validación local.
* En las 8 teselas con cicatrices de nivel 2 de La Convención, esta regla sin aprendizaje alcanza **$\text{Recall} = 83{,}80\%$**, pero **$\text{Precisión} = 18{,}04\%$**, **$\text{IoU} = 0{,}1743$** y **$\text{F1} = 0{,}2968$**, demostrando cuantitativamente que el color y la pendiente por sí solos confunden caminos, chacras y pasto seco, y justificando el uso de segmentación profunda para discriminar la morfología espacial de la cicatriz.

---

## Parte II. Textos corregidos para la Memoria Descriptiva del Artefacto (OE2)

> **Instrucciones de uso:** Los siguientes apartados reemplazan las secciones equivalentes de `Memoria_Descriptiva_Artefacto_UPCH.docx` donde se describían $256\text{ px}$, 14 canales, Google Earth Engine y FastAPI sin detalles de despliegue.

### Texto 1. Arquitectura general y flujo de datos del artefacto (reemplaza diseño de entrada y plataforma)

> **Arquitectura del sistema desplegado.** El artefacto de software se implementó bajo una arquitectura cliente-servidor desacoplada en la nube: una interfaz web interactiva alojada en *Hugging Face Spaces* (`NikolaiVictor/deslizamientos-laconvencion`) y un servicio de procesamiento e inferencia geoespacial desplegado en *Render* (`deslizamientos-laconvencion.onrender.com`). A diferencia de un flujo dependiente de cuentas autenticadas de Google Earth Engine, la adquisición de datos se realiza en tiempo de ejecución y sin requerir credenciales de usuario mediante el catálogo público **STAC de Microsoft Planetary Computer**.
>
> Dado un polígono o caja delimitadora (de hasta $8 \times 8\text{ km}$ dentro de la provincia de La Convención) y un rango de fechas en temporada seca (junio a septiembre, a partir de abril de 2017), el módulo `nucleo.datos` consulta la colección `sentinel-2-l2a`, selecciona hasta 6 escenas con menor cobertura nubosa según el metadato `eo:cloud_cover`, enmascara píxeles defectuosos o nubosos empleando la banda de clasificación de escena (`SCL` $\in \{0, 1, 3, 8, 9, 10\}$), corrige el desplazamiento radiométrico de $-1000\text{ DN}$ introducido en la *baseline* 04.00 de enero de 2022 y construye un compuesto temporal por mediana en la proyección métrica local **UTM Zona 18S (`EPSG:32718`) a $10\text{ m}$ de resolución espacial**.

### Texto 2. Especificación del tensor de entrada, relieve y normalización (reemplaza 256 px y 14 canales)

> **Representación de entrada ($128 \times 128\text{ px}$, 6 canales + 2 índices internos).** Para garantizar la compatibilidad geométrica exacta con el benchmark internacional *Landslide4Sense* (L4S) y preservar la resolución nativa de $10\text{ m/px}$ sin distorsión por remuestreo, el sistema opera sobre ventanas de **$128 \times 128\text{ píxeles}$** ($1{,}28 \times 1{,}28\text{ km}$) con un solape deslizante del **$50\%$** ($64\text{ px}$) durante la inferencia.
>
> Cada ventana se compone de **6 canales de entrada**:
> 1. **Cuatro bandas espectrales de Sentinel-2 (`B01`, `B02`, `B03`, `B04`)**, llevadas de niveles digitales (DN) de reflectancia de superficie L2A a unidades equivalentes de L4S dividiendo cada banda por su media provincial de temporada seca (`escala_laconvencion.json`: $\text{B01} = 381{,}62$; $\text{B02} = 442{,}57$; $\text{B03} = 576{,}95$; $\text{B04} = 465{,}56$).
> 2. **Dos canales topográficos (`B13` pendiente en grados y `B14` elevación en metros)**, derivados de un modelo digital de elevación (DEM) global de $30\text{ m}$ (NASADEM / Copernicus GLO-30) consultado con su propia ventana temporal estática para evitar fallos por fechas de escena. Ambos canales se escalan mediante las constantes físicas de Landslide4Sense ($\text{pendiente} / 20{,}292$ y $\text{elevación} / 772{,}82$).
>
> Posteriormente, los 6 canales se estandarizan mediante media y desviación estándar ($\mu, \sigma$) calculadas estrictamente sobre el conjunto de entrenamiento oficial de L4S (3799 parches). En la versión 4 de la arquitectura **U-Net con codificador ResNet-34**, el grafo exportado a **ONNX (opset 17)** incorpora una capa inicial determinista (`ConIndices`) que calcula internamente dos índices espectrales de vegetación a partir de las bandas visibles: el *Green-Red Vegetation Index* ($\text{GRVI} = (\text{B03} - \text{B04}) / (|\text{B03} + \text{B04}| + 10^{-3})$) y el *Excess Green* ($\text{ExG} = (2\,\text{B03} - \text{B04} - \text{B02}) / (|\text{B03} + \text{B04} + \text{B02}| + 10^{-3})$), entregando 8 mapas de características al codificador sin alterar la interfaz de 6 canales del servidor.

### Texto 3. Verificación forense de pesos, optimización de memoria y entregables (OE3)

> **Integridad de modelos y optimización de recursos.** Antes de habilitar cualquier modelo en el catálogo (`/modelos`), el backend ejecuta una verificación estadística sobre los tensores convolucionales del archivo `.onnx` (`revisar_pesos`), comprobando su desviación estándar, cota máxima y curtosis frente a las distribuciones de inicialización uniforme y normal de Kaiming/Xavier ($1/\sqrt{\text{fan\_in}}$) y verificando su huella criptográfica SHA-256. Para operar de manera estable dentro del límite estricto de **$512\text{ MB}$ de RAM** del plan gratuito de Render, se aplicaron cuatro optimizaciones de ingeniería: (i) separación de tensores de pesos a un archivo externo mapeado en disco (`.onnx.data`), ahorrando $\sim 95\text{ MB}$ de memoria residente; (ii) nivel de optimización de grafo `ORT_ENABLE_EXTENDED` en lugar de `ALL` para evitar la duplicación de pesos al formato interno `NCHWc` ($\sim 100\text{ MB}$ menos); (iii) inferencia por lotes de 4 parches con liberación inmediata de la sesión ONNX (`SOLTAR_SESION = 1`) y devolución explícita de páginas al sistema operativo (`malloc_trim`); y (iv) cola de ejecución secuencial de un trabajo a la vez, reduciendo el pico de memoria de $759\text{ MB}$ a $440\text{--}480\text{ MB}$. Como salida, cada corrida entrega la máscara binaria y el mapa de probabilidad en formato **GeoTIFF (`EPSG:32718`)**, los polígonos vectorizados ($\ge 4\text{ px}$, equivalente a $\ge 0{,}04\text{ ha}$) en **GeoJSON (`EPSG:4326`)**, una vista previa **PNG** y un **manifiesto JSON** auditable con las escenas empleadas y los tiempos por etapa.

---

## Parte III. Redefinición fundamentada de metas y reporte de resultados (OE4)

> **Por qué debe incluirse en el capítulo de Metodología / Discusión:** Las metas originales del anexo ($\text{IoU} \ge 0{,}70$, $\text{Recall} > 85\%$, $\text{FP} \le 15\%$, $\text{F1} > 75{,}5\%$) son matemáticamente redundantes ($\text{Recall} > 85\%$ con $\text{Precisión} = 1 - \text{FP} \ge 85\%$ ya obliga a $\text{F1} > 85\%$ e $\text{IoU} > 73{,}9\%$) y superan por más de 10 puntos el estado del arte mundial publicado sobre la prueba oficial de Landslide4Sense.

### Texto 4. Contextualización de métricas frente al benchmark internacional Landslide4Sense

> **Desempeño en el benchmark oficial y contextualización frente a la literatura.** En la partición de prueba oficial del benchmark internacional *Landslide4Sense* (800 parches de $128 \times 128\text{ px}$), utilizando únicamente los 6 canales compatibles con Sentinel-2 L2A y DEM global (`B01`–`B04`, pendiente y elevación) de los 14 canales disponibles en el concurso original, el modelo **U-Net v4** alcanzó un **$\text{IoU} = 0{,}4602$** y **$\text{F1} = 0{,}6303$** (precisión $59{,}1\%$, exhaustividad $67{,}6\%$), superando de forma estadísticamente significativa a la versión base U-Net v1 ($\text{IoU} = 0{,}4245$; diferencia por *bootstrap* de 2000 repeticiones: $\Delta\text{IoU} \in [+0{,}011; +0{,}063]$ al $95\%$ de confianza, ganando en el $100\%$ de las réplicas). Por su parte, el **ensamble mixto (U-Net v4 + DeepLabV3+ v1, umbral $0{,}35$)** alcanzó **$\text{IoU} = 0{,}4656$** y **$\text{F1} = 0{,}6354$** (exhaustividad $71{,}2\%$).
>
> Estos resultados deben interpretarse a la luz de la literatura oficial del benchmark (Ghorbanzadeh et al., *IEEE JSTARS*, 2022): la línea base oficial publicada por los organizadores con U-Net y los 14 canales completos obtuvo un **$\text{F1} = 59{,}92\%$** ($\text{IoU} \approx 0{,}4278$), mientras que el primer lugar mundial de la competencia 2022 —empleando los 14 canales y ensambles pesados— alcanzó **$\text{F1} = 74{,}54\%$** ($\text{IoU} \approx 0{,}5941$). En consecuencia, nuestra U-Net v4 con solo 6 canales supera en $+3{,}1$ puntos de F1 a la U-Net de referencia de 14 canales del artículo original, y el análisis estratificado por área demuestra que en deslizamientos medianos ($0{,}5\text{ a }2\text{ ha}$) y grandes ($> 2\text{ ha}$) la tasa de detección por evento alcanza el **$90{,}7\%$** y **$97{,}4\%$**, respectivamente, concentrándose la pérdida de sensibilidad en micro-cicatrices menores a $0{,}1\text{ ha}$ ($< 10\text{ píxeles}$ a $10\text{ m}$ de resolución).

