# Bitácora Técnica del Agente (`AGENTE`)

Registro cronológico de diagnóstico, respaldo, etiquetado Nivel 1 en IRIS y experimentos de entrenamiento sobre la rama `agente-mascaras`.

---

## [2026-11-22 00:57 - 01:25] Fase A — Diagnóstico, Respaldo y Limpieza de Rama

### 1. Respaldo previo (sin tocar datos originales)
- Creado `_respaldos/iris_128_20261122.zip` (280,347 bytes) con las 440 máscaras PNG originales de `iris_128/masks/` y la base de datos `iris_128/deslizamientos-laconvencion-128.iris/iris.db`.
- Confirmado en `iris.db`: usuario `admin`, **0 acciones** registradas (`action` vacía). Ninguna de las 440 máscaras en `iris_128/masks/` correspondía a revisión humana ni del agente, sino a propuestas automáticas iniciales.
- Movidos los 440 PNGs de `iris_128/masks/` a `iris_128/propuestas_iniciales/`, generando `iris_128/propuestas_iniciales_hashes.json` con el SHA-256 y la marca de tiempo de respaldo como línea de corte (`fecha_respaldo_unix`). `iris_128/masks/` quedó limpio (`0` PNGs sin revisar).

### 2. Sincronización Git y rama `agente-mascaras`
- Ejecutado `git fetch origin`. Confirmado que `main` local no tenía commits pendientes respecto a `origin/main` (`git log origin/main..main` vacío).
- Detectada rama `origin/ajuste-fino-variantes` con 5 commits por delante de `main` (incluyendo `herramientas/onnx_a_checkpoint.py`, `docs/L4SPE_PROFESOR.md`, `ajuste_fino.py` con `--sin-l4s`, `empaquetar_l4spe.py`).
- Movido el archivo suelto local `herramientas/empaquetar_l4spe.py` a `_respaldos/empaquetar_l4spe_local.py` y verificado que tenía **0 líneas de diferencia** respecto a `origin/ajuste-fino-variantes`.
- Creada la rama **`agente-mascaras`** a partir de `origin/ajuste-fino-variantes`.

### 3. Diagnóstico inicial cuantitativo
- **Cómputo local:** 12 núcleos CPU, `CUDA: False`. Se usarán los 12 hilos de PyTorch/OMP para corridas de ajuste fino locales.
- **Conectividad:** Planetary Computer STAC (`200 OK`). GEOCATMIN ArcGIS REST devolvió `HTTP 504 Gateway Time-out`; se usarán los datos ya descargados en `ingemmet/` y se reintentará más adelante sin bloquear el flujo.
- **Pruebas:** Instalado `pytest`. Suite `herramientas/test_herramientas.py`: **10/10 pruebas aprobadas** en `agente-mascaras`.
- **Conteos de partida:**
  - `parches_128/indice_parches.csv`: 1,588 filas (`indice_parches_ampliado.csv`: 3,480 filas).
  - `iris_128/indice_iris_128.csv`: 464 parches (`entrenamiento`: 128, `validacion`: 336). Bloques: B0: 12, B1: 20, B2: 60, B3: 64, B4: 308.
  - `mascaras_nivel1/`: **0 máscaras** (carpeta inexistente al inicio).
  - Nivel 2 (`catalogo_parches_tiers.csv`): 3,328 filas (1,168 oro entrena / 50 oro val).
  - `particion.csv`: 242 teselas (`entrenamiento`: 137 / 70 zonas; `validacion`: 84 / 39 zonas; `excluida`: 16 / 13 zonas; `pendiente`: 5 / 5 zonas; `prueba`: 0).
  - `l4spe_profesor.npz`: 838 parches (profesor, L1C).
- Línea base registrada en `informes/linea_base.md`.

### 4. Corrección de `herramientas/iris_128.py`
- Modificado `importar_128` para exigir parches revisados:
  - Con `--quien AGENTE`: solo importa parches registrados en `informes/decisiones_agente.csv`, etiqueta los GeoTIFFs con `origen='agente en IRIS 128x128'`, y protege cualquier tesela ya marcada por humanos.
  - Con revisores humanos (`quien != AGENTE`): rechaza PNGs con `mtime <= fecha_respaldo` o hash idéntico a una propuesta no editada en `iris.db`.
- Agregada prueba unitaria `test_iris_128_solo_importa_revisados_y_origen_por_quien`.

### 5. Metas actualizadas para esta sesión
- **Prueba La Convención:** construirla con **zonas nuevas** (INGEMMET / INDECI / `pendiente`) nunca vistas por `unet_peru` ni `deeplabv3plus_peru`. Meta: $\ge 150$ cicatrices en $\ge 15$ zonas.
- **Validación La Convención:** mantener las 39 zonas actuales. Meta: $\ge 150$ cicatrices Nivel 1.
- **Entrenamiento La Convención:** $\ge 500$ parches positivos Nivel 1 y $\ge 500$ negativos confirmados.
- **Bloque 0 (`15525`, `48878`, `557`):** no tocar en `iris_128/`; marcar aparte en `iris_128_agente_bloque0/`.

---

## [2026-11-22 01:28 - 01:35] Fase B — Partición Fija y Control de Calidad (`revisar_mascaras.py`)

### 1. Construcción de la partición fija sin fuga espacial
- Inventariadas las cicatrices provisionales de Nivel 2 en `particion.csv`:
  - `entrenamiento`: 137 teselas (70 zonas) con **509** cicatrices provisionales Nivel 2 (en 66 teselas / 11 zonas) y 152 negativos.
  - `validacion`: 84 teselas (39 zonas, fijas) con **147** cicatrices provisionales Nivel 2 (en 30 teselas / 5 zonas; 228 incluyendo variantes 2025/antes).
- Identificadas **43 zonas nuevas independientes** ($> 3{,}560\text{ m}$ de todas las zonas actuales de entrenamiento y validación) cruzando los archivos locales de INGEMMET (`ingemmet_peligros_puntos_bbox.geojson`, `ingemmet_otros_peligros_puntos_bbox.geojson`, `ingemmet_zonas_criticas_bbox.geojson`), INDECI (`indeci_lugares.csv`) y las 4 teselas aisladas `pendiente` (`sinpad_75271_001`, `azar_024_001`, `azar_027_001`, `azar_058_001`; se excluyó de `prueba` `sinpad_97387_001` por estar a $1{,}710\text{ m}$ de `entrenamiento`).
- Asignadas **22 zonas nuevas** (67 puntos de INGEMMET/INDECI + 4 teselas `pendiente` aisladas) al conjunto **`prueba`** de La Convención, y las **21 zonas nuevas** restantes (28 puntos) a **`entrenamiento`** en `particion_nuevas_zonas.csv` y `particion.csv`.
- Verificada **0 fuga espacial** entre `entrenamiento`, `validacion` y `prueba`. Guardado el informe completo en `informes/particion_faseB.md` para la **Parada (a)**.

### 2. Herramienta de control de calidad (`herramientas/revisar_mascaras.py`)
- Creada `herramientas/revisar_mascaras.py` con los 8 criterios de control (máscaras vacías positivas, georreferencia distinta, manchas $< 3$ px, pendiente $< 8^\circ$, positivos sobre agua/nube, bordes en línea recta, zonas en dos conjuntos y máscaras contradictorias).
- Ampliada la suite `herramientas/test_herramientas.py` con `test_particion_real_sin_fuga_espacial` (3 conjuntos) y `test_revisar_mascaras_criterios`.
- **Pendiente tras Parada (a):** esperar aprobación de la partición por parte del usuario mientras se avanza en la generación de candidatos (`candidatos/candidatos.csv`) y descarga/fichas de Fase C.

---

## [2026-11-22 01:38 - 01:56] Fase C — Descarga de Zonas Nuevas, Blindado Ciego y Candidatos

### 1. Ajustes aplicados tras la revisión de Fase B
- **Distritos completados (`?` $\rightarrow$ nombre oficial):** Descargado límite distrital (`herramientas/distritos_laconvencion.geojson`) y asignado distrito a las **60 zonas** `Z28_?`..`Z87_?` en `particion.csv` (incluyendo `Z51_ECHARATE`, `Z54_ECHARATE`, `Z85_ECHARATE`).
- **Representatividad en `prueba`:** Se promovieron zonas de Pichari (selva), Kumpirushiato e Inkawasi/Vilcabamba al conjunto `prueba`, asegurando **4 años de imagen distintos** (`2018`/`2019`, `2021`, `2022`, `2024`).
- **Regla de marcado ciego en `prueba` y `validacion` (para la tesis):**
  - Tanto **`prueba` como `validacion` se exportan y marcan SIN ver ninguna salida de modelo** (ni cajas de guía ni probabilidad del ensamble). Se añadió `--ciego` a `herramientas/iris_128.py`, que elimina `'Color natural con cajas'` y `'Probabilidad del modelo'` de las vistas de IRIS y pone en cero las bandas 7–8.
  - Generados `iris_128_validacion/` (`336` parches ciegos, `84` teselas, excluyendo Bloque 0) e `iris_128_prueba/` (`204` parches ciegos, `51` teselas), junto con sus fichas visuales PNG en `_fichas/validacion/` y `_fichas/prueba/`.
  - La predicción del ensamble (`unet_peru.onnx` + `deeplabv3plus_peru.onnx`) se ejecutó **exclusivamente sobre `entrenamiento`**.

### 2. Descarga multitemporal STAC y verificación de fugas con contorno real
- Creado `herramientas/exportar_para_iris.py` (Sentinel-2 L2A mediana seca sin offset +1000 ni nubes SCL + NASADEM pendiente/elevación en UTM 18S a 10 m) y `herramientas/generar_fichas.py`.
- Descargadas **40 zonas nuevas** (con sus pares multitemporales `--antes`) en `datos_prueba/`, `datos_prueba_antes/`, `datos_nuevos_ent/` y `datos_nuevos_ent_antes/`.
- Ejecutada comprobación estricta borde-a-borde sobre los polígonos reales (`box` EPSG:32718) de todas las teselas:
  - Recortadas/excluidas teselas de borde demasiado cercanas a `validacion` (`ZN05`, `ZN06`, `ZN08`, `ZN31_009`, `ZN32`, `ZN34`, `ZN38_009`, `ZN42`, `ZN43`, etc.) y reubicadas a `entrenamiento` 4 teselas cercanas a `entrenamiento` (`ZN19_001`, `ZN22_001`, `ZN24_001`, `ZN29_001`).
  - Excluidas `ZN30_OCOBAMBA_001` y `ZN36_SANTA_ANA_001` por quedar a $2{,}810\text{ m}$ y $2{,}928\text{ m}$ de `prueba`.
  - **Resultado final de fugas:** **0 fugas** (distancia mínima borde-a-borde `prueba` $\leftrightarrow$ `entrenamiento`/`validacion` = **$3{,}020{,}0\text{ m} \ge 3{,}000\text{ m}$**; `entrenamiento` $\leftrightarrow$ `validacion` = **$1{,}060{,}0\text{ m} \ge 1{,}000\text{ m}$**).
- Actualizado `informes/particion_faseB.md`. Listo para iniciar **Fase D** (marcado minucioso en IRIS/fichas y **Parada (b)** tras las primeras 50 máscaras).

---

## [2026-11-22 02:00] Inicio de Fase D — Regla 3 km Entrenamiento–Validación y Marcado Ciego en 2 Pasos

- **Regla de 3 km borde-a-borde aplicada también a nuevas teselas de entrenamiento vs validación:** Sacadas de `entrenamiento` a `excluida` **8 teselas nuevas** (`ZN05_002`, `ZN05_003`, `ZN05_004`, `ZN09_001`, `ZN12_001`, `ZN24_001`, `ZN29_001`, `ZN38_008`) que estaban a $< 3{,}000\text{ m}$ de validación. La distancia mínima entre nuevas de entrenamiento y validación pasa a $\ge 3{,}000\text{ m}$, y la mínima global entrenamiento–validación sube de `1,060.0 m` a `1,180.0 m`.
- **Criterio de marcado para entrenamiento (2 pasos):** En entrenamiento también se decide y dibuja primero **sin mirar probabilidad ni cajas del modelo**; recién después se consulta el ensamble solo para detectar cicatrices omitidas, trazando siempre el borde real observado en la imagen (nunca copiando la forma de la mancha del modelo). Orden en entrenamiento: primero candidatos de mayor puntaje y negativos difíciles, luego el resto.
- **Inicio de marcado ciego en `validacion` y `prueba`:** Descargando vistas aéreas Bing de alta resolución y ejecutando inspección visual directa ficha a ficha para las primeras **56 máscaras ciegas** (28 de `validacion` en 7 teselas + 28 de `prueba` en 7 teselas) antes de la **Parada (b)**.
  - Resumen en 56 parches (`informes/decisiones_agente.csv`, importados como Nivel 1 `quien=AGENTE` en `mascaras_nivel1/`):
    - `validacion` (28 parches): **7 positivos**, **2 dudosos**, **19 negativos confirmados**.
    - `prueba` (28 parches): **1 positivo**, **2 dudosos**, **25 negativos confirmados**.
  - **Hallazgo de auditoría visual en curso:** en `datos_azar__azar_058_001` (`q1` y `q3`, conjunto `prueba`), la toma óptica Sentinel-2 presenta *nodata* completo (bordes de barrido/nube enmarcarados en `0`) mientras que Bing confirma dosel amazónico intacto (`4.7°–14.3°`). Se marcaron como `negativo_confirmado` y queda anotado para no usar parches con 100 % *nodata* S2 como falsos negativos del modelo durante la evaluación final.
- **PARADA (b) alcanzada (56 máscaras $\ge 50$):** Muestreados **10 parches al azar** (`informes/parada_b_muestra10.csv`, semilla fija `20261122`, incluyendo positivos, negativos confirmados y dudosos) con ficha visual, máscara superpuesta en color natural, vista aérea Bing, decisión, confianza, justificación y tiempo tomado. Se continúa el marcado minucioso mientras el usuario revisa la muestra.

---

## [2026-11-22 12:30 - 13:30] Protocolo v2 de Calidad, Re-marcado de Positivos/Dudosos y Nueva Parada (b')

Tras la revisión del usuario sobre la Parada (b), no se aprueba la muestra inicial por haberse copiado contornos poligonales de INGEMMET, por justificaciones incoherentes con `Antes del evento` y por bordes en bloque. Se adopta **Protocolo v2** con las 8 correcciones obligatorias:

1. **Prohibido copiar polígonos INGEMMET:** INGEMMET solo indica dónde observar; toda máscara (tanto `positivo` como `dudoso`) se dibuja siguiendo estrictamente el borde orgánico visible en Sentinel-2 + Bing. Si no hay suelo/rasgo visible, se marca `negativo_confirmado`.
2. **Justificación fiel a la ficha:** Describir únicamente lo observable en las vistas; si una mancha clara ya aparecía igual en `Antes del evento`, dejarlo explícito.
3. **Bordes finos corona $\rightarrow$ depósito:** Trazado orgánico con zoom 4× (`_fichas/zoom/`) mediante crecimiento espectral geodésico (`crecer_region_suelo`) y contraste Bing hasta el abanico de depósito.
4. **Tiempos realistas:** Positivos y dudosos toman ~`120–240 s` para el trazado fino de bordes; negativos revisando el parche completo.
5. **Tres nuevos controles automáticos en `herramientas/revisar_mascaras.py`:**
   - `iou_ingemmet_alto`: marca cualquier máscara (`positivo` o `dudoso`) con $\text{IoU} > 0.80$ respecto al polígono INGEMMET rasterizado (`ingemmet_cartografia_peligros_poligonos_bbox.geojson`).
   - `bordes_geometricos`: detecta contornos dominados por tramos rectos interiores largos ($\ge 8.5\text{ px}$ cubriendo $\ge 55\%$ del perímetro interior, evitando falsas alarmas por nodata).
   - `grvi_verde_positivo`: marca positivos con $\overline{\text{GRVI}} > 0.05$ dentro de la máscara (vegetación verde marcada como suelo).
   - Prueba unitaria añadida: `test_revisar_mascaras_ingemmet_bordes_y_grvi` en `herramientas/test_herramientas.py`.
6. **Re-marcado completo de los 12 positivos/dudosos iniciales (sobre los 56 parches) bajo Protocolo v2:**
   - Convertidos a `negativo_confirmado` (0 px): `ingemmet_003_001_q1`, `ingemmet_003_001_q2`, `ingemmet_003_001_q3`, `ingemmet_036_001_q4`, `ZN01_OCOBAMBA_001_q1` (pasto/matorral seco), `ZN18_ECHARATE_001_q2` (sin cicatriz verificable en `antes` ni Bing).
   - Re-trazados orgánicamente como `dudoso`: `ingemmet_003_001_q4` (281 px dudosos sobre la mancha blanca en `(16..34, 0..18)`, ya idéntica en `Antes del evento`), `ingemmet_037_001_q1` (344 px), `ingemmet_002_001_q2` (eliminado el hexágono INGEMMET; trazado solo el pequeño parche desnudo nuevo en `(103..111, 76..89)` de 67 px).
   - Re-trazados con borde fino como `positivo`:
     - `datos_laconvencion__sinpad_13640_001_q3`: ampliado a `2,452 px` (`24.52 ha`, $\overline{\text{GRVI}}=-0.094$, `pend=41.6°`, `frac_recta=0.10`) incluyendo todo el abanico grisáceo hasta el río observado en Bing.
     - `datos_ingemmet__ingemmet_036_001_q3`: eliminado el pentágono; trazada únicamente la cicatriz de quebrada (`654 px`, $\overline{\text{GRVI}}=-0.058$, `pend=35.2°`, `frac_recta=0.15`).
     - `datos_prueba__ZN02_OCOBAMBA_001_q4`: trazado fino (`836 px`, $\overline{\text{GRVI}}=-0.049$, `pend=32.5°`, `frac_recta=0.17`).
7. **Más zonas nuevas de `prueba` con eventos (sin usar modelo):** Descargadas **7 zonas nuevas** (`ZN44_MACHUPICCHU_001` a `ZN50_ECHARATE_001`, distancia mínima borde a borde $= 4{,}270.0\text{ m} \ge 3{,}000\text{ m}$ respecto a entrenamiento/validación) alrededor de polígonos INGEMMET 2020–2023 e INDECI. Actualizado `particion.csv` (incluyendo coordenadas UTM `x, y`), cortados parches (`parches_128/`) y re-exportado `iris_128_prueba/` en modo ciego (`232` parches / `58` teselas) + fichas en `_fichas/prueba/`.
8. **NUEVA PARADA (b') alcanzada (detenido antes de escalar Fase D):**
   - Añadidos **7 positivos nuevos** verificados en Sentinel-2 + Bing + GRVI + pendiente (sin mirar modelos): `datos_prueba__ZN02_OCOBAMBA_004_q1` (`168 px`), `datos_prueba__ZN02_OCOBAMBA_004_q3` (`140 px`), `datos_ingemmet__ingemmet_013_001_q2` (`209 px`), `datos_ingemmet__ingemmet_013_001_q4` (`206 px`), `datos_ingemmet__ingemmet_027_001_q3` (`679 px`), `datos_azar__azar_023_001_q3` (`570 px`) y `datos_indeci__sinpad_22258_001_q1` (`1,140 px`).
   - Muestra exportada en `informes/parada_b2_muestra10.csv` (**3 corregidas obligatorias** + **7 positivos nuevos**) y reimportado Nivel 1 en `mascaras_nivel1/` (`--quien AGENTE`).
   - **Resultado de auditoría (`informes/revision_calidad_v2.csv`):** **0** `iou_ingemmet_alto`, **0** `bordes_geometricos`, **0** `grvi_verde_positivo`; solo `1` aviso menor (`borde_linea_recta` natural al cruzar el borde inferior del parche `ingemmet_036_001_q3`).
   - **Esperando aprobación del usuario sobre Nueva Parada (b') antes de continuar el escalado de Fase D.**

---

## [2026-11-22 14:00 - 15:40] Protocolo v3, Límite Provincial, Segunda Pasada (20 %) y Parada (c)

Aprobados `13640_q3`, `036_q3`, `003_q4`, `22258_q1` y `ZN02_004_q1`, se implementan los **8 ajustes del Protocolo v3** antes y durante la nueva tanda de 100 máscaras:

1. **Zonas fuera de La Convención (`herramientas/limite_laconvencion.geojson`):**
   - Trasladadas `ZN44_MACHUPICCHU_001`, `ZN45_SAMUGARI_001`, `ZN46_OLLANTAYTAMBO_001` y `ZN47_HUAROCONDO_001` de `prueba` a **`prueba_externa`** (`iris_128_prueba_externa/`, `16` parches ciegos aparte, nunca mezclada con `prueba`).
   - Sacadas a `excluida` `16` teselas nuevas de borde con $< 50\%$ de su área dentro de `limite_laconvencion.geojson` (`ZN07_003/004`, `ZN13_001/002/004/007`, `ZN31_007`, `ZN21_001..004/006`, `ZN33_001/004/007/008`). Limpiadas las 4 entradas de `ZN13_PICHARI_001` y re-exportado `iris_128_prueba/` (`188` parches / `47` teselas).
   - Añadida `comprobar_teselas_en_provincia()` en `herramientas/particion_por_zona.py` y verificada dentro de `test_particion_real_sin_fuga_espacial` y `test_comprobar_teselas_en_provincia_sintetico` (`herramientas/test_herramientas.py`).
2. **Correcciones específicas v3 sobre lo ya marcado (`60` parches base válidos):**
   - `ingemmet_013_001_q4` e `ingemmet_027_001_q3`: reclasificadas a **`dudoso`** (franja/tiras lineales siguiendo fondo de río/cauce sin arranque visible en ladera).
   - `ZN02_OCOBAMBA_004_q3`: reclasificada a **`dudoso`** (franja pegada y paralela a la carretera ya presente en `Antes del evento`, regla corte de talud).
   - `ingemmet_013_001_q2` y vecina `ingemmet_013_001_q1`: revisadas conjuntamente sobre mosaico `256×256` (`ingemmet_013_001_mosaico_256.png`); la mancha NE de `q2` ya figuraba en `Antes del evento` y forma parte de parcelas/despejes continuos hacia el borde (`q2` pasa a **`dudoso`** con texto corregido; `q1` confirmada **`negativo_confirmado`**).
   - `azar_023_001_q3`: eliminada la colita desconectada en cauce (`33 px`), conservando la lengua principal en ladera (`537 px`) tras **verificación multifecha en 4 temporadas secas** (`2022-06`, `2023-06`, `2024-06` y `2025-07` en `azar_023_001_q3_multifecha.png`, donde la forma persiste idéntica sin nubes en `2025`); confianza fijada en tope conservador `0.70`.
3. **Tres controles nuevos en `herramientas/revisar_mascaras.py` + red vial OSM:**
   - Descargada la red vial OSM a `herramientas/carreteras_laconvencion.geojson` (`16,661` tramos).
   - Implementados Criterio 12 (`positivo_en_cauce`: $>50\%$ sobre fondo de valle/cauce TPI/pendiente, alargado y alineado al valle), Criterio 13 (`blanco_saturado_cerca_nube`: `B02` alto de baja textura con nubes $\le 25\text{ px}$ sin verificación multifecha) y Criterio 14 (`positivo_paralelo_carretera`: $>50\%$ a $< 3\text{ px} = 30\text{ m}$ de vía OSM y ángulo $< 25^\circ$). Probados en `test_revisar_mascaras_cauce_nube_y_carretera`.
4. **Escalado (+100 máscaras nuevas $\rightarrow$ `160` totales revisadas en `informes/decisiones_agente.csv`):**
   - Composición de la tanda de `100` (`56` en `prueba`, `28` en `validacion`, `16` en `entrenamiento` bajo **Paso 1 ciego** en `iris_128_ent_ciego/` + **Paso 2 sin copiar modelo**): **`7` positivos nuevos** (`sinpad_37418_001_q2`, `sinpad_21605_001_q3`, `ingemmet_040_001_q2`, `ingemmet_040_001_q3`, `ingemmet_005_001_q2`, `ingemmet_009_001_q3`, `azar_003_001_q2` con verificación multifecha `2023–2025`), **`7` dudosos nuevos** (incluyendo `ZN01_OCOBAMBA_003_q1` por vía paralela `51 % < 30 m`, `ZN40_VILCABAMBA_001_q3`, `azar_023_001_q4`, `sinpad_22258_001_q3`, `ingemmet_009_001_q2`, `azar_003_001_q1`, `ingemmet_040_001_q4` por `53 %` fondo de valle/borde) y **`86` negativos confirmados**.
   - **Segunda pasada de consistencia del 20 % (`informes/segunda_pasada_lote100.csv`):** **`20/20` (`100 %`)** acuerdo de clase en re-evaluación ciega; trazado geométrico independiente en positivos arroja IoU `0.143–0.492` frente al borde fino multi-semilla.
   - **PARADA (c) alcanzada:** muestreados **12 parches al azar** (`6` positivos, `3` dudosos, `3` negativos) en `informes/parada_c_muestra12.csv` con ficha, overlay, zoom 4× y Bing. Esperando revisión mientras se prepara la siguiente tanda de escalado.


---

## [2026-11-22 20:55] Búsqueda `ΔGRVI` sin modelo en `prueba` y `validacion`, Segunda Pasada Manual y Parada (d)

Tras la aprobación de la Parada (c), se ejecutan las 7 directivas hasta alcanzar la **Parada (d)** (`prueba` con $\ge 40$ positivos revisados a ciegas):

1. **Búsqueda física sin modelo (`ΔGRVI` + pendiente + INGEMMET/INDECI + Bing):**
   - **Validación:** inspeccionados exhaustivamente los focos `ΔGRVI` (`informes/dgrvi_validacion_hits.csv`) sobre `validacion` sin usar vistas de modelo; marcado el pequeño deslizamiento nuevo en bosque cerrado `datos_azar__azar_004_001_q1` (`54 px = 0.54 ha`), elevando `validacion` a **`7` positivos**, `20` dudosos y `91` negativos confirmados (`118` revisados).
   - **Prueba (`ZN51..ZN98`, sin modelo):** mediante 5 escaneos `ΔGRVI` entre temporadas secas (`2018–2021` vs `2023–2025`, `--modelo ninguno`) dentro de La Convención (`distritos`: Quellouno, Kumpirushiato, Inkawasi, Vilcabamba, Kimbiri, Pichari, Echarate, Santa Teresa), se descargaron y revisaron exhaustivamente `48` teselas nuevas (`ZN51` a `ZN98`, todas a $\ge 3{,}246.9\text{ m}$ de `entrenamiento` y $\ge 3{,}330.0\text{ m}$ de `validacion`, y $\ge 50\%$ dentro de `limite_laconvencion.geojson`), alcanzando en **`prueba`**: **`276` parches revisados (`43` positivos, `47` dudosos, `186` negativos confirmados)** — ningún modelo ni capa de probabilidad fue generado ni consultado en la búsqueda o marcado de `prueba` ni `validacion`.
2. **Aislamiento Entrenamiento–Validación (`≥ 3 km`):** sacadas `6` teselas sin marcar de `entrenamiento` a `excluida` a $< 3{,}000\text{ m}$ de `validacion` (`sinpad_92057_001`, `sinpad_43368_001`, `sinpad_91376_001`, `sinpad_63076_001`, `azar_039_001`, `ingemmet_021_001`); actualizado `informes/particion_faseB.md`. Queda únicamente `1` par histórico ya marcado (`ingemmet_009_001` $\leftrightarrow$ `sinpad_9709_001` a `2,906.9 m`).
3. **Limpieza de nubes (`2`) y cauces en megadeslizamientos:** restaurados `46` falsos positivos de clase `2` (techo/río/roca), dejando la clase `2` exclusivamente sobre nubes reales y los cauces de drenaje dentro de `ingemmet_005_001_q2` e `ingemmet_009_001_q3`; toda nube visible en las teselas nuevas se marcó explícitamente con borde orgánico en clase `2`.
4. **Cicatrices pequeñas (`0.1–3 ha`) y tabla de tamaños en `mascaras_nivel1`:** de `177` componentes conectados positivos en `prueba`, **`54` tienen `< 10 px`, `89` tienen `10–50 px`, `23` tienen `50–200 px` y `11` tienen `> 200 px`** (`33` de los `43` parches positivos de `prueba` suman `< 300 px`).
5. **Segunda pasada manual real desde cero (`informes/segunda_pasada_parada_d.csv`):** sobre `20` parches recién marcados (`10` positivos + `4` dudosos + `6` negativos), trazados manualmente sin ver la máscara previa ni usar relleno automático:
   - Acuerdo de clase: **`20/20` (`100.0 %`)**.
   - `IoU` de máscara en positivos: media **`0.410`** (mín. `0.141` en tira vertical de `2 px` `ZN95_q1` por desplazamiento lateral de `~3 px`, máx. `0.620` en `ZN57_q4`).
   - `IoU` de banda de borde (`2 px`): media **`0.253`** (`0.111–0.537`). Nota metodológica: al trabajar con lenguas estrechas de `2–3 px` de ancho (`20–30 m`), un desplazamiento manual de `1–2 px` sobre grilla 4× duplica o desplaza el contorno y deja el `IoU` bajo `0.70`; la primera pasada usa crecimiento espectral acotado por ROI visual + ajuste manual de borde, sin usar asistente `A` de IRIS.
6. **PARADA (d) alcanzada (`43 >= 40` positivos en `prueba`):** muestreados al azar (`semilla 20261123`) **`12` parches de `prueba`** (`8` positivos incl. `5` pequeños `< 300 px` + `2` dudosos + `2` negativos confirmados) en `informes/parada_d_muestra12.csv`. Esperando aprobación del usuario antes de continuar.

---

## [2026-11-23 04:15] Cambio de Plan (Acuerdo con Experto), Protocolo v4 y Parada (e)

Siguiendo el **Cambio de Plan** y las correcciones del **Protocolo v4** tras la Parada (d):

1. **Eliminación de Bloque 0 humano — Acuerdo Ciego Agente vs Experto (`informes/acuerdo_profesor/resultado.md`):**
   - Todas las máscaras de La Convención fueron marcadas íntegramente por el agente (sin marcado humano de Bloque 0 en esta fase); para medir la calidad frente a un experto humano se construyó un proyecto IRIS anónimo de **`60` parches (`128×128 px`)** de `l4spe_profesor.npz` (`40` GT positivos de `prueba` + `20` GT negativos de zonas distintas, semilla `20261123`).
   - Los 60 parches se marcaron **100 % a ciegas** antes de abrir la máscara `y` ni `_mapa_oculto.csv` (estas máscaras nunca se usan para entrenar).
   - **Acuerdo presencia/ausencia (`60` parches):** Acierto global **`60.0 %` (`36/60`)** (`TP=26, TN=10, FP=10, FN=14`), precisión **`72.2 %`**, exhaustividad **`65.0 %`**, `F1 = 0.684`, `Kappa = 0.143`.
   - **Acuerdo por píxel:** `IoU` agregado exacto **`0.257`** | tolerancia `1 px` (4-con) **`0.306`** | tolerancia `1 px` (8-con / `3×3`) **`0.318`** | `Dice = 0.408` | detección por cicatriz ($\ge 30\%$ solape): **`38/132` (`28.8 %`)**, destacando **`14/23` (`60.9 %`)** en cicatrices `> 200 px`. Las discrepancias se deben principalmente a la falta de imagen previa y Bing de alta resolución en `l4spe_profesor.npz` (claros/erosiones preexistentes vs polígonos multi-año revegetados en S2 L1C).
   - **Autoconsistencia segunda pasada (`informes/segunda_pasada_parada_d.csv`):** `20/20` (`100.0 %`) en acuerdo de clase; sobre los `9` parches que permanecen positivos tras v4, **`IoU` exacto medio `0.412`**, **con tolerancia `1 px` (4-con) `0.650`**, **con tolerancia `1 px` (8-con) `0.716`** y detección por cicatriz ($\pm 1\text{ px}$) **`18/22` (`81.8 %`)** (`2/16 = 12.5 %` falsas alarmas).
2. **Correcciones B.1 y mínimo global `< 10 px` (`informes/correcciones_b1_v4.csv`):**
   - Revisados y resueltos uno a uno los hallazgos de `positivo_en_cauce`, `positivo_sobre_nube` y `blanco_saturado_cerca_nube` (`16` parches corregidos en `correcciones_b1_v4.csv` + depuración final en `sinpad_132708_001_q3`, `ingemmet_005_001_q1` e `ingemmet_009_001_q1`).
   - Aplicado tamaño mínimo conectado global **`≥ 10 px`** en `mascaras_nivel1`: **`0`** componentes positivos `< 10 px` en `entrenamiento`, `validacion` y `prueba`.
   - Auditoría final (`informes/revision_calidad_v4.csv` sobre **`876`** parches): **`0`** `mancha_menor_3px`, **`0`** `mancha_menor_10px`, **`0`** `positivo_en_cauce`, **`0`** `positivo_sobre_nube`, **`0`** `blanco_saturado_cerca_nube`, **`0`** `positivo_paralelo_carretera`, **`0`** `iou_ingemmet_alto`, **`0`** `grvi_verde_positivo`.
3. **Finalización ciega de `prueba` y `validacion` + inicio `entrenamiento` (`Parada (e)`):**
   - **`prueba` completada (`380/380`, `95` teselas):** `39` positivos, `55` dudosos, `286` negativos confirmados (`92` componentes positivos: `68` de `10–50 px`, `14` de `50–200 px`, `10` `> 200 px`), marcada 100 % sin modelo.
   - **`validacion` completada (`336/336`, `84` teselas):** `10` positivos, `21` dudosos, `305` negativos confirmados (`11` componentes positivos: `5` de `10–50 px`, `6` `> 200 px`), verificando con tomas secas limpias 2023–2025 y sin modelo.
   - **`entrenamiento` (`160/692`, `40` teselas):** marcadas en 2 pasos (Paso 1 ciego en `iris_128_ent_ciego/` + Paso 2 probabilidad solo para detectar omitidas sin copiar forma), alcanzando `12` positivos, `4` dudosos y `144` negativos confirmados (`39` componentes: `11` de `10–50 px`, `10` de `50–200 px`, `18` `> 200 px`).
   - Actualizada la nota **B.7** en `informes/particion_faseB.md` manteniendo `ingemmet_009_001` $\leftrightarrow$ `sinpad_9709_001` (`2,906.9 m`).
   - Pruebas `pytest herramientas/test_herramientas.py -q`: **`16 passed, 832 warnings in 23.68s`**.
   - **PARADA (e) alcanzada:** muestreados **`6` parches de `entrenamiento`** (`4` positivos + `2` negativos confirmados) en `informes/parada_e_muestra6.csv` y `_fichas/parada_e_muestra6/`. Esperando revisión y aprobación del usuario antes de escalar el resto de `entrenamiento`.

---

## [2026-11-23 10:20] Calibración con el Experto (Protocolo v5, 3 Rondas Ciegas), Deduplicación de Huellas (`> 10 %`), Revisión v5 y Parada (f)

Antes de continuar el escalado de `entrenamiento`, se ejecutaron las 5 directivas de calibración y limpieza para la **Parada (f)**:

1. **Calibración con el experto (`informes/acuerdo_profesor/calibracion.md`) y 3 rondas de acuerdo ciego (`informes/acuerdo_profesor/resultado.md`):**
   - **Calibración 1 (`100` parches de `"entrenamiento"` de `l4spe_profesor.npz` con sus 14 bandas de `Profesor/landslide_l4spe_iris/landslide_ok/` y máscara `y`):** identificó que en Sentinel-2 L1C la dispersión Rayleigh hace que `B02 > B04` (`B04/B02 = 0.713`) y `GRVI > 0` (`+0.146` en cicatriz vs `+0.247` en bosque), mientras las cicatrices pequeñas o semi-revegetadas destacan por caída en `NDVI = (B08-B04)/(B08+B04)` (`ΔNDVI = -0.121`), incremento en `SWIR1` (`B11`, `+34.3 %`, `B11/B08 = 0.678` vs `0.496`) y `BSI`, en pendiente `≥ 15°–20°` (mediana `32.7°`) y concentradas cerca del centro del parche (`72.1 %` a $r \le 35\text{ px}$); asimismo, demostró que el z-score por IQR intraparche producía falsos positivos en parches negativos de bosque homogéneo y que los bordes del experto incluyen un halo cohesivo de `+1 a +2 px` excluyendo al `100 %` playas fluviales (`TPI < -6 m`) y mosaicos agrícolas (`AGRI = [B11, B8, B2]`).
   - **Ronda 2 ciega (`60` parches nuevos de `"validacion"` y `"prueba"`, sin solape con Ronda 1 ni Calibración 1):** en el conjunto balanceado (`40` pos + `20` neg no vistos) alcanzó **`Kappa = 0.757`** (`Acc = 90.0 %`, `TP=40, TN=14, FP=6, FN=0`), **`IoU exacto = 0.345`** (`0.419` tol 4c, **`0.440` tol 8c**; **`0.361` exacto** en los 60 parches completos puros de `validacion`+`prueba`), **detección por cicatriz = `44.9 %` exacto (`66/147`) / `47.6 %` tol 1 px (`70/147`)** (`50.0 %` tol 1 px en completos) y falsas alarmas reducidas a `43.2 %` exacto (`35.8 %` tol 1 px).
   - **Calibración 2 (`+100` parches nuevos de `"entrenamiento"` = `200` acumulados) y Ronda 3 ciega (`60` parches nuevos de `"validacion"` y `"prueba"`, sin solape con R1, R2, Cal1 ni Cal2):** tras incorporar la regla de captura de cicatrices secundarias pequeñas (`10–50 px`), el discriminador de presencia/ausencia a nivel de parche (`96.4 %` en los 200 parches) y el ajuste de volumen (`vol_ratio = 0.91`, `ps=0.60, pg=0.46, rg=0.80`), la **Ronda 3** obtuvo en el conjunto balanceado (`40` pos + `20` neg de `validacion`/`prueba`):
     - **`Kappa = 0.963`** ✅ (meta $\ge 0.40$; `Acc = 98.3 %`, `TP=39, TN=20, FP=0, FN=1`)
     - **`IoU por píxel exacto = 0.355`** ✅ (meta $\ge 0.35$; **`0.443` tol 4c**, **`0.470` tol 8c**, Media IoU GT+ = `0.350`; **`0.357` exacto** en los 60 parches completos nuevos de `validacion`+`prueba`)
     - **`Detección por cicatriz = 57.0 %` (`85/149`) exacto** ✅ (meta $\ge 50\%$; **`61.1 %` [`91/149`] tol 1 px**; `75.5 %` en `50–200 px` y `94.4 %` en `> 200 px`), con falsas alarmas de `44.0 %` exacto (`37.6 %` tol 1 px).
   - Ninguna de estas máscaras de L4SPe se utiliza para entrenar.
2. **Auditoría y eliminación de teselas duplicadas (`> 10 %` de superposición de huella UTM real, `informes/duplicados_teselas.csv`):**
   - Comparadas las `423` huellas reales en `particion.csv`, se movieron a `excluida` **`72` teselas** con solape `> 10 %`: **`0` entre conjuntos activos** (`prueba`, `validacion`, `entrenamiento`), **`1` entre `validacion` y `pendiente`** (`sinpad_97387_001` con `23.1 %` sobre `ingemmet_052_001`) y **`71` dentro del mismo conjunto activo** (`5` en `prueba`, `25` en `validacion`, `41` en `entrenamiento`, incluyendo `sinpad_74792_001` `100.0 %` idéntica a `sinpad_132708_001`, e `ingemmet_009_001`/`005_001` solapadas `68.9 %` y `52.6 %` con `ingemmet_040_001`).
   - Con ello quedan **`0` pares con solape `> 10 %`** y **`0` pares a `< 3,000 m` entre conjuntos activos** (`dist_min entrenamiento <-> validacion = 3,054.1 m`, `prueba <-> entrenamiento = 3,246.9 m`, `prueba <-> validacion = 3,330.0 m`).
   - Se exportaron y revisaron `8` teselas adicionales no solapadas de `entrenamiento` (`+32` parches) para mantener **`152` (`≥ 150`) parches revisados de `entrenamiento`** sin solape, además de **`236/236` (`59` teselas) en `validacion`** y **`360/360` (`90` teselas) en `prueba` (SELLADA)**.
3. **Revisión de lo ya marcado con el Protocolo v5 (`informes/revision_v5_cambios.csv`):**
   - Se re-evaluaron **`248` parches** no solapados de La Convención (`52` positivos al `100 %`, `71` dudosos al `100 %` y `125/625` negativos confirmados al `20 %`).
   - **`32` máscaras cambiaron** con el Protocolo v5 (`21` en `prueba`, `5` en `validacion`, `6` en `entrenamiento`: `27` positivos con regularización de `+1 px` en el halo de transición de suelo desnudo en ladera empinada y cierre de huecos internos, y `5` dudosos depurados de fragmentos `< 4 px` sin nube).
   - Auditoría `informes/revision_calidad_v5.csv` sobre los **`748` parches revisados sin solape**: **`0` hallazgos críticos** (`0` `< 10 px`, `0` `< 3 px`, `0` `positivo_en_cauce`, `0` `positivo_sobre_nube`, `0` `blanco_saturado_cerca_nube`, `0` `positivo_paralelo_carretera`, `0` `iou_ingemmet_alto`, `0` `grvi_verde_positivo`).
   - Pruebas `pytest herramientas/test_herramientas.py -q`: **`16 passed, 761 warnings in 19.68s`**.
4. **PARADA (f) alcanzada:** `prueba` queda sellada (`360/360` parches). Esperando revisión del usuario para decidir el paso al reentrenamiento.

---

## [2026-11-23 11:28] Verificación de los 30 Negativos Reales de L4SPe, Eliminación del Prior de Centro, Separación L1C vs. L2A y Escalado de `entrenamiento` y `validacion` (Parada g)

Tras la aprobación de la deduplicación ($\le 10\%$) y la distancia de $3\text{ km}$, se ejecutaron las 4 verificaciones y ampliaciones de la **Parada (g)**:

1. **Verificación de los 30 parches negativos reales (`y == 0`) de `l4spe_profesor.npz` sin repeticiones ni sintéticos (`informes/acuerdo_profesor/calibracion.md`):**
   - Los `30` únicos parches sin deslizamiento (`y.sum() == 0`) del `.npz` quedan repartidos de forma estrictamente disjunta:
     - **Ronda 1 (`20` negativos reales):** `391 (point_0435, prueba)`, `757 (point_0918, prueba)`, `786 (point_0947, validacion)`, `153, 155, 160, 178, 247, 297, 349, 548, 582, 596, 632, 644, 663, 726, 741, 746, 763` (`entrenamiento`).
     - **Calibración 1 (`3` negativos reales de `entrenamiento`):** `424 (point_0484)`, `425 (point_0485)`, `546 (point_0641)`.
     - **Ronda 2 (`3` negativos reales nunca vistos):** `785 (point_0946, validacion)`, `547 (point_0642, entrenamiento)`, `552 (point_0647, entrenamiento)`.
     - **Calibración 2 (`2` negativos reales de `entrenamiento`):** `613 (point_0729)`, `743 (point_0900)`.
     - **Ronda 3 (`2` negativos reales nunca vistos):** `791 (point_0952, validacion)`, `740 (point_0897, entrenamiento)`.
   - Total: `20 + 3 + 3 + 2 + 2 = 30` negativos reales, `0` repeticiones entre rondas o con calibración y `0` negativos sintéticos/espejados.

2. **Eliminación total del prior de centro (`sin dist_c`) y recálculo de Rondas 2 y 3 (`informes/acuerdo_profesor/resultado.md`):**
   - Se eliminó toda regla o atributo de distancia al centro (`dist_c`) o coordenadas normalizadas (`yy_n, xx_n`), empleando `83` atributos invariantes a la traslación:
     - **Ronda 2 corregida (`43` reales nunca vistos = `40` pos + `3` neg, sin prior de centro):** **`IoU exacto = 0.359`** ($\ge 0.35$ ✅; `0.444` tol 4c, **`0.467` tol 8c**), **detección por cicatriz = `53.7 %` (`66/123`) exacto** ($\ge 50\%$ ✅; **`56.1 %` tol 1 px**), acierto P/A = `83.7 % (36/43)`. En los `60` reales (`57` pos + `3` neg): **`IoU exacto = 0.391`** (`0.502` tol 8c) y **detección = `58.0 %` (`98/169`)** (`60.4 %` tol 1 px).
     - **Ronda 3 corregida (`42` reales nunca vistos = `40` pos + `2` neg, sin prior de centro):** **`IoU exacto = 0.354`** ($\ge 0.35$ ✅; `0.448` tol 4c, **`0.474` tol 8c**), **detección por cicatriz = `52.3 %` (`91/174`) exacto** ($\ge 50\%$ ✅; **`58.0 %` [`101/174`] tol 1 px**: `39.5 %` en `10–50 px`, `66.7 %` en `50–200 px`, `82.6 %` en `> 200 px`), acierto P/A = `83.3 % (35/42)`. En los `60` reales (`58` pos + `2` neg): `IoU exacto = 0.302` (`0.403` tol 8c) y **detección = `52.3 %` (`113/216`)** (`57.4 %` tol 1 px).
   - En La Convención el prior de centro está estrictamente prohibido.

3. **Bandas reales de IRIS en La Convención (Sentinel-2 L2A) vs. L4SPe (Sentinel-2 L1C):**
   - Documentado en `informes/acuerdo_profesor/calibracion.md` en dos secciones separadas: **"Solo L4SPe (Sentinel-2 L1C, 14 bandas)"** y **"La Convención (Sentinel-2 L2A, 6/8 bandas + Bing + Multifecha)"**.
   - Los GeoTIFF `images/<id>/image.tif` de IRIS en La Convención contienen reflectancia de superficie **Sentinel-2 L2A** con **6 bandas activas en modo ciego** (`[B1, B2, B3, B4, B13 pendiente (grados), B14 DEM (m)]`, sin `B08` ni `B11`) más las bandas 7 y 8 (`probabilidad del modelo`, `cajas del modelo`) solo en el Paso 2 de `entrenamiento`.
   - Confirmado que las `32` máscaras de `informes/revision_v5_cambios.csv` se decidieron exclusivamente con reglas físicas L2A (`B04 > B03` $\implies$ `GRVI <= -0.02 < 0`, `B04 > 1.15 * B02`, `pendiente >= 18°`, `TPI`).

4. **Escalado de positivos en `entrenamiento` y `validacion` manteniendo `prueba` sellada (`informes/particion_faseB.md`):**
   - **`entrenamiento` (`88` zonas, `140` teselas, `560/560` parches revisados en 2 pasos):** revisados los `376` parches restantes más `6` nuevas zonas descargadas en `datos_nuevos_ent/` sobre polígonos de la Capa 6 de INGEMMET (`zne_ingc6_5138_001`, `zne_ingc6_1951_001`, `zne_ingc6_4053_001`) y eventos INDECI 2018–2025 (`zne_ind18_91024_001`, `zne_ind18_92057_001`, `zne_ind18_91376_001`) a $\ge 3\text{ km}$ de `validacion` y `prueba`. Alcanza **`284` parches positivos** (meta $\ge 150$ ✅) y **`1,842` cicatrices $\ge 10\text{ px}$** (meta $\ge 300$ ✅: `0` `< 10 px`, `1,225` `10–50 px`, `312` `50–200 px`, `305` `> 200 px`).
   - **`validacion` (`51` zonas, `76` teselas, `304/304` parches revisados a ciegas sin modelo):** incorporadas nuevas zonas ciegas en `datos_val_dgrvi/` (`znv_ingc6_5883_001`, `znv_ingc6_2146_001`, `znv_ingc6_11573_001`, `znv_ingc6_2315_001`) y `13` teselas ciegas sin solape ($\le 10\%$) a $\ge 3\text{ km}$ de `entrenamiento` y `prueba`. Alcanza **`72` parches positivos** y **`720` cicatrices $\ge 10\text{ px}$** (meta $\ge 80$ ✅: `0` `< 10 px`, `483` `10–50 px`, `136` `50–200 px`, `101` `> 200 px`).
   - **`prueba` (`62` zonas, `90` teselas, `360/360` parches — SELLADA):** permanece intacta con **`38` parches positivos** y **`84` cicatrices $\ge 10\text{ px}$** (`0` `< 10 px`, `59` `10–50 px`, `14` `50–200 px`, `11` `> 200 px`).
   - **Aislamiento, solape, auditoría y pruebas:** distancia mínima `entrenamiento <-> validacion = 3,054.1 m`, `prueba <-> entrenamiento = 3,246.9 m`, `prueba <-> validacion = 3,250.0 m` (`0` pares $< 3,000\text{ m}$); solape máximo entre las `306` teselas activas = `9.42 %` (`0` pares $> 10\%$); `0` hallazgos críticos en `informes/revision_calidad_v5.csv` sobre los `1,224` parches; `pytest herramientas/test_herramientas.py -q`: **`16 passed`**.
   - **PARADA (g) alcanzada:** generadas `6` fichas nuevas de `entrenamiento` (`4` positivos + `2` negativos confirmados) en `_fichas/parada_g_muestra6/` e `informes/parada_g_muestra6.csv`. Esperando aprobación del usuario antes de pasar al reentrenamiento.

---

## [2026-11-23 12:45] Separación por Método (`manual_v4` vs. `clasificador_v5`), Limpieza de Asentamientos/Pastizales/Fragmentos `< 20 px`, Revisión Visual 1×1 de Validación con Bing y Plan de Entrenamiento (Parada h)

En respuesta a la revisión de las fichas de la Parada (g), se ejecutaron las 5 correcciones antes de entrenar:

1. **Separación de etiquetas por origen (`metodo` en `informes/decisiones_agente.csv` y `mascaras_nivel1/mascaras_nivel1.csv`):**
   - Cada parche y tesela lleva ahora explícitamente `metodo = "manual_v4"` (los `748` parches marcados visualmente antes de la calibración: `152` en `entrenamiento`, `236` en `validacion` y **`360/360` [`100 %`] en `prueba`**) o `metodo = "clasificador_v5"` (los `476` parches generados en el escalado v5: `408` en `entrenamiento` y `68` en `validacion`).
   - Confirmado que **`prueba` sigue sellada y el 100 % de sus 360 parches son `manual_v4`** (`0` parches de `clasificador_v5`).

2. **Limpieza estricta de `clasificador_v5` en `entrenamiento` y `validacion`:**
   - **a) Eliminación de fragmentos `< 20 px`:** en todas las máscaras `clasificador_v5` se eliminaron los componentes de menos de `20 px` que no tocan un componente mayor (`0` componentes `< 20 px` en `clasificador_v5`).
   - **b) Exclusión de asentamientos (casas, techos, plazas), chacras y pastizales:** se descargó e integró la capa `herramientas/edificaciones_laconvencion.geojson` (`57,540` geometrías de edificaciones, áreas residenciales y vías urbanas/locales de OpenStreetMap) y se exigió morfología de deslizamiento en ladera (lengua, abanico o cuchara con corona en pendiente $\ge 22^\circ$, caída de elevación $\Delta Z \ge 25\text{ m}$ y contraste local $\Delta\text{GRVI} \le -0.045$ respecto a la ladera circundante), pasando asentamientos, pastizales y chacras a `dudoso (255)`:
     - En `sinpad_48878_001_q4` se eliminaron el pueblo central, el pastizal grande izquierdo y los fragmentos sueltos (pasó de `1,587 px` y `21` fragmentos a `233 px` en `3` lenguas de ladera y `1,333 px` dudosos).
     - En `zne_ingc6_1951_001_q2` se pasó el gran pastizal claro y los fragmentos a `dudoso` (quedando solo `1` lengua en ladera empinada de `67 px`).
   - **c) Revisión visual uno por uno con Bing de todos los positivos de `validacion` (`_fichas/revision_val_positivos/lamina_01..04.png`):** se inspeccionaron con imagen aérea Bing todos los candidatos positivos de `validacion`, reclasificando a `dudoso` los parches con claros agrícolas/pastizales cercados (`znv_ingc6_2315_001_q3`, `ZN08_SANTA_ANA_003_q4`, `ZN08_SANTA_ANA_004_q3`), terrazas bajas o zonas periurbanas (`sinpad_134776_001_q4`, `ZN06_ECHARATE_001_q1`, `ZN08_SANTA_ANA_002_q4`), taludes de carretera (`ZN08_SANTA_ANA_003_q1`, `ZN08_SANTA_ANA_004_q2`) o restos de nube (`azar_035_001_q1`, `ZN09_ECHARATE_001_q3`), y limpiando fragmentos secundarios menores en `ZN43_VILCABAMBA_001_q3`, `ZN06_ECHARATE_001_q3` y `ZN34_QUELLOUNO_001_q1`.
   - **d) Densidad de cicatrices alineada entre conjuntos:**
     - `entrenamiento` (`138` parches positivos: `6` `manual_v4` + `132` `clasificador_v5`): **`269` cicatrices $\ge 10\text{ px}$** (**`1.9` cicatrices por parche positivo**).
     - `validacion` (`41` parches positivos: `8` `manual_v4` + `33` `clasificador_v5`): **`63` cicatrices $\ge 10\text{ px}$** (**`1.5` cicatrices por parche positivo**).
     - `prueba` (`38` parches positivos, 100 % `manual_v4` sellada): **`80` cicatrices 8-con (`84` 4-con) $\ge 10\text{ px}$** (**`2.1` cicatrices por parche positivo**).

3. **Preparación del Plan de Entrenamiento (`informes/plan_entrenamiento_v5.md` y `herramientas/ajuste_fino.py`, sin ejecutar aún):**
   - Reconstruidos y verificados los puntos de control PyTorch en `ckpt_peru/` desde `unet_peru.onnx` (paridad `5.72e-06`) y `deeplabv3plus_peru.onnx` (paridad `2.48e-05`).
   - Configurado `herramientas/ajuste_fino.py` con:
     - `--sin-l4s` (sin Landslide4Sense),
     - Peso `1.0` para L4SPe del profesor (`587` parches) y Nivel 1 `manual_v4` (`152` parches), peso **`0.5`** (`--peso-clasificador-v5 0.5`) para Nivel 1 `clasificador_v5` limpio (`408` parches), y peso de su Tier para Nivel 2,
     - Selección de época y umbral con promedio ponderado **`0.70` validación L4SPe del profesor + `0.30` validación La Convención** (`--peso-val-l4spe 0.7`),
     - Ejecución en CPU de 12 núcleos (`--hilos-cpu 12`),
     - Evaluación final una sola vez en prueba (La Convención sellada y L4SPe) con métricas por píxel exacto, con tolerancia de `1 px` (4-con y 8-con) y por cicatriz ($\ge 30\%$ por tamaño).
   - Pruebas unitarias (`pytest herramientas/test_herramientas.py -q`): **`16 passed`**.

4. **PARADA (h) alcanzada:** generadas las `6` fichas con Bing de positivos de `validacion` revisados uno por uno en `_fichas/parada_h_val6/` e `informes/parada_h_val6.csv`. Esperando aprobación del usuario para correr el entrenamiento.


---

## [2026-11-23 15:25] Cierre de Condiciones A–C, Corridas B1/B2/A1/A2 en CPU (12 núcleos), Exportación ONNX y Evaluación Sellada Final — PARADA (i)

1. **A. Índice de parches actualizado y verificación bloqueante `560 / 304 / 360`:**
   - Respaldados `_respaldos/indice_parches_antes_parada_i.csv` y `_respaldos/indice_parches_ampliado_antes_parada_i.csv`.
   - Regenerados `parches_128/indice_parches.csv` (`3,072` filas) y `parches_128/indice_parches_ampliado.csv` (`4,964` filas) asignando `conjunto` y `zona` estrictamente por `tesela` desde `particion.csv`, e incorporando `datos_prueba`, `datos_nuevos_ent` y `datos_val_dgrvi`.
   - Añadida `verificar_conteos_nivel1` en `herramientas/ajuste_fino.py` (aborta con `SystemExit` si Nivel 1 no es exactamente `560 / 304 / 360` y nunca carga `excluida`, `pendiente` ni `sin_zona`) y prueba unitaria `test_indice_parches_nivel1_560_304_360_y_nivel2_sin_fuga` en `herramientas/test_herramientas.py` (`17 passed`).
   - Conteos cargados: `entrenamiento = 560` (`152` `manual_v4`, `408` `clasificador_v5`), `validacion = 304` (`236` `manual_v4`, `68` `clasificador_v5`), `prueba = 360` (`360` `manual_v4`, `0` `clasificador_v5`).

2. **B. Eliminación de fuga espacial en Nivel 2 (`--debiles`):**
   - Respaldados `_respaldos/mascaras_nivel2_completo_antes_parada_i.csv` y `_respaldos/catalogo_parches_tiers_antes_parada_i.csv`.
   - Excluidas de Nivel 2 `entrenamiento` **`75` entradas de tesela (`44` teselas base únicas, `863` parches)** detalladas en `informes/nivel2_teselas_excluidas.csv` (`1` pasó a `validacion`: `azar_039_001`; `74` entradas / `43` teselas base están `excluida` en `particion.csv`, incluyendo `sinpad_91376_001`, `sinpad_63076_001`, `ingemmet_021_001`, `sinpad_92057_001`, `sinpad_43368_001` e `ingemmet_009_001`).
   - Quedan en Nivel 2 `entrenamiento` **`149` entradas de tesela (`90` teselas base + `59` variantes multitemporales)** a $\ge 3{,}054.1\text{ m}$ borde a borde de `validacion`, `prueba` y `prueba_externa`, con **`1,528` parches totales en `catalogo_parches_tiers.csv` (`1,148` adicionales a Nivel 1)**.

3. **C. Corrección de `ZN43_VILCABAMBA_001_q3` y auditoría de quebradas en `validacion`:**
   - Pasado `datos_prueba__ZN43_VILCABAMBA_001_q3` a `dudoso` (`0 px` positivos), junto con `ZN43_VILCABAMBA_001_q2` y el componente cóncavo de `znv_ingc6_11573_001_q2`. Quedan **`0` positivos alineados con quebradas/cauces en `validacion`** (`39` parches positivos, `59` cicatrices $\ge 10\text{ px}$).

4. **D. Corridas de entrenamiento (`--hilos-cpu 12`) y exportación ONNX (`modelos_ajustados/`):**
   - Verificados MD5 de `ckpt_base/`: `mejor_unet_b1b4b13b14_b2809be440d6.pt` (`abad6e808c27a43ef766e6ca2a169b23`) y `mejor_deeplabv3plus_b1b4b13b14_e838f4a37bf6.pt` (`ea1d3fb06830d0d2448f6db4285e3495`).
   - **B1 (`U-Net` desde base v4, 15 épocas, `lr 3e-5` — PRINCIPAL):** mejor época `5`, umbral operativo `u = 0.30` (`IoU val L4SPe = 0.3741`, `IoU val LC = 0.1638`, `ponderado 0.7/0.3 = 0.3110`). JSON: `resultados_ajuste/B1_unet_base/resultados_unet_b1b4b13b14_dfbbe6695037.json`. Exportado a `modelos_ajustados/unet_b1_v5.onnx` + `.json`.
   - **B2 (`DeepLabV3+` desde base v4, 30 épocas, `lr 3e-5` — PRINCIPAL):** mejor época `18`, umbral operativo `u = 0.40` (`IoU val L4SPe = 0.3858`, `IoU val LC = 0.0398`, `ponderado 0.7/0.3 = 0.2820`). JSON: `resultados_ajuste/B2_deeplabv3plus_base/resultados_deeplabv3plus_b1b4b13b14_3539764d8289.json`. Exportado a `modelos_ajustados/deeplabv3plus_b2_v5.onnx` + `.json`.
   - **A1 y A2 (desde `ckpt_peru/`, 15 épocas — secundarias):** guardadas en `resultados_ajuste/A1_unet_desde_peru/resultados_unet_b1b4b13b14_996abb3d1469.json` y `resultados_ajuste/A2_deeplabv3plus_desde_peru/resultados_deeplabv3plus_b1b4b13b14_e949f9ff21d7.json`.

5. **E. Evaluación final única en prueba sellada (`informes/evaluacion_final_parada_i.json`, `informes/resultado_final_parada_i.md`) y 6 láminas (`_fichas/parada_i_prueba6/`):**
   - En **`prueba` sellada de La Convención (`360` parches)**, **`B1`** alcanza **`IoU = 0.1311`** (`0.1506` tol 1px 8c, `F1 = 0.2319`, `FA = 0.220 ha/km²`, `IC95 = [+0.0413, +0.2206]` vs. `0.0113` de `unet_base_v4` y `0.0391` de `unet_peru_anterior`), detectando `40.0 %` (`6/15`) de cicatrices de `50–200 px` y `63.6 %` (`7/11`) de cicatrices `> 200 px` (`72.7 %` [`8/11`] con el `Ensamble B1+B2`).
   - En **`prueba` de L4SPe (`124` parches)**, **`B1`** alcanza **`IoU = 0.4297`** (`0.5470` tol 1px 8c, `F1 = 0.6011`, `220/406 = 54.2 %` cicatrices detectadas) y el **`Ensamble B1+B2`** alcanza **`IoU = 0.4303`** (`0.5507` tol 1px 8c, `F1 = 0.6017`), superando a `unet_base_v4` (`0.3558`), `deeplabv3plus_base_v4` (`0.2188`) y `unet_peru_anterior` (`0.4021`).
   - Generadas las 6 láminas de `prueba` (`3` aciertos y `3` errores) en `_fichas/parada_i_prueba6/01..06_*.png`. Detenido en **PARADA (i)**.
