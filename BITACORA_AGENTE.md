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
