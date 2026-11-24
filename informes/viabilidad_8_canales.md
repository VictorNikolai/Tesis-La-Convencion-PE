# Estudio de Viabilidad — Modelo de 8 Canales (`B02, B03, B04, B08, B11, B12, pendiente, elevación`)

> [!NOTE]
> **Alcance (Tarea 7):** Este documento contiene únicamente el análisis técnico y las mediciones de 1 época de prueba en CPU local y descarga de 3 teselas en Planetary Computer. **No se ha reentrenado ningún modelo de 8 canales**; la decisión de ejecutarlo queda pendiente de aprobación.

---

## a) ¿Landslide4Sense trae `B08`, `B11` y `B12`?

**Sí.** El conjunto oficial `Landslide4Sense` (`ibm-nasa-geospatial/Landslide4sense`, `3 799` parches de entrenamiento, `245` de validación y `800` de prueba a `128×128 px`) contiene **`14` canales** en cada archivo `.h5` (verificado en `herramientas/entrenar.py` y `herramientas/exportar_onnx.py`, lista `NOMBRES_14`):

| Índice L4S | Canal | Resolución nativa Sentinel-2 | ¿Disponible en L4S? |
|---:|---|---|:-:|
| 0–3 | `B01, B02, B03, B04` | 60 m / 10 m | Sí (usados hoy) |
| 4–6 | `B05, B06, B07` | 20 m | Sí |
| **7** | **`B08` (NIR)** | **10 m** | **Sí** |
| 8–9 | `B09, B10` | 60 m | Sí |
| **10–11** | **`B11` (SWIR-1), `B12` (SWIR-2)** | **20 m** | **Sí** |
| 12–13 | `B13` (pendiente), `B14` (elevación DEM ALOS) | 12.5 m | Sí (usados hoy) |

Por lo tanto, el **modelo base en Landslide4Sense sí puede entrenarse con los 8 canales** (`B02, B03, B04, B08, B11, B12, pendiente, elevacion`), obteniendo además `NDVI` (`B08/B04`) y `NBR / NDMI` (`B08/B11/B12`) reales para separar suelo desnudo de vegetación seca y humedad.

---

## b) ¿`l4spe_profesor.npz` trae esas bandas o solo 6 canales?

**Solo trae 6 canales.** Verificado abriendo directamente `l4spe_profesor.npz`:
- `x.shape = (838, 6, 128, 128)` (`dtype=float32`)
- `escala_json = {"B01": 938.122, "B02": 698.203, "B03": 620.596, "B04": 399.646}` (más `pendiente` y `elevacion`).
- **Implicación crítica:** `l4spe_profesor.npz` **NO contiene `B08`, `B11` ni `B12`**. Para usar L4SPe en el ajuste fino de un modelo de 8 canales habría que:
  1. Re-descargar las bandas `B08, B11, B12` desde Sentinel-2 L1C/L2A para las `838` coordenadas `(lat, lon)` del archivo `l4spe_profesor.npz` (~`1–1.5 h` adicionales y con riesgo de discrepancia de fecha exacta frente a las escenas originales del profesor), **o**
  2. Hacer el ajuste fino de 8 canales **sin L4SPe** (solo con La Convención + ancla de Landslide4Sense), perdiendo el ancla manual humana del profesor (`587` parches de entrenamiento y `127` de validación) que ha sido clave para estabilizar `U-Net v6`.

---

## c) ¿Cuánto tomaría descargar `B08`, `B11` y `B12` para las 451 teselas de La Convención?

Se midió la descarga real desde Planetary Computer (`sentinel-2-l2a`, mediana de hasta 10 escenas secas `2023–2025` con `eo:cloud_cover < 35 %`, remuestreadas a `10 m` en `EPSG:32718`, bandas `B08, B11, B12, SCL`) sobre 3 teselas reales (`ZN59_INKAWASI_001`, `ZN64_QUELLOUNO_001`, `ZN57_VILCABAMBA_001`):

- **Tiempos medidos:** `8.07 s`, `7.76 s`, `4.69 s` → **media = `6.84 s / tesela`**.
- **Extrapolación para `451` teselas (secuencial):** **`~51.4 minutos`** (~`15–20 minutos` con 4 hilos en paralelo).
- **Espacio en disco estimado:** `451` teselas × `3` bandas × `256×256` `float32` comprimido (`deflate`) ≈ **`~120–180 MB`** (manejable).

---

## d) ¿Cuánto tomaría en esta CPU entrenar U-Net de 8 canales (base L4S + ajuste fino peruano)?

Se midió un ensayo real en esta CPU (`12` hilos PyTorch, `lote = 16`, `U-Net ResNet-34`, `in_channels = 8`, `128×128 px`, AdamW + forward/backward completo):

- **Velocidad medida:** **`1.061 s / paso`** (20 pasos tras calentamiento).
- **1 época completa de Landslide4Sense (`3 799` parches = `238` pasos):** **`4.21 min`** (más ~`0.4 min` de validación en `245` parches ≈ **`4.6 min / época`**).
- **Entrenamiento base completo en Landslide4Sense (`50` épocas):** **`3.51 h`** de entrenamiento puro (**`~3.8 – 4.2 horas`** con validación por época).
- **Ajuste fino peruano (`25` épocas × `45` pasos/época):** **`19.9 min`** de entrenamiento puro (**`~30 – 35 minutos`** por semilla con validación; **`~1.5 – 1.8 horas`** para 3 semillas).
- **Nota adicional:** si además se necesitara descargar Landslide4Sense completo (`~3 GB` en `.h5`) porque no está en disco local en este equipo, sumar ~`15–30 min` de descarga y preparación.

---

## e) ¿Qué cambiaría en el servidor (`Proyecto_Tesis_Deslizamiento`, Render gratuito `< 450 MB`)?

1. **Bandas a descargar desde Planetary Computer (`nucleo/datos.py`):**
   - Actualmente descarga `4` bandas ópticas (`B01, B02, B03, B04`) + `SCL` + DEM (`cop-dem-glo-30`).
   - Con 8 canales pasaría a descargar `6` bandas ópticas (`B02, B03, B04, B08` a `10 m` + `B11, B12` a `20 m` remuestreadas bilinealmente a `10 m`) + `SCL` + DEM.
   - **Impacto en tiempo por consulta (`3 × 3 km`):** `+25 %` a `+40 %` en descarga STAC (~`+2 a +4 s` por consulta); la inferencia ONNX en CPU apenas cambia (`< +1 ms/parche`, porque solo cambia la primera capa `conv1` de 6 a 8 canales).
2. **Memoria RAM en pico (`Render` límite `512 MB`, objetivo `< 450 MB`):**
   - El tamaño del archivo `.onnx` de `U-Net ResNet-34` prácticamente **no cambia** (`97.8 MB` vs `97.8 MB`, diferencia de `< 50 KB` en `conv1`).
   - El tensor de entrada en memoria para una escena de `3 × 3 km` (`300 × 300 px`) sube de `6 × 300 × 300 × 4 B = 2.16 MB` a `8 × 300 × 300 × 4 B = 2.88 MB` (`+0.72 MB`).
   - Sin embargo, durante la lectura `stackstac` de `6` bandas en vez de `4` sobre `10–12` escenas, el pico temporal de RAM sube ~`+35 a +60 MB`, por lo que conviene leer banda por banda o limitar escenas a `6–8` para mantenerse holgado bajo `450 MB`.
3. **Receta y compatibilidad (`nucleo/recetas.py`):**
   - El servidor ya soporta recetas declarativas por canal; habría que añadir `B08, B11, B12` a `ESCALA_L4S` (calculando la media DN de La Convención para `B08, B11, B12`) y mantener retrocompatibilidad con los modelos de 6 canales existentes.

---

## Conclusión de Viabilidad

| Aspecto | Estimación | Riesgo principal |
|---|---|---|
| Descarga `B08, B11, B12` (`451` teselas LC) | `~51 min` (`~15–20 min` paralelo) | Bajo |
| Entrenamiento base L4S (50 ep, CPU 12 hilos) | `~3.8 – 4.2 h` | Medio (requiere L4S local o Colab GPU ~15 min) |
| Ajuste fino peruano (3 semillas × 25 ep) | `~1.5 – 1.8 h` | **Alto sin `B08/B11/B12` en `l4spe_profesor.npz`** (se pierde el ancla del profesor o requiere re-descargar `838` parches L1C/L2A con posible desfase de fechas) |
| Cambios en servidor / Render (`< 450 MB`) | `+2–4 s/consulta`, `+35–60 MB` pico STAC | Bajo si se lee por banda |

**Recomendación técnica:** dado que **`l4spe_profesor.npz` solo tiene 6 canales** y que el beneficio principal en `prueba2` (`−78 %` de falsas alarmas) ya se logró con `unet_v6` + `wc_dem_3200_agua_urb`, el experimento de 8 canales solo conviene ejecutarlo si dispones de GPU (Colab) y decides aceptar o bien entrenar sin L4SPe o re-descargar `B08/B11/B12` para las fechas exactas de L4SPe. Queda detenido a tu decisión.
