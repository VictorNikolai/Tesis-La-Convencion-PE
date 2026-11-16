# Segmentación de deslizamientos en La Convención (Cusco) — Conjunto local y ajuste fino

Repositorio local de datos, herramientas de preparación, control de calidad y ajuste fino (*fine-tuning*) del sistema de detección de deslizamientos mediante imágenes Sentinel-2 L2A y modelos de segmentación profunda (**U-Net v4** y **DeepLabV3+ v1**).

---

## 1. Arquitectura real del artefacto (alineada con producción)

| Componente | Especificación en producción y ajuste fino |
|---|---|
| **Fuente satelital** | Sentinel-2 L2A (Planetary Computer STAC), mediana sin nubes en temporada seca (junio–septiembre), corrección de *offset* BOA ($-1000$ desde *baseline* 04.00) |
| **Fuente de relieve** | DEM de 30 m (NASADEM / Copernicus GLO-30) con ventana temporal propia, remuestreado a la malla UTM 18S de 10 m |
| **Tamaño de entrada** | Ventanas de $128 \times 128\text{ px}$ a $10\text{ m/px}$ ($1{,}28 \times 1{,}28\text{ km}$), inferencia con solape del $50\%$ |
| **Canales (6)** | `B01`, `B02`, `B03`, `B04` (escalados por la media provincial `escala_laconvencion.json`), `B13` pendiente ($\text{grados} / 20{,}292$) y `B14` elevación ($\text{metros} / 772{,}82$) |
| **Índices internos (v4)** | `GRVI` y `ExG` calculados dentro del grafo ONNX a partir de `B03`, `B04` y `B02` |
| **Modelos desplegados** | **U-Net · ResNet34 v4** (umbral $0{,}45$, IoU prueba L4S $0{,}4602$; ensamble mixto $0{,}4656$) y **DeepLabV3+ v1** |

---

## 2. Estructura de carpetas y los tres niveles de máscara

Solo el **Nivel 1** es verdad de terreno humana y es el único que se usa para medir el desempeño local (OE4).

| Nivel | Carpeta | Origen | Uso permitido |
|---|---|---|---|
| **Nivel 1** | `mascaras_nivel1/` | Polígonos dibujados y confirmados por personas en QGIS (`marcas.geojson` + `revisadas.csv` con `si`) o IRIS (`iris_128.py`) | **Verdad de referencia:** entrenamiento y medición oficial local |
| **Nivel 2** | `mascaras_provisionales/nivel2/` | Polígonos de campo de INGEMMET intersectados con suelo desnudo visible y revisados visualmente (`decisiones.csv`) | Opcional en entrenamiento con peso bajo (`--debiles --peso-debil 0.3`); **nunca para medir** |
| **Nivel 3** | `mascaras_provisionales/nivel3/` | Reglas automáticas sin supervisión (suelo desnudo en ladera y pérdida de vegetación entre fechas) | **Solo pista visual** para guiar el marcado; nunca para entrenar ni medir |

### Conjuntos de teselas ($256 \times 256\text{ px}$ a $10\text{ m}$) y parches ($128 \times 128\text{ px}$)
* `datos_laconvencion/` y `datos_laconvencion_2025/`: 66 teselas en torno a eventos SINPAD (2003–2017) y su par de 2025.
* `datos_indeci/` y `datos_indeci_antes/`: 64 teselas de eventos INDECI (2017–2021 y antiguos) con imagen previa al evento.
* `datos_ingemmet/` y `datos_ingemmet_antes/`: 52 teselas sobre polígonos de peligros geológicos de INGEMMET con imagen previa.
* `datos_azar/`: 60 teselas sorteadas al azar dentro del límite provincial (`limite_laconvencion.geojson`) a $> 6\text{ km}$ entre sí para medir falsas alarmas reales y aportar negativos confirmados.
* `parches_128/`: 1588 parches de $128 \times 128\text{ px}$ (6 canales) cortados sin redimensionar desde las 397 imágenes, indexados en `parches_128/indice_parches.csv`.

---

## 3. Flujo de trabajo reproducible (`herramientas/`)

### Paso 1. Partición espacial por zonas y corte de parches (ya ejecutado)
Agrupa en una misma zona todas las teselas cuyos centros distan menos de $3560\text{ m}$ en $x$ e $y$ (encadenadas), garantizando cero fuga espacial entre entrenamiento y validación:
```powershell
python herramientas/particion_por_zona.py
python herramientas/cortar_parches_128.py
```

### Paso 2. Marcado humano en QGIS (o en IRIS a $128 \times 128$)
* Seguir las reglas de [`datos_laconvencion/GUIA_marcado.md`](datos_laconvencion/GUIA_marcado.md) sobre `datos_laconvencion/proyecto_marcado.qgz`.
* Priorizar las **zonas de validación con cicatrices** (`Z04_SANTA_ANA`, `Z08_QUELLOUNO`, `Z09_INKAWASI` en [`particion.md`](particion.md)) y las cicatrices claras de Mesacancha (`005, 009, 040`), Sillapata (`002, 003, 036, 037`) y `046`.
* Exportar la capa `marcas` en `EPSG:4326` como `datos_laconvencion/marcas.geojson` y anotar `si` en la columna `revisada` del `revisadas.csv` correspondiente.
* *(Alternativa IRIS $128 \times 128$)*:
  ```powershell
  python herramientas/iris_128.py exportar --salida iris_128
  iris label iris_128/deslizamientos_128.json
  python herramientas/iris_128.py importar --iris iris_128 --salida mascaras_nivel1
  ```

### Paso 3. Acuerdo entre anotadores (control de calidad)
Con las 3 teselas marcadas de forma independiente por dos personas (`sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`):
```powershell
python herramientas/acuerdo_anotadores.py marcas_A.geojson marcas_B.geojson --teselas sinpad_15525_001,sinpad_48878_001,sinpad_557_001
```

### Paso 4. Generación de máscaras de Nivel 1 y paquete para Colab
```powershell
python herramientas/mascaras_desde_marcas.py
python herramientas/empaquetar_para_colab.py
```
Esto genera `datos_locales_ajuste.zip`. Subirlo a Google Drive y ejecutar [`herramientas/Ajuste_Fino_LaConvencion.ipynb`](herramientas/Ajuste_Fino_LaConvencion.ipynb) en Google Colab con GPU T4.

### Paso 5. Pruebas automáticas de verificación
Para ejecutar la batería de pruebas unitarias e integrales (partición sin fuga espacial, máscaras de Nivel 1, regla de nubes sobre suelo brillante, exportador IRIS 128 y componentes de ajuste fino + comparador sin aprendizaje):
```powershell
python herramientas/test_herramientas.py
```
