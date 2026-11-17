# Detección y Segmentación de Deslizamientos en La Convención (Cusco, Perú)

Repositorio de investigación para la detección semiautomática de cicatrices de deslizamientos mediante imágenes satelitales **Sentinel-2 L2A**, modelos de elevación **ALOS PALSAR / NASADEM** y redes neuronales de segmentación (**U-Net ResNet-34** y **DeepLabV3+**).

---

## Estructura del repositorio

```text
Tesis-La-Convencion-PE/
├── docs/                           # Documentación y material visual de referencia
│   ├── GUIA_IRIS_128.md            # Manual operativo de etiquetado a 128x128 px
│   └── img/                        # Figuras de auditoría y demostración en IRIS
├── herramientas/                   # Scripts de procesamiento, entrenamiento y pruebas
│   ├── ajuste_fino.py              # Entrenamiento con pesos estratificados y geomorfometría
│   ├── aumentar_d4_tiers.py        # Aumento geofísico invariante (grupo diedral D4)
│   ├── clasificar_tiers_y_expandir.py
│   ├── exportar_poligonos_qgis.py  # Vectorización suavizada (Chaikin) y estilo .qml
│   ├── minar_hard_negatives.py     # Minería activa de negativos difíciles
│   ├── nutrir_mascaras.py          # Extracción multitemporal y geológica
│   ├── empaquetar_para_colab.py    # Empaquetado para ejecución en GPU (Colab)
│   ├── Ajuste_Fino_LaConvencion.ipynb
│   └── test_herramientas.py        # Suite de pruebas unitarias e integrales (9/9)
├── nucleo/                         # Módulo base de configuración, modelos y recetas
│   ├── config.py
│   ├── modelos.py
│   └── recetas.py                  # Normalización espectral y cálculo de CURVATURA / TPI
├── resultados/                     # Salidas cartográficas y catálogos consolidados
│   ├── cicatrices_consolidadas_qgis.geojson
│   ├── cicatrices_consolidadas_qgis.qml
│   ├── cicatrices_consolidadas_resumen.csv
│   ├── catalogo_parches_tiers.csv
│   ├── mascaras_nivel2_completo.csv
│   └── particion.csv
├── escala_laconvencion.json        # Constantes de normalización espectral provincial
├── iniciar_iris_128.bat            # Lanzador local de la interfaz de etiquetado IRIS
└── README.md
```

---

## Metodología y especificaciones técnicas

| Componente | Especificación |
| :--- | :--- |
| **Fuente satelital** | Sentinel-2 L2A (temporada seca junio–septiembre), corrección BOA |
| **Relieve** | DEM ALOS PALSAR / NASADEM remuestreado a malla UTM 18S (10 m/px) |
| **Parches de entrada** | Ventanas de $128 \times 128\text{ px}$ ($1{,}28 \times 1{,}28\text{ km}$) |
| **Canales base (6)** | `B01`, `B02`, `B03`, `B04`, `pendiente` (grados) y `elevacion` (m) |
| **Variables derivadas** | `GRVI`, `ExG`, `CURVATURA` laplaciana del DEM y `TPI` (ventana $5 \times 5$) |
| **Arquitecturas** | U-Net ResNet-34 v4 (IoU L4S: $0{,}4602$) y DeepLabV3+ ResNet-34 (ensamble: $0{,}4656$) |

---

## Etiquetado en IRIS ($128 \times 128$)

Para la validación y refinamiento humano de máscaras se utiliza la interfaz **IRIS** (*Intelligent Remote Sensing Image Segmentation*) configurada a $128 \times 128\text{ px}$ (guía detallada en [`docs/GUIA_IRIS_128.md`](docs/GUIA_IRIS_128.md)).

### Instalación y ejecución de IRIS

Requiere **Python 3.10+** y **Node.js (npm)**:

```powershell
# 1. Clonar el código fuente de IRIS en iris_app/
git clone https://github.com/gt-csse/iris.git iris_app

# 2. Instalar dependencias de Python y el paquete IRIS
pip install -e iris_app
pip install rasterio scikit-learn matplotlib pillow scipy

# 3. Compilar la interfaz web (React / Vite)
npm --prefix iris_app install
npm --prefix iris_app run build
```

Para exportar los parches, iniciar el servidor local e importar las máscaras validadas:

```powershell
# Exportar proyecto a iris_128/
python herramientas/iris_128.py exportar --salida iris_128

# Iniciar el servidor web local (o hacer doble clic en iniciar_iris_128.bat)
python herramientas/iris_128.py abrir --iris iris_128
# Abrir en el navegador: http://127.0.0.1:5000/segmentation/ (usuario: admin | clave: laconvencion)

# Importar las máscaras guardadas a Nivel 1
python herramientas/iris_128.py importar --iris iris_128 --salida mascaras_nivel1
```

![Demostración de etiquetado en IRIS 128](docs/img/video_tutorial_iris.gif)

![Mapa de interfaz IRIS 128](docs/img/mapa_interfaz_iris.png)

---

## Auditoría de calidad y resultados consolidados

Las máscaras multitemporales (2018–2025, eventos INDECI y cartografía geológica de INGEMMET) se filtran mediante compacidad morfológica ($\text{área}/\text{perímetro} \ge 1{,}38\text{ px}$) y se estratifican en niveles de confianza (**Oro**, **Plata**, **Bronce** y **Hard Negatives**):

![Auditoría de calidad y variables geomorfométricas](docs/img/auditoria_calidad_tesis.png)

### Resumen cuantitativo del conjunto de entrenamiento

| Etapa / Artefacto | Resultado | Archivo en repositorio |
| :--- | :--- | :--- |
| **Teselas procesadas** | 353 teselas (96 con cicatrices propuestas, 783,3 ha) | [`resultados/mascaras_nivel2_completo.csv`](resultados/mascaras_nivel2_completo.csv) |
| **Ventana deslizante + Tiers** | 310 parches centrados nuevos (196 Tier Oro, 125 Tier Plata) | [`resultados/catalogo_parches_tiers.csv`](resultados/catalogo_parches_tiers.csv) |
| **Aumento geofísico $D_4$** | 2,003 parches positivos vs 577 negativos (total: 2,580 parches) | [`herramientas/aumentar_d4_tiers.py`](herramientas/aumentar_d4_tiers.py) |
| **Minería de negativos difíciles** | 443 negativos difíciles identificados (peso de pérdida `0.8`) | [`herramientas/minar_hard_negatives.py`](herramientas/minar_hard_negatives.py) |
| **Polígonos suavizados (QGIS)** | 499 polígonos (409 Oro, 58 Plata, 32 Bronce; 609,1 ha) | [`resultados/cicatrices_consolidadas_qgis.geojson`](resultados/cicatrices_consolidadas_qgis.geojson) |
| **Control ciego inter-anotador** | 3 teselas reservadas intactas (`15525`, `48878`, `557`) | [`herramientas/acuerdo_anotadores.py`](herramientas/acuerdo_anotadores.py) |

---

## Verificación automática

```powershell
python herramientas/test_herramientas.py
```

Todas las pruebas unitarias e integrales (partición espacial sin fuga, regla de nubes, métricas de acuerdo inter-anotador, cálculo geomorfométrico de `CURVATURA`/`TPI`, pesos por Tier y suavizado cartográfico de Chaikin) pasan al 100% (`9/9 OK`).
