# Datos propios de La Convención: de los puntos de SINPAD al entrenamiento

Qué necesitas al empezar: solo el archivo de puntos de SINPAD-INDECI (por ejemplo
`Deslizamiento0327.csv`). Las imágenes las baja el script, gratis y sin cuenta, del mismo
catálogo de Sentinel-2 que usa el sistema en Render.

## Paso 1. Preparar la computadora (una vez)

Instala Python 3.12 (marca «Add python.exe to PATH»), Git y QGIS. En PowerShell:

```
git clone https://github.com/VictorNikolai/Proyecto_Tesis_Deslizamiento
cd Proyecto_Tesis_Deslizamiento
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copia la carpeta `herramientas` de este paquete dentro de `Proyecto_Tesis_Deslizamiento` y:

```
pip install -r herramientas\requirements-herramientas.txt
```

Copia también tu CSV de SINPAD en esa carpeta. Si PowerShell no deja activar el entorno:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Paso 2. Probar con dos puntos (unos 5 minutos)

```
python herramientas\descargar_desde_puntos.py Deslizamiento0327.csv --salida datos_laconvencion --desde-anio-evento 2016 --limite 2
```

Debe terminar con «2 puntos con imágenes».

## Paso 3. Bajar las imágenes de todos los puntos (2 a 4 horas)

```
python herramientas\descargar_desde_puntos.py Deslizamiento0327.csv --salida datos_laconvencion
```

- Si se corta, vuelve a correrlo: salta lo hecho.
- `datos_laconvencion\puntos_resumen.csv` dice, por punto, qué año usó y si quedó con nubes.

## Paso 4. Dejar las imágenes listas para QGIS

```
python herramientas\preparar_para_qgis.py datos_laconvencion
```

Crea `datos_laconvencion\qgis\` con la imagen a color y la pendiente de cada tesela,
`cajas_modelo.geojson` (lo que marca hoy el modelo, solo como guía) y `teselas.geojson`.
Crea también `datos_laconvencion\revisadas.csv`.

## Paso 5. Dibujar los deslizamientos en QGIS

1. Abre QGIS y arrastra: todas las `*_color.tif`, `teselas.geojson` y `cajas_modelo.geojson`.
   A las dos capas GeoJSON ponles «sin relleno» y borde de color (teselas en blanco, cajas en amarillo).
2. Crea la capa donde se dibuja (una sola para todo): Capa → Crear capa → Nueva capa GeoPackage.
   - Archivo: `marcas.gpkg`; tipo de geometría: Polígono; SRC: EPSG:4326.
   - Campo nuevo: nombre `clase`, tipo Texto.
3. Con la capa `marcas` seleccionada: Conmutar edición (lápiz) → Añadir polígono.
   Dibuja el borde de cada deslizamiento y en `clase` escribe `deslizamiento`.
   Si no se puede decidir, escribe `dudoso`.
4. Reglas para que sirva:
   - marca lo que se ve en la imagen, no un círculo en el punto de SINPAD;
   - revisa toda la tesela, no solo las cajas amarillas;
   - lo que no tiene polígono cuenta como «sin deslizamiento»: chacras, techos y arena de
     río no se dibujan;
   - si un punto antiguo no muestra cicatriz, no la inventes.
5. Al terminar una tesela, abre `revisadas.csv` (Excel) y pon `si` en la columna `revisada`.
   Una tesela sin «si» no se usa.
6. Guarda la edición. Para entregar: clic derecho en `marcas` → Exportar → Guardar objetos como…
   → formato GeoJSON → `marcas.geojson` → SRC EPSG:4326.
7. Que dos personas marquen las mismas 10 a 20 teselas (cada una en su propia capa), para
   medir cuánto coinciden.

## Paso 6. Convertir lo dibujado y medir el modelo en La Convención

```
python herramientas\importar_de_qgis.py datos_laconvencion marcas.geojson
python herramientas\importar_de_iris.py datos_laconvencion --umbral 0.45
```

El primero crea las máscaras de entrenamiento; el segundo deja en
`datos_laconvencion\resultados_iris\` la medición del sistema en la provincia
(precisión, recall, IoU y F1 en `metricas.json`).

## Paso 7. Entrenar

Con unas 50 teselas revisadas ya se puede medir; para ajustar el modelo hacen falta varios
cientos de parches con deslizamiento. Con la carpeta `datos_laconvencion` (images, masks y
revisadas.csv) se arma el entrenamiento con partición por zonas y mezcla con Landslide4Sense.
