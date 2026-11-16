# Guía completa de etiquetado en IRIS a $128 \times 128\text{ px}$ (La Convención)

Esta guía explica paso a paso cómo abrir y usar **IRIS** (`ESA-PhiLab / fork gt-csse`) directamente sobre los parches nativos de $128 \times 128\text{ px}$ ($10\text{ m/px}$, $1{,}28 \times 1{,}28\text{ km}$) que consume la red neuronal **U-Net v4** y **DeepLabV3+ v1**, cómo se interpreta el ejemplo práctico incluido en `iris_ejemplo_demo/`, y cómo convertir tus trazos a máscaras oficiales de **Nivel 1** para el ajuste fino en Google Colab.

---

## 1. Inicio rápido (en 1 clic o por terminal)

El software **IRIS** (`iris_app/`) y su interfaz React ya están instalados, compilados y verificados en tu computadora, y el proyecto de La Convención ya está exportado en `iris_128/` con **464 parches priorizados (116 teselas)**.

### Opción A — Con doble clic
Haz doble clic en el archivo de la raíz del proyecto:
- [`iniciar_iris_128.bat`](../iniciar_iris_128.bat)

### Opción B — Desde PowerShell
```powershell
python herramientas/iris_128.py abrir --iris iris_128
```

Luego abre en tu navegador:
- **URL:** `http://127.0.0.1:5000/segmentation/`
- **Usuario:** `admin`
- **Contraseña:** `laconvencion`

*(Si deseas probar primero sobre el proyecto de ejemplo de 4 parches sin tocar `iris_128/`, corre `python herramientas/iris_128.py abrir --iris iris_ejemplo_demo/proyecto_iris`).*

---

## 2. Orden inteligente de los parches en `iris_128/`

El proyecto está configurado con `shuffle_images: false` para que **los 4 cuadrantes de cada tesela (`q1`, `q2`, `q3`, `q4`) salgan siempre juntos y en orden de prioridad de la tesis**:

| Bloque | Orden en IRIS | Teselas incluidas | Propósito en la tesis |
|---|---|---|---|
| **Bloque 0** | Parches `1` a `12` (3 teselas) | `sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001` | **Acuerdo inter-anotador (OE2):** marcar por dos personas sin verse para medir IoU y $\kappa$ de Cohen |
| **Bloque 1** | Parches `13` a `48` (9 teselas) | Sillapata (`ingemmet_002, 003, 036, 037, 046`) y Mesacancha (`indeci_005, 009, 040, 046`) | **Cicatrices claras confirmadas:** aportan la mayor cantidad de píxeles positivos limpios |
| **Bloque 2** | Parches `49` a `132` (21 teselas) | Zonas `Z04_SANTA_ANA`, `Z08_QUELLOUNO`, `Z09_INKAWASI` | **Validación con cicatrices (OE4):** indispensables para medir IoU y F1 locales |
| **Bloque 3** | Parches `133` a `208` (19 teselas) | Resto de `ronda1` + 10 negativos difíciles (`13640, 7448, 56208, 9709, 61214, 96, 36869, 62691, 86528, 5559`) | **Entrenamiento prioritario y control de falsas alarmas** (playas de río, pasto seco, ciudad, caminos) |
| **Bloque 4** | Parches `209` a `464` (64 teselas) | `ronda2` y resto de validación | Ampliación del conjunto local |

Los cuadrantes de cada tesela de $256 \times 256\text{ px}$ se recorren siempre en este orden:
- `q1`: arriba-izquierda (Noroeste)
- `q2`: arriba-derecha (Noreste)
- `q3`: abajo-izquierda (Suroeste)
- `q4`: abajo-derecha (Sureste)

---

## 3. Las 3 clases de pintado y qué significa cada una

Al abrir un parche en IRIS, **toda la imagen empieza por defecto en la Clase 1 (`Sin deslizamiento`, valor interno `0`, transparente)**. Eso significa que **no necesitas pintar el bosque ni el fondo**: solo pintas donde hay cicatriz (`2`) o zona dudosa (`3`).

| Tecla | Nombre en IRIS | Color en pantalla | Valor en Nivel 1 (`.tif`) | Qué debes pintar con esta clase |
|---|---|---|---|---|
| **`1`** | **Sin deslizamiento** | Transparente (o cian suave donde pasas el pincel) | `0` (fondo confirmado) | Sirve para borrar o corregir si te saliste del borde, o para darle ejemplos de fondo al asistente IA (`A`). Todo lo que dejes sin pintar ya vale `0`. |
| **`2`** | **Deslizamiento** | **Rojo** | `1` (positivo) | **Cicatriz de deslizamiento o derrumbe que arranca en ladera** ($\ge 10^\circ\text{--}15^\circ$): mancha de suelo desnudo con forma de lengua, cuchara o abanico, desde la corona superior hasta el depósito visible. |
| **`3`** | **Dudoso** | **Amarillo / Ámbar** | `255` (ignorar) | Nubes y sus sombras, cauces de huaico/aluvión donde no se ve el arranque en la ladera, bruma espesa o parcelas donde no se puede decidir. No cuenta ni a favor ni en contra en el entrenamiento ni en las métricas. |

---

## 4. Vistas multibanda y atajos de teclado en IRIS

En pantalla verás **3 paneles simultáneos**. Presiona la tecla **`B`** para rotar entre los 5 grupos de vistas configurados:

1. **Grupo `default` (al abrir):**
   - *Panel izquierdo:* **Color natural con cajas** (las cajas amarillas muestran dónde la U-Net v4 actual vio algo $\ge 0{,}30$; úsalas solo como guía).
   - *Panel central:* **Color natural** (`B4`, `B3`, `B2` limpio sin cajas, para ver el borde exacto de la cicatriz).
   - *Panel derecho:* **Pendiente** (`B13` en grados, escala `viridis`: morado = plano/río, verde-amarillo = ladera empinada).
2. **Grupo `probabilidad` (tecla `B`):** `Color natural` | `Probabilidad del modelo` | `Pendiente`.
3. **Grupo `espectral` (tecla `B`):** `Color natural` | `Suelo vs Vegetacion` (índice GRVI: rojo = suelo desnudo, verde = vegetación) | `Pendiente`.
4. **Grupo `relieve` (tecla `B`):** `Color natural` | `Relieve (DEM m)` | `Bordes (Sobel)`.
5. **Grupo `mapa` (tecla `B`):** `Color natural` | `Mapa aéreo Bing`.

### Tabla de atajos de teclado esenciales

| Tecla | Acción |
|---|---|
| **`2`** | Seleccionar clase **Deslizamiento** (rojo) |
| **`3`** | Seleccionar clase **Dudoso** (amarillo) |
| **`1`** | Seleccionar clase **Sin deslizamiento** (fondo / corregir) |
| **`D`** | Herramienta Pincel (*Draw*) |
| **`E`** | Herramienta Borrador (*Eraser*) |
| **`W`** | Mover / arrastrar la vista (*Pan*) |
| **Rueda del ratón** | Hacer zoom (o cambiar tamaño del pincel con el control superior) |
| **`Z`** | Restablecer zoom de las vistas |
| **`Espacio`** | Ocultar / mostrar temporalmente la máscara pintada para ver la imagen limpia debajo |
| **`B`** | Cambiar al siguiente grupo de vistas (`default` $\rightarrow$ `probabilidad` $\rightarrow$ `espectral` $\rightarrow$ `relieve` $\rightarrow$ `mapa`) |
| **`U` / `R`** | Deshacer (*Undo*) / Rehacer (*Redo*) |
| **`N`** | Limpiar toda la máscara del parche actual (*Reset*) |
| **`A`** | **Asistente IA local (LightGBM):** tras pintar unos trazos de `Deslizamiento` (`2`) y unos trazos de `Sin deslizamiento` (`1`), rellena el resto por similitud usando solo las 6 bandas físicas (`B1..B4`, pendiente y DEM) |
| **`S`** | **GUARDAR la máscara actual en `masks/<id>.png` (¡Obligatorio en cada parche revisado!)** |
| **`Enter`** | Pasar al siguiente parche |
| **`Backspace`** | Volver al parche anterior |

> [!IMPORTANT]
> **Regla crítica para guardar (`S` antes de `Enter`):**
> En IRIS, la tecla **`Enter`** avanza al siguiente parche, pero **solo la tecla `S` (o el botón del disquete *Save mask*) escribe el archivo `.png` en `iris_128/masks/`**.
> - Si revisas un parche que **sí tiene cicatriz**: píntala con `2`, presiona **`S`** (verás el mensaje *"Masks successfully saved!"*) y luego **`Enter`**.
> - Si revisas un parche que **NO tiene ningún deslizamiento** (negativo confirmado): míralo completo, **no pintes nada y presiona `S`** antes de `Enter`. Así se guarda su máscara toda en `0` como **negativo confirmado**. Los parches donde nunca presiones `S` se consideran "aún no revisados" y no se importan.

---

## 5. Ejemplo práctico paso a paso (`ingemmet_003_001` — Sillapata)

En la carpeta [`iris_ejemplo_demo/`](../iris_ejemplo_demo) ya se ejecutó un flujo completo sobre los 4 cuadrantes de la tesela **`ingemmet_003_001` (Sillapata)** y se generó la lámina visual [`iris_ejemplo_demo/lamina_ejemplo_iris_128.png`](../iris_ejemplo_demo/lamina_ejemplo_iris_128.png):

1. **Qué muestra la Columna 1 (`Color natural con cajas`):**
   - La U-Net v4 actual encierra en cajas amarillas tanto la gran cicatriz real de Sillapata (en `q3` arriba-derecha y `q4` arriba-izquierda) como varios caminos rurales y cauces en `q1` y `q3`.
2. **Qué muestra la Columna 2 (`Color natural`) y la vista `Pendiente`:**
   - En `q3` (esquina superior derecha) y `q4` (esquina superior izquierda) se observa una gran cicatriz brillante con forma de abanico/lengua sobre ladera fuerte ($> 25^\circ$).
   - En la mitad izquierda de `q1` y `q3` se ven líneas blancas finas y continuas: son **carreteras y caminos en zigzag**, no deslizamientos.
3. **Qué se pinta en IRIS (Columna 3):**
   - Con la tecla **`2` (`Deslizamiento`, rojo)** se pinta únicamente el cuerpo de suelo desnudo de la cicatriz de ladera (`421 px = 4,21 ha` en `q3` y `318 px = 3,18 ha` en `q4`).
   - Con la tecla **`3` (`Dudoso`, amarillo)** se cubren los cauces o laderas ambiguas donde no se distingue si es erosión de quebrada o cicatriz.
   - Los caminos, bosques y pastos se dejan sin pintar (**`Sin deslizamiento`, valor `0`**).
4. **Qué produce `iris_128.py importar` (Columna 4 — Máscara Nivel 1 `.tif`):**
   - Convierte cada `.png` de IRIS en un GeoTIFF georreferenciado (`uint8`) en `mascaras_nivel1/parches_128/` con valores `0` (fondo confirmado), `1` (deslizamiento) y `255` (ignorar: dudosos + nubes blancas reales, respetando el suelo desnudo brillante de Sillapata sin borrarlo).
   - Además reconstruye la máscara completa de $256 \times 256\text{ px}$ en `mascaras_nivel1/mascaras_256/` y exporta los polígonos a `mascaras_nivel1/marcas_desde_iris.geojson` por si quieres visualizarlos también en QGIS.

---

## 6. Cómo convertir tus marcas de IRIS y pasar al Ajuste Fino

Cada vez que avances una tanda de parches en `iris_128/`:

### Paso A — Importar las máscaras de IRIS a Nivel 1
```powershell
python herramientas/iris_128.py importar --iris iris_128 --salida mascaras_nivel1 --quien VN
```
*(Este comando es acumulativo: si también generaste máscaras desde QGIS con `mascaras_desde_marcas.py`, fusiona ambas fuentes en `mascaras_nivel1/mascaras_nivel1.csv` sin borrar ninguna).*

### Paso B — Medir el acuerdo entre dos anotadores (en las 3 teselas del Bloque 0)
Si la Persona A marcó en `iris_128_A` y la Persona B marcó en `iris_128_B` (o en archivos GeoJSON de QGIS):
```powershell
python herramientas/acuerdo_anotadores.py iris_128_A iris_128_B --teselas sinpad_15525_001,sinpad_48878_001,sinpad_557_001
```

### Paso C — Empaquetar para Google Colab
```powershell
python herramientas/empaquetar_para_colab.py
```
Esto crea `datos_locales_ajuste.zip` listo para subir a Google Drive y correr [`herramientas/Ajuste_Fino_LaConvencion.ipynb`](../herramientas/Ajuste_Fino_LaConvencion.ipynb) con GPU T4.

