# Máscaras provisionales: NO son etiquetas humanas

Hay tres niveles de máscara en el proyecto. Solo el nivel 1 es verdad de terreno.

| Nivel | Carpeta | De dónde sale | Para qué sirve |
|---|---|---|---|
| 1 | `mascaras_nivel1/` (aparece cuando marquen) | Lo que ustedes dibujan y confirman en QGIS | Entrenar y medir. Es lo único que se reporta en la tesis |
| 2 | `mascaras_provisionales/nivel2/` | Polígonos de campo de INGEMMET cruzados con el suelo desnudo visible, y revisados por Claude sobre la imagen | Probar el código y, con peso bajo, entrenamiento débil. Nunca para medir |
| 3 | `mascaras_provisionales/nivel3/` | Reglas automáticas (suelo desnudo en ladera y cambios antes/después), sin revisar | Solo pistas para buscar cicatrices. **No usar para entrenar** |

## Nivel 2

- `propuestas_nivel2.geojson`: las 598 piezas, cada una con `pid`, `decision` (aceptada, dudosa,
  descartada), `motivo` y `tipo_ingemmet`. Está cargada en QGIS dentro del grupo INGEMMET.
- `decisiones.csv`: lo que decidió Claude al mirar cada tesela, con el motivo. Se puede editar y
  volver a correr `python herramientas/mascaras_provisionales.py --hojas`.
- `mascaras_256/` y `parches_128/`: la máscara de cada tesela y sus 4 cortes.
  Valores: 0 fondo, 1 propuesta aceptada, 255 ignorar.
- `mascaras_nivel2.csv`: resumen por tesela.
- `_revision/`: una hoja por tesela con la imagen de antes, la de después y los contornos.

Resultado de la revisión: 84 piezas aceptadas (185 ha) en 8 teselas (Mesacancha 005, 009 y 040;
Sillapata 002, 003, 036 y 037; y la 046), 468 dudosas (91 ha) y 46 descartadas (8 ha). Las teselas
005, 009 y 040 muestran las mismas cicatrices en fechas distintas, y las de Sillapata se solapan:
las cicatrices independientes son bastante menos de lo que sugieren las hectáreas.

Reglas de la revisión:
- Flujo de detritos, aluvión, inundación y erosión en cárcava quedan como dudosa. La guía de
  marcado deja el cauce como dudoso salvo que se vea dónde arrancó en la ladera.
- Piezas de menos de 0,10 ha quedan como dudosa.
- Caminos, chacras, techos y zonas urbanas quedan como descartada.
- Dentro del polígono de INGEMMET, lo que no es pieza aceptada vale 255: no se afirma que la
  parte vegetada sea fondo.
- La regla de nube (B2 > 1000) no se aplica sobre una pieza aceptada, porque el suelo desnudo
  muy claro de una cicatriz también supera ese valor.

Para que una pieza pase a ser etiqueta de verdad, una persona la revisa en QGIS y la copia a la
capa «marcas» (seleccionar, Ctrl+C, «marcas» en edición, Ctrl+V), recortándola si hace falta.

## Nivel 3

`nivel3/<proyecto>/<tesela>.tif`: 0 nada, 1 suelo desnudo en ladera, 2 cambio de vegetación a suelo
desnudo. `mascaras_nivel3.csv` ordena las teselas por hectáreas de cambio. No incluye la
probabilidad de la U-Net a propósito: entrenar con ella sería enseñarle sus propios errores.
