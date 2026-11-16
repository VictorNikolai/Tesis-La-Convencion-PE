# datos_azar: 60 teselas elegidas al azar en la provincia

Todas las demás teselas nacen de un punto donde se registró un evento, así que están sesgadas
hacia lugares donde algo pasó. Estas 60 se sortearon dentro de La Convención para medir cuánto
pinta el modelo donde no hay ningún registro, y para dar negativos variados al entrenamiento.

Sorteo (semilla fija, `herramientas/elegir_teselas_azar.py`; puntos en `azar/puntos_azar.csv`):
- la tesela entera cae dentro de la provincia y está a más de 6 km de cualquier otra tesela
- 40 «cerca» (a menos de 20 km de alguna tesela con evento) y 20 «lejos» (el resto de la provincia)
- imágenes de temporada seca de 2019 a 2025

Qué hay:
- `images/`, `qgis/`: igual que en las otras carpetas. En QGIS están en el grupo
  «Teselas al azar (negativos por confirmar)», con un marcador espacial por tesela (grupo «Al azar»).
- `revision_teselas.csv`: una fila por tesela con lo que se ve, la pendiente, la altitud y las
  hectáreas que la U-Net v4 marca a umbral 0,45. Ordenado de más a menos hectáreas marcadas.
- `revisadas.csv`: aquí va el «si» cuando una persona la revisó.
- `qgis/_revision/hoja_1.png` a `hoja_5.png`: las 60 de un vistazo, con el contorno rojo de lo
  que marca el modelo.

Lo que salió (lectura de Claude sobre las imágenes, sin confirmar por una persona):
- 55 usables, 3 para volver a bajar por nubes o huecos (024, 027, 058) y 2 descartadas por nieve
  (025, 035).
- En las 53 usables sin contar dos casos raros, la U-Net v4 marca 448 ha de 34 734 ha (1,3 %).
  En selva baja casi no marca nada (0,7 ha en 14 teselas); en puna y alta montaña marca mucho
  (133 ha en 12 teselas): pasto seco, roca clara y suelo desnudo natural.
- **Dos teselas, 012 y 015, salen marcadas enteras** con probabilidad cercana a 0,73 en todos
  los píxeles, sobre bosque y chacras normales. Es un fallo del modelo que hay que investigar
  (la misma respuesta aparecería en la aplicación web).
- Posibles cicatrices que nadie registró: 008 y 003 (franja parda alargada en bosque) y 023.
  Mirarlas primero.

Cómo usarlas: abrir cada tesela, y si no hay ninguna cicatriz, poner «si» en `revisadas.csv` sin
dibujar nada. Eso la convierte en un negativo confirmado (máscara en cero). Si hay algo, se
marca como en cualquier otra tesela. La mitad de las zonas al azar va a validación y la otra
mitad a entrenamiento (`particion.csv`).
