# parches_128: las teselas cortadas al tamaño del modelo

El modelo (U-Net v4), Landslide4Sense y la aplicación trabajan con parches de 128×128 px a 10 m.
Las teselas de La Convención miden 256×256 px. Aquí cada tesela está **cortada** (no encogida) en
4 parches sin solape; cada píxel sigue midiendo 10 m.

- `q1` arriba izquierda, `q2` arriba derecha, `q3` abajo izquierda, `q4` abajo derecha.
- Cada parche es un GeoTIFF con 6 bandas: B1, B2, B3, B4, pendiente (grados) y DEM (m). La
  probabilidad del modelo y las cajas de la tesela original no se copian: no son entradas.
- `indice_parches.csv` tiene una fila por parche: tesela, cuadrante, fecha (base, 2025 o antes del
  evento), zona, conjunto (entrenamiento, validacion, excluida, pendiente), coordenadas y cuánto
  del parche está limpio.

Las teselas originales no se tocaron. Para rehacerlo todo:

    python herramientas/particion_por_zona.py
    python herramientas/cortar_parches_128.py

Las máscaras cortadas igual que estos parches están en `mascaras_nivel1/parches_128` (cuando
marquen) y en `mascaras_provisionales/nivel2/parches_128`.
