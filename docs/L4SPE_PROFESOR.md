# Máscaras manuales del profesor (L4SPe) en el ajuste fino

## Qué son
838 parches de 128 × 128 px con 14 bandas (formato Landslide4Sense) y una máscara manual por parche
(`image_ls4pe/point_XXXX.h5` y `mask_ls4pe/point_XXXX.h5`; el proyecto IRIS con las mismas máscaras está en
`Profesor/landslide_l4spe_iris`). 808 parches tienen deslizamiento y 30 no; el 2,9 % de los píxeles es deslizamiento.
Los puntos están en el centro del Perú (lat −11,6 a −8,2; lon −76,9 a −74,8): **ninguno cae en La Convención**.

## Cómo se usan
1. `python herramientas/empaquetar_l4spe.py --carpeta <l4s> --iris <landslide_ok> --salida l4spe_profesor.npz`
   deja un solo archivo (≈ 100 MB) y `l4spe_profesor_particion.csv`.
2. Las **zonas** (puntos a menos de 3 km encadenados) se reparten enteras: 70 % entrenamiento, 15 % validación,
   15 % prueba (587 / 127 / 124 parches; 181 / 31 / 68 zonas).
3. El ajuste fino (`--l4spe`) mezcla en cada lote La Convención, L4SPe y Landslide4Sense (ancla); elige época y umbral
   con la validación de L4SPe (máscara manual) y mide al final en la **prueba de L4SPe**, que no se usó para decidir nada.

## Qué hay que tener presente
- Las bandas vienen en niveles digitales de Sentinel-2 L1C (B10 ≈ 10), no normalizadas por escena. Se llevan a unidades
  de Landslide4Sense dividiendo por la media del entrenamiento (guardada en el `.npz`), y la pendiente (radianes, comprobado
  contra el DEM) se convierte a grados / 20,292.
- El servidor trabaja con L2A. Que el modelo mejore en L4SPe (L1C, otras zonas) **no garantiza** mejora en La Convención.
  Por eso el informe muestra por separado L4SPe, validación local provisional (nivel 2) y Landslide4Sense.
- En el `landslides.json` del proyecto IRIS, `elevation` apunta a `B13.tif` y `slope` a `B14.tif`, pero en los datos B13 es la
  pendiente y B14 la elevación (como en Landslide4Sense). Las vistas `elevation` y `slope` de IRIS salen intercambiadas.
- Confirmar con el profesor el origen y la licencia de estos datos, y cómo citarlos (en los documentos del proyecto
  «Landslide4SensePe» figura como un conjunto de 47 candidatos, 10 aceptados; este es distinto y mucho mayor).
