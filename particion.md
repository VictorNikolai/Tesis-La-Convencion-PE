# Partición por zona (propuesta)

Generado por `herramientas/particion_por_zona.py`. La regla y el porqué están al inicio de ese archivo.
Una zona = teselas que se solapan o están a menos de 1 km, encadenadas. Toda la zona va a un solo conjunto.

- Teselas base: 242 en 123 zonas.
- Entrenamiento: 137 teselas (31 de ronda).
- Validación: 84 teselas (8 de ronda, 13 negativos de la guía).
- Excluidas: 16; pendientes de rebajar: 5.

Solo cuentan las teselas que se lleguen a marcar; las de reserva sin marcar no entran a ningún cálculo.
La prueba final sigue siendo aparte: el test de Landslide4Sense y, si se arma, un conjunto local nunca usado para decidir nada.

| Zona | Teselas | Conjunto | Motivo | Para marcar (rondas y validación de la guía) |
|---|---|---|---|---|
| Z01_SANTA_TERESA | 27 | entrenamiento | tiene teselas de ronda | ingemmet_005 ingemmet_009 ingemmet_040 ingemmet_043 ingemmet_045 sinpad_119409 sinpad_21608 sinpad_48878 sinpad_557 sinpad_75170 sinpad_75302 sinpad_92395 sinpad_92622 sinpad_25956 |
| Z02_ECHARATE | 20 | validacion | negativos de validación de la guía | sinpad_134745 sinpad_36869 |
| Z03_SANTA_TERESA | 16 | entrenamiento | tiene teselas de ronda | ingemmet_017 ingemmet_025 ingemmet_029 ingemmet_046 ingemmet_050 sinpad_5330 sinpad_69126 sinpad_91210 |
| Z04_SANTA_ANA | 12 | validacion | ancla: Santa Ana (Quillabamba): taludes, Paraíso Alto y el negativo urbano 61214 | ingemmet_027 sinpad_37418 sinpad_61214 |
| Z05_ECHARATE | 9 | validacion | negativos de validación de la guía | sinpad_62691 |
| Z07_VILCABAMBA | 5 | entrenamiento | tiene teselas de ronda | ingemmet_023 sinpad_74792 |
| Z08_QUELLOUNO | 5 | validacion | ancla: Quellouno/Echarate: dos teselas de ronda junto a dos negativos de validación | sinpad_11049 sinpad_21603 sinpad_134852 sinpad_56208 |
| Z09_INKAWASI | 4 | validacion | ancla: Sillapata (Inkawasi): cicatrices claras de INGEMMET en un distrito aparte | ingemmet_002 ingemmet_003 ingemmet_036 ingemmet_037 |
| Z10_OCOBAMBA | 4 | entrenamiento | tiene teselas de ronda | sinpad_134371 sinpad_44342 sinpad_57603 |
| Z11_ECHARATE | 3 | entrenamiento | tiene teselas de ronda | ingemmet_011 |
| Z13_HUAYOPATA | 3 | entrenamiento | tiene teselas de ronda | sinpad_43368 |
| Z16_HUAYOPATA | 2 | entrenamiento | tiene teselas de ronda | sinpad_133797 |
| Z22_QUELLOUNO | 2 | validacion | negativos de validación de la guía | sinpad_7448 |
| Z27_ECHARATE | 2 | validacion | negativos de validación de la guía | sinpad_91757 |
| Z92_QUELLOUNO | 1 | validacion | negativos de validación de la guía | sinpad_13640 |
| Z97_QUELLOUNO | 1 | entrenamiento | tiene teselas de ronda | sinpad_15525 |
| Z98_OCOBAMBA | 1 | entrenamiento | tiene teselas de ronda | sinpad_21604 |
| Z105_QUELLOUNO | 1 | validacion | negativos de validación de la guía | sinpad_5559 |
| Z114_ECHARATE | 1 | validacion | negativos de validación de la guía | sinpad_86528 |
| Z122_HUAYOPATA | 1 | validacion | negativos de validación de la guía | sinpad_96 |
| Z123_MARANURA | 1 | validacion | negativos de validación de la guía | sinpad_9709 |

Las otras 102 zonas solo tienen teselas de reserva o al azar: 30 a validación y 72 a entrenamiento (detalle en particion.csv).

## Cambios respecto de la guía

- `sinpad_25956_001` figuraba como validación, pero está en la zona Z01_SANTA_TERESA, que entrena. Se marca igual; cuenta como entrenamiento.
