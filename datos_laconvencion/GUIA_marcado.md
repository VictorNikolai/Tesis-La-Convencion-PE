# Guía de marcado de deslizamientos en QGIS (La Convención)

Proyecto: `datos_laconvencion\proyecto_marcado.qgz`. Se dibuja en la capa **«marcas (dibujar aquí)»**.

## 1. Qué se marca

| Clase | Cuándo | Cómo se ve en la imagen (10 m, temporada seca) |
|---|---|---|
| **deslizamiento** | Hay una cicatriz de deslizamiento, derrumbe o huaico que arranca en una ladera | Mancha de suelo desnudo (café claro, gris o anaranjado) **alargada ladera abajo**, con forma de lengua, abanico o cuchara; bordes netos arriba (la corona) y difusos abajo; sin surcos paralelos ni formas rectas |
| **dudoso** | No se puede decidir con la imagen | Nubes y sus sombras, huecos sin dato (negros), bruma, cauces con depósitos de huaico, chacras recién abiertas que podrían ser cicatrices |
| *(nada)* | Todo lo demás | Bosque, pastos, chacras con bordes rectos, caminos, techos, playas de río, roca de alta montaña |

Se marca **la cicatriz completa**: zona de arranque y el material que bajó, hasta donde se ve suelo desnudo. No se marca un círculo alrededor del punto de SINPAD: ese punto es el lugar afectado (un pueblo, una carretera), no la cicatriz.

## 2. Confusores que hay en estas teselas (ya vistos)

- **Pasto seco en ladera** (Quellouno, Santa Ana, Maranura): grandes zonas pardas, textura uniforme, sin forma de lengua. No es deslizamiento. El modelo las pinta como «probable»: no copiar.
- **Playas y cauces trenzados** (Urubamba, Apurímac, Santa Teresa): planos, blancos o grises, junto al río. No es deslizamiento. Un **cauce de huaico** que baja por una quebrada va como `dudoso`, salvo que se vea claramente dónde arrancó en la ladera: eso sí es `deslizamiento`.
- **Caminos** (zigzag de Quellouno, Kiteni): líneas finas y continuas.
- **Chacras y terrazas**: bordes rectos, parcelas repetidas, cerca de casas.
- **Té y café** (Huayopata): verde amarillento brillante.
- **Roca de alta montaña** (Santa Teresa, hacia el Salkantay): toda la tesela es roca y nieve. Esas teselas están descartadas.
- **Bruma fina**: aclara la imagen y hace que pastos y terrazas parezcan suelo desnudo (pasó en 15525 y 3566 en 2019). Comparar con la imagen 2025.
- **Sombras de ladera**: oscuras, del lado opuesto al sol (el noroeste en estas imágenes).

Regla práctica: si la mancha parda tiene **forma de lengua y pendiente** (prende la capa «Pendiente» si dudas), es candidata; si es **plana, recta o cuadrada**, no.

## 3. Capas de ayuda (ninguna es la verdad)

- **Candidatos (posible deslizamiento)**: rojo = la U-Net da ≥ 0,45; naranja a rayas = entre 0,30 y 0,45; magenta punteado = suelo desnudo en ladera por una regla simple. Los que terminan en `_plano` están en terreno llano y casi siempre son cauces o pastos.
- **Cambios 2018 → 2025** (cuando esté cargada): `nuevo_desnudo` = tenía vegetación y ahora es suelo desnudo en ladera. **Es la mejor pista de cicatriz reciente**; aun así hay que mirarla.
- **Imágenes 2025** (grupo aparte): para comparar con la imagen base. Si la cicatriz solo se ve en 2025, se marca igual, pero se anota `2025` en el campo `nota` (la máscara se emparejará con la imagen de 2025).
- **Esri World Imagery** (fondo, apagado por defecto): imagen de muy alta resolución de otra fecha. Sirve para confirmar si una mancha es una cicatriz, una chacra o un techo. Ojo: su fecha no es la de Sentinel-2; lo que se marca es lo que se ve en la imagen Sentinel-2.
- **Probabilidad del modelo (mapa de calor)** y **Pendiente**: apagadas por defecto.

Atajo: seleccionar candidatos buenos → `Ctrl+C` → activar «marcas» en edición → `Ctrl+V`. Entran como `deslizamiento`; cambiar a `dudoso` si hace falta y **recortar el polígono** a lo que realmente se ve (el candidato suele ser más grande o más chico que la cicatriz).

## 4. Procedimiento por tesela

1. Ir a la tesela con **Marcadores espaciales** (panel Navegador).
2. Mirarla entera a escala 1:10 000 y luego recorrerla a 1:5 000.
3. Dibujar o copiar los polígonos. Guardar la edición (disquete).
4. En `revisadas.csv`, poner `si` en `revisada`, las iniciales en `quien` y, si hay algo raro, una nota.
5. Nunca marcar una tesela a medias: o se revisa entera o no se pone `si`.

## 5. Acuerdo entre los dos anotadores (hacerlo al inicio)

- Los dos marcan **las mismas 3 teselas** sin verse: `sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`. Cada uno guarda su capa con otro nombre (`marcas_NH.gpkg`, `marcas_LM.gpkg`).
- Se comparan superpuestas y se discuten las diferencias hasta acordar reglas escritas (se agregan a esta guía).
- Yo calculo el acuerdo (IoU entre anotadores) y queda en la tesis como control de calidad del etiquetado.

## 6. Primera ronda (20 teselas) y conjunto de validación

Marcar en este orden. **Ronda 1 (13 teselas, las más prometedoras):**
15525, 48878, 75170, 44342, 557, 21603, 37418, 21608, 43368, 74792, 57603, 5330, 21604.

**Validación local (10 teselas, se marcan igual de bien pero solo sirven para medir; nunca para entrenar):**
13640, 7448, 56208, 9709, 61214, 96, 36869, 62691, 86528, 5559.
Son negativos difíciles: pasto seco, ciudad, té, playas, pista de aterrizaje, caminos. Si el modelo ajustado deja de pintarlos, se nota ahí.

Descartadas (no marcar): 49054, 15262, 43587, 44383, 63280, 43111, 57225, 60684, 63635, 66653. Llano, urbano o alta montaña.
Pendiente de volver a bajar sin nubes: 75271.

El detalle tesela por tesela está en `revision_teselas.csv` (columna `uso_propuesto`).

## 6b. Ronda 2: teselas de INDECI 2017 a 2021 (carpeta `datos_indeci`)

Vienen de los shapefiles de INDECI 2003 a 2021 (404 movimientos en masa en la provincia, 136 lugares distintos). Se bajaron 64 zonas nuevas: 42 con eventos de 2017 a 2021 (con imagen **antes** y **después** del evento) y 22 antiguas. En QGIS están en el grupo **«INDECI 2017-2021 y antiguos»**, con sus propias capas de cambios, candidatos, cajas y bordes, y las imágenes previas en «Imágenes INDECI (antes del evento)».

**Ronda 2 (9 teselas, en este orden):** 133797 (Huayopata: cicatriz nueva clarísima arriba a la derecha, el mejor ejemplo del lote), 92622, 91210, 75302, 134371, 92395, 69126, 11049, 119409.
**Validación adicional:** 25956, 134745, 134852, 91757 (los cambios de 91757 son chacras nuevas, no deslizamientos).
**Descartadas:** 134776, 137127, 145162 (ciudades), 68867 (alta montaña). **Rebajar:** 97387 (nubes).
El detalle está en `datos_indeci\revision_teselas.csv`. El resto (46) queda de reserva.

Al terminar una tesela de INDECI, el `si` va en `datos_indeci\revisadas.csv` (cada carpeta tiene el suyo). La capa «marcas» es una sola para todo.

**Regla para la partición:** varias teselas se solapan o están a menos de 2,5 km (lista en `indeci\teselas_que_se_solapan.csv`; por ejemplo 557, 75170, 119409 y 75302 son todas Santa Teresa). Las teselas que se solapan van **juntas** al mismo conjunto (entrenamiento o validación); nunca una a cada lado.

## 6c. Ronda 3: polígonos de INGEMMET (carpeta `datos_ingemmet`, grupo «INGEMMET» en QGIS)

INGEMMET publica en GEOCATMIN 333 polígonos de peligros geológicos levantados en campo en la provincia (2020 a 2024): 143 deslizamientos, 135 caídas, 39 flujos. Se agruparon en 55 zonas; se bajaron 52 (3 quedan en el borde provincial) con imagen **antes** y **después** de la evaluación.

**Importante:** el polígono de INGEMMET delimita el **cuerpo del movimiento** (incluida la parte vegetada o latente), no solo la cicatriz visible. Por eso en QGIS hay dos capas:
- **«Polígonos INGEMMET (inventario de campo)»**: amarillo = deslizamiento, naranja = caída, celeste = flujo. Es la referencia del geólogo. No se copia tal cual.
- **«PROPUESTAS nivel 2»** (ver 6d): la parte de cada polígono donde la imagen sí muestra suelo desnudo, ya revisada. **Esto es lo que se revisa y se copia a «marcas»** (seleccionar → Ctrl+C → marcas en edición → Ctrl+V), recortando o completando lo que haga falta. Ya traen `clase = deslizamiento` y una `nota` con el código del informe de INGEMMET.

De los 305 polígonos que caen en las teselas, 56 muestran cicatriz clara, 39 parcial y 210 están vegetados (archivo `ingemmet\ingemmet_poligonos_visibilidad.csv`). Los vegetados **no se marcan** como deslizamiento (el modelo no puede verlos); si acaso, `dudoso`.

**Ronda 3 (orden en `datos_ingemmet\revision_teselas.csv`):** empezar por Mesacancha (005, 009, 040: el mejor conjunto de cicatrices grandes de toda la provincia), Sillapata en Inkawasi (002, 003, 036, 037) y la 046. En Wiracchuaycco (023), Ichiquiato (011) y las teselas del aluvión de Salkantay (043, 045, 017, 029, 050, 025) casi todo lo propuesto es **cauce**: va como `dudoso`, salvo que se vea el arranque en la ladera.

Las teselas de INGEMMET se solapan entre sí (misma zona evaluada en varios años) y algunas con las de SINPAD: misma regla, van juntas al mismo conjunto. El `si` va en `datos_ingemmet\revisadas.csv`.

## 6d. Propuestas revisadas (nivel 2) y teselas al azar (15/11/2026)

**Capa nueva en el grupo INGEMMET: «PROPUESTAS nivel 2».** Son las mismas 598 piezas, pero ya miradas una por una sobre la imagen: rojo = aceptada (84 piezas, cicatriz clara en ladera), naranja a rayas = dudosa (cauces de huaico y del aluvión, cárcavas, piezas de menos de 0,1 ha), gris = descartada (caminos, chacras, techos). El campo `motivo` dice por qué. La capa anterior «PROPUESTAS visibles» quedó apagada. Siguen siendo propuestas: lo que vale es lo que ustedes copien a «marcas».
- Empezar por las rojas de Mesacancha (005, 009, 040), Sillapata (002, 003, 036, 037) y la 046.
- En la 045 hay una **cicatriz nueva grande al noreste, fuera de los polígonos de INGEMMET**: marcarla a mano.
- Las hojas de `mascaras_provisionales\nivel2\_revision` muestran cada tesela antes, después y con contornos.

**Grupo nuevo: «Teselas al azar (negativos por confirmar)».** 60 teselas sorteadas en la provincia (carpeta `datos_azar`, marcadores en el grupo «Al azar»). Si una tesela no tiene ninguna cicatriz, basta poner `si` en `datos_azar\revisadas.csv` sin dibujar nada: queda como negativo confirmado. Orden sugerido en `datos_azar\revision_teselas.csv`: primero las que más pinta el modelo. Mirar con cuidado 008, 003 y 023 (posibles cicatrices sin registro). Las 012 y 015 salen marcadas enteras por el modelo: son un fallo a investigar, no un deslizamiento.

**Cambio en la validación.** La validación de la guía solo tenía negativos, y así no se puede medir IoU. Ahora tres zonas con cicatrices van a validación: Sillapata (002, 003, 036, 037), Quellouno/Echarate (11049, 21603, con 134852 y 56208) y Santa Ana (027, 37418, 61214). La 25956 pasa a entrenamiento porque su zona entrena. La lista completa, por zona, está en `particion.md`. **Marquen también las zonas de validación**: sin ellas no se puede decir si el ajuste mejoró.

**Suelo desnudo muy claro.** La regla de nube (azul por encima de 1000) también se dispara sobre cicatrices muy claras, como las de Sillapata y Mesacancha. Dentro de un polígono de `deslizamiento` esa regla ya no se aplica: lo que ustedes marcan manda.

## 7. Al terminar una tanda

1. En QGIS, guardar la edición y exportar la capa: clic derecho en «marcas» → Exportar → Guardar objetos como → GeoJSON, EPSG:4326, en `datos_laconvencion\marcas.geojson`.
2. Poner `si` en el `revisadas.csv` de cada carpeta para las teselas terminadas (también las que no tienen nada: son negativos).
3. Avisar a Claude, o correr en `Tesis-Proyecto`:

        python herramientas/mascaras_desde_marcas.py      (máscaras de nivel 1 y sus parches de 128)
        python herramientas/empaquetar_para_colab.py      (arma datos_locales_ajuste.zip)

4. Subir `datos_locales_ajuste.zip` a Google Drive y abrir `herramientas\Ajuste_Fino_LaConvencion.ipynb` en Colab. El cuaderno mide la U-Net v4 antes y después del ajuste en las zonas de validación y en la prueba de Landslide4Sense.

Para el acuerdo entre anotadores (sección 5): cada uno exporta su capa como GeoJSON y se corre `python herramientas/acuerdo_anotadores.py marcas_NH.geojson marcas_LM.geojson --teselas sinpad_15525_001,sinpad_48878_001,sinpad_557_001`.
