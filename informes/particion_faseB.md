# Partición Espacial por Zonas — Fase B (Actualizada en Parada h: Separación por Método `manual_v4` vs. `clasificador_v5`)

## 1. Reglas de Aislamiento Espacial, Provincia y Deduplicación Verificadas

1. **Pertenencia provincial ($\ge 50\%$ dentro de La Convención):**
   - Comprobación geométrica automática contra `herramientas/limite_laconvencion.geojson`.
   - Las 4 teselas fuera del límite provincial (`ZN44_MACHUPICCHU_001`, `ZN45_SAMUGARI_001`, `ZN46_OLLANTAYTAMBO_001`, `ZN47_HUAROCONDO_001`) permanecen aisladas en `prueba_externa`.
   - **100 % de las 306 teselas activas** (`140` `entrenamiento`, `76` `validacion`, `90` `prueba`) tienen $\ge 50\%$ de su huella real dentro de la provincia de La Convención.

2. **Aislamiento espacial borde a borde ($\ge 3,000\text{ m}$) entre conjuntos activos:**
   - Medido sobre los contornos reales UTM (`EPSG:32718`, $2,560\text{ m} \times 2,560\text{ m}$) de las 306 teselas activas:
     - **`entrenamiento` $\leftrightarrow$ `validacion`:** distancia mínima borde a borde = **`3,054.1 m`** (`0` pares $< 3,000\text{ m}$).
     - **`prueba` $\leftrightarrow$ `entrenamiento`:** distancia mínima borde a borde = **`3,246.9 m`** (`0` pares $< 3,000\text{ m}$).
     - **`prueba` $\leftrightarrow$ `validacion`:** distancia mínima borde a borde = **`3,250.0 m`** (`0` pares $< 3,000\text{ m}$).

3. **Sin teselas duplicadas ($\le 10\%$ de solape de área):**
   - Las `72` teselas redundantes con solape $> 10\%$ permanecen en `excluida` y documentadas en `informes/duplicados_teselas.csv`.
   - Entre las **306 teselas activas**, el solape máximo entre cualquier par de teselas (dentro del mismo conjunto o entre conjuntos distintos) es **`9.42 %`** (**`0` pares $> 10\%$**).

---

## 2. Tabla Limpia por Conjunto, Método (`manual_v4` vs. `clasificador_v5`) y Tamaño de Cicatriz (Parada h)

En `informes/decisiones_agente.csv` y `mascaras_nivel1/mascaras_nivel1.csv` se agregó la columna `metodo` (`manual_v4` vs. `clasificador_v5`). En los parches `clasificador_v5` de `entrenamiento` y `validacion` se aplicó la limpieza morfológica y cartográfica:
- **a)** Eliminados todos los componentes de menos de `20 px` que no tocan un componente mayor (`0` componentes $< 20\text{ px}$ en `clasificador_v5`).
- **b)** Excluidos asentamientos (usando las `57,540` geometrías de `herramientas/edificaciones_laconvencion.geojson` + vías residenciales OSM + Bing), chacras y pastizales (exigiendo morfología de lengua, abanico o cuchara con corona en ladera empinada $\ge 21^\circ\text{--}22^\circ$, caída de elevación $\Delta Z \ge 25\text{ m}$ y contraste local $\Delta\text{GRVI} \le -0.045$; lo demás pasa a `dudoso`).
- **c)** Revisados visualmente uno por uno con Bing todos los positivos de `validacion`.
- **d)** Confirmado que **`prueba` permanece sellada y el 100 % de sus 360 parches (`90` teselas) son `manual_v4`** (`0` parches de `clasificador_v5`).

| Conjunto | Método (`metodo`) | Peso Pérdida | Teselas (`256×256`) | Parches (`128×128`) | Parches Positivos | Parches Dudosos | Parches Negativos | Cicatrices `< 10 px` | Cicatrices `10–19 px` | Cicatrices `20–50 px` | **Subtotal `10–50 px`** | Cicatrices `50–200 px` | Cicatrices `> 200 px` | **Total Cicatrices $\ge 10\text{ px}$** | Cicatrices / Parche Pos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`entrenamiento`** | `manual_v4` | **1.0** | 38 | 152 | **6** | 2 | 144 | **0** | 1 | 3 | **4** | 3 | 9 | **16** | 2.7 |
| **`entrenamiento`** | `clasificador_v5` *(limpio)* | **0.5** | 102 | 408 | **132** | 162 | 114 | **0** | **0** | 147 | **147** | 87 | 19 | **253** | **1.9** |
| **Subtotal `entrenamiento`** | Ambos | 1.0 / 0.5 | **140** | **560** | **138** | **164** | **258** | **0** | **1** | **150** | **151** | **90** | **28** | **269** | **1.9** |
| **`validacion`** *(ciego)* | `manual_v4` | — | 59 | 236 | **8** | 18 | 210 | **0** | 0 | 5 | **5** | 0 | 4 | **9** | 1.1 |
| **`validacion`** *(ciego, rev. Bing 1×1)* | `clasificador_v5` *(limpio)* | — | 17 | 68 | **33** | 30 | 5 | **0** | **0** | 14 | **14** | 27 | 13 | **54** | **1.6** |
| **Subtotal `validacion`** | Ambos | — | **76** | **304** | **41** | **48** | **215** | **0** | **0** | **19** | **19** | **27** | **17** | **63** | **1.5** |
| **`prueba`** *(ciego, **sellada**)* | **`manual_v4` (100 %)** | — | **90** | **360** | **38** | **51** | **271** | **0** | **27** | **27** | **54** | **15** | **11** | **80** *(84 en 4-con)* | **2.1** |
| **Total activo La Convención** | Todos | — | **306** | **1,224** | **217** | **263** | **744** | **0** | **28** | **196** | **224** | **132** | **56** | **412** | **1.9** |
