# Partición Fija por Zona — Fase B + Verificación de Contornos Reales (Fase C)

Fecha de actualización: 2026-11-22  
Criterio estricto de aislamiento espacial sobre el **contorno real de cada tesela descargada** (`EPSG:32718`):
- **Distancia mínima borde-a-borde `prueba` $\leftrightarrow$ `entrenamiento`/`validacion`:** **`3,020.0 m`** ($\ge 3{,}000\text{ m}$, **0 fugas**).
- **Distancia mínima borde-a-borde `entrenamiento` $\leftrightarrow$ `validacion`:** **`1,060.0 m`** ($\ge 1{,}000\text{ m}$, **0 fugas**).

---

## 1. Resumen cuantitativo final tras descarga y recorte real

| Conjunto | Zonas | Teselas activas | Parches (128×128 px) | Cicatrices Nivel 1 actuales | Cicatrices Nivel 2 provisionales | Modo de marcado |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`entrenamiento`** | **88** | **196** | **784** | 0 | 509 (en existentes) | Con apoyo de ensamble + $\Delta\text{GRVI}$ + INGEMMET |
| **`validacion`** *(fija)* | **39** | **84** | **336** | 0 | 147 (228 c/ variantes) | **CIEGO** (`iris_128_validacion/`, sin vistas de modelo, revisión exhaustiva) |
| **`prueba`** *(zonas nuevas)* | **17** | **51** | **204** | 0 | 0 *(nunca vista por ningún modelo)* | **CIEGO** (`iris_128_prueba/`, sin vistas de modelo, revisión exhaustiva) |
| **`pendiente`** | 1 | 1 (`sinpad_97387_001`) | — | — | — | A 1,710 m de entrenamiento (no se usa en prueba) |
| **`excluida`** | — | 36 | — | — | — | Descartadas en guía / recortadas por contorno $<3$ km |

---

## 2. Detalle de las 17 zonas de `prueba` (51 teselas = 204 parches)

Distribución temporal en `prueba`: **4 años distintos** (`2019`: 12 teselas, `2021`: 7 teselas, `2022`: 26 teselas, `2024`: 6 teselas).

| Zona | Teselas | Carpeta | Distrito / Sector | Año imagen |
| :--- | :---: | :--- | :--- | :---: |
| `Z111_ECHARATE` | 9 | `datos_prueba` | Echarate (INDECI + `sinpad_75271`) | 2022 |
| `Z51_ECHARATE` | 1 | `datos_azar` | Echarate (control aleatorio) | 2021 |
| `Z54_ECHARATE` | 1 | `datos_azar` | Echarate (control aleatorio) | 2021 |
| `Z85_ECHARATE` | 1 | `datos_azar` | Echarate / Bajo Urubamba (control aleatorio) | 2021 |
| `ZN01_OCOBAMBA` | 4 | `datos_prueba` | Ocobamba / Cielopunco / Combapata | 2019 |
| `ZN02_OCOBAMBA` | 4 | `datos_prueba` | Ocobamba / Yanamayo | 2022 |
| `ZN03_AYNA` | 2 | `datos_prueba` | Kimbiri / Ayna | 2024 |
| `ZN04_VILCABAMBA` | 4 | `datos_prueba` | Vilcabamba / Oyara / Tarqui | 2024 |
| `ZN07_QUELLOUNO` | 2 | `datos_prueba` | Quellouno / Cº Pantorrilla | 2019 |
| `ZN11_HUAYOPATA` | 1 | `datos_prueba` | Huayopata / Abra Málaga | 2022 |
| `ZN13_PICHARI` | 9 | `datos_prueba` | Pichari (selva, caja INDECI) | 2022 |
| `ZN18_ECHARATE` | 1 | `datos_prueba` | Kumpirushiato (río Cumpirusiato) | 2019 |
| `ZN28_KIMBIRI` | 1 | `datos_prueba` | Kimbiri / Samaniato | 2021 |
| `ZN31_PICHARI` | 8 | `datos_prueba` | Pichari (selva, caja INDECI) | 2021/2022 |
| `ZN35_QUELLOUNO` | 1 | `datos_prueba` | Quellouno / Yavero Chico | 2019 |
| `ZN40_VILCABAMBA` | 1 | `datos_prueba` | Inkawasi / río Apurímac | 2019 |
| `ZN41_VILCABAMBA` | 1 | `datos_prueba` | Inkawasi / San Fernando / río Yauyacu | 2019 |

---

## 3. Reglas de aislamiento y Bloque 0
- Las 60 zonas con `"?"` (`Z28_?`..`Z87_?`) en `particion.csv` recibieron su distrito oficial cruzado con `herramientas/distritos_laconvencion.geojson`.
- El Bloque 0 (`sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`) queda intacto en `iris_128/`.
