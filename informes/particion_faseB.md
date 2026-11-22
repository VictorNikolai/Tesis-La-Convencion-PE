# Partición Fija por Zona — Fase B (Parada A)

Fecha de congelamiento: 2026-11-22  
Criterio de aislamiento espacial: encadenamiento completo con umbral $\Delta x < 3{,}560\text{ m}$ y $\Delta y < 3{,}560\text{ m}$. **Cero fugas verificadas entre los tres conjuntos (`entrenamiento`, `validacion`, `prueba`).**

---

## 1. Resumen cuantitativo por conjunto

| Conjunto | Zonas actuales (con tesela en disco) | Zonas nuevas planificadas (`particion_nuevas_zonas.csv`) | Total Zonas | Cicatrices actuales Nivel 1 | Cicatrices provisionales Nivel 2 (actuales) | Puntos INGEMMET / INDECI nuevos | Meta Nivel 1 para la tesis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`entrenamiento`** | 70 (137 teselas) | +21 | **91 zonas** | 0 | 509 (en 66 teselas / 11 zonas) | +28 puntos | $\ge 500$ parches positivos + $\ge 500$ negativos confirmados |
| **`validacion`** *(fija)* | 39 (84 teselas) | 0 (se mantienen intactas las 39 zonas) | **39 zonas** | 0 | 147 (en 30 teselas / 5 zonas; 228 incluyendo 2025/antes) | — | $\ge 150$ cicatrices Nivel 1 |
| **`prueba`** *(zonas nuevas)* | 4 aisladas de `pendiente` (`Z111_ECHARATE`, `Z51_?`, `Z54_?`, `Z85_?`) | +18 zonas INGEMMET/INDECI nuevas | **22 zonas** | 0 | 0 *(ningún modelo la ha visto)* | 67 puntos | $\ge 150$ cicatrices en $\ge 15$ zonas |
| **`pendiente`** | 1 (`sinpad_97387_001`, a 1,710 m de `entrenamiento`) | — | 1 | — | — | — | No se usa en `prueba` (cercana a `entrenamiento`) |
| **`excluida`** | 16 teselas (13 zonas) | — | 13 | — | — | — | Descartadas en guía |

---

## 2. Detalle de las 22 zonas de `prueba` (100% nuevas, nunca vistas por `unet_peru` ni `deeplabv3plus_peru`)

Todas están a $> 3{,}560\text{ m}$ (las 4 de `pendiente` a $> 6{,}840\text{ m}$) de cualquier tesela de `entrenamiento` y `validacion`:

| Zona | Puntos / Fuente | Distrito(s) | Peligros / propósito |
| :--- | :---: | :--- | :--- |
| `Z111_ECHARATE` | `sinpad_75271_001` + INDECI (2 pts) | Echarate | Negativo difícil + evento INDECI |
| `Z51_?` | `azar_024_001` (ya descargada) | La Convención | Control aleatorio / negativos difíciles |
| `Z54_?` | `azar_027_001` (ya descargada) | La Convención | Control aleatorio / negativos difíciles |
| `Z85_?` | `azar_058_001` (ya descargada) | Bajo Urubamba / Echarate | Control aleatorio selva baja / ríos |
| `ZN01_OCOBAMBA` | 20 pts INGEMMET | Ocobamba, Quellouno, Yanatile | Alta densidad de deslizamientos (`1302`), derrumbes (`1301`), caídas (`1103`) |
| `ZN02_OCOBAMBA` | 10 pts INGEMMET | Ocobamba | Deslizamientos (`1302`), flujos (`1504`), sector Yanamayo |
| `ZN03_AYNA` | 6 pts INGEMMET + INDECI | Ayna, Kimbiri | Deslizamientos / derrumbes / carretera San Francisco |
| `ZN04_VILCABAMBA` | 5 pts INGEMMET | Vilcabamba | Deslizamientos (`1302`) en Oyara, Tarqui, Quinuayarca |
| `ZN06_ECHARATE` | 3 pts INGEMMET | Echarate | Deslizamientos (`1302`), flujos (`1504`), Quellomayo |
| `ZN07_QUELLOUNO` | 3 pts INGEMMET | Quellouno | Deslizamientos (`1302`), derrumbes (`1301`), Cº Pantorrilla |
| `ZN08_SANTA_ANA` | 3 pts INGEMMET | Santa Ana | Flujos (`1504`) y borde urbano Quillabamba (control falsos positivos) |
| `ZN11_HUAYOPATA` | 2 pts INGEMMET | Huayopata | Deslizamientos (`1302`) en Abra Málaga |
| `ZN18_ECHARATE` | 1 pt INGEMMET | Echarate | Derrumbe (`1301`) |
| `ZN22_ECHARATE` | 1 pt INGEMMET | Echarate | Deslizamiento (`1302`) / Kepashiato |
| `ZN24_ECHARATE` | 1 pt INGEMMET | Echarate | Derrumbe / Deslizamiento (`1302`) Cº Yuracmayo |
| `ZN28_KIMBIRI` | 1 pt INGEMMET | Kimbiri | Derrumbe (`1301`) Samaniato |
| `ZN29_OCOBAMBA` | 1 pt INGEMMET | Ocobamba | Derrumbe (`1301`) Cº La Florida |
| `ZN34_QUELLOUNO` | 1 pt INGEMMET | Quellouno | Deslizamiento (`1302`) Quebrada Honda |
| `ZN35_QUELLOUNO` | 1 pt INGEMMET | Quellouno | Deslizamiento (`1302`) Yavero Chico |
| `ZN40_VILCABAMBA` | 1 pt INGEMMET | Vilcabamba | Río Yauyacu |
| `ZN41_VILCABAMBA` | 1 pt INGEMMET | Vilcabamba | Deslizamiento (`1302`) San Fernando |
| `ZN42_VILCABAMBA` | 1 pt INGEMMET | Vilcabamba | Derrumbe (`1301`) Cº Campanayoc |

---

## 3. Reglas de congelamiento y Bloque 0
- Las **39 zonas de `validacion` no cambian**. Se completarán hasta $\ge 150$ cicatrices Nivel 1 marcando sus teselas en IRIS (`validacion` ya tiene 336 parches en `iris_128`).
- El **Bloque 0** (`sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`) permanece **sin tocar** en `iris_128/` para Víctor y Leonardo; se marcará aparte en `iris_128_agente_bloque0/`.
- Los archivos completos de la partición se encuentran en [`particion.csv`](../particion.csv) y [`particion_nuevas_zonas.csv`](../particion_nuevas_zonas.csv).
