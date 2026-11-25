# Guía Corta de Remarcado de la Prueba Sellada (`docs/GUIA_REMARCADO.md`)

> [!IMPORTANT]
> **Propósito:** Unificar el criterio de etiquetado entre revisores sobre los parches con mayor desacuerdo de la prueba sellada (`360` parches), sin modificar ninguna máscara existente hasta contar con revisión conjunta.

## Criterio Común (4 reglas)

1. **Tamaño mínimo (`≥ 0.5 ha`):** marcar como **positivo (`1`)** únicamente cicatrices claras de **`0.5 ha` o más (`≥ 50` píxeles de `10 × 10 m`)**, con arranque o cuerpo bien definido en pendiente y cambio visible respecto a la imagen anterior (`2018–2020`).
2. **Bordes inciertos y fragmentos menores (`255`):** los **bordes difusos**, zonas parcialmente revegetadas con duda y **cicatrices menores de `0.5 ha` (`< 50 px`)** se marcan como **dudoso (`255`)** para no penalizar ni premiar desplazamientos de `1–2 px` en Sentinel-2.
3. **Qué NO se marca (queda como fondo `0`):**
   - **Cauces activos y playas de río** (lechos planos o trenzados en fondo de valle, TPI negativo).
   - **Chacras, pastos y cultivos** (parcelas agrícolas, mosaicos con bordes rectos o rebrote estacional).
   - **Carreteras, trochas, canteras de obra y construcciones** (cortes lineales paralelos a vías o techos).
   - **Nubes y sombras de nube** (verificar siempre en la vista aérea Bing y en la escena anterior).
4. **Prioridad de revisión:** comenzar por los parches marcados como **`ALTA (FA v9 tipo a)`** en `informes/remarcado_lista.csv` (posibles cicatrices reales detectadas por `v9` que no figuran en la referencia actual), y continuar en orden de zona.

## Flujo de Revisión con `informes/remarcado_laminas.pdf`

Cada parche muestra 5 vistas alineadas:
1. **Antes (`2018–2020`)** — cobertura previa al evento.
2. **Después (`2023–2025`)** — escena actual usada por el modelo.
3. **Aérea (Bing)** — verificación geométrica de alta resolución (caminos, techos, chacras, cauces).
4. **Referencia actual** — etiqueta vigente (`manual_v4`).
5. **Modelo `v9`** — predicción actual para contrastar omisiones o falsas alarmas.
