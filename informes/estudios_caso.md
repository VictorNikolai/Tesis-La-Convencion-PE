# Estudios de Caso Cualitativos — Eventos Reales de INDECI (3 × 3 km)

> [!NOTE]
> **Material cualitativo (no métrica de prueba):** Los 3 eventos fueron localizados exclusivamente mediante **ΔGRVI** entre escenas secas de Sentinel-2 anteriores y posteriores al evento en un radio de `15 km` de la capital distrital reportada por INDECI, **sin usar ningún modelo**, y confirmados visualmente en Bing aéreo y Sentinel-2 a 10 m.
> 
> **Nota sobre selección de eventos:** Se inspeccionó inicialmente `Inkawasi (2021-04-14, ID 138302)`, pero las anomalías de ΔGRVI en radio de 15 km correspondían a secado estacional de pastizales/chacras altoandinas sin una cicatriz inequívoca en Bing; siguiendo el protocolo, se sustituyó por **`Vilcabamba (2021-03-27, ID 137416)`**, donde sí se verificó una cicatriz nueva clara sobre ladera.

## Tabla Comparativa (Área de `3 × 3 km` = `900 ha` por caso)

| Caso (Distrito, Fecha, ID INDECI) | Centro localizado (ΔGRVI, dist. capital) | Modelo | Umbral | Total marcado (`ha`) | En foco del evento (`ha`) | Fuera del foco / FA (`ha`) | ¿Cubre la cicatriz? |
|---|---|---|---:|---:|---:|---:|:---:|
| **Santa Teresa** (2020-02-23) | `(-72.6230, -13.2050)` (9.0 km) | U-Net L4S | 0.45 | 7.16 | 2.06 | 5.10 | Sí |
| **Santa Teresa** (2020-02-23) | `(-72.6230, -13.2050)` (9.0 km) | U-Net B1 v5 | 0.30 | 0.76 | 0.61 | 0.15 | Sí |
| **Santa Teresa** (2020-02-23) | `(-72.6230, -13.2050)` (9.0 km) | U-Net v6 (sin TTA+flt) | 0.45 | 0.71 | 0.48 | 0.23 | Parcial |
| **Echarate** (2021-02-16) | `(-72.5620, -12.7470)` (3.0 km) | U-Net L4S | 0.45 | 86.69 | 2.91 | 83.78 | Sí |
| **Echarate** (2021-02-16) | `(-72.5620, -12.7470)` (3.0 km) | U-Net B1 v5 | 0.30 | 3.48 | 0.41 | 3.07 | Parcial |
| **Echarate** (2021-02-16) | `(-72.5620, -12.7470)` (3.0 km) | U-Net v6 (sin TTA+flt) | 0.45 | 0.98 | 0.00 | 0.98 | No |
| **Vilcabamba** (2021-03-27) | `(-73.0589, -13.0938)` (13.9 km) | U-Net L4S | 0.45 | 2.88 | 1.42 | 1.46 | Sí |
| **Vilcabamba** (2021-03-27) | `(-73.0589, -13.0938)` (13.9 km) | U-Net B1 v5 | 0.30 | 1.55 | 1.55 | 0.00 | Sí |
| **Vilcabamba** (2021-03-27) | `(-73.0589, -13.0938)` (13.9 km) | U-Net v6 (sin TTA+flt) | 0.45 | 0.38 | 0.38 | 0.00 | Parcial |

## Láminas por Caso

1. **Caso 1 — Santa Teresa (23/02/2020, Alud Salkantay–Sahuayaco):** `_fichas/estudios_caso/caso_01_santa_teresa_2020.png`
2. **Caso 2 — Echarate (16/02/2021, Huaico en quebrada Echarate):** `_fichas/estudios_caso/caso_02_echarate_2021.png`
3. **Caso 3 — Vilcabamba (27/03/2021, Deslizamiento en ladera):** `_fichas/estudios_caso/caso_03_vilcabamba_2021.png`

## Observaciones Cualitativas (Verificación Visual de las 3 Láminas)

- **U-Net Landslide4Sense (`modelos/unet.onnx`, `u = 0.45`):** sobremarca laderas secas y chacras fuera del evento en los 3 casos (`5.10 ha` de FA en Santa Teresa, **`83.78 ha` de FA** en la ladera marrón seca de Echarate, y `1.46 ha` de FA en Vilcabamba).
- **U-Net B1 v5 (`unet_b1_v5.onnx`, `u = 0.30`, sin filtro):** recorta drásticamente las falsas alarmas frente a Landslide4Sense (`0.15 ha` en Santa Teresa, `3.07 ha` en Echarate —27× menos—, y **`0.00 ha` de FA** en Vilcabamba), cubriendo con claridad la cicatriz de ladera de Vilcabamba (`1.55 ha`) y parcialmente los bordes erosionados del alud de Santa Teresa (`0.61 ha`) y del huaico de Echarate (`0.41 ha`).
- **U-Net v6 sin TTA + filtro `wc_dem_3200_agua_urb` (`unet_v6.onnx`, `u = 0.45`):** entrega los mapas con menor tasa de falsas alarmas (`0.23 ha` en Santa Teresa, `0.98 ha` en Echarate —88× menos que L4S—, y **`0.00 ha`** en Vilcabamba), y expone dos límites reales valiosos para la discusión de tesis:
  1. **Flujos de detritos / huaicos canalizados de brillo alto en fondo de quebrada (Echarate y cauce central de Santa Teresa):** al haberse entrenado para rechazar cauces/ríos brillantes y evitar falsas alarmas fluviales, el modelo tiende a no marcar el fondo blanco del cauce (solo márgenes erosionados en Santa Teresa, `0.48 ha`, y `0.00 ha` en el canal de Echarate).
  2. **Compromiso del filtro altoandino a `≥ 3 200 m` (Vilcabamba, elevación media `3 535 m`):** el filtro `wc_dem_3200_agua_urb` mantiene `0.00 ha` de falsas alarmas, pero recorta parte de la cicatriz (`1.55 → 0.38 ha`) donde ESA WorldCover 2021 clasificó la ladera afectada como pastizal/suelo desnudo.
