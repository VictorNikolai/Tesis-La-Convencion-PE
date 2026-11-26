# Entrega Final de Mejoras — Resumen Comparativo (`mejora-iou`)

> [!IMPORTANT]
> Toda selección de hiperparámetros, épocas, umbrales y reglas se realizó **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Toda cifra sobre `L4SPe test` (`124`), `La Convención sellada` (`360`) y `prueba2` (`349` negativos confirmados) es **«post hoc»** con **IC 95 % por bootstrap de zonas**.

## 1. Tabla Resumen Comparativa

| Modelo / Versión | Conjunto | Marca | IoU 0 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] | `prueba2` FA (`ha/km²` / `%`) |
|---|---|---|---|---|---|---:|
| **v6** (`U-Net 6b`, sin TTA + `wc_urb_agua`) | `L4SPe test` | post hoc | 0.4229 [0.3684, 0.4794] | 66.7 % [61.6, 72.2] | 79.5 % [70.7, 89.4] | — |
| **v6** (`U-Net 6b`, sin TTA + `wc_urb_agua`) | `La Convención sellada` | post hoc | 0.1425 [0.0496, 0.2761] | 25.0 % [9.5, 39.7] | 72.7 % [27.3, 100.0] | 0.0305 (6.6 %) |
| **v9** (`v6+P3`, TTA4, `E1`, `E4b/E4c`, `hist`, `min20`) | `Val` (`L4SPe` \| `LC`) | validación | `0.3568` \| `0.1633` | — | `85.4 %` (`41/48` en `L4SPe`) | — (`LC val`: `0.0263`) |
| **v9** | `L4SPe test` | post hoc | **0.4298 [0.3753, 0.4837]** | **74.5 % [69.4, 80.0]** | **83.6 % [75.0, 94.1]** | — |
| **v9** | `La Convención sellada` | post hoc | 0.1455 [0.0456, 0.2860] | **34.7 % [13.6, 51.0]** | **72.7 % [27.3, 100.0]** | 0.0157 (2.3 %) |
| **v9 + filtro IR (`NDVI > 0.40`)** | `LC val` | validación | `0.1702` | — | — (`0` perdidas) | — (`LC val`: `0.0158`) |
| **v9 + filtro IR (`NDVI > 0.40`)** *(solo LC)* | `La Convención sellada` | post hoc | **0.1519 [0.0353, 0.3044]** | 30.0 % [10.2, 50.0] | 63.6 % [14.3, 92.3] | **0.0050 (0.9 %)** |
| **9 bandas v2** (`8+8`, dropout IR `0.5`, ep `14`, `u=0.75`) | `Val` (`L4SPe` \| `LC`) | validación | `0.3463` \| `0.2172` | — | — | — |
| **9 bandas v2** *(no usado por sobreajuste)* | `L4SPe test` | post hoc | 0.3849 [0.3239, 0.4463] | 62.4 % [56.0, 68.2] | 72.6 % [62.2, 84.1] | — |
| **9 bandas v2** *(no usado por sobreajuste)* | `La Convención sellada` | post hoc | 0.0367 [0.0000, 0.1905] | 14.0 % [0.0, 29.8] | 27.3 % [0.0, 46.7] | 0.0239 (1.1 %) |

## 2. Recomendación Final

- **Para la tesis (resultados principales y mapas fuera de línea):** usar **`v9`** como modelo principal equilibrado (mejor `F1` por objeto y detección `≥ 2 ha` tanto en `L4SPe test` [`74.5 %` / `83.6 %`] como en `La Convención sellada` [`34.7 %` / `72.7 %`], con `0.0157 ha/km²` en `prueba2`), y reportar **`v9 + filtro infrarrojo (NDVI > 0.40)`** como mejora opcional de alta pureza para La Convención cuando se cuente con `B08` (baja las falsas alarmas en `prueba2` a **`0.0050 ha/km²`, `0.9 %`**, a costa de perder cicatrices parcialmente revegetadas). Descartar ambos modelos de 9 bandas reentrenados (`unet_v10_9b` y `9 bandas v2`).
- **Para el despliegue en servidor (`Proyecto_Tesis_Deslizamiento` en Render, 512 MB RAM):** mantener **`unet_v6` (`6` canales sin TTA + filtro `wc_urb_agua`)** como receta operativa ligera (`< 20 ms/parche`, `< 371 MB` de pico en `3 × 3 km`), ya que no requiere `P3` ni descargar `B08` adicional ni TTA de 4 espejos, y ofrece falsas alarmas bajas (`0.0305 ha/km²` en `prueba2`) con latencia de servidor gratuito.
