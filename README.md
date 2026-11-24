# Deslizamientos en La Convención (Cusco, Perú) con datos peruanos

Conjunto de datos propio, protocolo de marcado y modelos de segmentación de deslizamientos ajustados **solo con datos del Perú** para la provincia de La Convención, Cusco. Imágenes **Sentinel-2 L2A** (10 m) y relieve **NASADEM** (pendiente y elevación).

Tesis de Ingeniería Informática, Universidad Peruana Cayetano Heredia (2026):
*Sistema de segmentación semántica basado en imágenes satelitales Sentinel-2 para la identificación de deslizamientos en La Convención, Cusco.*

- **Autores:** Víctor Nikolai Huarcaya Pumacayo y Leonardo Camilo Mejía López
- **Asesor:** Mg. Roy Marco Yali Samaniego
- **Sistema web:** repositorio [Proyecto_Tesis_Deslizamiento](https://github.com/VictorNikolai/Proyecto_Tesis_Deslizamiento)

---

## Qué contiene

| Parte | Descripción |
| :--- | :--- |
| **Conjunto de La Convención** | 1 224 parches de 128 × 128 px con máscara de referencia en 306 teselas, más 564 parches de la prueba 2 y 16 de la prueba externa |
| **Partición espacial** | Entrenamiento, validación y prueba separados al menos 3 km borde a borde, sin teselas duplicadas |
| **Protocolo de marcado** | Marcado en IRIS a 128 × 128 px, ciego en validación y prueba, con revisión por lotes (`docs/GUIA_IRIS_128.md`) |
| **Ajuste fino** | U-Net y DeepLabV3+ (ResNet-34) ajustados con La Convención y con L4SPe, sin datos extranjeros (`herramientas/ajuste_fino.py`) |
| **Post-procesamiento** | Filtro de ESA WorldCover (urbano y agua; opcional altitud) y filtro de cambio entre fechas (ΔGRVI) |
| **Evaluación** | Prueba sellada evaluada una sola vez, prueba 2 de falsas alarmas, métricas por objeto y estudios de caso de INDECI (`informes/`) |

## Conjunto de datos

| Conjunto | Teselas | Parches | Parches con deslizamiento | Cicatrices ≥ 10 px | Uso |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Entrenamiento | 140 | 560 | 138 | 269 | Ajuste fino |
| Validación | 76 | 304 | 39 | 59 | Época, umbral y post-procesamiento |
| Prueba sellada | 90 | 360 | 38 | 80 | Evaluación final única |
| Prueba 2 | 141 | 564 | 4 (cualitativos) | — | Falsas alarmas en 349 negativos difíciles |
| Prueba externa | 4 | 16 | — | — | Solo referencial |
| **L4SPe** (asesor) | — | 838 | — | — | Referencia peruana marcada por especialista (587 / 127 / 124) |

- Cada parche: imagen GeoTIFF de 6 canales (B01, B02, B03, B04, pendiente, elevación) y máscara con **1** = deslizamiento, **0** = fondo, **255** = dudoso (nube, sombra, cauce, carretera; no cuenta).
- Partición: `particion.csv` (por tesela). Decisiones de marcado por parche: `informes/decisiones_agente.csv` (columna `metodo`: `manual_v4` o `clasificador_v5`).
- Las imágenes y máscaras no se suben al repositorio por su tamaño (paquete de 418 MB, disponible a pedido de los autores).

## Resultados principales

Prueba sellada de La Convención (360 parches, 80 cicatrices) y prueba del especialista (L4SPe, 124 parches):

| Modelo | IoU La Convención | Falsas alarmas (ha/km²) | Cicatrices (de 80) | IoU L4SPe |
| :--- | ---: | ---: | ---: | ---: |
| U-Net base (sin datos peruanos) | 0.011 | 12.47 | 71 | 0.356 |
| **U-Net ajustada (B1)** | **0.131** | **0.22** | 18 | **0.430** |
| DeepLabV3+ ajustada (B2) | 0.019 | 1.43 | 10 | 0.403 |

- La mejora de la U-Net en La Convención tiene un intervalo de confianza de [+0.041, +0.221] (bootstrap por zonas).
- **Versión v6** (U-Net + filtro WorldCover): en 349 parches negativos difíciles nuevos, las falsas alarmas bajan de 0.095 a 0.021 ha/km² (−78 %).
- Detección de cicatrices mayores de 2 ha: 64 % en La Convención y 82 % en L4SPe.

### Límites conocidos

- La referencia de La Convención coincide con la del especialista en un IoU de 0.30 a 0.39: es el techo de lo medible en la provincia.
- Las cicatrices menores de ~0.2 ha (10–19 px) no se marcan de forma consistente a 10 m y casi no se detectan.
- Los modelos ajustados no marcan los huaicos canalizados en el fondo del cauce.
- Las metas iniciales del proyecto (IoU ≥ 0.70) no se alcanzan; ver `informes/metricas_reformuladas.md`.

## Modelos

Las recetas (`.json`) están en `modelos_ajustados/`; los archivos `.onnx` se publican como *Release* del repositorio.

| Modelo | Umbral | Uso recomendado |
| :--- | :---: | :--- |
| `unet_v6` | 0.45 | Principal: sin espejos, con filtro WorldCover |
| `deeplabv3plus_v6` | 0.60 | Alternativa (más falsas alarmas en laderas altoandinas) |
| `unet_b1_v5`, `deeplabv3plus_b2_v5` | 0.30 / 0.40 | Modelos de la evaluación final sellada |

Los modelos parten de una base preentrenada; el ajuste usa solo datos del Perú (La Convención + L4SPe).

## Estructura

```text
herramientas/   descarga, corte de parches, marcado con IRIS, partición, ajuste fino,
                exportación a ONNX, post-procesamiento, evaluación y pruebas (pytest)
nucleo/         configuración, modelos y recetas de normalización
informes/       resultados de cada etapa (CSV, JSON y Markdown)
modelos_ajustados/  recetas y resultados de los modelos ajustados
resultados_ajuste/  historial y métricas de cada corrida
docs/           guía de IRIS y documentación del conjunto L4SPe
```

## Uso rápido

```powershell
pip install torch segmentation-models-pytorch rasterio onnxruntime pystac-client planetary-computer pytest
pytest herramientas/test_herramientas.py -q

# Ajuste fino con datos peruanos (sin datos extranjeros)
python herramientas/ajuste_fino.py --sin-l4s --desde ckpt_base/<modelo>.pt --stats ckpt_base/<stats>.json `
  --parches parches_128 --mascaras mascaras_nivel1/parches_128 --l4spe l4spe_profesor.npz
```

## Cómo citar

Huarcaya Pumacayo, V. N. y Mejía López, L. C. (2026). *Sistema de segmentación semántica basado en imágenes satelitales Sentinel-2 para la identificación de deslizamientos en La Convención, Cusco* [Tesis de pregrado]. Universidad Peruana Cayetano Heredia.

Fuentes de datos: Sentinel-2 (ESA/Copernicus, vía Microsoft Planetary Computer), NASADEM, ESA WorldCover 2021, INGEMMET, INDECI y el conjunto L4SPe del Mg. Roy Marco Yali Samaniego.
