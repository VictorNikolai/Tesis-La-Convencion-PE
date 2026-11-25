# Resultado Final — Mejora del IoU (`mejora-iou`)

## Tabla Resumen de IoU por Formulación y Modelo

| Formulación / Etapa | Conjunto | Marca | IoU [IC 95 %] | Referencia |
|---|---|---|---|---|
| IoU original (`v6`, `≥ 0.1 ha`, `0 px`) | `L4SPe test` | post hoc | 0.4229 [0.3684, 0.4794] | `iou_reformulado.md` |
| IoU original (`v9`, `≥ 0.1 ha`, `0 px`) | `L4SPe test` | post hoc | 0.4298 [0.3753, 0.4837] | `iou_reformulado.md` |
| IoU original (`v6`, `≥ 0.1 ha`, `0 px`) | `La Convención sellada` | post hoc | 0.1425 [0.0496, 0.2761] | `iou_reformulado.md` |
| IoU original (`v9`, `≥ 0.1 ha`, `0 px`) | `La Convención sellada` | post hoc | 0.1455 [0.0456, 0.2860] | `iou_reformulado.md` |
| IoU reformulado (`v9`, `≥ 0.5 ha` + tol. `2 px`) | `L4SPe test` | post hoc | **0.6085 [0.5336, 0.6799]** | `iou_reformulado.md` |
| IoU reformulado (`v9`, `≥ 0.5 ha` + tol. `2 px`) | `La Convención sellada` | post hoc | 0.1768 [0.0510, 0.3565] | `iou_reformulado.md` |
| IoU con bordes refinados (`guided_rgb` sobre `v9`) | Validación (`L4SPe` \| `LC`) | validación | `0.3593` \| `0.1604` *(no aplicado, `Δ=+0.0009`)* | `refinado_bordes.md` |
| IoU modelo 9 bandas (`unet_v10_9b`, ep `0`, `u=0.40`) | `La Convención val` | validación | `0.1366` | `modelo_9_bandas.md` |
| IoU modelo 9 bandas (`unet_v10_9b`) | `La Convención sellada` | post hoc | 0.1211 [0.0292, 0.2416] | `modelo_9_bandas.md` |

## Conclusión Honesta respecto a la Meta `IoU ≥ 0.70`

- **Bajo la formulación estricta a píxel exacto (`0 px`, todas `≥ 0.1 ha`), ningún modelo alcanza `0.70`:** `v9` llega a **`0.4298`** en `L4SPe test` y **`0.1455`** en `La Convención sellada` (`unet_v10_9b`: `0.1211`). La propia concordancia inter-anotador humana (`0.30–0.39` en Sentinel-2 a `10 m`) fija un techo físico por debajo de `0.70` a píxel exacto.
- **Bajo la formulación reformulada por cicatrices medibles (`≥ 0.5 ha`) y tolerancia de borde (`2 px`), `v9` llega a `0.6085` (IC 95 % `[0.5336, 0.6799]`) y `F1 = 75.7 %` en `L4SPe test`**, quedando a `~0.09` de `0.70` y dentro de `0.02` en el extremo superior del IC.
- **En La Convención sellada**, el diagnóstico (`iou_diagnostico.md`) y la inspección visual (`fa_clasificacion_v9.md`) demuestran que la brecha restante no es de bordes ni de posproceso, sino de **etiquetado inconsistente** (`21.83 ha` de posibles cicatrices reales marcadas como fondo en la referencia) y de **omisiones en sustratos pardo-rojizos**, por lo que el paso siguiente natural es el **remarcado conjunto** usando `docs/GUIA_REMARCADO.md` e `informes/remarcado_laminas.pdf`.
