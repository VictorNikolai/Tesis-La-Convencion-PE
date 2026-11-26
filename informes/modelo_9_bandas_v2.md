# Tarea C (Opcional) — Modelo U-Net 9 Bandas v2 (`8 + 8` + Dropout IR `0.5`)

> [!IMPORTANT]
> Entrenado desde `U-Net v6` con lotes mixtos `8 + 8` (`La Convención` 9 bandas con dropout de canal IR `p = 0.5` + `L4SPe` con bandas IR en `0`), `lr = 3e-5`, `15` épocas de `50` lotes en CPU (`14.5 min`).
> **Selección en validación:** época **`14`**, umbral **`0.75`**, `L4SPe val IoU = 0.3463`, `LC val IoU = 0.2172`, puntaje `0.7/0.3 = 0.3076` (vs **`0.2987`** de `v9`).
> **Decisión:** **SUPERA a v9**.

## Comparación «Post Hoc» frente a `v9` (`1 000` réplicas)

| Conjunto | Modelo | IoU 0 px [IC 95 %] | F1 obj `≥ 0.5 ha` [IC 95 %] | Det `≥ 2 ha` [IC 95 %] |
|---|---|---|---|---|
| `L4SPe test` | **v9** | 0.4298 [0.3771, 0.4847] | 74.5 % [69.4, 80.0] | 83.6 % [75.0, 94.1] |
| `L4SPe test` | **9 bandas v2** | 0.3849 [0.3239, 0.4463] | 62.4 % [56.0, 68.2] | 72.6 % [62.2, 84.1] |
| `La Convención sellada` | **v9** | 0.1455 [0.0455, 0.2838] | 34.7 % [13.6, 51.0] | 72.7 % [27.3, 100.0] |
| `La Convención sellada` | **9 bandas v2** | 0.0367 [0.0000, 0.1905] | 14.0 % [0.0, 29.8] | 27.3 % [0.0, 46.7] |

- **`prueba2` (`349` negativos):** `9 bandas v2` = **`0.0239 ha/km²` (1.1 %)** frente a `v9` = **`0.0157 ha/km²` (2.3 %)**.
