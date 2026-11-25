# Tarea 3 — Refinado de Bordes sobre `v9`

> [!IMPORTANT]
> Selección realizada **exclusivamente en validación** (`L4SPe val` 127 + `La Convención val` 304). Los resultados sobre `L4SPe test` (`124`) y `La Convención sellada` (`360`) son **«post hoc»** con bootstrap por zonas (`1 000` réplicas).

## 1. Barrido en Validación

- **Referencia `v9` en validación:** `L4SPe IoU = 0.3568`, `LC IoU = 0.1633`, puntaje `0.7/0.3 = 0.2987`, `FA_LC = 0.0263 ha/km²`.

| Método | L4SPe val IoU | LC val IoU | FA LC (`ha/km²`) | Puntaje `0.7/0.3` | Δ vs `v9` |
|---|---:|---:|---:|---:|---:|
| `crecimiento_2px(u=0.30)` | 0.3574 | 0.1628 | 0.0268 | 0.2990 | `+0.0003` |
| `crecimiento_2px(u=0.35)` | 0.3575 | 0.1626 | 0.0268 | 0.2990 | `+0.0003` |
| `crecimiento_2px(u=0.25)` | 0.3573 | 0.1628 | 0.0270 | 0.2989 | `+0.0002` |
| `crecimiento_2px(u=0.20)` | 0.3525 | 0.1665 | 0.0303 | 0.2967 | `-0.0020` |
| `guided_rgb(r=2,eps=0.05,u=0.35)` | 0.3593 | 0.1604 | 0.0197 | 0.2996 | `+0.0009` |
| `guided_rgb(r=2,eps=0.001,u=0.30)` | 0.3575 | 0.1631 | 0.0237 | 0.2992 | `+0.0005` |
| `guided_rgb(r=2,eps=0.05,u=0.30)` | 0.3565 | 0.1648 | 0.0232 | 0.2990 | `+0.0003` |
| `guided_rgb(r=2,eps=0.001,u=0.35)` | 0.3592 | 0.1583 | 0.0209 | 0.2989 | `+0.0002` |

## 2. Decisión en Validación: **NO SE APLICA** (mejora nula o negativa en validación; diferencia dentro del ruido)
- Mejor candidato en validación: `guided_rgb(r=2,eps=0.05,u=0.35)` (`puntaje 0.7/0.3 = 0.2996`, `Δ = +0.0009`, `FA_LC = 0.0197`).

## 3. Comparación «Post Hoc» frente a `v9` (IC 95 %, `1 000` réplicas)

| Conjunto | `v9` IoU [IC 95 %] | Mejor refinado IoU [IC 95 %] | Δ IoU [IC 95 %] |
|---|---|---|---|
| `L4SPe test` | 0.4298 [0.3753, 0.4837] | 0.4283 [0.3724, 0.4847] | `-0.0015` [-0.0073, +0.0049] |
| `La Convención sellada` | 0.1455 [0.0456, 0.2860] | 0.1587 [0.0517, 0.3025] | `+0.0132` [+0.0015, +0.0304] |
