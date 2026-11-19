"""
Reconstruye un punto de control de PyTorch (.pt) a partir del ONNX publicado de la U-Net v4, para poder
ajustarla sin el .pt original.

Al exportar en modo evaluación, PyTorch funde cada BatchNorm en la convolución anterior (peso y sesgo nuevos).
Aquí se hace lo inverso: cada Conv2d recibe el peso fundido y su BatchNorm queda como identidad con el sesgo
fundido (gamma 1, media 0, varianza 1 - eps, beta = sesgo). Con BatchNorm fijo (lo normal en ajuste_fino.py,
sin --bn-libre) la red es exactamente la misma; se comprueba contra onnxruntime.

Uso:
  python herramientas/onnx_a_checkpoint.py --onnx modelos/unet.onnx --receta modelos/unet.json --salida ckpt/
Deja mejor_unet_b1b4b13b14_<firma>.pt y stats_norm_<firma>.json (lo que leen ajuste_fino.py y exportar_onnx.py).
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--receta', required=True)
    ap.add_argument('--salida', required=True)
    ap.add_argument('--ohem', type=float, default=0.25, help='OHEM con que se entrenó el modelo (va a la config)')
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    import torch
    import segmentation_models_pytorch as smp
    from onnx import numpy_helper
    from entrenar import modelo_con_indices

    rec = json.load(open(a.receta, encoding='utf-8'))
    canales = rec['canales']
    ent = rec.get('entrenamiento', {})
    firma = ent.get('firma', 'onnx')
    m = onnx.load(a.onnx)
    ini = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = [n for n in m.graph.node if n.op_type == 'Conv']

    base = smp.Unet(encoder_name=ent.get('codificador', 'resnet34'), encoder_weights=None,
                    in_channels=len(canales) + 2, classes=1)
    modelo = modelo_con_indices(base, canales)
    for k in ('media', 'desv', 'media_ind', 'desv_ind'):
        getattr(modelo, k).copy_(torch.from_numpy(ini[k].astype(np.float32)).view_as(getattr(modelo, k)))

    # orden de ejecución de las Conv2d en PyTorch
    orden, ganchos = [], []
    nombres = {mod: n for n, mod in modelo.named_modules()}
    for mod in modelo.modules():
        if isinstance(mod, torch.nn.Conv2d):
            ganchos.append(mod.register_forward_hook(lambda mm, i, o: orden.append(mm)))
    modelo.eval()
    with torch.no_grad():
        modelo(torch.zeros(1, len(canales), 128, 128))
    for g in ganchos:
        g.remove()
    if len(orden) != len(convs):
        raise SystemExit(f'{len(orden)} convoluciones en PyTorch y {len(convs)} en el ONNX')
    modulos = dict(modelo.named_modules())

    def bn_de(nombre):
        partes = nombre.split('.')
        ult = partes[-1]
        cand = {'conv1': 'bn1', 'conv2': 'bn2', '0': '1'}.get(ult)
        if cand is None:
            return None
        bn = modulos.get('.'.join(partes[:-1] + [cand]))
        return bn if isinstance(bn, torch.nn.BatchNorm2d) else None

    with torch.no_grad():
        for mod, nodo in zip(orden, convs):
            W = ini[nodo.input[1]]
            B = ini[nodo.input[2]] if len(nodo.input) > 2 else np.zeros(W.shape[0], np.float32)
            if tuple(mod.weight.shape) != W.shape:
                raise SystemExit(f'{nombres[mod]}: forma {tuple(mod.weight.shape)} y en el ONNX {W.shape}')
            mod.weight.copy_(torch.from_numpy(W))
            if mod.bias is not None:
                mod.bias.copy_(torch.from_numpy(B))
                continue
            bn = bn_de(nombres[mod])
            if bn is None:
                raise SystemExit(f'{nombres[mod]} no tiene sesgo ni BatchNorm a continuación')
            bn.weight.fill_(1.0); bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - bn.eps); bn.bias.copy_(torch.from_numpy(B))

    # paridad con onnxruntime
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (4, len(canales), 128, 128)).astype(np.float32)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    y_onnx = s.run(None, {s.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        y_pt = modelo(torch.from_numpy(x)).numpy()
    dif = float(np.abs(y_onnx - y_pt).max())
    print(f'paridad PyTorch vs ONNX: diferencia máxima de logit {dif:.2e}')
    if dif > 1e-3:
        raise SystemExit('la reconstrucción no reproduce el ONNX')

    os.makedirs(a.salida, exist_ok=True)
    por_canal = rec['normalizacion']['por_canal']
    config = {'canales': canales, 'encoder': ent.get('codificador', 'resnet34'),
              'v4': {'indices_verdor': True, 'ohem': a.ohem},
              'reconstruido_desde_onnx': os.path.basename(a.onnx), 'paridad_max_logit': dif}
    ruta = os.path.join(a.salida, f'mejor_unet_b1b4b13b14_{firma}.pt')
    torch.save({'modelo': modelo.state_dict(), 'arquitectura': 'unet', 'epoca': ent.get('epoca'),
                'iou_val': ent.get('iou_validacion'), 'firma': firma, 'config': config}, ruta)
    json.dump(por_canal, open(os.path.join(a.salida, f'stats_norm_{firma}.json'), 'w', encoding='utf-8'), indent=1)
    print('escrito', ruta)


if __name__ == '__main__':
    main()
