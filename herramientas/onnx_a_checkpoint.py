"""
Reconstruye un punto de control de PyTorch (.pt) a partir del ONNX publicado de la U-Net o la
DeepLabV3+ v4, para poder ajustarla sin el .pt original.

Al exportar en modo evaluación, PyTorch funde cada BatchNorm en la convolución anterior (peso y sesgo nuevos).
Aquí se hace lo inverso: cada Conv2d recibe el peso fundido y su BatchNorm queda como identidad con el sesgo
fundido (gamma 1, media 0, varianza 1 - eps, beta = sesgo). Con BatchNorm fijo (lo normal en ajuste_fino.py,
sin --bn-libre) la red es exactamente la misma; se comprueba contra onnxruntime.

La arquitectura se toma de receta['entrenamiento']['arquitectura'] ('unet' o 'deeplabv3plus'). Para
emparejar cada Conv2d sin sesgo con su BatchNorm fundido no se usa el nombre del módulo (el decodificador
de DeepLabV3+ tiene bloques ASPP y convoluciones separables que no siguen un patrón fijo de nombres), sino
el ORDEN DE EJECUCIÓN real: se registra un gancho en todas las Conv2d y BatchNorm2d, y el BatchNorm que
sigue a una Conv2d sin sesgo en ese orden es, por construcción de la red (conv seguida siempre de su bn en
el forward), el suyo.

Uso:
  python herramientas/onnx_a_checkpoint.py --onnx modelos/unet.onnx --receta modelos/unet.json --salida ckpt/
Deja mejor_<arquitectura>_b1b4b13b14_<firma>.pt y stats_norm_<firma>.json (lo que leen ajuste_fino.py y
exportar_onnx.py).
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
    arq = ent.get('arquitectura', 'unet')
    m = onnx.load(a.onnx)
    ini = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = [n for n in m.graph.node if n.op_type == 'Conv']

    tiene_ind = all(k in ini for k in ('media', 'desv', 'media_ind', 'desv_ind'))
    clase = {'unet': smp.Unet, 'deeplabv3plus': smp.DeepLabV3Plus}[arq]
    base = clase(encoder_name=ent.get('codificador', 'resnet34'), encoder_weights=None,
                 in_channels=len(canales) + (2 if tiene_ind else 0), classes=1)
    if tiene_ind:
        modelo = modelo_con_indices(base, canales)
        for k in ('media', 'desv', 'media_ind', 'desv_ind'):
            getattr(modelo, k).copy_(torch.from_numpy(ini[k].astype(np.float32)).view_as(getattr(modelo, k)))
    else:
        modelo = base

    # orden de ejecución, en PyTorch, de TODAS las Conv2d y BatchNorm2d (en un solo gancho, para poder
    # saber qué módulo se ejecuta justo después de cada Conv2d)
    orden_todo, orden_conv, ganchos = [], [], []
    nombres = {mod: n for n, mod in modelo.named_modules()}
    for mod in modelo.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            ganchos.append(mod.register_forward_hook(lambda mm, i, o: orden_todo.append(mm)))
        if isinstance(mod, torch.nn.Conv2d):
            ganchos.append(mod.register_forward_hook(lambda mm, i, o: orden_conv.append(mm)))
    modelo.eval()
    with torch.no_grad():
        modelo(torch.zeros(1, len(canales), 128, 128))
    for g in ganchos:
        g.remove()
    if len(orden_conv) != len(convs):
        raise SystemExit(f'{len(orden_conv)} convoluciones en PyTorch ({arq}) y {len(convs)} en el ONNX')
    siguiente = {mod: orden_todo[i + 1] for i, mod in enumerate(orden_todo[:-1])}

    def bn_de(mod):
        idx = orden_todo.index(mod)
        if idx + 1 < len(orden_todo) and isinstance(orden_todo[idx + 1], torch.nn.BatchNorm2d):
            return orden_todo[idx + 1]
        if idx + 2 < len(orden_todo) and isinstance(orden_todo[idx + 2], torch.nn.BatchNorm2d):
            return orden_todo[idx + 2]
        return None

    with torch.no_grad():
        for mod, nodo in zip(orden_conv, convs):
            W = ini[nodo.input[1]]
            B = ini[nodo.input[2]] if len(nodo.input) > 2 else np.zeros(W.shape[0], np.float32)
            if tuple(mod.weight.shape) != W.shape:
                raise SystemExit(f'{nombres[mod]}: forma {tuple(mod.weight.shape)} y en el ONNX {W.shape}')
            mod.weight.copy_(torch.from_numpy(W))
            if mod.bias is not None:
                mod.bias.copy_(torch.from_numpy(B))
                continue
            bn = bn_de(mod)
            if bn is None or bn.num_features != B.shape[0]:
                if float(np.abs(B).max()) <= 1e-6:
                    continue
                raise SystemExit(f'{nombres[mod]} no tiene sesgo ni BatchNorm compatible a continuación en el orden de ejecución')
            bn.weight.fill_(1.0); bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - bn.eps); bn.bias.copy_(torch.from_numpy(B))

    # paridad con onnxruntime
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (4, len(canales), 128, 128)).astype(np.float32)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    y_onnx = s.run(None, {s.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        y_pt_logits = modelo(torch.from_numpy(x)).numpy()
    if rec.get('salida') == 'sigmoide' and float(y_onnx.min()) >= 0.0 and float(y_onnx.max()) <= 1.0:
        y_pt_cmp = 1.0 / (1.0 + np.exp(-np.clip(y_pt_logits, -60, 60)))
    else:
        y_pt_cmp = y_pt_logits
    dif = float(np.abs(y_onnx - y_pt_cmp).max())
    print(f'paridad PyTorch vs ONNX: diferencia máxima {dif:.2e} (indices_verdor={tiene_ind})')
    if dif > 1e-3:
        raise SystemExit('la reconstrucción no reproduce el ONNX')

    os.makedirs(a.salida, exist_ok=True)
    por_canal = rec['normalizacion']['por_canal']
    config = {'canales': canales, 'encoder': ent.get('codificador', 'resnet34'),
              'v4': {'indices_verdor': tiene_ind, 'ohem': a.ohem},
              'reconstruido_desde_onnx': os.path.basename(a.onnx), 'paridad_max_logit': dif}
    ruta = os.path.join(a.salida, f'mejor_{arq}_b1b4b13b14_{firma}.pt')
    torch.save({'modelo': modelo.state_dict(), 'arquitectura': arq, 'epoca': ent.get('epoca'),
                'iou_val': ent.get('iou_validacion'), 'firma': firma, 'config': config}, ruta)
    json.dump(por_canal, open(os.path.join(a.salida, f'stats_norm_{firma}.json'), 'w', encoding='utf-8'), indent=1)
    print('escrito', ruta)


if __name__ == '__main__':
    main()
