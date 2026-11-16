"""
Exporta un checkpoint de la Fase 4 a ONNX, con su receta, y comprueba que el
ONNX resultante está entrenado y da lo mismo que PyTorch.

Corrige el fallo de los ONNX del repositorio, que se exportaron sin cargar el
entrenamiento. Aquí la carga es estricta (cualquier clave que no calce es un
error, no un aviso), se comprueba que los pesos cambiaron al cargar, y el ONNX
pasa la misma prueba forense que hace el backend antes de usarlo.

Uso, en Colab o en local:

  python3 exportar_onnx.py \
      --checkpoint mejor_deeplabv3plus_6ch_23c2a9ca2368.pt \
      --stats stats_norm_23c2a9ca2368.json \
      --resultados resultados_deeplabv3plus_6ch_23c2a9ca2368.json \
      --salida modelos/ --nombre deeplabv3plus \
      --autor "Huarcaya Pumacayo, V. N."

Para la U-Net de Leonardo es lo mismo con sus tres archivos y --nombre unet.
Si el checkpoint no trae la configuración de la Fase 4 (por ejemplo, un
state_dict suelto), se dice a mano. Para las entradas de la especificación
(Sentinel-2 B1 a B4, pendiente B13 y DEM B14 de ALOS PALSAR) y salida de 2 clases:

  python3 exportar_onnx.py --checkpoint unet.pt --salida modelos/ --nombre unet \
      --arquitectura unet --encoder resnet34 --lado 128 \
      --canales B01,B02,B03,B04,B13,B14 --clases 2 \
      --normalizacion media_desv --stats stats_unet.json

--stats acepta {"0": {"lo":..,"hi":..}} (índices de L4S, como la Fase 4),
{"B01": {"media":..,"desv":..}} o {"mean": [...], "std": [...]} en el orden de
--canales. Con --normalizacion ninguna no hace falta.

Con --tta 4 el ONNX lleva dentro el promedio de la imagen y sus tres espejos (lo que
entrenar.py mide como «con TTA») y usa el umbral elegido para eso en validación. El
servidor no cambia: recibe el mismo tensor y devuelve la misma forma, pero cada parche
cuesta 4 veces más de cálculo.
Requiere: torch, segmentation-models-pytorch, onnx, onnxruntime.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'backend'))
sys.path.insert(0, os.path.join(AQUI, '..'))          # repositorio v2: nucleo/ está en la raíz

CANALES_6CH = ['B02', 'B03', 'B04', 'B08', 'NDVI', 'NDWI']
NOMBRES_14 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10',
              'B11', 'B12', 'pendiente', 'elevacion']
IDX_L4S = {n: i for i, n in enumerate(NOMBRES_14)}
ALIAS = {'B13': 'pendiente', 'B14': 'elevacion', 'B1': 'B01', 'B2': 'B02', 'B3': 'B03',
         'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', 'B8': 'B08', 'B9': 'B09'}
# Constantes de Landslide4Sense: pendiente = grados / 20.292, DEM = metros / 772.82
ESCALA_TERRENO_L4S = {'pendiente': 20.292, 'elevacion': 772.82}


def construir(arquitectura, encoder, n_canales, n_clases=1):
    import segmentation_models_pytorch as smp
    clase = {'deeplabv3plus': smp.DeepLabV3Plus, 'unet': smp.Unet}.get(arquitectura)
    if clase is None:
        raise SystemExit(f'arquitectura "{arquitectura}" no reconocida')
    return clase(encoder_name=encoder, encoder_weights=None, in_channels=n_canales,
                 classes=n_clases)


def con_tta(modelo, n):
    """Envuelve el modelo: logits promediados sobre la imagen y sus espejos."""
    import torch

    class ConTTA(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            z = self.m(x)
            if n >= 2:
                z = z + torch.flip(self.m(torch.flip(x, [3])), [3])
            if n >= 4:
                z = z + torch.flip(self.m(torch.flip(x, [2])), [2])
                z = z + torch.flip(self.m(torch.flip(x, [2, 3])), [2, 3])
            return z / n

    return ConTTA(modelo).eval()


def estado_del_checkpoint(ck):
    """El state_dict, venga en el formato de la Fase 4 o suelto."""
    if isinstance(ck, dict):
        for clave in ('modelo', 'model_state_dict', 'state_dict', 'model'):
            if clave in ck and isinstance(ck[clave], dict):
                return ck[clave]
        if ck and all(hasattr(v, 'shape') for v in ck.values()):
            return ck
    raise SystemExit('no encuentro los pesos en el checkpoint (claves: '
                     f'{list(ck)[:8] if isinstance(ck, dict) else type(ck).__name__})')


def constantes(stats, canales, metodo):
    """Constantes de normalización por canal desde los formatos de --stats."""
    out = {}
    for i, c in enumerate(canales):
        if c not in IDX_L4S:
            continue                        # índices como NDVI: no se normalizan
        if metodo == 'percentil':
            k = stats.get(str(IDX_L4S[c])) or stats.get(c)
            if not k or 'lo' not in k:
                raise SystemExit(f'--stats no trae lo y hi de {c}')
            out[c] = {'lo': float(k['lo']), 'hi': float(k['hi'])}
        else:
            if 'mean' in stats and 'std' in stats:
                out[c] = {'media': float(stats['mean'][i]), 'desv': float(stats['std'][i])}
            else:
                k = stats.get(c) or stats.get(str(IDX_L4S[c]))
                if not k or 'media' not in k:
                    raise SystemExit(f'--stats no trae media y desv de {c}')
                out[c] = {'media': float(k['media']), 'desv': float(k['desv'])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--stats')
    ap.add_argument('--resultados')
    ap.add_argument('--salida', required=True)
    ap.add_argument('--nombre', required=True)
    ap.add_argument('--etiqueta')
    ap.add_argument('--autor')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--arquitectura', choices=['unet', 'deeplabv3plus'])
    ap.add_argument('--encoder')
    ap.add_argument('--lado', type=int)
    ap.add_argument('--canales', help='p. ej. B01,B02,B03,B04,B13,B14')
    ap.add_argument('--clases', type=int, choices=[1, 2],
                    help='1: salida sigmoide; 2: softmax (fondo, deslizamiento)')
    ap.add_argument('--normalizacion', choices=['percentil', 'media_desv', 'ninguna'])
    ap.add_argument('--umbral', type=float)
    ap.add_argument('--tta', type=int, choices=[1, 4], default=1,
                    help='4: el ONNX promedia la imagen y sus tres espejos (4 veces más lento)')
    a = ap.parse_args()

    import torch
    from nucleo import modelos as backend_modelos

    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
    arq = a.arquitectura or (ck.get('arquitectura') if isinstance(ck, dict) else None)
    if not arq:
        raise SystemExit('el checkpoint no dice su arquitectura: usa --arquitectura')
    encoder = a.encoder or cfg.get('encoder', 'resnet34')
    lado = int(a.lado or cfg.get('tam_parche', 128))
    if a.canales:
        canales = [ALIAS.get(c.strip(), c.strip()) for c in a.canales.split(',') if c.strip()]
    elif cfg.get('canales'):            # checkpoints de entrenar.py
        canales = [ALIAS.get(c, c) for c in cfg['canales']]
    elif cfg.get('config_canales'):
        canales = CANALES_6CH if cfg['config_canales'] == '6ch' else NOMBRES_14
    else:
        raise SystemExit('el checkpoint no dice sus canales: usa --canales')
    desconocidos = [c for c in canales if c not in IDX_L4S and c not in ('NDVI', 'NDWI')]
    if desconocidos:
        raise SystemExit(f'canales desconocidos: {desconocidos}')
    n = len(canales)
    estado = estado_del_checkpoint(ck)
    if any(k.startswith('module.') for k in estado):
        estado = {k[len('module.'):]: v for k, v in estado.items()}
    cabeza = [v for k, v in estado.items() if 'segmentation_head' in k and v.ndim == 4]
    # versión 4: el modelo calcula dentro dos índices de verdor (entrenar.py --indices-verdor)
    indices = bool((cfg.get('v4') or {}).get('indices_verdor')) or any(k.startswith('base.') for k in estado)
    n_clases = a.clases or (int(cabeza[0].shape[0]) if cabeza else 1)
    print(f'checkpoint : {os.path.basename(a.checkpoint)}')
    print(f'modelo     : {arq}, codificador {encoder}, {n} canales {canales}, {lado}x{lado}, '
          f'{n_clases} clase(s)')
    if isinstance(ck, dict):
        print(f'época {ck.get("epoca")}, IoU de validación {ck.get("iou_val")}, firma {ck.get("firma")}')

    modelo = construir(arq, encoder, n + (2 if indices else 0), n_clases)
    if indices:
        sys.path.insert(0, AQUI)
        from entrenar import modelo_con_indices
        modelo = modelo_con_indices(modelo, canales)
        print('el modelo calcula dentro los índices de verdor GRVI y ExG (entra con los mismos canales)')
    antes = {k: v.clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(estado, strict=True)
    # solo los pesos que se inicializan al azar delatan si la carga ocurrió; los de
    # BatchNorm arrancan en 1 y 0 en cualquier modelo nuevo
    aleatorios = [k for k, v in antes.items() if v.dtype.is_floating_point and v.ndim >= 2]
    cambiadas = sum(1 for k in aleatorios if not torch.equal(modelo.state_dict()[k], antes[k]))
    print(f'carga estricta: {cambiadas} de {len(aleatorios)} pesos aleatorios reemplazados')
    if cambiadas < len(aleatorios):
        raise SystemExit('el checkpoint no reemplazó todos los pesos: no se exporta')
    modelo.eval()
    if a.tta > 1:
        modelo = con_tta(modelo, a.tta)
        print(f'TTA: el ONNX promedia {a.tta} versiones de cada parche')

    os.makedirs(a.salida, exist_ok=True)
    ruta_onnx = os.path.join(a.salida, f'{a.nombre}.onnx')
    x = torch.randn(2, n, lado, lado)
    torch.onnx.export(modelo, x, ruta_onnx, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'N'}, 'output': {0: 'N'}},
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)
    print(f'ONNX escrito: {ruta_onnx} ({os.path.getsize(ruta_onnx)/1e6:.1f} MB)')

    import onnxruntime as ort
    ses = ort.InferenceSession(ruta_onnx, providers=['CPUExecutionProvider'])
    xs = torch.rand(4, n, lado, lado)
    with torch.no_grad():
        y_t = modelo(xs).numpy()
    y_o = ses.run(None, {'input': xs.numpy()})[0]
    dif = float(np.abs(y_t - y_o).max())
    print(f'paridad PyTorch / ONNX: diferencia máxima de logit {dif:.2e}')
    if dif > 1e-3:
        raise SystemExit('el ONNX no reproduce a PyTorch')

    pesos = {k: v for k, v in backend_modelos.revisar_pesos(ruta_onnx).items()
             if not k.startswith('_')}
    print(f'prueba de pesos: {pesos}')
    if pesos['sin_entrenar']:
        raise SystemExit('el ONNX tiene pesos de inicialización: o el checkpoint no se cargó, '
                         'o el modelo se entrenó tan pocos pasos que sus capas finales siguen '
                         'como al inicio (pasa con 2 o 3 épocas de prueba, no con un '
                         'entrenamiento completo)')

    metodo = a.normalizacion or cfg.get('normalizacion') or \
        ('percentil' if cfg.get('config_canales') else None)
    if metodo is None:
        raise SystemExit('di cómo se normalizó al entrenar: --normalizacion')
    if metodo != 'ninguna' and not a.stats:
        raise SystemExit(f'--normalizacion {metodo} necesita --stats')
    stats = json.load(open(a.stats, encoding='utf-8')) if a.stats else {}
    por_canal = constantes(stats, canales, metodo) if metodo != 'ninguna' else {}
    res = json.load(open(a.resultados, encoding='utf-8')) if a.resultados else {}
    if a.tta == 4:
        if 'tta4' not in res:
            raise SystemExit('--tta 4 necesita --resultados de entrenar.py versión 3 (trae el umbral con TTA)')
        res = dict(res, umbral_operacion=res['tta4']['umbral_operacion'],
                   validacion=res['tta4']['validacion'], prueba=res['tta4']['prueba'])
    terreno = [c for c in canales if c in ESCALA_TERRENO_L4S]
    receta = {
        'nombre': a.nombre,
        'etiqueta': a.etiqueta or {'deeplabv3plus': 'DeepLabV3+', 'unet': 'U-Net'}[arq]
                    + f' · {encoder}',
        'autor': a.autor,
        'onnx': os.path.basename(ruta_onnx),
        'canales': canales, 'lado_px': lado,
        'normalizacion': {
            'metodo': metodo, 'percentiles': cfg.get('percentiles', [2, 98]),
            'canales': list(por_canal),
            'por_canal': por_canal,
            'nota': 'constantes en unidades de Landslide4Sense; los índices NDVI y NDWI '
                    'se calculan sobre esas unidades y no se normalizan'},
        'salida': 'sigmoide' if n_clases == 1 else 'softmax',
        'umbral': float(a.umbral or res.get('umbral_operacion', cfg.get('umbral_entren', 0.5))),
        **({'tta': a.tta} if a.tta > 1 else {}),
        **({'indices_internos': 'GRVI y ExG de B03, B04 y B02, calculados dentro del ONNX'} if indices else {}),
        'entrenamiento': {
            'firma': ck.get('firma') if isinstance(ck, dict) else None,
            'arquitectura': arq, 'codificador': encoder,
            'dataset': cfg.get('dataset', 'Landslide4Sense'),
            'epoca': ck.get('epoca') if isinstance(ck, dict) else None,
            **({'promedio_de_epocas': ck['promedio_de_epocas']}
               if isinstance(ck, dict) and ck.get('promedio_de_epocas') else {}),
            'iou_validacion': res.get('validacion', {}).get('iou'),
            'prueba': {k: res.get('prueba', {}).get(k)
                       for k in ('iou', 'f1', 'precision', 'recall')},
            'particion': cfg.get('particion') or
                         ('grupos espaciales de TrainData (Fase 4)' if cfg else None)},
        **({'rango_entrenamiento': {c: {'min': stats[c]['min'], 'max': stats[c]['max']}
                                    for c in canales if isinstance(stats.get(c), dict)
                                    and 'min' in stats[c]}}
           if any(isinstance(stats.get(c), dict) and 'min' in stats[c] for c in canales) else {}),
        **({'escala_terreno': {c: ESCALA_TERRENO_L4S[c] for c in terreno},
            'terreno_entrenamiento': 'ALOS PALSAR (ASF) en Landslide4Sense: B13 pendiente, '
                                     'B14 DEM'} if terreno else {}),
        'exportacion': {'opset': a.opset, 'paridad_max_logit': dif, 'pesos': pesos,
                        'sha256_onnx': hashlib.sha256(open(ruta_onnx, 'rb').read()).hexdigest()},
    }
    ruta_json = os.path.join(a.salida, f'{a.nombre}.json')
    json.dump(receta, open(ruta_json, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'receta escrita: {ruta_json}')


if __name__ == '__main__':
    main()
