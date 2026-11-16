"""
Verifica un ONNX con su receta sobre parches de Landslide4Sense con máscara.

Usa la misma función que el backend (`recetas.desde_l4s` -> `recetas.armar`),
así que lo que se mide aquí es lo que corre en producción, no una copia.

  python3 verificar_onnx.py --modelos ../backend/modelos --l4s /ruta/l4s \
      [--splits validation test] [--lista split_grupos.csv --grupo test] \
      [--escribir]

--l4s apunta a la estructura del dataset de Hugging Face
(ibm-nasa-geospatial/Landslide4sense): images/<split>/image_N.h5 y
annotations/<split>/mask_N.h5.
--escribir guarda el resultado en la receta (campo verificacion_l4s), que el
backend publica en /modelos y en el manifiesto de cada ZIP.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'backend'))
sys.path.insert(0, os.path.join(AQUI, '..'))          # repositorio v2: nucleo/ está en la raíz


def pares(raiz, split, nombres=None):
    imgs = glob.glob(os.path.join(raiz, 'images', split, 'image_*.h5'))
    imgs.sort(key=lambda p: int(p.rsplit('_', 1)[1][:-3]))
    for p in imgs:
        if nombres is not None and os.path.basename(p) not in nombres:
            continue
        q = p.replace(os.sep + 'images' + os.sep, os.sep + 'annotations' + os.sep) \
             .replace('image_', 'mask_')
        if os.path.exists(q):
            yield p, q


def evaluar(ses, receta, lista, lote=16):
    import h5py
    from nucleo import recetas
    tp = fp = fn = tn = 0
    ious = []
    nombre = ses.get_inputs()[0].name
    for i in range(0, len(lista), lote):
        X, Y, V = [], [], []
        for p, q in lista[i:i + lote]:
            with h5py.File(p, 'r') as f:
                t, v = recetas.desde_l4s(f['img'][:], receta)
            with h5py.File(q, 'r') as f:
                y = np.asarray(f['mask'][:]) > 0
            X.append(t); Y.append(y & v); V.append(v)
        prob = recetas.probabilidad(ses.run(None, {nombre: np.stack(X)})[0], receta['salida'])
        for pk, y, v in zip(prob, Y, V):
            m = (pk >= receta['umbral']) & v
            a, b, c = int((m & y).sum()), int((m & ~y).sum()), int((~m & y & v).sum())
            tp += a; fp += b; fn += c; tn += int((~m & ~y & v).sum())
            if y.any():
                ious.append(a / max(a + b + c, 1))
    iou = tp / max(tp + fp + fn, 1)
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return {'parches': len(lista), 'iou': round(iou, 4),
            'f1': round(2 * pr * rc / max(pr + rc, 1e-9), 4),
            'precision': round(pr, 4), 'recall': round(rc, 4),
            'iou_medio_parches_con_deslizamiento': round(float(np.mean(ious)), 4) if ious else None,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelos', required=True)
    ap.add_argument('--l4s', required=True)
    ap.add_argument('--splits', nargs='+', default=['validation', 'test'])
    ap.add_argument('--lista', help='CSV de la Fase 4 con columnas de archivo y split')
    ap.add_argument('--grupo', default='test')
    ap.add_argument('--escribir', action='store_true')
    a = ap.parse_args()

    import onnxruntime as ort
    from nucleo import recetas
    from nucleo.modelos import revisar_pesos

    for ruta_r in sorted(glob.glob(os.path.join(a.modelos, '*.json'))):
        if json.load(open(ruta_r, encoding='utf-8')).get('tipo') == 'ensamble':
            print(f'\n===== {os.path.basename(ruta_r)}: ensamble, sin ONNX propio; se salta')
            continue
        r = recetas.leer(ruta_r)
        onnx = os.path.join(a.modelos, r['onnx'])
        print(f'\n===== {r["nombre"]} ({os.path.basename(onnx)})')
        pesos = {k: v for k, v in revisar_pesos(onnx).items() if not k.startswith('_')}
        print(f'  pesos: {pesos}')
        if pesos['sin_entrenar']:
            print('  SIN ENTRENAR: no tiene sentido medirlo'); continue
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        ses = ort.InferenceSession(onnx, so, providers=['CPUExecutionProvider'])
        resultado = {'fecha': dt.date.today().isoformat(), 'umbral': r['umbral'],
                     'dataset': 'ibm-nasa-geospatial/Landslide4sense (Hugging Face)'}
        if a.lista:
            import pandas as pd
            t = pd.read_csv(a.lista)
            col_split = 'split' if 'split' in t.columns else t.columns[-1]
            col_img = next(c for c in t.columns if 'img' in c.lower() or 'imag' in c.lower())
            sel = t[t[col_split] == a.grupo]
            nombres = {os.path.basename(str(x)) for x in sel[col_img]}
            lista = list(pares(a.l4s, 'train', nombres))
            m = evaluar(ses, r, lista)
            print(f'  {a.grupo} de la Fase 4 ({len(lista)} de {len(nombres)} parches): {m}')
            resultado[f'fase4_{a.grupo}'] = m
        for s in a.splits:
            lista = list(pares(a.l4s, s))
            if not lista:
                continue
            m = evaluar(ses, r, lista)
            print(f'  {s}: {m}')
            resultado[s] = m
        ref = (r.get('entrenamiento') or {}).get('prueba', {})
        if ref.get('iou') is not None:
            print(f'  referencia del entrenamiento, prueba: IoU {ref["iou"]:.4f}, F1 {ref["f1"]:.4f}')
            medidos = [resultado[s]['iou'] for s in a.splits if s in resultado]
            if medidos and max(medidos) < 0.6 * ref['iou']:
                print('  ATENCIÓN: el IoU medido queda muy por debajo del de entrenamiento. '
                      'Revisa que el checkpoint, las estadísticas y la receta sean de la misma '
                      'firma; con un ONNX bien exportado la diferencia es pequeña.')
                resultado['alerta'] = 'IoU muy por debajo del de entrenamiento'
        if a.escribir:
            crudo = json.load(open(ruta_r, encoding='utf-8'))
            crudo['verificacion_l4s'] = resultado
            json.dump(crudo, open(ruta_r, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
            print(f'  escrito en {os.path.basename(ruta_r)}')


if __name__ == '__main__':
    main()
