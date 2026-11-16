"""
Entrena U-Net o DeepLabV3+ con Landslide4Sense, con las entradas que pide el
proyecto: Sentinel-2 B1, B2, B3, B4, pendiente (B13) y DEM (B14).

Qué hace, en orden:

  1. Lee los h5 de Landslide4Sense (128 x 128 x 14) de los tres conjuntos
     oficiales: entrenamiento (3799), validación (245) y prueba (800).
  2. Calcula la media y la desviación de cada canal sobre el entrenamiento.
  3. Arma cada parche con `recetas.desde_l4s`, la MISMA función que usa el
     servidor en Render. Entrenamiento y detección normalizan igual por
     construcción, no por cuidado.
  4. Entrena con BCE + Dice, AdamW, coseno y giros/espejos (como la Fase 4).
     Guarda la época con mejor IoU en validación.
     Opciones para deslizamientos pequeños y zonas verdes (versión 2, apagadas por
     defecto, así que sin ellas el entrenamiento es idéntico al original):
       --peso-tamano    más peso en la pérdida a los deslizamientos pequeños
       --peso-lovasz    suma la pérdida Lovász, que optimiza el IoU directamente
       --copiar-pegar   pega deslizamientos pequeños sobre laderas verdes y empinadas
       --sobremuestreo  muestra más seguido los parches con deslizamientos pequeños
       --brillo         varía brillo y contraste de B1 a B4 (no del relieve)
       --ema            elige y guarda un promedio móvil de los pesos, más estable
       --seleccion iou_max  elige la época por su mejor IoU en cualquier umbral
       --desde          continúa desde un punto de control ya entrenado
     Opciones para un entrenamiento largo (versión 3, también apagadas por defecto):
       --calentamiento  épocas con la tasa de aprendizaje subiendo de a poco al inicio
       --lr-min         tasa mínima al final del coseno (en vez de llegar a cero)
       --recorte-grad   recorta gradientes muy grandes, para que un lote raro no
                        desarme lo aprendido
       --promediar K    guarda las K mejores épocas y al final prueba su promedio de
                        pesos; se queda con el promedio si gana en validación
       --reanudar       guarda el avance cada pocas épocas y, si Colab se corta,
                        sigue desde ahí al volver a correr el mismo comando
     Opciones para bajar las falsas alarmas (versión 4, apagadas por defecto):
       --ohem F         la BCE solo cuenta la fracción F de píxeles con más error
                        (ejemplos difíciles), sobre todo falsas alarmas y bordes
       --indices-verdor calcula dos índices de verdor de B2, B3 y B4 (GRVI y ExG)
                        dentro del modelo; van dentro del ONNX y el servidor sigue
                        enviando los mismos 6 canales
  5. Elige el umbral que maximiza el IoU en validación y recién entonces mide
     una sola vez en prueba. Mide además con TTA (promedio de la imagen y sus
     espejos), que exportar_onnx.py puede meter dentro del ONNX con --tta.
  6. Deja tres archivos con la misma firma, que exportar_onnx.py lee sin
     opciones extra:
        mejor_<arq>_b1b4b13b14_<firma>.pt
        stats_norm_<firma>.json
        resultados_<arq>_b1b4b13b14_<firma>.json

Uso (Colab con GPU):
  python3 entrenar.py --l4s /content/l4s --arquitectura unet --salida /content/drive/MyDrive/entrenamiento
  python3 entrenar.py --l4s /content/l4s --arquitectura deeplabv3plus --salida ...

Requiere: torch, segmentation-models-pytorch, h5py, numpy.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', 'backend'))
sys.path.insert(0, os.path.join(AQUI, '..'))          # repositorio v2: nucleo/ está en la raíz

CANALES_PROYECTO = ['B01', 'B02', 'B03', 'B04', 'pendiente', 'elevacion']
ETIQUETA_CANALES = 'b1b4b13b14'


# ------------------------------------------------------------------- datos
def pares(raiz, split, limite=None):
    """(imagen, máscara) emparejados por nombre: images/<split>/image_12.h5 <->
    annotations/<split>/mask_12.h5. Solo esos nombres: el espejo de Hugging Face
    trae en annotations/train cinco archivos vacíos llamados image_*.h5 que no son
    máscaras. Un par con algún archivo vacío o roto se salta y se avisa."""
    import h5py
    num = lambda f: int(re.search(r'_(\d+)\.h5$', os.path.basename(f)).group(1))
    imgs = {num(f): f for f in glob.glob(os.path.join(raiz, 'images', split, 'image_*.h5'))}
    msks = {num(f): f for f in glob.glob(os.path.join(raiz, 'annotations', split, 'mask_*.h5'))}
    comunes = sorted(set(imgs) & set(msks))
    if not comunes:
        raise SystemExit(f'no hay pares imagen/máscara en {raiz}/images/{split}')
    sin_par = len(set(imgs) ^ set(msks))
    if sin_par:
        print(f'  aviso: {sin_par} archivo(s) de {split} sin su par; no se usan')
    if limite:
        comunes = comunes[:limite]
    buenos, malos = [], []
    for k in comunes:
        try:
            with h5py.File(imgs[k], 'r') as f:
                ok_i = len(f.keys()) > 0 and f[list(f.keys())[0]].shape[-1] == 14
            with h5py.File(msks[k], 'r') as f:
                ok_m = len(f.keys()) > 0 and f[list(f.keys())[0]].size == 128 * 128
        except OSError:
            ok_i = ok_m = False
        (buenos if ok_i and ok_m else malos).append(k)
    if malos:
        print(f'  aviso: {len(malos)} par(es) de {split} con archivos vacíos o rotos; se '
              f'saltan: {[os.path.basename(imgs[k]) for k in malos[:10]]}')
    return [(imgs[k], msks[k]) for k in buenos]


def leer_h5(ruta):
    import h5py
    with h5py.File(ruta, 'r') as f:
        return np.asarray(f[list(f.keys())[0]])


def estadisticas(lista, canales):
    """Media y desviación de cada canal sobre los píxeles válidos del entrenamiento,
    en unidades de Landslide4Sense."""
    from nucleo.recetas import IDX_L4S
    idx = [IDX_L4S[c] for c in canales]
    s = np.zeros(len(idx)); s2 = np.zeros(len(idx)); n = 0
    muestras = [[] for _ in idx]
    for img, _ in lista:
        a = leer_h5(img).astype(np.float64)
        if a.shape[-1] == 14:
            a = a.transpose(2, 0, 1)
        a = a[idx]
        v = np.isfinite(a).all(axis=0)
        x = a[:, v]
        s += x.sum(axis=1); s2 += (x ** 2).sum(axis=1); n += x.shape[1]
        for k in range(len(idx)):
            muestras[k].append(x[k, ::37])
    media = s / n
    desv = np.sqrt(np.maximum(s2 / n - media ** 2, 1e-12))
    # el rango que vio el modelo: el servidor avisa si un área cae fuera de él
    rango = [np.percentile(np.concatenate(m), [0.1, 99.9]) for m in muestras]
    return {c: {'media': round(float(m), 6), 'desv': round(float(d), 6),
                'min': round(float(r[0]), 6), 'max': round(float(r[1]), 6)}
            for c, m, d, r in zip(canales, media, desv, rango)}


def receta_de(canales, stats):
    """La receta con la que se arma el tensor: la misma que irá junto al ONNX."""
    return {'nombre': 'entrenamiento', 'onnx': '-', 'canales': canales, 'lado_px': 128,
            'salida': 'sigmoide', 'umbral': 0.5,
            'normalizacion': {'metodo': 'media_desv',
                              'canales': [c for c in canales if c in stats],
                              'por_canal': stats}}


def cargar(lista, receta):
    """Tensores (N, C, 128, 128) float32, máscaras (N, 128, 128) y válidos."""
    from nucleo import recetas
    X = np.zeros((len(lista), len(receta['canales']), 128, 128), np.float32)
    Y = np.zeros((len(lista), 128, 128), np.float32)
    V = np.zeros((len(lista), 128, 128), np.float32)
    for i, (img, msk) in enumerate(lista):
        t, v = recetas.desde_l4s(leer_h5(img), receta)
        X[i], V[i] = t, v
        Y[i] = (np.asarray(leer_h5(msk)).squeeze() > 0)
    return X, Y, V


# ------------------------------------------------------------------ modelo
def perdida_mixta(logits, y, v, peso_dice=0.5, ohem=0.0):
    import torch.nn.functional as F
    import torch
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    if ohem > 0:
        # ejemplos difíciles: solo cuenta la fracción `ohem` de píxeles válidos con más error
        # (casi siempre falsas alarmas y bordes), en vez del promedio de todos
        l = (bce * v).flatten()
        k = max(int(ohem * float((v > 0).sum())), 1)
        bce = torch.topk(l, min(k, l.numel())).values.mean()
    else:
        bce = (bce * v).sum() / v.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * v
    yt = y * v
    dice = 1.0 - (2.0 * (p * yt).sum() + 1.0) / (p.sum() + yt.sum() + 1.0)
    return (1.0 - peso_dice) * bce + peso_dice * dice


def indices_verdor_np(X, canales, por_canal):
    """GRVI = (G - R) / (G + R) y ExG = (2G - R - B) / (G + R + B), en unidades de
    Landslide4Sense, a partir del tensor ya normalizado. Igual que ConIndices."""
    def crudo(c):
        return X[:, canales.index(c)] * por_canal[c]['desv'] + por_canal[c]['media']
    g, r, b = crudo('B03'), crudo('B04'), crudo('B02')
    grvi = (g - r) / (np.abs(g + r) + 1e-3)
    exg = (2 * g - r - b) / (np.abs(g + r + b) + 1e-3)
    return np.clip(np.stack([grvi, exg], 1), -1, 1)


def modelo_con_indices(base, canales, por_canal=None, med_ind=None, des_ind=None):
    """Envuelve el modelo: recibe los 6 canales de siempre, calcula dentro dos índices
    de verdor de B2, B3 y B4 y le pasa 8 canales al modelo. Va dentro del ONNX, así
    que el servidor no cambia. Sin por_canal (al exportar) las constantes llegan con
    el state_dict."""
    import torch
    sel = [canales.index(c) for c in ('B03', 'B04', 'B02')]

    class ConIndices(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.base = m
            if por_canal:
                me = [por_canal[c]['media'] for c in ('B03', 'B04', 'B02')]
                de = [por_canal[c]['desv'] for c in ('B03', 'B04', 'B02')]
            else:
                me, de = [0.0] * 3, [1.0] * 3
            self.register_buffer('media', torch.tensor(me, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('desv', torch.tensor(de, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('media_ind', torch.tensor(med_ind if med_ind is not None else [0.0, 0.0],
                                                           dtype=torch.float32).view(1, 2, 1, 1))
            self.register_buffer('desv_ind', torch.tensor(des_ind if des_ind is not None else [1.0, 1.0],
                                                          dtype=torch.float32).view(1, 2, 1, 1))

        def forward(self, x):
            rgb = x[:, sel] * self.desv + self.media
            g, r, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            grvi = (g - r) / (torch.abs(g + r) + 1e-3)
            exg = (2 * g - r - b) / (torch.abs(g + r + b) + 1e-3)
            ind = torch.clamp(torch.cat([grvi, exg], 1), -1.0, 1.0)
            ind = (ind - self.media_ind) / self.desv_ind
            return self.base(torch.cat([x, ind], 1))

    return ConIndices(base)


def mapa_pesos_tamano(Y, ref=50.0, maximo=5.0):
    """Peso por píxel: 1 en el fondo y en deslizamientos grandes; hasta `maximo`
    en los pequeños, según sqrt(ref / tamaño). Así perder uno chico cuesta más."""
    from scipy import ndimage
    W = np.ones_like(Y, np.float32)
    for i in range(len(Y)):
        lab, n = ndimage.label(Y[i] > 0)
        if not n:
            continue
        tam = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        w = np.clip(np.sqrt(ref / np.maximum(tam, 1)), 1.0, maximo).astype(np.float32)
        W[i][lab > 0] = w[lab[lab > 0] - 1]
    return W


def objetos(Y, max_px=None):
    """Lista de deslizamientos (componentes conectados): (parche, filas, columnas, tamaño)."""
    from scipy import ndimage
    out = []
    for i in range(len(Y)):
        lab, n = ndimage.label(Y[i] > 0)
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            m = lab[sl] == j
            if max_px is None or m.sum() <= max_px:
                out.append((i, sl, m, int(m.sum())))
    return out


def recall_por_tamano(prob, Y, V, umbral, objs=None):
    """Fracción de deslizamientos tocados por lo marcado, por tamaño (px a 10 m)."""
    objs = objs if objs is not None else objetos(Y)
    tramos = {'<10': (1, 10), '10-50': (10, 50), '50-200': (50, 200), '>200': (200, 10 ** 9)}
    hit = {k: [0, 0] for k in tramos}
    for i, sl, m, t in objs:
        toca = bool(((prob[i][sl] >= umbral) & (V[i][sl] > 0) & m).any())
        for k, (a, b) in tramos.items():
            if a <= t < b:
                hit[k][0] += toca; hit[k][1] += 1
    return {k: round(h / max(n, 1), 4) for k, (h, n) in hit.items()}


class Pegador:
    """Copia-pega: lleva deslizamientos pequeños de un parche a otro, solo en las
    bandas espectrales y solo sobre píxeles empinados y verdes del destino, para que
    el modelo vea más ejemplos de lo que hoy se le escapa. El relieve del destino no
    se toca: el deslizamiento queda donde el terreno lo permite."""
    def __init__(self, X, Y, receta, rng, max_px=50, pend_min=15.0):
        st = receta['normalizacion']['por_canal']; can = receta['canales']
        self.esp = [k for k, c in enumerate(can) if c.startswith('B')]
        self.ip = can.index('pendiente') if 'pendiente' in can else None
        self.ig, self.ir = (can.index('B03') if 'B03' in can else None), (can.index('B04') if 'B04' in can else None)
        self.m = {k: st[c]['media'] for k, c in enumerate(can) if c in st}
        self.s = {k: st[c]['desv'] for k, c in enumerate(can) if c in st}
        self.X, self.Y, self.rng, self.pend_min = X, Y, rng, pend_min
        self.objs = objetos(Y, max_px)

    def crudo(self, x, k):
        return x[k] * self.s[k] + self.m[k]

    def aplicar(self, x, y, w, n=2):
        if not self.objs:
            return x, y, w
        apto = y < 0.5
        if self.ip is not None:
            apto &= self.crudo(x, self.ip) * 20.292 >= self.pend_min
        if self.ig is not None and self.ir is not None:
            apto &= self.crudo(x, self.ig) >= self.crudo(x, self.ir)
        cand = np.argwhere(apto)
        if len(cand) == 0:
            return x, y, w
        for _ in range(n):
            i, sl, m, t = self.objs[int(self.rng.integers(len(self.objs)))]
            h, ww = m.shape
            dst = None
            for _intento in range(20):
                # el centro del deslizamiento cae en un píxel apto, y todo él tiene que
                # quedar sobre píxeles aptos (empinados, verdes y sin otro deslizamiento)
                fc, cc = cand[int(self.rng.integers(len(cand)))]
                f0 = int(np.clip(fc - h // 2, 0, 128 - h)); c0 = int(np.clip(cc - ww // 2, 0, 128 - ww))
                prueba = (slice(f0, f0 + h), slice(c0, c0 + ww))
                if apto[prueba][m].all():
                    dst = prueba
                    break
            if dst is None:
                continue
            apto[dst][m] = False
            for k in self.esp:
                x[k][dst][m] = self.X[i, k][sl][m]
            y[dst][m] = 1.0
            w[dst][m] = np.clip(np.sqrt(50.0 / max(t, 1)), 1.0, 5.0)
        return x, y, w


class Conteo:
    """tp, fp, fn, tn globales (sobre píxeles) para varios umbrales a la vez."""
    def __init__(self, umbrales):
        self.u = np.asarray(umbrales, np.float32)
        self.c = np.zeros((len(self.u), 4), np.int64)

    def sumar(self, prob, y, v):
        ok = v > 0
        p, t = prob[ok], y[ok] > 0
        for i, u in enumerate(self.u):
            m = p >= u
            self.c[i] += [np.sum(m & t), np.sum(m & ~t), np.sum(~m & t), np.sum(~m & ~t)]

    def metricas(self, i):
        tp, fp, fn, tn = (int(x) for x in self.c[i])
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return {'umbral': round(float(self.u[i]), 2), 'iou': round(tp / max(tp + fp + fn, 1), 4),
                'f1': round(2 * pr * rc / max(pr + rc, 1e-9), 4), 'precision': round(pr, 4),
                'recall': round(rc, 4), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def logits_tta(modelo, x, tta=1):
    """Logits promediados sobre la imagen y sus espejos: 1 = sin TTA, 2 = más espejo
    horizontal, 4 = más espejo vertical y ambos. Igual que el envoltorio del ONNX."""
    import torch
    vueltas = [(), (-1,), (-2,), (-2, -1)][:tta]
    suma = None
    for d in vueltas:
        z = modelo(torch.flip(x, d) if d else x)
        z = torch.flip(z, d) if d else z
        suma = z if suma is None else suma + z
    return suma / len(vueltas)


def predecir(modelo, X, disp, lote=32, tta=1):
    import torch
    modelo.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(logits_tta(modelo, torch.from_numpy(X[i:i + lote]).to(disp), tta)[:, 0])
                               .float().cpu().numpy() for i in range(0, len(X), lote)])


def evaluar(modelo, X, Y, V, umbrales, disp, lote=32, tta=1):
    cont = Conteo(umbrales)
    for i in range(0, len(X), lote):
        prob = predecir(modelo, X[i:i + lote], disp, lote, tta)
        cont.sumar(prob, Y[i:i + lote], V[i:i + lote])
    return cont


def factor_lr(ep, total, calentamiento, lr_min_rel):
    """Multiplicador de la tasa de aprendizaje en la época ep (desde 0): sube en línea
    recta durante el calentamiento y después baja en coseno hasta lr_min_rel."""
    import math
    if ep < calentamiento:
        return (ep + 1) / calentamiento
    t = (ep - calentamiento) / max(total - calentamiento, 1)
    return lr_min_rel + (1.0 - lr_min_rel) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def promedio_de_pesos(estados):
    """Promedio simple de varios state_dict (los enteros, como num_batches_tracked,
    se toman del primero)."""
    out = {}
    for k, v in estados[0].items():
        if v.dtype.is_floating_point:
            out[k] = sum(e[k].float() for e in estados) / len(estados)
            out[k] = out[k].to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def recalcular_bn(modelo, X, disp, lote=32):
    """Las medias y varianzas de BatchNorm de un promedio de pesos no son las de
    ninguna época: se vuelven a medir pasando el entrenamiento (sin aumentos)."""
    import torch
    bns = [m for m in modelo.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    guardado = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None        # promedio acumulado exacto
    modelo.train()
    with torch.no_grad():
        for i in range(0, len(X) - 1, lote):
            if len(X[i:i + lote]) < 2:
                continue
            modelo(torch.from_numpy(X[i:i + lote]).to(disp))
    for m, mo in zip(bns, guardado):
        m.momentum = mo
    modelo.eval()


def guardar_seguro(obj, ruta):
    """Escribe primero a un temporal y lo renombra: si Colab se corta a mitad, el
    archivo anterior sigue entero."""
    import torch
    tmp = ruta + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--l4s', required=True, help='carpeta con images/ y annotations/')
    ap.add_argument('--arquitectura', required=True, choices=['unet', 'deeplabv3plus'])
    ap.add_argument('--salida', required=True)
    ap.add_argument('--canales', default=','.join(CANALES_PROYECTO))
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--encoder-weights', dest='encoder_weights', default='imagenet',
                    help="'imagenet' o 'ninguno'")
    ap.add_argument('--epocas', type=int, default=100)
    ap.add_argument('--paciencia', type=int, default=30)
    ap.add_argument('--lote', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', dest='wd', type=float, default=1e-4)
    ap.add_argument('--peso-dice', dest='peso_dice', type=float, default=0.5)
    ap.add_argument('--semilla', type=int, default=42)
    ap.add_argument('--limite', type=int, help='solo para probar: parches por conjunto')
    # versión 2: deslizamientos pequeños y zonas verdes (todo apagado por defecto)
    ap.add_argument('--peso-tamano', dest='peso_tamano', type=float, default=0.0,
                    help='peso máximo para deslizamientos pequeños (0 = apagado; sugerido 5)')
    ap.add_argument('--peso-lovasz', dest='peso_lovasz', type=float, default=0.0,
                    help='peso de la pérdida Lovász que se suma (0 = apagado; sugerido 0.5)')
    ap.add_argument('--copiar-pegar', dest='copiar_pegar', type=float, default=0.0,
                    help='probabilidad de pegar deslizamientos pequeños en cada parche (sugerido 0.3)')
    ap.add_argument('--sobremuestreo', type=float, default=1.0,
                    help='cuántas veces más se muestran los parches con deslizamientos pequeños')
    ap.add_argument('--brillo', type=float, default=0.0,
                    help='variación máxima de brillo y contraste en B1 a B4 (sugerido 0.1)')
    ap.add_argument('--ema', type=float, default=0.0, help='decaimiento del promedio de pesos (sugerido 0.999)')
    ap.add_argument('--seleccion', choices=['iou05', 'iou_max'], default='iou05')
    ap.add_argument('--desde', help='punto de control .pt de entrenar.py para continuar desde él')
    # versión 3: entrenamiento largo (todo apagado por defecto)
    ap.add_argument('--calentamiento', type=int, default=0, help='épocas de calentamiento (sugerido 5)')
    ap.add_argument('--lr-min', dest='lr_min', type=float, default=0.0, help='tasa mínima al final (sugerido 1e-6)')
    ap.add_argument('--recorte-grad', dest='recorte_grad', type=float, default=0.0,
                    help='norma máxima del gradiente (0 = apagado; sugerido 1.0)')
    ap.add_argument('--promediar', type=int, default=0,
                    help='promedia los pesos de las K mejores épocas (0 = apagado; sugerido 5)')
    ap.add_argument('--reanudar', action='store_true',
                    help='guarda el avance y, si existe, continúa desde él')
    ap.add_argument('--guardar-cada', dest='guardar_cada', type=int, default=5,
                    help='con --reanudar, cada cuántas épocas se guarda el avance')
    # versión 4: menos falsas alarmas y mejor en zonas verdes (apagado por defecto)
    ap.add_argument('--ohem', type=float, default=0.0,
                    help='fracción de píxeles más difíciles que cuenta en la BCE (0 = todos; sugerido 0.25)')
    ap.add_argument('--indices-verdor', dest='indices_verdor', action='store_true',
                    help='calcula GRVI y ExG de B2, B3 y B4 dentro del modelo (van en el ONNX)')
    a = ap.parse_args()

    import torch
    import segmentation_models_pytorch as smp
    from nucleo.recetas import ALIAS, IDX_L4S

    canales = [ALIAS.get(c.strip(), c.strip()) for c in a.canales.split(',') if c.strip()]
    malos = [c for c in canales if c not in IDX_L4S]
    if malos:
        raise SystemExit(f'canales que no están en Landslide4Sense: {malos}')
    torch.manual_seed(a.semilla); np.random.seed(a.semilla)
    rng = np.random.default_rng(a.semilla)
    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = disp == 'cuda'
    os.makedirs(a.salida, exist_ok=True)

    config = {'dataset': 'Landslide4Sense (ibm-nasa-geospatial/Landslide4sense, conjuntos oficiales)',
              'canales': canales, 'normalizacion': 'media_desv', 'encoder': a.encoder,
              'encoder_weights': a.encoder_weights, 'tam_parche': 128, 'lote': a.lote,
              'epocas': a.epocas, 'paciencia': a.paciencia, 'lr': a.lr, 'weight_decay': a.wd,
              'peso_dice': a.peso_dice, 'semilla': a.semilla,
              'particion': 'oficial: entrenamiento, validación (época y umbral), prueba (una vez)',
              'aumentos': 'giros de 90° y espejo horizontal', 'limite': a.limite}
    v2 = {k: getattr(a, k) for k, d in (('peso_tamano', 0.0), ('peso_lovasz', 0.0), ('copiar_pegar', 0.0),
                                         ('sobremuestreo', 1.0), ('brillo', 0.0), ('ema', 0.0),
                                         ('seleccion', 'iou05'), ('desde', None)) if getattr(a, k) != d}
    if v2:
        config['v2'] = dict(v2, desde=os.path.basename(a.desde) if a.desde else None)
    v3 = {k: getattr(a, k) for k, d in (('calentamiento', 0), ('lr_min', 0.0), ('recorte_grad', 0.0),
                                         ('promediar', 0)) if getattr(a, k) != d}
    if v3:
        config['v3'] = v3
    v4 = {k: getattr(a, k) for k, d in (('ohem', 0.0), ('indices_verdor', False)) if getattr(a, k) != d}
    if v4:
        config['v4'] = v4
    if a.indices_verdor and not all(c in canales for c in ('B02', 'B03', 'B04')):
        raise SystemExit('--indices-verdor necesita B02, B03 y B04 entre los canales')
    if a.indices_verdor and a.desde:
        raise SystemExit('--indices-verdor cambia la entrada del modelo: no se puede continuar con --desde')
    firma = hashlib.sha1(json.dumps(dict({k: config[k] for k in config if k != 'epocas'},
                                         arquitectura=a.arquitectura),
                                    sort_keys=True).encode()).hexdigest()[:12]
    base = f'{a.arquitectura}_{ETIQUETA_CANALES}_{firma}'
    print(f'firma {firma} | {a.arquitectura} {a.encoder} | canales {canales} | {disp}')
    ruta_res = os.path.join(a.salida, f'resultados_{base}.json')
    ruta_ultimo = os.path.join(a.salida, f'ultimo_{base}.pt')
    if a.reanudar and os.path.exists(ruta_res) and not os.path.exists(ruta_ultimo):
        print(f'este entrenamiento ya terminó antes ({os.path.basename(ruta_res)}): no se repite')
        return

    t0 = time.time()
    tr, va, te = (pares(a.l4s, s, a.limite) for s in ('train', 'validation', 'test'))
    print(f'parches: entrenamiento {len(tr)}, validación {len(va)}, prueba {len(te)}')
    stats = estadisticas(tr, canales)
    ruta_stats = os.path.join(a.salida, f'stats_norm_{firma}.json')
    json.dump(stats, open(ruta_stats, 'w', encoding='utf-8'), indent=1)
    print('media y desviación del entrenamiento:', json.dumps(stats))
    if 'elevacion' in stats:
        print(f"rango del DEM en el entrenamiento: {stats['elevacion']['min']*772.82:.0f} a "
              f"{stats['elevacion']['max']*772.82:.0f} m; pendiente hasta "
              f"{stats.get('pendiente', {}).get('max', 0)*20.292:.0f}°")
    receta = receta_de(canales, stats)
    Xtr, Ytr, Vtr = cargar(tr, receta)
    Xva, Yva, Vva = cargar(va, receta)
    Xte, Yte, Vte = cargar(te, receta)
    print(f'datos en memoria en {time.time()-t0:.0f} s; píxeles de deslizamiento en '
          f'entrenamiento: {Ytr.mean()*100:.2f} %')

    Wtr = mapa_pesos_tamano(Ytr, maximo=a.peso_tamano) if a.peso_tamano > 1 else None
    objs_va = objetos(Yva)
    p_muestra = None
    if a.sobremuestreo > 1:
        chicos = {i for i, _, _, t in objetos(Ytr, 50)}
        p_muestra = np.array([a.sobremuestreo if i in chicos else 1.0 for i in range(len(Xtr))])
        p_muestra /= p_muestra.sum()
        print(f'sobremuestreo: {len(chicos)} parches con deslizamientos pequeños, x{a.sobremuestreo:g}')
    pegador = Pegador(Xtr, Ytr, receta, rng) if a.copiar_pegar > 0 else None
    if pegador:
        print(f'copiar-pegar: {len(pegador.objs)} deslizamientos pequeños disponibles')
    esp = [k for k, c in enumerate(canales) if c.startswith('B')]
    lovasz = None
    if a.peso_lovasz > 0:
        lovasz = smp.losses.LovaszLoss('binary', per_image=False, ignore_index=2, from_logits=True)

    pesos = None if a.encoder_weights in ('ninguno', 'none', '') else a.encoder_weights
    clase = {'unet': smp.Unet, 'deeplabv3plus': smp.DeepLabV3Plus}[a.arquitectura]
    extra = 2 if a.indices_verdor else 0
    modelo = clase(encoder_name=a.encoder, encoder_weights=pesos, in_channels=len(canales) + extra,
                   classes=1)
    if a.indices_verdor:
        pc = receta['normalizacion']['por_canal']
        ind = indices_verdor_np(Xtr[::5], canales, pc)
        med_ind = [float(ind[:, k].mean()) for k in range(2)]
        des_ind = [float(max(ind[:, k].std(), 1e-3)) for k in range(2)]
        modelo = modelo_con_indices(modelo, canales, pc, med_ind, des_ind)
        print(f'índices de verdor dentro del modelo: GRVI media {med_ind[0]:.3f} desv {des_ind[0]:.3f}, '
              f'ExG media {med_ind[1]:.3f} desv {des_ind[1]:.3f}')
    modelo = modelo.to(disp)
    if a.desde:
        ck0 = torch.load(a.desde, map_location=disp, weights_only=False)
        if ck0.get('arquitectura') not in (None, a.arquitectura):
            raise SystemExit(f"--desde es de {ck0.get('arquitectura')}, no de {a.arquitectura}")
        c0 = (ck0.get('config') or {}).get('canales')
        if c0 and list(c0) != canales:
            raise SystemExit(f'--desde usa los canales {c0}, no {canales}')
        modelo.load_state_dict(ck0['modelo'], strict=True)
        print(f"continúa desde {os.path.basename(a.desde)} (época {ck0.get('epoca')}, IoU val {ck0.get('iou_val')})")
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(modelo).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.calentamiento > 0 or a.lr_min > 0:
        rel = a.lr_min / a.lr
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: factor_lr(e, a.epocas, a.calentamiento, rel))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas)
    escalador = torch.amp.GradScaler('cuda', enabled=amp)
    ruta_ckpt = os.path.join(a.salida, f'mejor_{base}.pt')
    carpeta_top = os.path.join(a.salida, f'mejores_epocas_{base}')
    top = []                                   # [(iou_val, época, ruta)] de las K mejores
    historial = []
    mejor, mejor_ep, sin_mejora = -1.0, 0, 0
    ep_inicial = 1
    if a.reanudar and os.path.exists(ruta_ultimo):
        u = torch.load(ruta_ultimo, map_location=disp, weights_only=False)
        if u.get('firma') != firma:
            raise SystemExit(f'{os.path.basename(ruta_ultimo)} es de otra configuración')
        modelo.load_state_dict(u['modelo'])
        if ema is not None and u.get('ema') is not None:
            ema.load_state_dict(u['ema'])
        opt.load_state_dict(u['opt']); sched.load_state_dict(u['sched']); escalador.load_state_dict(u['escalador'])
        rng.bit_generator.state = u['rng']
        torch.set_rng_state(u['torch_rng'].cpu())
        if disp == 'cuda' and u.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(u['cuda_rng'].cpu())
        historial, top = u['historial'], [tuple(t) for t in u['top']]
        mejor, mejor_ep, sin_mejora = u['mejor'], u['mejor_ep'], u['sin_mejora']
        ep_inicial = u['epoca'] + 1
        print(f'retoma desde la época {u["epoca"]} (mejor hasta ahora: época {mejor_ep}, IoU val {mejor:.4f})')

    def guardar_avance(ep):
        guardar_seguro({'firma': firma, 'epoca': ep, 'modelo': modelo.state_dict(),
                        'ema': ema.state_dict() if ema is not None else None,
                        'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'escalador': escalador.state_dict(), 'rng': rng.bit_generator.state,
                        'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state() if disp == 'cuda' else None,
                        'historial': historial, 'top': top, 'mejor': mejor, 'mejor_ep': mejor_ep,
                        'sin_mejora': sin_mejora}, ruta_ultimo)

    for ep in range(ep_inicial, a.epocas + 1):
        modelo.train(); perdidas = []
        orden = rng.choice(len(Xtr), len(Xtr), p=p_muestra) if p_muestra is not None else rng.permutation(len(Xtr))
        te0 = time.time()
        for i in range(0, len(orden) - 1, a.lote):
            j = orden[i:i + a.lote]
            if len(j) < 2:
                continue                         # BatchNorm necesita al menos 2
            x, y, v = Xtr[j].copy(), Ytr[j].copy(), Vtr[j]
            w = Wtr[j].copy() if Wtr is not None else np.ones_like(y)
            if pegador is not None or a.brillo > 0:
                for b in range(len(j)):
                    if pegador is not None and rng.random() < a.copiar_pegar:
                        x[b], y[b], w[b] = pegador.aplicar(x[b], y[b], w[b])
                    if a.brillo > 0:
                        for kk in esp:
                            f = 1.0 + rng.uniform(-a.brillo, a.brillo)
                            mu = receta['normalizacion']['por_canal'][canales[kk]]['media']
                            sd = receta['normalizacion']['por_canal'][canales[kk]]['desv']
                            x[b, kk] = x[b, kk] * f + mu * (f - 1.0) / sd
            k = int(rng.integers(0, 4)); espejo = bool(rng.integers(0, 2))
            x, y, v, w = (np.rot90(m, k, axes=(-2, -1)) for m in (x, y, v, w))
            if espejo:
                x, y, v, w = x[..., ::-1], y[..., ::-1], v[..., ::-1], w[..., ::-1]
            x, y, v, w = (torch.from_numpy(np.ascontiguousarray(m)).to(disp) for m in (x, y, v, w))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=amp):
                logits = modelo(x)[:, 0]
            loss = perdida_mixta(logits.float(), y, v * w if a.peso_tamano > 1 else v, a.peso_dice, a.ohem)
            if lovasz is not None:
                loss = loss + a.peso_lovasz * lovasz(logits.float(), torch.where(v > 0, y, torch.full_like(y, 2.0)))
            escalador.scale(loss).backward()
            if a.recorte_grad > 0:
                escalador.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), a.recorte_grad)
            escalador.step(opt); escalador.update()
            if ema is not None:
                with torch.no_grad():
                    for qe, q in zip(ema.parameters(), modelo.parameters()):
                        qe.mul_(a.ema).add_(q.detach(), alpha=1.0 - a.ema)
                    for be, bq in zip(ema.buffers(), modelo.buffers()):
                        be.copy_(bq)
            perdidas.append(float(loss.detach()))
        sched.step()
        evaluado = ema if ema is not None else modelo
        if a.seleccion == 'iou_max':
            cv_ep = evaluar(evaluado, Xva, Yva, Vva, np.round(np.arange(0.05, 0.96, 0.05), 2), disp)
            m = max((cv_ep.metricas(i) for i in range(len(cv_ep.u))), key=lambda r: r['iou'])
        else:
            m = evaluar(evaluado, Xva, Yva, Vva, [0.5], disp).metricas(0)
        historial.append({'epoca': ep, 'perdida': round(float(np.mean(perdidas)), 5),
                          'lr': opt.param_groups[0]['lr'], **{k: m[k] for k in ('iou', 'f1', 'precision', 'recall')},
                          'segundos': round(time.time() - te0, 1)})
        marca = ''
        if m['iou'] > mejor:
            mejor, mejor_ep, sin_mejora = m['iou'], ep, 0
            torch.save({'modelo': evaluado.state_dict(), 'arquitectura': a.arquitectura,
                        'epoca': ep, 'iou_val': m['iou'], 'firma': firma, 'config': config},
                       ruta_ckpt)
            marca = '  <- mejor, guardado'
        else:
            sin_mejora += 1
        if a.promediar > 0 and (len(top) < a.promediar or m['iou'] > min(t[0] for t in top)):
            os.makedirs(carpeta_top, exist_ok=True)
            ruta_ep = os.path.join(carpeta_top, f'epoca_{ep:04d}.pt')
            guardar_seguro({k: v.detach().cpu().clone() for k, v in evaluado.state_dict().items()}, ruta_ep)
            top.append((m['iou'], ep, ruta_ep))
            top.sort(key=lambda t: -t[0])
            for _, _, r in top[a.promediar:]:
                if os.path.exists(r):
                    os.remove(r)
            top = top[:a.promediar]
        chicos = ''
        if a.peso_tamano > 1 or a.copiar_pegar > 0 or a.sobremuestreo > 1:
            evaluado.eval()
            with torch.no_grad():
                pv = np.concatenate([torch.sigmoid(evaluado(torch.from_numpy(Xva[i:i + 32]).to(disp))[:, 0]).float().cpu().numpy()
                                     for i in range(0, len(Xva), 32)])
            r = recall_por_tamano(pv, Yva, Vva, m['umbral'], objs_va)
            chicos = f" | pequeños <0.1 ha {r['<10']*100:.0f} %, 0.1-0.5 ha {r['10-50']*100:.0f} %"
        print(f"época {ep:3d} | pérdida {np.mean(perdidas):.4f} | IoU val {m['iou']:.4f} | "
              f"F1 {m['f1']:.4f}{chicos} | lr {opt.param_groups[0]['lr']:.1e} | {time.time()-te0:.0f} s{marca}",
              flush=True)
        if a.reanudar and (ep % max(a.guardar_cada, 1) == 0 or ep == a.epocas):
            guardar_avance(ep)
        if sin_mejora >= a.paciencia:
            print(f'sin mejora en {a.paciencia} épocas: se detiene'); break
    with open(os.path.join(a.salida, f'historial_{base}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0])); w.writeheader(); w.writerows(historial)

    # el mejor modelo, el umbral elegido en validación y una sola medición en prueba
    umbrales = np.round(np.arange(0.05, 0.96, 0.05), 2)
    ck = torch.load(ruta_ckpt, map_location=disp, weights_only=False)
    modelo.load_state_dict(ck['modelo'])
    cv = evaluar(modelo, Xva, Yva, Vva, umbrales, disp)
    barrido = [cv.metricas(i) for i in range(len(umbrales))]
    elegido = max(barrido, key=lambda r: r['iou'])
    promedio = None
    if a.promediar > 1 and len(top) > 1:
        # promedio de pesos de las K mejores épocas, con BatchNorm medido de nuevo;
        # se queda solo si en validación supera a la mejor época sola
        estados = [torch.load(r, map_location='cpu', weights_only=False) for _, _, r in top]
        modelo.load_state_dict(promedio_de_pesos(estados))
        modelo.to(disp)
        recalcular_bn(modelo, Xtr, disp)
        cvp = evaluar(modelo, Xva, Yva, Vva, umbrales, disp)
        barrido_p = [cvp.metricas(i) for i in range(len(umbrales))]
        elegido_p = max(barrido_p, key=lambda r: r['iou'])
        eps = sorted(t[1] for t in top)
        print(f"\npromedio de las épocas {eps}: IoU val {elegido_p['iou']:.4f} "
              f"(la mejor época sola: {elegido['iou']:.4f})")
        promedio = {'epocas': eps, 'iou_val': elegido_p['iou'], 'iou_val_mejor_epoca': elegido['iou'],
                    'elegido': elegido_p['iou'] > elegido['iou']}
        if promedio['elegido']:
            print('  -> se usa el promedio')
            barrido, elegido = barrido_p, elegido_p
            guardar_seguro({'modelo': modelo.state_dict(), 'arquitectura': a.arquitectura,
                            'epoca': mejor_ep, 'promedio_de_epocas': eps, 'iou_val': elegido_p['iou'],
                            'firma': firma, 'config': config}, ruta_ckpt)
        else:
            print('  -> se usa la mejor época sola')
            modelo.load_state_dict(ck['modelo'])
    prueba = evaluar(modelo, Xte, Yte, Vte, [elegido['umbral']], disp).metricas(0)
    pt = predecir(modelo, Xte, disp)
    por_tamano = recall_por_tamano(pt, Yte, Vte, elegido['umbral'])
    # con TTA (imagen y sus tres espejos): umbral propio, elegido también en validación
    cvt = evaluar(modelo, Xva, Yva, Vva, umbrales, disp, tta=4)
    elegido_t = max((cvt.metricas(i) for i in range(len(umbrales))), key=lambda r: r['iou'])
    prueba_t = evaluar(modelo, Xte, Yte, Vte, [elegido_t['umbral']], disp, tta=4).metricas(0)
    por_tamano_t = recall_por_tamano(predecir(modelo, Xte, disp, tta=4), Yte, Vte, elegido_t['umbral'])
    resultados = {'firma': firma, 'arquitectura': a.arquitectura, 'mejor_epoca': mejor_ep,
                  'umbral_operacion': elegido['umbral'], 'validacion': elegido,
                  'prueba': prueba, 'prueba_deslizamientos_detectados_por_tamano_px': por_tamano,
                  'barrido_validacion': barrido, 'config': config,
                  'epocas_corridas': len(historial),
                  **({'promedio_de_pesos': promedio} if promedio else {}),
                  'tta4': {'umbral_operacion': elegido_t['umbral'], 'validacion': elegido_t,
                           'prueba': prueba_t, 'prueba_deslizamientos_detectados_por_tamano_px': por_tamano_t},
                  'minutos': round((time.time() - t0) / 60, 1)}
    json.dump(resultados, open(ruta_res, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"\nmejor época {mejor_ep}; umbral {elegido['umbral']} (elegido en validación)")
    print(f"validación: IoU {elegido['iou']:.4f}  F1 {elegido['f1']:.4f}")
    print(f"prueba    : IoU {prueba['iou']:.4f}  F1 {prueba['f1']:.4f}  "
          f"precisión {prueba['precision']:.4f}  exhaustividad {prueba['recall']:.4f}")
    print('deslizamientos detectados en prueba por tamaño (px a 10 m):', por_tamano)
    print(f"con TTA (umbral {elegido_t['umbral']}): IoU val {elegido_t['iou']:.4f} | prueba IoU {prueba_t['iou']:.4f}  "
          f"F1 {prueba_t['f1']:.4f}")
    if os.path.exists(ruta_ultimo):
        os.remove(ruta_ultimo)                 # terminado: el avance ya no hace falta
    if os.path.isdir(carpeta_top):
        import shutil
        shutil.rmtree(carpeta_top)             # las K épocas ya se usaron; ocupan mucho en Drive
    print('archivos:', ruta_ckpt, ruta_stats, ruta_res, sep='\n  ')


if __name__ == '__main__':
    main()
