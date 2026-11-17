"""
Pipeline integral de entrenamiento, ajuste fino y exportación ONNX
para segmentación de deslizamientos en La Convención (Cusco, Perú).

Propósito:
  Orquesta de forma reproducible el entrenamiento comparativo de las arquitecturas
  U-Net (ResNet-34) y DeepLabV3+ (ResNet-34), integrando los 2,580 parches locales
  estratificados por Tiers (Oro, Plata, Bronce y Hard Negatives), la opción
  geomorfométrica (CURVATURA y TPI) y la generación automática de modelos ONNX
  con sus recetas JSON listas para despliegue en producción.

Modos de ejecución:
  1. base         Entrena un modelo base sobre Landslide4Sense y exporta su ONNX.
  2. ajuste       Ejecuta el ajuste fino sobre La Convención y exporta el ONNX ajustado.
  3. comparativo  Ejecuta el benchmark completo U-Net ResNet-34 vs. DeepLabV3+ ResNet-34,
                  exportando ambos modelos ONNX y verificando sus métricas oficiales.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)


def _ejecutar(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ultimo_archivo(patron):
    coincidencias = glob.glob(patron)
    if not coincidencias:
        return None
    return max(coincidencias, key=os.path.getmtime)


def _exportar_desde_salida(carpeta_salida, carpeta_onnx, nombre_modelo, etiqueta, arquitectura):
    """Localiza el último checkpoint generado y lo exporta a formato ONNX + receta JSON."""
    ckpt = _ultimo_archivo(os.path.join(carpeta_salida, 'mejor_*.pt'))
    stats = _ultimo_archivo(os.path.join(carpeta_salida, 'stats_norm_*.json'))
    resultados = _ultimo_archivo(os.path.join(carpeta_salida, 'resultados_*.json'))
    if not ckpt or not stats:
        print(f'AVISO: no se encontró checkpoint o stats en {carpeta_salida}; se omite exportación ONNX.')
        return False

    os.makedirs(carpeta_onnx, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(AQUI, 'exportar_onnx.py'),
        '--checkpoint', ckpt,
        '--stats', stats,
        '--salida', carpeta_onnx,
        '--nombre', nombre_modelo,
        '--etiqueta', etiqueta,
        '--arquitectura', arquitectura,
    ]
    if resultados:
        cmd += ['--resultados', resultados]
    _ejecutar(cmd)
    return True


def ejecutar_base(a):
    """Entrenamiento base en Landslide4Sense + exportación automática a ONNX."""
    os.makedirs(a.salida, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(AQUI, 'entrenar.py'),
        '--l4s', a.l4s,
        '--arquitectura', a.arquitectura,
        '--encoder', a.encoder,
        '--encoder-weights', 'imagenet',
        '--canales', 'B01,B02,B03,B04,pendiente,elevacion',
        '--epocas', str(a.epocas),
        '--paciencia', '15',
        '--lote', str(a.lote),
        '--lr', '3e-4',
        '--peso-dice', '0.5',
        '--ohem', '0.25',
        '--indices-verdor',
        '--ema', '0.999',
        '--calentamiento', '5',
        '--promediar', '5',
        '--salida', a.salida,
    ]
    if a.limite:
        cmd += ['--limite', str(a.limite)]
    _ejecutar(cmd)

    if not a.sin_onnx:
        _exportar_desde_salida(
            a.salida, a.salida_onnx,
            nombre_modelo=f'{a.arquitectura}_{a.encoder}_base',
            etiqueta=f'{a.arquitectura.upper()} {a.encoder} (Base L4S)',
            arquitectura=a.arquitectura,
        )


def ejecutar_ajuste(a):
    """Ajuste fino en La Convención (Tiers + Hard Negatives) + exportación ONNX."""
    os.makedirs(a.salida, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(AQUI, 'ajuste_fino.py'),
        '--l4s', a.l4s,
        '--desde', a.desde,
        '--stats', a.stats,
        '--parches', a.parches,
        '--mascaras', a.mascaras,
        '--debiles', a.debiles,
        '--peso-debil', str(a.peso_debil),
        '--escala', a.escala,
        '--epocas', str(a.epocas),
        '--lote', str(a.lote),
        '--lr', str(a.lr),
        '--frac-local', '0.5',
        '--tolerancia-l4s', '0.03',
        '--salida', a.salida,
    ]
    if a.geomorfologia:
        cmd.append('--geomorfologia')
    if a.congelar_encoder:
        cmd.append('--congelar-encoder')
    if a.limite:
        cmd += ['--limite', str(a.limite)]
    _ejecutar(cmd)

    if not a.sin_onnx:
        res_path = _ultimo_archivo(os.path.join(a.salida, 'resultados_*.json'))
        if res_path:
            res = json.load(open(res_path, encoding='utf-8'))
            if res.get('mejoro'):
                arq = res.get('arquitectura', 'unet')
                suf = '_geom' if a.geomorfologia else ''
                _exportar_desde_salida(
                    a.salida, a.salida_onnx,
                    nombre_modelo=f'{arq}_laconvencion{suf}',
                    etiqueta=f'{arq.upper()} ResNet-34 Ajustado (La Convención)',
                    arquitectura=arq,
                )


def ejecutar_comparativo(a):
    """Entrena U-Net ResNet-34 y DeepLabV3+ ResNet-34, exporta ambos ONNX y los verifica."""
    for arq in ('unet', 'deeplabv3plus'):
        sub_salida = os.path.join(a.salida, arq)
        args_base = argparse.Namespace(
            l4s=a.l4s,
            arquitectura=arq,
            encoder='resnet34',
            epocas=a.epocas,
            lote=a.lote,
            limite=a.limite,
            salida=sub_salida,
            salida_onnx=a.salida_onnx,
            sin_onnx=False,
        )
        ejecutar_base(args_base)

    _ejecutar([
        sys.executable, os.path.join(AQUI, 'verificar_onnx.py'),
        '--modelos', a.salida_onnx,
        '--l4s', a.l4s,
        '--splits', 'test',
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='modo', required=True)

    p_base = sub.add_parser('base', help='Entrenamiento base en Landslide4Sense y exportación ONNX')
    p_base.add_argument('--l4s', required=True, help='carpeta raíz de Landslide4Sense')
    p_base.add_argument('--arquitectura', choices=['unet', 'deeplabv3plus'], default='unet')
    p_base.add_argument('--encoder', default='resnet34')
    p_base.add_argument('--epocas', type=int, default=60)
    p_base.add_argument('--lote', type=int, default=16)
    p_base.add_argument('--salida', default=os.path.join(RAIZ, 'modelos_base'))
    p_base.add_argument('--salida-onnx', dest='salida_onnx', default=os.path.join(RAIZ, 'modelos_onnx'))
    p_base.add_argument('--sin-onnx', action='store_true')
    p_base.add_argument('--limite', type=int, default=None)

    p_aj = sub.add_parser('ajuste', help='Ajuste fino sobre La Convención y exportación ONNX')
    p_aj.add_argument('--l4s', required=True)
    p_aj.add_argument('--desde', required=True, help='checkpoint .pt del modelo base')
    p_aj.add_argument('--stats', required=True, help='stats_norm_<firma>.json del modelo base')
    p_aj.add_argument('--parches', default=os.path.join(RAIZ, 'parches_128'))
    p_aj.add_argument('--mascaras', default=os.path.join(RAIZ, 'mascaras_nivel1', 'parches_128'))
    p_aj.add_argument('--debiles', default=os.path.join(RAIZ, 'mascaras_provisionales', 'nivel2', 'parches_128'))
    p_aj.add_argument('--escala', default=os.path.join(RAIZ, 'escala_laconvencion.json'))
    p_aj.add_argument('--peso-debil', type=float, default=0.3)
    p_aj.add_argument('--epocas', type=int, default=40)
    p_aj.add_argument('--lote', type=int, default=16)
    p_aj.add_argument('--lr', type=float, default=3e-5)
    p_aj.add_argument('--geomorfologia', action='store_true', help='activa CURVATURA y TPI desde el DEM')
    p_aj.add_argument('--congelar-encoder', action='store_true')
    p_aj.add_argument('--salida', default=os.path.join(RAIZ, 'modelos_ajustados'))
    p_aj.add_argument('--salida-onnx', dest='salida_onnx', default=os.path.join(RAIZ, 'modelos_onnx'))
    p_aj.add_argument('--sin-onnx', action='store_true')
    p_aj.add_argument('--limite', type=int, default=None)

    p_comp = sub.add_parser('comparativo', help='Benchmark U-Net vs DeepLabV3+ con exportación y verificación ONNX')
    p_comp.add_argument('--l4s', required=True)
    p_comp.add_argument('--epocas', type=int, default=60)
    p_comp.add_argument('--lote', type=int, default=16)
    p_comp.add_argument('--salida', default=os.path.join(RAIZ, 'modelos_comparativo'))
    p_comp.add_argument('--salida-onnx', dest='salida_onnx', default=os.path.join(RAIZ, 'modelos_onnx'))
    p_comp.add_argument('--limite', type=int, default=None)

    a = ap.parse_args()
    if a.modo == 'base':
        ejecutar_base(a)
    elif a.modo == 'ajuste':
        ejecutar_ajuste(a)
    else:
        ejecutar_comparativo(a)


if __name__ == '__main__':
    main()
