"""
Orquestador reproducible de entrenamiento y ajuste fino para la tesis (La Convención, Cusco).

Permite ejecutar desde línea de comandos (en GPU local o Google Colab):
  1. Entrenamiento base sobre Landslide4Sense (U-Net ResNet-34 v4 y DeepLabV3+ ResNet-34).
  2. Ajuste fino (fine-tuning) con los 2,580 parches estratificados por Tiers (Oro, Plata,
     Bronce y Hard Negatives), con opción geomorfométrica (--geomorfologia: CURVATURA + TPI).
  3. Exportación a ONNX y verificación automática.

Ejemplo de uso completo (ajuste fino desde checkpoint base):
  python herramientas/correr_entrenamiento_tesis.py ajuste \\
      --l4s /content/l4s \\
      --desde /content/drive/MyDrive/entrenamiento_v4/mejor_unet_b1b4b13b14_b2809be440d6.pt \\
      --stats /content/drive/MyDrive/entrenamiento_v4/stats_norm_b2809be440d6.json \\
      --geomorfologia

Ejemplo de entrenamiento base comparativo (U-Net v4 vs DeepLabV3+):
  python herramientas/correr_entrenamiento_tesis.py base --l4s /content/l4s --arquitectura unet
  python herramientas/correr_entrenamiento_tesis.py base --l4s /content/l4s --arquitectura deeplabv3plus
"""
import argparse
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)


def _correr(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_base(a):
    """Entrena el modelo base en Landslide4Sense (U-Net v4 o DeepLabV3+ ResNet-34)."""
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
    _correr(cmd)


def cmd_ajuste(a):
    """Ejecuta el ajuste fino sobre La Convención usando Tiers (Oro/Plata/HardNeg) y L4S."""
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
    _correr(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='modo', required=True)

    p_base = sub.add_parser('base', help='Entrenamiento base sobre Landslide4Sense')
    p_base.add_argument('--l4s', required=True, help='carpeta raíz de Landslide4Sense')
    p_base.add_argument('--arquitectura', choices=['unet', 'deeplabv3plus'], default='unet')
    p_base.add_argument('--encoder', default='resnet34')
    p_base.add_argument('--epocas', type=int, default=60)
    p_base.add_argument('--lote', type=int, default=16)
    p_base.add_argument('--salida', default=os.path.join(RAIZ, 'modelos_base'))
    p_base.add_argument('--limite', type=int, default=None)

    p_aj = sub.add_parser('ajuste', help='Ajuste fino sobre La Convención con Tiers y geomorfometría')
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
    p_aj.add_argument('--limite', type=int, default=None)

    a = ap.parse_args()
    if a.modo == 'base':
        cmd_base(a)
    else:
        cmd_ajuste(a)


if __name__ == '__main__':
    main()
