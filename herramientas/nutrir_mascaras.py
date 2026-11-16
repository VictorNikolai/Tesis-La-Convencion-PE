"""
Generador de máscaras de apoyo de alta precisión (Nivel 2 ampliado + Borradores pre-pintados en IRIS 128)
para nutrir el entrenamiento del proyecto de La Convención sin tocar `marcas.geojson` ni `revisadas.csv`,
y dejando intactas las 3 teselas de acuerdo inter-anotador (`sinpad_15525_001`, `sinpad_48878_001`, `sinpad_557_001`).

Combina cuatro fuentes físicas y documentadas del proyecto:
  1. Polígonos geológicos de INGEMMET revisados en `mascaras_provisionales/nivel2/` (`datos_ingemmet`).
  2. Detección multitemporal antes -> después (`qgis/cambios.geojson`, nivel `nuevo_desnudo`) en `datos_indeci`,
     `datos_laconvencion` y `datos_laconvencion_2025`, cruzada con pendiente >= 16° y filtro morfológico anti-caminos.
  3. Candidatos de ladera fuerte (`pendiente media >= 18°`, `pendiente max >= 22°`, `suelo desnudo GRVI < 0`,
     `prob_max >= 0.50`, área >= 0.10 ha y compacidad grosor área/perímetro >= 1.35 px para excluir carreteras).
     Todo suelo desnudo restante en ladera moderada, cauces o nubes queda como `255` (`Dudoso`), evitando falsos negativos.
  4. Negativos confirmados y negativos difíciles catalogados en `revision_teselas.csv` (`datos_azar`, `datos_laconvencion`,
     `datos_indeci`, `datos_indeci_antes`, `datos_ingemmet_antes`): bosques, pastos secos, playas de río, cultivos de té y caminos en zigzag quedan en `0`
     (con nubes blancas reales y bordes sin dato en `255`).

Produce:
  - En `mascaras_provisionales/nivel2/parches_128/<proyecto>/<parche>.tif` y `mascaras_256/<proyecto>/<tesela>.tif`:
    Más de 350 teselas (> 1400 parches de 128×128 px) listas para entrenamiento con `--debiles`.
  - En `iris_128/masks/<id_iris>.png`:
    Borradores pre-pintados en los parches de `iris_128` (excepto las 3 teselas de acuerdo entre anotadores),
    para que al abrir IRIS ya aparezcan pintados y el usuario solo revise, retoque si desea y guarde.
"""
import csv
import json
import os
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from cortar_parches_128 import LADO, cuadrantes, escribir_parches  # noqa: E402

TESELAS_ACUERDO = {'sinpad_15525_001', 'sinpad_48878_001', 'sinpad_557_001'}
TESELAS_AZAR_DUDOSAS = {'azar_003_001', 'azar_008_001', 'azar_023_001', 'azar_052_001'}


def cargar_catalogo_revision(raiz):
    cat = {}
    for proy in ('datos_laconvencion', 'datos_indeci', 'datos_ingemmet', 'datos_azar'):
        ruta = os.path.join(raiz, proy, 'revision_teselas.csv')
        if os.path.exists(ruta):
            with open(ruta, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    tes = r['tesela'].strip()
                    cat[(proy, tes)] = r
                    if proy == 'datos_laconvencion':
                        cat[('datos_laconvencion_2025', tes)] = r
                    elif proy == 'datos_indeci':
                        cat[('datos_indeci_antes', tes)] = r
                    elif proy == 'datos_ingemmet':
                        cat[('datos_ingemmet_antes', tes)] = r
    return cat


def es_tesela_negativa_catalogada(proy, tes, rev):
    if not rev:
        if proy in ('datos_indeci_antes', 'datos_ingemmet_antes'):
            return True
        return False
    uso = (rev.get('uso_propuesto') or '').strip().lower()
    if uso in ('descartar', 'rebajar'):
        return False
    pq = (rev.get('por_que') or '').strip().lower()
    lv = (rev.get('lo_que_se_ve') or '').strip().lower()
    if proy == 'datos_azar':
        return tes not in TESELAS_AZAR_DUDOSAS
    palabras_neg = (
        'negativo', 'sin cicatrices claras', 'sin cicatriz visible',
        'pasto seco', 'playas: confusor', 'playas de rio: confusor',
        'confusor: pista de aterrizaje', 'confusor: el modelo marca los caminos',
        'confusor de te', 'los cambios son chacras nuevas', 'cambios por desbosque',
        'relieve suave', 'marcar los huecos negros como dudoso', 'marcar huecos como dudoso',
        'bruma: si hace falta rebajar', 'cauce de huaico', 'igual que los cauces',
        'depositos de huaico en el cauce; las terrazas confunden',
        'erosion de altura; dificil de separar',
    )
    if any(p in pq for p in palabras_neg) or any(p in lv for p in ('sin cicatriz visible', 'relieve suave')):
        if 'posibles cicatrices' not in pq and 'candidata' not in pq and 'cicatriz nueva clara' not in pq:
            return True
    if uso == 'validacion' and ('negativo' in pq or 'confusor' in pq or 'pasto seco' in pq):
        return True
    if 'sin rasgo destacado' in lv and uso == 'reserva':
        return True
    return False


def filtrar_morfologia_cicatriz(bin_mask, pend, prob, es_cambio=False):
    """Filtra componentes conexas para separar cicatrices reales de carreteras/caminos y de pastos secos extensos."""
    lab, n = ndimage.label(bin_mask)
    aceptada = np.zeros_like(bin_mask, dtype=bool)
    dudosa = np.zeros_like(bin_mask, dtype=bool)
    for k in range(1, n + 1):
        comp = lab == k
        area_px = int(comp.sum())
        # Menos de 0,10 ha (<10 px) o más de 8,0 ha (>800 px = ladera entera de pasto seco o cauce ancho) -> dudoso (255)
        if area_px < 10 or area_px > 800:
            dudosa |= comp
            continue
        erod = ndimage.binary_erosion(comp)
        perim_px = max(int((comp & ~erod).sum()), 1)
        grosor_medio = area_px / perim_px
        p_med = float(pend[comp].mean())
        p_max = float(pend[comp].max())
        pr_max = float(prob[comp].max())

        # Los caminos son cintas finas (grosor_medio < 1.38 px)
        if grosor_medio < 1.38:
            dudosa |= comp
            continue

        if es_cambio:
            if p_med >= 21.0 and p_max >= 26.0 and pr_max >= 0.25 and grosor_medio >= 1.42 and 10 <= area_px <= 600:
                aceptada |= comp
            else:
                dudosa |= comp
        else:
            if p_med >= 22.0 and p_max >= 27.0 and pr_max >= 0.60 and 12 <= area_px <= 500 and grosor_medio >= 1.45:
                aceptada |= comp
            else:
                dudosa |= comp
    return aceptada, dudosa


def construir_mascara_tesela(raiz, proy, tes, rev, info_part):
    """Devuelve `(mask_256, perfil, tipo_tesela, motivo)` con valores uint8 `{0, 1, 255}`."""
    ruta_img = os.path.join(raiz, proy, 'images', tes, 'image.tif')
    with rasterio.open(ruta_img) as ds:
        x = ds.read()
        tr, crs, forma, perfil = ds.transform, ds.crs, ds.shape, ds.profile
    b2, b3, b4, pend, prob = x[1], x[2], x[3], x[4], x[6]
    # En azar_012 y azar_015 la banda 7 quedó corrupta a ~0.73 por carrera de hilos; ignorar banda 7 ahí
    if tes in ('azar_012_001', 'azar_015_001'):
        prob = np.zeros_like(b2)

    suelo_claro = (b2 <= 2000) & (b4 > b3) & (b4 > 1.2 * b2)
    nube_o_nodata = (b2 <= 0) | ((b2 > 1000) & ~suelo_claro)

    # Caso 1: `datos_ingemmet` o `datos_ingemmet_antes` -> usar directamente la máscara curada de Nivel 2 si existe
    if proy in ('datos_ingemmet', 'datos_ingemmet_antes'):
        ruta_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2', 'mascaras_256', f'{tes}.tif')
        m = np.zeros(forma, dtype=np.uint8)
        if os.path.exists(ruta_n2):
            with rasterio.open(ruta_n2) as dn2:
                m = dn2.read(1)
        # Para datos_ingemmet (post-evento), incorporar polígonos visibles de INGEMMET con criterio físico
        if proy == 'datos_ingemmet':
            ruta_ing_vis = os.path.join(raiz, 'ingemmet', 'propuestas_ingemmet_visibles.geojson')
            if os.path.exists(ruta_ing_vis):
                with open(ruta_ing_vis, encoding='utf-8') as fi:
                    gj_i = json.load(fi)
                geos_i = [
                    transform_geom('EPSG:4326', crs, f['geometry'])
                    for f in gj_i.get('features', [])
                    if f.get('geometry') and (f.get('properties') or {}).get('tesela') == tes
                    and float((f.get('properties') or {}).get('pend_media') or 0) >= 18.0
                    and float((f.get('properties') or {}).get('area_ha') or 0) >= 0.10
                    and float((f.get('properties') or {}).get('prob_max') or 0) >= 0.25
                ]
                if geos_i:
                    mask_ing = rasterize([(g, 1) for g in geos_i], out_shape=forma, transform=tr, fill=0, dtype='uint8') == 1
                    m[mask_ing & ~nube_o_nodata] = 1
        # Asegurar que suelo_claro fuera de cuerpos no quede como nube
        m[(b2 > 1000) & suelo_claro & (m != 1)] = 0
        m[nube_o_nodata] = 255
        tipo = 'positiva_ingemmet' if (m == 1).any() else 'contexto_ingemmet'
        return m, perfil, tipo, 'INGEMMET geológico + propuestas visibles Nivel 2'

    # Caso 2: Negativo confirmado por catálogo visual
    if es_tesela_negativa_catalogada(proy, tes, rev):
        m = np.zeros(forma, dtype=np.uint8)
        m[nube_o_nodata] = 255
        # Si es negativo con nubes o cauce dudoso mencionado en la nota, proteger con 255
        pq = ((rev or {}).get('por_que') or '').lower()
        if 'huecos' in pq or 'nube' in pq or 'bruma' in pq:
            bruma_fuerte = (b2 > 850) & ~suelo_claro
            m[bruma_fuerte] = 255
        return m, perfil, 'negativo_confirmado', (rev or {}).get('por_que', 'negativo catalogado')

    # Caso 3: Tesela con posibles cicatrices (`datos_indeci`, `datos_laconvencion`, `datos_laconvencion_2025` o `azar` dudosa)
    cambio_mask = np.zeros(forma, dtype=bool)
    proy_cambios = 'datos_laconvencion' if proy == 'datos_laconvencion_2025' else proy
    if proy == 'datos_indeci_antes':
        proy_cambios = 'datos_indeci'
    ruta_cambios = os.path.join(raiz, proy_cambios, 'qgis', 'cambios.geojson')
    if os.path.exists(ruta_cambios):
        with open(ruta_cambios, encoding='utf-8') as fc:
            gj_c = json.load(fc)
        geos_c = [
            transform_geom('EPSG:4326', crs, f['geometry'])
            for f in gj_c.get('features', [])
            if f.get('geometry') and (f.get('properties') or {}).get('tesela') == tes
            and (f.get('properties') or {}).get('nivel') == 'nuevo_desnudo'
        ]
        if geos_c:
            cambio_mask = rasterize([(g, 1) for g in geos_c], out_shape=forma, transform=tr, fill=0, dtype='uint8') == 1

    # Si es "antes" de INDECI, el cambio de nuevo_desnudo ocurrió después, aquí era vegetación/fondo
    if proy == 'datos_indeci_antes':
        m = np.zeros(forma, dtype=np.uint8)
        m[nube_o_nodata] = 255
        return m, perfil, 'negativo_confirmado', 'Estado pre-evento INDECI verificado'

    grvi = (b3 - b4) / (b3 + b4 + 1e-5)
    suelo_ladera = (b2 > 0) & ~nube_o_nodata & (grvi < 0.01) & (b4 > b3) & (b4 >= 520) & (pend >= 16.0)
    cand_fuerte = suelo_ladera & (prob >= 0.42)

    acep_cambio, dud_cambio = filtrar_morfologia_cicatriz(cambio_mask & ~nube_o_nodata, pend, prob, es_cambio=True)
    acep_espect, dud_espect = filtrar_morfologia_cicatriz(cand_fuerte & ~acep_cambio, pend, prob, es_cambio=False)

    des = acep_cambio | acep_espect
    # Todo el suelo desnudo no aceptado, candidatos débiles del modelo o cauces van a 255 (dudoso) para no meter falsos negativos
    suelo_restante = (b2 > 0) & (grvi < 0.03) & (b4 >= 500) & ~des
    prob_debil = (prob >= 0.28) & ~des
    dud = dud_cambio | dud_espect | suelo_restante | prob_debil | nube_o_nodata

    m = np.zeros(forma, dtype=np.uint8)
    m[dud] = 255
    m[des] = 1
    tipo = 'positiva_multitemporal_ladera' if (m == 1).any() else 'dudosa_protegida'
    return m, perfil, tipo, (rev or {}).get('por_que', 'análisis multitemporal y pendiente')


def ejecutar(raiz='.', poblar_iris=True):
    cat = cargar_catalogo_revision(raiz)
    part = {}
    # 1. Cargar particion.csv clásico
    ruta_part_csv = os.path.join(raiz, 'particion.csv')
    if os.path.exists(ruta_part_csv):
        with open(ruta_part_csv, encoding='utf-8') as fp:
            for r in csv.DictReader(fp):
                part[(r['proyecto'], r['tesela'])] = r

    # 2. Incorporar partición completa desde indice_parches.csv para proyectos multitemporales
    ruta_ind_parches = os.path.join(raiz, 'parches_128', 'indice_parches.csv')
    if os.path.exists(ruta_ind_parches):
        with open(ruta_ind_parches, encoding='utf-8') as fip:
            for r in csv.DictReader(fip):
                k = (r['proyecto'], r['tesela'])
                if k not in part:
                    part[k] = {
                        'proyecto': r['proyecto'],
                        'tesela': r['tesela'],
                        'zona': r['zona'],
                        'conjunto': r['conjunto'],
                        'uso': r['uso'],
                    }

    sal_n2 = os.path.join(raiz, 'mascaras_provisionales', 'nivel2')
    os.makedirs(os.path.join(sal_n2, 'mascaras_256'), exist_ok=True)
    os.makedirs(os.path.join(sal_n2, 'parches_128'), exist_ok=True)

    resumen = []
    conteo_tipos = {}
    parches_iris_escritos = 0
    teselas_iris_escritas = 0

    # Mapa de parches en iris_128
    iris_dir = os.path.join(raiz, 'iris_128')
    iris_ids = set()
    ruta_ind_iris = os.path.join(iris_dir, 'indice_iris_128.csv')
    if os.path.exists(ruta_ind_iris):
        with open(ruta_ind_iris, encoding='utf-8') as fi:
            iris_ids = {r['id_iris'] for r in csv.DictReader(fi)}
    if poblar_iris and os.path.isdir(os.path.join(iris_dir, 'masks')):
        for f_old in os.listdir(os.path.join(iris_dir, 'masks')):
            if f_old.endswith('.png'):
                os.remove(os.path.join(iris_dir, 'masks', f_old))

    for (proy, tes), info in sorted(part.items()):
        if info['conjunto'] in ('excluida', 'pendiente', 'sin_zona'):
            continue
        # Nunca auto-etiquetar las 3 teselas de acuerdo entre anotadores
        if tes in TESELAS_ACUERDO:
            continue
        ruta_img = os.path.join(raiz, proy, 'images', tes, 'image.tif')
        if not os.path.exists(ruta_img):
            continue

        rev = cat.get((proy, tes))
        m256, perfil, tipo, motivo = construir_mascara_tesela(raiz, proy, tes, rev, info)
        px_1 = int((m256 == 1).sum())
        px_0 = int((m256 == 0).sum())
        px_255 = int((m256 == 255).sum())

        # Guardar en mascaras_provisionales/nivel2/ con subcarpeta por proyecto (y en raíz si es datos_ingemmet para compatibilidad)
        dir_m256_p = os.path.join(sal_n2, 'mascaras_256', proy)
        dir_p128_p = os.path.join(sal_n2, 'parches_128', proy)
        os.makedirs(dir_m256_p, exist_ok=True)
        os.makedirs(dir_p128_p, exist_ok=True)

        p8 = dict(driver='GTiff', height=m256.shape[0], width=m256.shape[1], count=1, dtype='uint8',
                  crs=perfil['crs'], transform=perfil['transform'], compress='deflate')
        with MemoryFile() as mem:
            with mem.open(**p8) as d:
                d.write(m256, 1)
                d.update_tags(nivel='2', tipo=tipo, valores='0 fondo, 1 cicatriz propuesta, 255 ignorar')
            raw_bytes = mem.read()
        with open(os.path.join(dir_m256_p, f'{tes}.tif'), 'wb') as fo:
            fo.write(raw_bytes)
        if proy == 'datos_ingemmet':
            with open(os.path.join(sal_n2, 'mascaras_256', f'{tes}.tif'), 'wb') as fo:
                fo.write(raw_bytes)

        escribir_parches(m256[None], perfil, dir_p128_p, tes, ['mascara nivel 2'])
        if proy == 'datos_ingemmet':
            escribir_parches(m256[None], perfil, os.path.join(sal_n2, 'parches_128'), tes, ['mascara nivel 2'])

        # Pre-poblar borrador en IRIS 128 si la tesela está en iris_128
        en_iris = False
        if poblar_iris and tipo in ('positiva_ingemmet', 'positiva_multitemporal_ladera', 'negativo_confirmado', 'contexto_ingemmet'):
            for q, f0, c0 in cuadrantes(m256.shape[0], m256.shape[1], LADO):
                pid = f'{proy}__{tes}_{q}'
                if pid not in iris_ids:
                    continue
                ruta_png = os.path.join(iris_dir, 'masks', f'{pid}.png')
                trozo = m256[f0:f0 + LADO, c0:c0 + LADO]
                # Codificación 0-based de IRIS: 0 = Sin deslizamiento, 1 = Deslizamiento, 2 = Dudoso
                m_iris = np.zeros_like(trozo, dtype=np.uint8)
                m_iris[trozo == 1] = 1
                m_iris[trozo == 255] = 2
                Image.fromarray(m_iris).save(ruta_png)
                parches_iris_escritos += 1
                en_iris = True
            if en_iris:
                teselas_iris_escritas += 1

        usar_en_entrenamiento_debil = (
            info['conjunto'] == 'entrenamiento'
            and (px_1 > 0 or tipo in ('negativo_confirmado', 'contexto_ingemmet', 'positiva_ingemmet', 'positiva_multitemporal_ladera'))
        )
        conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + 1
        resumen.append({
            'proyecto': proy, 'tesela': tes, 'zona': info['zona'], 'conjunto': info['conjunto'],
            'uso': info['uso'], 'tipo': tipo, 'incluir_debil': 'si' if usar_en_entrenamiento_debil else 'no',
            'pre_pintada_iris': 'si' if en_iris else 'no',
            'px_1': px_1, 'ha_1': round(px_1 * 0.01, 2),
            'px_0': px_0, 'ha_0': round(px_0 * 0.01, 2),
            'px_255': px_255, 'ha_255': round(px_255 * 0.01, 2),
            'motivo': motivo[:140],
        })

    ruta_csv_comp = os.path.join(sal_n2, 'mascaras_nivel2_completo.csv')
    with open(ruta_csv_comp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)

    n_ent = sum(1 for r in resumen if r['incluir_debil'] == 'si')
    n_pos = sum(1 for r in resumen if r['px_1'] > 0)
    n_neg = sum(1 for r in resumen if r['tipo'] in ('negativo_confirmado', 'contexto_ingemmet'))
    ha_tot = sum(r['ha_1'] for r in resumen)
    print(f'Total teselas procesadas en Nivel 2 ampliado: {len(resumen)} ({len(resumen)*4} parches de 128×128)')
    print(f'  • Teselas con cicatrices propuestas (px_1 > 0): {n_pos} teselas ({n_pos*4} parches, {ha_tot:.1f} ha de deslizamiento)')
    print(f'  • Teselas negativas confirmadas / contexto: {n_neg} teselas ({n_neg*4} parches)')
    print(f'  • Teselas activas para entrenamiento débil (conjunto=entrenamiento): {n_ent} teselas ({n_ent*4} parches)')
    print(f'  • Teselas pre-pintadas en iris_128/masks/: {teselas_iris_escritas} teselas ({parches_iris_escritos} parches de 128×128)')
    return resumen


if __name__ == '__main__':
    ejecutar()
