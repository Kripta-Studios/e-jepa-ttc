# Resolución de la geometría para el detector eAP independiente

Fecha: 2026-10-29. La correspondencia **original eAP → HF train** ya tiene evidencia
directa independiente de GarlTTC. El contrato permite construir targets geométricos,
entrenar un overfit funcional de 16 frames e inferir en el formato interno HF.
No certifica todavía compatibilidad con el scorer privado ni cierra todos los
subgates de la integración detector → asociación → PHASE.

## Evidencia que cambia la decisión

Se recuperó exclusivamente `2cyv0Oedzg/annotations.pkl` del ZIP publicado en el
[catálogo oficial](https://nail-hnu.github.io/eAP_dataset/assets/data/release_catalog.json).
La [URL del ZIP original](https://www.dropbox.com/scl/fi/v9bs0pbs0iz3ygdvky0ov/2cyv0Oedzg.zip?rlkey=sos77nxp53b5bmdsol4sbychd&dl=1)
respondió HTTP 206 a solicitudes Range. No se descargaron sus 6.558.664.264 bytes.
Se leyeron el header local de 30 bytes, el nombre de 26 bytes y el segmento DEFLATE
`bytes=56-1555058`: **1.555.003 bytes comprimidos**, 8.525.395 descomprimidos.
CRC32 comprobado: `1581563834`. SHA-256 del pickle:
`b81c6c2585284ed891520dd25120b5c24f3005a998083a2a319771d7f46d832e`.

El verificador admite exclusivamente valores builtin del pickle y rechaza globals
ejecutables. Los originales permanecen en `output/local_geometry_sources/`, fuera
de los archivos versionados. El recibo versionado contiene hashes y diagnósticos,
sin copiar cajas de anotaciones ni imágenes.

La unión por `(sequence_id, file_name, instance_id)` recuperó **17.611 de 17.611
filas**, sin duplicados ni ausentes. Son exactamente iguales entre original y HF:

- `bbox_3d` y `bbox_3d_ego`;
- categoría y velocidad 3D;
- `K_event`, `T_event_ego` y ambos timestamps de exposición;
- `size` HF y `bbox_3d[[5,3,4]]`.

La igualdad directa se comprobó para esta secuencia train. La auditoría anterior
de todas las etiquetas sigue aportando consistencia estructural de las cuarenta
secuencias; no se presenta como comparación directa con cuarenta ZIP originales.
Los resultados regenerables están en
[geometry_original_hf_real.json](evidence/geometry_original_hf_real.json).

## Contrato geométrico establecido

Se aplicó la función del
[visualizador oficial fijado](https://github.com/NAIL-HNU/eAP_dataset/blob/2c5f3a74ba5b693b746820a0a355684e3597dc2d/release_visualizer/visualize_release_sequence.py),
con SHA-256 `e975941c5866af465dcf3897efcdae36d71306808a38a63b580311a35f5ed5a7`.
El [README del mismo commit](https://github.com/NAIL-HNU/eAP_dataset/blob/2c5f3a74ba5b693b746820a0a355684e3597dc2d/README.md)
documenta los campos originales y la transformación hacia cámara.

| Elemento | Interpretación y operación |
|---|---|
| `bbox_3d_ego` | `[x,y,z,length,height,width,yaw]` |
| Centro | Centro geométrico, no centro inferior |
| Ejes ego | X hacia delante, Y a la izquierda, Z hacia arriba |
| Cuboide local antes del yaw | Ancho en X, largo en Y, altura en Z |
| Rotación | `Rz(-yaw)`; positivo horario visto desde +Z |
| Cámara | `p_event = T_event_ego @ p_ego_homogeneous` |
| `size` HF | `[width,length,height]`; no copiar como `size_lwh_m` |
| Dimensiones semánticas `[length,width,height]` | `bbox_3d_ego[[3,5,4]]` |
| Yaw para entrenamiento interno | Valor original, sin reinterpretar `rotation` HF |
| Velocidad para regresión | Valor publicado sin cambiar marco ni derivar TTC |

La matriz de rotación física anterior no se sustituye por el cuaternión HF
`[cos(yaw/2),0,0,sin(yaw/2)]`: copiar la fórmula de ese campo como si fuera la
rotación geométrica del visualizador cambia el signo. Si se necesita una
convención estándar con largo en X y yaw antihorario, el rumbo del eje largo
original es `pi/2 - yaw`; esa es una conversión interna que debe invertirse al
recuperar el target original. No se aplica automáticamente al submission.

## Reproyección sobre imágenes reales

Se seleccionaron el primer frame, el central y el último mediante posiciones
equiespaciadas entre los tokens train ordenados, sin filtrar por resultados:

```text
2cyv0Oedzg:000000221_000512214
2cyv0Oedzg:000000371_000214375
2cyv0Oedzg:000000520_000916457
```

Los tres PNG reales tienen resolución 1280×720. Se proyectaron las **19 cajas**
presentes. La diferencia máxima entre nuestro cálculo y las corners oficiales
es `3,7208848e-6 m` —la implementación oficial usa float32—. Tras clipping a
`[0,1279]×[0,719]`, el máximo residual frente a la caja 2D original publicada es
**0,9575132 px**, compatible con su cuantización entera.

Se inspeccionaron los tres overlays generados: los cuboides corresponden
visualmente a los vehículos presentes, incluidos casos truncados y ocluidos.
Las proyecciones extensas de los vehículos cercanos son cuboides completos,
no máscaras de la superficie visible. Los JPEG quedan en
`output/local_geometry_sources/real_overlays/`; sus hashes y los PNG originales
leídos del TAR están vinculados en el recibo. La inspección visual es cualitativa.

La [sección III-B del artículo](https://arxiv.org/html/2603.16303v1#S3.SS2)
describe un mapeo RGB de profundidad infinita al plano **distorsionado** del evento,
con paralaje residual por la pequeña separación entre sensores. El visualizador
publicado usa `K_event` y una proyección pinhole. Reproducimos esa aproximación
para targets y overlays; no afirmamos rectificación perfecta, ni calibración
subpíxel ni correspondencia RGB-eventos exacta para fusión posterior.

## Qué puede ejecutarse y qué queda separado

**Detector independiente:** cargar RGB público, generar cajas 2D mediante la
proyección documentada, regresar clase y caja HF original, supervisar velocidad
como campo publicado y evaluar pérdidas y geometría sobre train/selección. El
overfit de 16 frames valida implementación, no generalización. La asociación con
GarlTTC no es prerrequisito de este experimento.

**Submission nativo:** `translation_m=bbox[:3]` y las dimensiones semánticas
`size_lwh_m=bbox[[3,5,4]]` están justificadas por el contrato original y el
[README HF](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/blob/a814a946e60d37f7242358af5f344542177e50c5/README.md).
Conservar `yaw_rad=bbox[6]` es la exportación candidata que preserva el target.
No se ha encontrado un scorer/adaptador público que certifique su interpretación
de yaw; el JSON sintácticamente válido no resuelve esa incertidumbre. No declarar
evaluación oficial a partir de una métrica de desarrollo.

**Puente PHASE:** siguen pendientes la identidad causal entre detecciones,
historia temporal, ROIs sensoriales, política de disponibilidad y semántica de
velocidad para cualquier relación TTC. La discrepancia de IDs Garl documentada
previamente no invalida la identidad original↔HF demostrada aquí. No se modifican
helpers históricos ni se recalculan resultados TTC publicados.

## Reproducción

Pillow 13.0.0 se instaló únicamente en `.venv-audit` para leer/renderizar imágenes.
Los datos originales y la fuente fijada están ya en la carpeta local indicada.

```powershell
.venv-audit/Scripts/python.exe scripts/verify_eap_geometry_real.py --eap-root E:/eAP_dataset --sequence 2cyv0Oedzg --original-annotations output/local_geometry_sources/2cyv0Oedzg_original_annotations.pkl --official-visualizer output/local_geometry_sources/visualize_release_sequence.py --frames 3 --overlays-dir output/local_geometry_sources/real_overlays --output docs/detection_local_20261029/evidence/geometry_original_hf_real.json
.venv-audit/Scripts/python.exe -m unittest tests.unit.test_eap_geometry_real -v
.venv-audit/Scripts/python.exe -m ruff check scripts/verify_eap_geometry_real.py tests/unit/test_eap_geometry_real.py
```

Ocho tests correctos: cuboide no cúbico con yaw conocido, transformación dirigida,
orden de dimensiones, plano de cámara, parejas exactas/ambiguas, discrepancias de
cajas/exposición y rechazo de globals pickle. Los tests y el recibo nuevo no
alteran ni reemplazan las evidencias históricas de los gates anteriores.
