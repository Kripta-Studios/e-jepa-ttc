# Perfil de evaluación local del detector

Fecha: 2026-10-29. Perfil: `eap_development_xy_ap101_envelope_v1`.

Este perfil permite medir el overfit funcional y comparar pilotos reproducibles
con etiquetas públicas train. **No es el scorer oficial de eAP** ni establece
paridad con su leaderboard. El resultado incluye siempre
`official_eap_score: false` y conserva los nombres `dev_*`.

La implementación está en
[development_metrics.py](../../src/e_jepa_ttc/detection/development_metrics.py)
y sus casos analíticos en
[test_detection_development_metrics.py](../../tests/unit/test_detection_development_metrics.py).

## Entrada y validaciones

`evaluate_detections(predictions, ground_truth)` recibe dos mappings con la forma
`sample_token -> lista de cajas`. Exige que ambos tengan exactamente el mismo
conjunto de tokens, que sean strings no vacíos y que exista al menos un frame.
Las listas vacías son válidas: permiten registrar frames sin detecciones o sin
objetos. No se omiten silenciosamente los frames difíciles.

Cada caja tiene `class_name`, `translation_m`, `size_lwh_m` y `yaw_rad`. Una
predicción necesita además `score`. Las ocho clases son `car`, `bicycle`, `truck`,
`pedestrian`, `motorcycle`, `bus`, `tricycle` y `other_vehicle`. Centro y dimensiones
deben contener tres números finitos; las dimensiones deben ser positivas. El yaw
debe ser finito. Los booleanos no se aceptan como números. El score debe estar en
`[0,1]`: expresa confianza para ordenar, sin presuponer calibración probabilística.

El caller debe proporcionar centros en un mismo sistema ego y dimensiones en un
orden común. El scorer no transforma coordenadas ni verifica calibración, clases
de origen, partición train/selección o procedencia de los targets. Tampoco convierte
quaternions ni interpreta los campos crudos de HF.

`velocity_mps` no se lee, valida ni puntúa. Esta función no es un validador del
JSON de submission y no comprueba `meta`. Las predicciones MonoDGP que usen
`p_clase * exp(-log_variance)` pueden superar uno; la elección de un score admitido
debe ser explícita y quedar registrada en el adaptador, sin clipping oculto.

## Matching y AP

Las predicciones se ordenan por confianza decreciente **por clase y sobre todos
los frames**. Los empates se resuelven por token lexicográfico y después por el
índice original de la predicción. Para cada predicción se busca el objeto todavía
no asignado de su misma clase y frame con menor distancia en ego XY. Los empates
entre targets se resuelven por su índice en la lista de ground truth.

La asignación es greedy y uno a uno. Se admite un match solo cuando
`distancia < umbral`, estrictamente. Los umbrales predeterminados son
`[0.5, 1.0, 2.0, 4.0]` metros y deben ser distintos, positivos y finitos. Cada
umbral se evalúa desde cero. El perfil no aplica filtros de rango, un límite de
cajas, NMS ni un corte mínimo de score; esas decisiones, si existen, pertenecen
al experimento y deben registrarse aparte.

Para una clase con `N > 0` objetos GT, tras cada predicción de rango `k` se calculan
`recall_k = TP_acumulados / N` y `precision_k = TP_acumulados / k`. Para cada uno
de los 101 niveles `r = 0, 0.01, ..., 1`, se toma la máxima precisión entre los
puntos con recall al menos `r`; si no existe ninguno se usa cero. `AP101` es la
media de esos 101 valores. No hay recorte por mínimos de recall o precisión.

La AP de cada clase es la media de las AP en los umbrales configurados. Se devuelven
también TP, FP y FN por clase y umbral. Una caja duplicada puede contar como FP
sin reducir necesariamente la AP cuando aparece después de recuperar todo el GT;
esto es consecuencia de la envolvente de precisión, no una omisión del duplicado.

## Errores de verdaderos positivos

Se repite el matching con `tp_threshold=2.0` metros por defecto, también estricto.
Para cada pareja asignada:

- `ATE_m`: distancia euclídea del centro en **XY**. Z no interviene en matching
  ni ATE; una predicción con Z incorrecta puede conservar AP y ATE perfectas.
- `ASE`: `1 - intersección / unión`, calculando intersección como el producto de
  los mínimos de las tres dimensiones. Esto compara tamaño después de alinear
  centros y orientación; no es IoU entre las cajas situadas en la escena.
- `AOE_rad`: diferencia angular absoluta envuelta en `[-pi, pi)`. Todas las clases
  usan periodo `2*pi`; no hay excepciones por simetría de clase.

Se promedian **las parejas reales asignadas** dentro de cada clase. Después se
hace una media uniforme entre las clases que tienen al menos un match. No se
interpolan errores sobre bins de recall ni se promedian todos los objetos como
un único grupo. Una clase sin matches tiene errores `null` y no contribuye a
esa macro. Por ello, una mejora aparente de error debe examinarse junto con AP,
recall y número de matches: perder los objetos difíciles no es una mejora global.

## Clases ausentes y comparaciones válidas

Se publican dos agregaciones diferentes:

| Campo | Regla |
|---|---|
| `dev_AP101_supported_classes` | Media de las clases con al menos un objeto GT |
| `dev_AP101_all_eight_absent_zero` | Suma de esas AP dividida entre ocho; clases sin GT aportan cero |
| `supported_class_count` | Número de clases con soporte GT |

Una clase con GT y sin detecciones tiene AP cero. Una clase sin GT tiene AP
`null`, aunque sus predicciones siguen contabilizándose como FP en el detalle.
Si ningún frame contiene objetos GT, la macro soportada es `null` y la macro
de ocho clases es cero. Los errores TP son `null` si no hay matches.

Las macros no deben compararse entre subconjuntos con diferente soporte como si
fueran el mismo protocolo. En overfit y pilotos se deben conservar el conjunto de
tokens, su hash, conteos por clase, configuración de decode y perfil de métricas.
Un overfit acredita capacidad de aprender los ejemplos, no generalización. La
selección de modelos requiere secuencias separadas y no utiliza test privado.

## Relación con las fuentes oficiales

El [artículo eAP, sección VII-A](https://arxiv.org/html/2603.16303v1#S7.SS1)
declara una adaptación de nuScenes con matching de centros en el plano del suelo,
ATE 2D, ASE tras alinear posición/orientación y diferencia mínima de yaw. Estas
definiciones fundamentan los errores geométricos usados aquí.

La [configuración nuScenes original](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json)
contiene los cuatro umbrales y un umbral TP de dos metros, pero también mínimos
de recall/precisión de 0.1, rangos por clase y límite de cajas. Su
[algoritmo de evaluación](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py)
interpola curvas y agrega errores de otra manera. **La AP101 y las medias TP de
este módulo no reproducen ese algoritmo ni el scorer eAP privado.**

El [README HF fijado](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/blob/a814a946e60d37f7242358af5f344542177e50c5/README.md)
define `size_lwh_m` como `[length, width, height]` y reserva las etiquetas test.
No especifica todos los detalles necesarios para certificar una conversión de
yaw al scorer. El [repositorio eAP](https://github.com/NAIL-HNU/eAP_dataset)
enlaza la competición de detección **16717**; 17289 corresponde al benchmark TTC.
En la comprobación HTTP de esta sesión, la página 16717 y su endpoint público
de competición devolvieron 404; la búsqueda pública por `eAP` no devolvió entradas.
No se concluye por ello que la competición haya desaparecido definitivamente.

Se inspeccionaron el árbol GitHub
`2c5f3a74ba5b693b746820a0a355684e3597dc2d` y el árbol HF
`a814a946e60d37f7242358af5f344542177e50c5`, sin encontrar el scorer de detección ni
un conversor de submission ejecutable. La ausencia de esos artefactos mantiene
pendiente la comparación oficial, pero permite medir el aprendizaje local con
este perfil explícito.

## Validación

Los **24 casos** focalizados pasan en `.venv-audit`; Ruff y formato pasan en ambos
archivos Python. Cubren ranking, recall parcial, duplicados, límite estricto,
clases incompatibles, yaw envuelto, escala y traslación conocidas, Z excluida,
macro TP por clase, tokens ausentes/extra, entradas inválidas y empates.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.venv-audit/Scripts/python.exe -X utf8 -m pytest -q tests/unit/test_detection_development_metrics.py
.venv-audit/Scripts/python.exe -m ruff check src/e_jepa_ttc/detection/development_metrics.py tests/unit/test_detection_development_metrics.py
.venv-audit/Scripts/python.exe -m ruff format --check src/e_jepa_ttc/detection/development_metrics.py tests/unit/test_detection_development_metrics.py
```
