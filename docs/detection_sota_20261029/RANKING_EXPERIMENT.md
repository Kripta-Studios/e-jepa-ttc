# S2 — experimento preregistrado de ranking de calidad

Estado: **implementado y probado, no ajustado**. No se entrenará este sidecar hasta que
existan predicciones y observaciones congeladas del nuevo baseline R0 sobre el rol TRAIN.
No se presenta ninguna mejora de AP.

## Hipótesis y alcance

S0 encontró un margen grande entre el score actual y el oráculo de ranking sobre el mismo
pool. S2 prueba una sola intervención atribuible: conservar cada propuesta 3D y aprender
exclusivamente su orden. No cambia clase, centro, dimensiones, yaw, velocidad, número de
propuestas ni orden de almacenamiento. Conserva `classification_score` y
`original_detection_score`; `score`, `detection_score` y `geometry_quality` contienen la
nueva calidad para evaluación y operación.

El candidato único es una regresión logística compartida por las ocho clases. No se porta
RARE ni otro framework. Las 16 features observables son los ocho logits en el orden exacto
`car, bicycle, truck, pedestrian, motorcycle, bus, tricycle, other_vehicle`, score de clase,
margen top1–top2, entropía de clase, profundidad predicha, log-varianza de profundidad y
logaritmos de las dimensiones L/W/H decodificadas. Media y desviación se estiman solo en
TRAIN. Una clase sin positivos queda declarada con soporte cero; no se inventa un modelo
por clase.

## Target TRAIN y máscaras

Por frame y clase, la asignación propuesta–GT minimiza la distancia total ego XY mediante
una correspondencia uno-a-uno independiente del score. La calidad de una asignación es la
fracción de umbrales locales `(0,5, 1, 2, 4 m)` que supera, por lo que pertenece a
`{0, 0,25, 0,5, 0,75, 1}`. Una propuesta no asignada próxima a un GT elegible ya consumido
es un duplicado negativo explícito; el resto es fondo.

Dos poblaciones no entrenan el ranker:

- predicciones fuera del dominio geométrico del protocolo v2;
- una propuesta por GT ignorado y clase, después de priorizar GT elegible, mediante una
  segunda correspondencia uno-a-uno.

Estas filas quedan con target nulo, no con calidad cero. El bundle registra filas totales,
ajustadas y enmascaradas, además de recuentos por clase y motivo. Así no se repite el error
de enseñar como fondo objetos que el protocolo suprime.

## Receta congelada

- semilla `7`, CPU/double;
- capa lineal de 16 entradas y sigmoide;
- BCE con target suave y peso por muestra `1 + 3·quality`;
- AdamW, LR `0,02`, weight decay `1e-4`, `400` épocas full-batch;
- floor de desviación `1e-6`.

No hay búsqueda de arquitectura ni hiperparámetros en esta versión. La receta, features,
normalizador, pesos, bias, hashes de TRAIN y soporte por clase quedan dentro del bundle
auto-hasheado. Cualquier futura matriz predefinida deberá congelarse antes de ver selección;
solo selección podrá elegirla. Test, calibración y audit no pueden participar en ajuste,
normalización ni selección. Los controles y el baseline se congelan antes de abrir
resultados de selección.

## Controles obligatorios

1. Score original sin cambios.
2. Control escalar fijo `logit(score_clase) − 0,5·logvar_profundidad`, sin llamarlo
   probabilidad calibrada.
3. Temperatura global fija en espacio logit. Es estrictamente monótona: debe conservar el
   orden y la AP del mismo pool; sirve para demostrar que una mera calibración monótona no
   resuelve ranking.

La comparación principal reejecuta el evaluador v2 completo tras reordenar. No reutiliza
flags TP/FP previos. Reportará AP/PR por clase, macro con soporte, recall v2, soporte y
grupos/secuencias. El recall class-agnostic v2 también recorre propuestas por score: una
reordenación puede cambiar legítimamente su asignación greedy y su cardinalidad aunque la
geometría sea idéntica. Por ello se añaden dos invariantes separados: hash de clase/geometría
del pool y recall máximo bipartito, independiente del score, en cada umbral. La mejora debe
persistir entre semillas/grupos para superar S2.

## Contrato y comandos futuros

El ajuste acepta un bundle GT v2 verificado y obliga `--dataset-role train`. Además comprueba
que cada token pertenece a una secuencia de `split.assignments.train`, que ninguna secuencia
pertenece a dos roles y que el prefijo del token coincide con el `sequence_id_by_token`
canónico del bundle GT. Etiquetar selección como TRAIN en la línea de comandos no permite
ajustar. El hash de esta propiedad y el hash del protocolo quedan dentro del bundle:

```powershell
python scripts/train_detection_ranker.py fit `
  --predictions <r0_train_predictions.json> `
  --sidecar <r0_train_sidecar.json> `
  --ground-truth <r0_train_ground_truth_v2.json> `
  --panel-recipe <ranking_panel_recipe.json> `
  --ground-truth-receipt <r0_train_ground_truth_receipt.json> `
  --split-manifest <split_v2.json> `
  --dataset-role train `
  --output <ranker_bundle.json>
```

La aplicación no tiene argumento de etiquetas:

```powershell
python scripts/train_detection_ranker.py apply `
  --predictions <r0_selection_predictions.json> `
  --sidecar <r0_selection_sidecar.json> `
  --bundle <ranker_bundle.json> `
  --output <r0_selection_ranked.json>
```

El sidecar debe estar ligado por hash al archivo de predicciones y al mismo checkpoint. Una
predicción decodificada se liga a un query raw solo si clase y score identifican exactamente
un candidato no usado. Esto soporta queries descartadas en medio del array; empates ambiguos
o campos ausentes fallan cerrados. Fit, apply y controles rechazan frames que excedan el cap
de propuestas del protocolo; apply rechaza también un protocolo diferente del usado al
crear los targets. La procedencia TRAIN se revalida al cargar el bundle, no solo mediante su
hash autocontenido. La box2D y el prior geométrico interno siguen sin reconstruirse.

## Criterio de decisión

Se conserva el ranker únicamente si mejora AP/PR sobre R0 con denominadores, hash geométrico
y recall máximo del pool idénticos, sin concentrar la ganancia en un solo grupo o degradar
clases comunes de forma oculta. El recall greedy v2 se reporta, pero no se exige idéntico. Si
la intervención falla, se registra el negativo y continúa el baseline R0 hacia temporalidad;
no se añaden más módulos para forzar un resultado positivo.
