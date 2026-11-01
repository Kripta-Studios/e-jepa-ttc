# Evaluación local alineada con el contrato observado de eAP

El detector Dfull con ranking alcanza **mAP local 0,06204229**, frente a
**0,05701726 sin ranking**, al reevaluar las mismas 23.958 imágenes de selección
con todas sus 190.469 anotaciones públicas. La ganancia es 0,00502503 absoluta,
8,81 % relativa. No se ha entrenado, repetido inferencia ni contratado cómputo.
El resultado oficial del modelo con ranking sigue siendo **0,05129508** en test.

Esto conserva evidencia útil del detector y del ranking, pero no acredita un
detector robusto de ocho clases ni preparación para producción. Coches concentra
la mayor parte de la AP. Bicicleta y other_vehicle tienen AP local cero bajo
este perfil; peatones está cerca de cero. El ranking empeora ligeramente
motorcycle: de 0,047429 a 0,045753. No ocultamos esa degradación.

## Qué se ha alineado

Nuevo módulo `src/e_jepa_ttc/detection/benchmark_evaluation.py`, perfil
`phase_ttc_eap_observed_reference_v1`, separado del evaluador histórico v2:

- Ocho clases y matching por distancia XY estrictamente menor que 0,5/1/2/4 m.
- Asignación greedy por confianza a GT de la misma clase, sin reutilizar objetos.
- Interpolación lineal sobre 101 puntos de recall; mínimos de recall y precisión
  de 0,1. La AP usa los puntos 11–100, resta 0,1 a precisión, recorta en cero y
  normaliza por 0,9. No aplica la envolvente de precisión del perfil histórico.
- ATE, ASE y AOE a 2 m; medias acumuladas interpoladas por confianza. Si no hay
  recall válido por encima del mínimo, el error de esa clase es 1.
- Macro sobre las ocho clases, incluidas clases sin soporte con AP cero y error
  uno. El compuesto usa peso 5 para AP y las tres métricas TP, denominador 8.
- Límite de 500 propuestas por muestra: se rechaza un exceso, no se trunca
  silenciosamente. Esta comparación conserva las 50 propuestas originales.
- Todas las anotaciones raw públicas. Se reincorporan 83.161 objetos antes
  ignorados por el perfil local, sin filtro de profundidad o imagen sobre GT
  ni predicciones. No se evalúa velocidad.

**Es un perfil local de referencia, no una copia certificada del scorer privado.**
La configuración y la aritmética de las 32 AP y los 24 errores TP se verifican
contra las curvas agregadas devueltas por el organizador. El matching, distancia,
tratamiento de empates y la interpolación siguen la
[referencia pública nuScenes](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py);
su [definición de errores](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/utils.py)
aclara distancia XY e IoU de tamaños alineados. No se conocen posibles filtros
adicionales ni implementación exacta del servicio eAP. No se accede a GT test.

## Resultado reproducible

| Variante congelada | mAP local alineada | ATE | ASE | AOE | Compuesto local |
|---|---:|---:|---:|---:|---:|
| Baseline | 0,057017 | 0,890311 | 0,484148 | 0,517191 | 0,174180 |
| Control incertidumbre escalar | 0,060531 | 0,875939 | 0,483311 | 0,511249 | 0,179019 |
| Ranking aprendido en TRAIN | 0,062042 | 0,869129 | 0,483038 | 0,507858 | 0,181273 |

Los números precisos y la tabla por clase se regeneran en
[REPORT.md](evidence/benchmark_local_alignment_20261101/REPORT.md), desde
[COMPARISON.json](evidence/benchmark_local_alignment_20261101/COMPARISON.json).
Se conserva cada curva de 101 puntos, recuentos, hashes de los 47 shards y
procedencia de las etiquetas públicas. No se guardan curvas por cada predicción
ni se cargan imágenes o checkpoints. Se comprueba el multiconjunto de geometría,
clase y velocidad: el ranking solo cambia score y metadatos asociados.

La AP histórica 0,18502078 del modelo con ranking **no era una estimación de la
AP oficial**. Sobre la población raw y exactamente los mismos matches del nuevo
perfil, cambiar únicamente la fórmula a AP101 con envolvente y sin mínimos da
0,10937358, frente a 0,06204229. Este control aísla el efecto de la definición
de AP en esa población. La diferencia restante respecto a la cifra histórica
incluye población, filtros e implementación de empates; no la atribuimos a una
única causa. Tampoco atribuimos el salto selección–test solo a generalización:
sus poblaciones difieren y no se ha certificado el scorer privado.

## Reproducción y verificaciones

```powershell
python scripts/evaluate_detection_benchmark_local.py `
  --shards D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards `
  --output D:/PHASE_TTC_BENCHMARK_ALIGNED_20261101/reproduction

python -m pytest tests/unit/test_detection_benchmark_evaluation.py `
  tests/unit/test_detection_official_result_analysis.py `
  tests/unit/test_detection_evaluation_v2.py `
  tests/unit/test_detection_evaluation_ap_fast.py -q
```

El destino debe ser nuevo. El runner verifica hashes, rol selection, ausencia de
etiquetas test, cobertura completa frente al recibo histórico, índices raw sin
huecos ni duplicados y checkpoint común. Los archivos fuente no pueden cambiar
durante la ejecución. La suma global se calcula después de unir los matches;
no se promedian AP por shard. El orden de entrada queda fijado por los manifiestos.

Las pruebas cubren fronteras estrictas, duplicados, empates de score y distancia,
profundidad fuera del rango antiguo, yaw circular, errores de tamaño, clases sin
GT, recall bajo, interpolación, rechazo de poblaciones incompletas y comparación
con un matcher global independiente. También verifican la aritmética oficial y
las regresiones del evaluador histórico. Ruff y Pyright forman parte del cierre.

## Decisión antes de volver a entrenar

Se conserva el candidato con ranking por su mejora pareada en selección completa,
sin afirmar significancia entre semillas ni mejora oficial sobre un baseline que
no se ha enviado a test. No se ajustan hiperparámetros mirando este resultado test.

Este perfil versionado pasa a ser la referencia primaria de comparación local
para decisiones futuras. El perfil v2 permanece como control histórico. Los
productores y runners congelados siguen conservando su configuración original;
esta entrega añade un evaluador externo, no modifica sus hashes o sus selecciones
de checkpoint retrospectivamente. Cualquier nuevo runner de entrenamiento debe
declarar y usar explícitamente este perfil para selección de checkpoints antes
de iniciar una campaña: no basta con renombrar la antigua AP.

La próxima intervención debe responder al recall y geometría por clase y rango
en desarrollo, especialmente clases raras y objetos excluidos por el rango del
entrenamiento anterior. Se conserva el presupuesto hasta disponer de ese
diagnóstico; sumar updates por sí solo no resuelve la discrepancia de tarea.
