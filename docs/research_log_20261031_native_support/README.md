# PHASE-TTC: soporte temporal nativo y control de capacidad

Esta comparación continúa tras el rechazo de la ampliación por reinicios de contexto. Se ejecuta sobre desarrollo con exposición histórica; no usa las consultas oficiales para ajustar ni constituye una evaluación SOTA.

## Información nueva y procedencia

Se recuperan diez valores existentes en el estado de 340 dimensiones: por modalidad, edad desde el último anchor, edad desde disponibilidad, último intervalo entre anchors, disponibilidad menos anchor y número de observaciones ejecutadas dividido por ocho. Son los índices 165:170 y 335:340. No codifican duración completa de historia ni primer evento del sensor.

Las tablas incluyen 2.608 consultas de desarrollo, 447 reinicios, 88.744 filas TRAIN40 y 6.762 filas lógicas Test12. Se recuperan sin ejecutar productores ni leer targets. Las 72 carreteras reextraídas y 76 consultas Slider disponibles reproducen exactamente las coordenadas históricas. Slider-500:76 y sus tres reinicios conservan ausencia explícita, con ceros como marcadores y máscaras de contexto ejecutado separadas para auditoría.

TRAIN40 contiene estados upstream físicos OOF de cinco folds; Test12 contiene estados de despliegue y carece de etiquetas. Los recuentos de carretera son constantes 5 RGB / 4 eventos; TRAIN40 tiene predominantemente 7 observaciones de eventos y Test12 predominantemente 4. Estas variables pueden informar del dominio: no interpretar correlación como causalidad.

## Comparación fijada antes de entrenar

IQ es el control original cuyo vector OOF se reprodujo bit a bit en la campaña anterior. F2 añade las diez entradas: 43→18→4, 868 parámetros. Fcap usa solo las 33 originales: 33→23→4, 878 parámetros. Fcap tiene 10 parámetros más; no se afirma igualdad de capacidad funcional.

Se ejecutan tres semillas (7,13,21), ocho folds por familia y 300 updates AdamW por ajuste. La pérdida conserva el algoritmo QA: mitad limpia, mitad con corrupción de un experto en el 25 % de las filas. Arquitecturas distintas consumen RNG distinto al inicializar: las perturbaciones tienen la misma distribución y semillas, no máscaras idénticas por actualización.

Normalización y escala del MAE usan exclusivamente train limpio. En F2, las coordenadas nuevas con varianza de train menor de 1e-12 quedan enmascaradas a cero también en inferencia. Cada máscara se guarda por fold y semilla. Los reinicios son diagnóstico secundario y no entran en train.

En el fold que reserva Slider, las carreteras de ajuste solo permiten aprender variación en el último intervalo RGB. Las otras nueve coordenadas nuevas, incluidos ambos recuentos y todo el soporte de eventos, quedan enmascaradas por ser constantes. Esta comparación no puede demostrar aprendizaje de una respuesta al arranque con pocos eventos en ese fold.

## Resultados primarios limpios

| Condición | RAE carretera macro (%) | MAE carretera macro (s) | RAE Slider-500 (%) | MAE Slider-500 (s) |
|---|---:|---:|---:|---:|
| IQ | 17.748872 | 0.543418 | 20.851858 | 0.317744 |
| F2 | 18.023060 | 0.562809 | 20.369949 | 0.311921 |
| Fcap | 17.141145 | 0.533926 | 21.666019 | 0.351925 |

Selección: `None`. Tiempo de campaña: 74.707 s.

Cada candidato exige cobertura completa y las cuatro métricas no peores que IQ. Si ambos pasan, se selecciona la menor media de las cuatro razones candidato/IQ, con desempate F2. La tabla completa por semilla, criterios y diagnósticos está en los recibos JSON.

Dominancia de F2 sobre Fcap: `False`. Incluso una dominancia es una asociación compatible con utilidad del soporte, no una prueba causal aislada de inicialización y dominio.

## Límite de esta entrega experimental

Esta campaña no exporta ni promueve automáticamente un modelo. Un seleccionado necesita refit y comprobación activa eAP frente a la entrega congelada; TRAIN40 sigue siendo resustitución. El diagnóstico Test12 sin etiquetas no certifica precisión. Resolver la discrepancia del evaluador Slider-750 y una evaluación comparable siguen siendo necesarios antes de afirmar SOTA.

El informe se regenera con `scripts/report_phase_native_support.py`. Los datos crudos, matrices y checkpoints permanecen fuera de Git; se publican código, metodología y recibos verificables.
