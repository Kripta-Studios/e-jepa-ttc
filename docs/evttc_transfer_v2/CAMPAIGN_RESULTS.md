# PHASE-TTC: cierre de campaña local

La cabeza F1 está entrenada y empaquetada. La validación adicional encuentra una mejora del error relativo y regresiones en MAE e incertidumbre. Se conserva como candidato exploratorio; no se promueve como ganador universal ni SOTA.

## Población y alcance

21 ajustes: siete familias × semillas 7/13/21. Cada fold reinicia la cabeza original y conserva TRAIN40 en entrenamiento. Las ocho grabaciones solapadas están excluidas de esta validación; quedan 24 grabaciones y 2.531 consultas. Los siete grupos son familias de escenario, no adquisiciones certificadas. La exposición histórica de los productores impide denominarlo test independiente.

El modelo final ya entregado usa 88.744 estados TRAIN40 + 3.571 estados EvTTC de 32 grabaciones: 92.315 filas, dos épocas completas y 1.444 actualizaciones por semilla. Sus métricas sobre esas filas son ajuste. Los ocho componentes anteriores y normalizadores permanecen congelados. No se declara convergencia ni fine-tuning de todos los productores.

## Validación agrupada

| Modelo | Macro RAE % | Micro RAE % | MAE s | Cobertura |
|---|---:|---:|---:|---:|
| F0 | 30.600 | 27.664 | 0.8494 | 100.0% |
| F1 seed 7 | 25.468 | 24.703 | 0.8585 | 100.0% |
| F1 seed 13 | 26.188 | 25.501 | 0.8925 | 100.0% |
| F1 seed 21 | 25.938 | 25.183 | 0.8653 | 100.0% |

Los intervalos siguientes remuestrean familias enteras y mantienen emparejadas las diferencias por grabación. Son exploratorios y condicionales a esta población; tres semillas no aportan tres adquisiciones nuevas.

| Semilla | Diferencia macro, pp | Intervalo 95%, pp | Cobertura nativa 80% | Reescala solo train |
|---|---:|---:|---:|---:|
| 7 | -5.133 | [-12.289, 0.741] | 67.88% | 70.17% |
| 13 | -4.412 | [-11.891, 1.810] | 65.78% | 68.63% |
| 21 | -4.663 | [-12.135, 1.043] | 67.72% | 70.01% |

La reescala usa residuos del entrenamiento de cada fold, nunca su familia reservada. No es split conformal y no ofrece una garantía de cobertura. No se incorpora al bundle.

## Regímenes de TTC, semilla predefinida 7

| Régimen | Consultas | RAE F0 % | RAE F1 % | MAE F0 s | MAE F1 s |
|---|---:|---:|---:|---:|---:|
| positive_under_1s | 92 | 39.516 | 28.435 | 0.3084 | 0.2192 |
| 1_to_4s | 1724 | 31.520 | 26.497 | 0.7104 | 0.6060 |
| over_4s | 715 | 16.841 | 19.896 | 1.2542 | 1.5496 |

No hay targets negativos en esta población: no permite validar generalización del signo negativo. La regresión en TTC largo queda visible en la tabla.

## Slider-500 externo al ajuste actual

Pesos y 77 consultas congelados antes de leer valores TTC; predicciones selladas antes del join por identidad de frame y reloj (tolerancia 2 μs). Extrínsecas de Slider provisionales, proyección solo rotacional, parentesco de adquisición sin resolver. No equivale al score oficial de las diez secuencias.

| Método | Targets válidos | Salidas evaluables | RAE % | MAE s |
|---|---:|---:|---:|---:|
| F0 | 77 | 76 | 83.887 | 1.1782 |
| F1_13 | 77 | 76 | 91.022 | 1.2047 |
| F1_21 | 77 | 76 | 87.330 | 1.1600 |
| F1_7 | 77 | 76 | 87.578 | 1.1657 |
| RGB | 77 | 76 | 89.689 | 1.3195 |
| bbox_area | 77 | 76 | 27.949 | 0.5253 |
| bbox_height | 77 | 75 | 47.815 | 0.6290 |
| bbox_width | 77 | 74 | 24.870 | 0.5093 |
| event | 77 | 76 | 97.669 | 1.7443 |

Las comparaciones sobre consultas compartidas están en Slider500/REPORT.json. No se ocultan los huecos de cada método. Los controles CMax y STRTTC lineal son implementaciones locales acotadas en doce consultas prefijadas, no reproducciones oficiales; sus resultados y motivos de ausencia están en geometry/REPORT.json.

## Robustez sobre sensores

48 consultas: dos cuartiles temporales por grabación elegible, seleccionados sin mirar errores. Nueve condiciones sobre eventos crudos o ROI, mismo target y pesos; la condición limpia comprueba paridad con los estados guardados.

| Condición | F0 RAE % | F1 RAE % | F1 MAE s | Salidas F1 | Cobertura intervalo 80% |
|---|---:|---:|---:|---:|---:|
| background_5 | 32.865 | 22.451 | 0.5853 | 48/48 | 87.50% |
| clean | 32.912 | 22.317 | 0.5805 | 48/48 | 85.42% |
| drop_30 | 33.064 | 22.443 | 0.5800 | 48/48 | 87.50% |
| drop_70 | 34.325 | 23.345 | 0.5942 | 48/48 | 83.33% |
| drop_negative | 39.149 | 25.473 | 0.6658 | 48/48 | 87.50% |
| drop_positive | 41.872 | 29.414 | 0.7496 | 48/48 | 83.33% |
| event_roi_75 | 33.456 | 23.197 | 0.5997 | 48/48 | 87.50% |
| jitter_1000 | 32.891 | 22.345 | 0.5819 | 48/48 | 85.42% |
| jitter_200 | 32.956 | 22.363 | 0.5821 | 48/48 | 85.42% |

## Decisiones experimentales

T0 fijo: RAE 27.664% → 26.111%, pero el fixture de aceleración retrasa 100 ms el cruce de alerta de 0,5 s. No se incorpora automáticamente.

F2/Fcap: no abiertos. La anchura nativa no muestra una asociación consistente con error dentro de grabación; no se afirma que toda variable de calidad sea inútil. Densidad, soporte y calidad adicional requieren una hipótesis validada y control de capacidad antes de ampliar la cabeza.

P0/J0/J1/J2: no abiertos. Los probes sobre contextos congelados recuperan inverse-TTC por encima de la media de train. Eso no prueba que el encoder sea óptimo, ni que JEPA aporte movimiento; no hay evidencia suficiente para atribuir el fallo a representación.

## Límites que siguen abiertos

- Genealogía completa de ancestros/teachers y certificación de adquisiciones independientes.
- Equivalencia exacta con evaluador oficial y cobertura de las 434 consultas: las 11 fuera de cobertura y los 69 targets Slider originales no verificados no desaparecen.
- Confirmación Slider con extrínsecas intersensor autoritativas y una adquisición nueva.
- Reproducción oficial de comparadores geométricos y márgenes prácticos de promoción.
- Incertidumbre con cobertura nominal verificada y robustez en TTC negativo.

No se reutiliza esta evaluación externa para elegir otra variante. Una siguiente iteración que use sus errores deberá declararla desarrollo expuesto.
