# Scientific Recovery V7: cierre del screen inicial

Fecha de corte: 2026-08-14. Rama:
`scientific-recovery-v7-balanced-oof`. Los cuatro brazos iniciales completaron
tres folds OOF con seed 7 sobre los mismos 8.192 tokens. Public validation,
private test, EvTTC test y CodaBench permanecieron cerrados.

## Resultado OOF firmado

A5 revaluado obtiene `158.449 MiD` a cobertura puntual completa y `155.374 MiD`
selectivo con `95.227%` de cobertura. Garl local obtiene `144.353 MiD`, cobertura
completa y `0%` de failure.

| Brazo | MiD puntual | Delta vs A5 | MiD selectivo | Cobertura selectiva | P(delta<0) |
|---|---:|---:|---:|---:|---:|
| V7-SOFT | 165.116 | +6.668 | 162.283 | 94.446% | 0.0004 |
| V7-C2F | 158.573 | +0.125 | 156.324 | 94.958% | 0.4460 |
| V7-T20 | 165.260 | +6.812 | 162.257 | 95.288% | 0.0020 |
| V7-CAP-S | 167.025 | +8.576 | 163.212 | 94.995% | 0.0002 |

Intervalos bootstrap emparejados, 422 clusters `sequence_id+track_id`, 5.000
réplicas y seed 20260813:

- SOFT: mediana `+6.703`, IC95% `[+3.084,+10.547]`.
- C2F: mediana `+0.215`, IC95% `[-3.025,+3.432]`.
- T20: mediana `+6.877`, IC95% `[+2.290,+11.514]`.
- CAP-S: mediana `+8.683`, IC95% `[+3.735,+13.630]`.

Los cuatro brazos produjeron exactamente 8.192 predicciones OOF únicas, tres
folds, nueve secuencias, `100%` de puntos finitos y `0%` de failure puntual. Todos
fallaron `mechanism_positive`, `geometry_positive` y `confirmation_candidate`.

## Geometría

La retención se calcula contra los parents A4 fold-locales. El gate exige al menos
60% y signo positivo en las cuatro medidas.

| Brazo | Slope bbox | Slope física | Std ratio bbox | Std ratio física |
|---|---:|---:|---:|---:|
| V7-SOFT | 20.6% | 19.8% | 28.6% | 28.6% |
| V7-C2F | 15.7% | 15.3% | 24.1% | 24.0% |
| V7-T20 | 12.8% | 11.3% | 24.4% | 24.3% |
| V7-CAP-S | 22.6% | 21.7% | 28.4% | 28.3% |

SOFT mejora algo la geometría relativa a los otros brazos, pero queda lejos del
umbral. La distillation no compensa la destrucción de señal temporal con el encoder
completamente entrenable.

## Inferencia y decisión

- C2F es compatible con efecto nulo: sus folds se cancelan y el intervalo cruza
  cero. No hay base para añadir radios o routers.
- T20 empeora en los tres folds. Más bins por polaridad no explica el gap con Garl
  en esta arquitectura; no autoriza ASTW.
- CAP-S empeora en los tres folds y no conserva geometría. Cap-M queda bloqueado.
- SOFT empeora TTC y falla geometría. El árbol preregistrado permite un solo
  control: congelar `encoder.features[0:3]` y mantener teacher, losses y pesos.

El control parcial está congelado antes de entrenar en
`configs/experiment/scientific_recovery_v7_fold_chain/soft_partial_freeze_manifest.json`,
artefacto `c41fabd0c4e12220d531bc39748aea40de078a316e14c9eca5b599bd93ecf174`.
El estudiante parte de cero; solo las tres primeras etapas dejan de recibir
gradientes. No se permite barrer capas ni pesos.

Si el control no cumple a la vez integridad, retención geométrica y mejora de al
menos 3 MiD con `P(delta<0)>=0.90`, V7 se cierra como negativo. No se ejecutarán
seeds 13/23 ni la ablation JEPA sin candidato.

## Fuentes auditables

- `artifacts/scientific_recovery_v7/results/{soft,c2f,t20,cap_s}_seed7_oof.json`
- `artifacts/scientific_recovery_v7/audit/{soft,c2f,t20,cap_s}_geometry.json`
- `artifacts/runs/scientific_recovery_v7_*_fold*_seed7/summary.json`
- `configs/protocol/scientific_recovery_v7_balanced_oof.json`

Los directorios `artifacts/` están ignorados por Git. Sus JSON incluyen hashes de
artefacto; deben preservarse junto al commit y verificarse antes de reproducir el
dictamen. No se hizo push, no se avanzó `main` y no se abrió ningún test externo.
