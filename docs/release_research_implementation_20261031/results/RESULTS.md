# Diagnóstico congelado O0/D1–D3

Población de desarrollo históricamente expuesta. No es confirmación independiente.
Los controles CLOCK usan PAIR de eventos; su referencia es EVENT_CONTEXT.
No se han reajustado pesos ni modificado el release oficial.

| Comparación | MAE macro referencia (s) | MAE macro control (s) | IC95 delta (s) | Nuevas ausencias |
|---|---:|---:|---|---:|
| CLOCK_ewma | 0.766024827 | 1.958482093 | [0.7840848048148972, 1.6484118202446318] | 0 |
| CLOCK_ewma_vs_event_context | 0.881464277 | 1.958482093 | [0.6056190490218107, 1.6040934746754272] | 0 |
| CLOCK_observer | 0.766024827 | 1.987680041 | [0.6792849725477006, 1.974475992651623] | 0 |
| CLOCK_observer_vs_event_context | 0.881464277 | 1.987680041 | [0.48134026335519653, 1.8946112410501947] | 0 |
| CLOCK_transport | 0.766024827 | 3.099852894 | [0.9987284169678103, 4.1251406635513925] | 0 |
| CLOCK_transport_vs_event_context | 0.881464277 | 3.099852894 | [0.8711256766055236, 4.067975781758625] | 0 |
| OBS_HALF | 0.766024827 | 0.802404742 | [-0.042885638395945225, 0.11983813406291755] | 0 |
| RGB | 0.766024827 | 1.947555324 | [0.4552877636465587, 2.272602649765318] | 0 |

El bootstrap conserva adquisiciones completas; los intervalos son descriptivos.
La pérdida original de cobertura se conserva en el denominador.
OBS_HALF es un control constante prefijado, no un gate de observabilidad.
No se infiere Overall nativo desde esta población ni se promociona un candidato.
