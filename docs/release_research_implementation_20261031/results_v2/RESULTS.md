# Diagnóstico congelado O0/D1–D3

Población de desarrollo históricamente expuesta. No es confirmación independiente.
Los controles CLOCK usan PAIR de eventos; su referencia es EVENT_CONTEXT.
No se han reajustado pesos ni modificado el release oficial.

| Comparación | MAE macro referencia (s) | MAE macro control (s) | IC95 delta (s) | Nuevas ausencias |
|---|---:|---:|---|---:|
| CLOCK_ewma | 0.766024827 | 1.958482093 | [0.7840848048148972, 1.6484118202446318] | 0 |
| CLOCK_ewma_vs_event_context | 1.947555324 | 1.958482093 | [-0.7931645354137086, 0.5871154207231702] | 0 |
| CLOCK_observer | 0.766024827 | 1.987680041 | [0.6792849725477006, 1.974475992651623] | 0 |
| CLOCK_observer_vs_event_context | 1.947555324 | 1.987680041 | [-0.3861329442262649, 0.39023782535812307] | 0 |
| CLOCK_transport | 0.766024827 | 3.099852894 | [0.9987284169678103, 4.1251406635513925] | 0 |
| CLOCK_transport_vs_event_context | 1.947555324 | 3.099852894 | [-0.25305021897951746, 2.716184594472248] | 0 |
| OBS_HALF | 0.766024827 | 0.802404742 | [-0.042885638395945225, 0.11983813406291755] | 0 |
| RGB | 0.766024827 | 0.881464277 | [-0.05335746595002057, 0.2972527024400862] | 0 |

El bootstrap conserva adquisiciones completas; los intervalos son descriptivos.
La pérdida original de cobertura se conserva en el denominador.
OBS_HALF es un control constante prefijado, no un gate de observabilidad.
No se infiere Overall nativo desde esta población ni se promociona un candidato.
