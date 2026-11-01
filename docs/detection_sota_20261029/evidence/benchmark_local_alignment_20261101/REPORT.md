# Evaluacion local alineada con eAP: seleccion completa

Perfil observado v1; equivalencia con el scorer privado NO certificada.
Solo etiquetas publicas de seleccion; pesos y scores congelados, sin entrenamiento.

| Variante | mAP | ATE | ASE | AOE | eAP compuesto |
|---|---:|---:|---:|---:|---:|
| baseline | 0.057017 | 0.890311 | 0.484148 | 0.517191 | 0.174180 |
| ranked | 0.062042 | 0.869129 | 0.483038 | 0.507858 | 0.181273 |
| scalar_uncertainty | 0.060531 | 0.875939 | 0.483311 | 0.511249 | 0.179019 |

| Clase | GT raw | Baseline AP | Ranked AP | Diferencia |
|---|---:|---:|---:|---:|
| car | 90884 | 0.319015 | 0.347893 | +0.028879 |
| bicycle | 789 | 0.000000 | 0.000000 | +0.000000 |
| truck | 38670 | 0.039700 | 0.043176 | +0.003476 |
| pedestrian | 14415 | 0.000392 | 0.000482 | +0.000089 |
| motorcycle | 39555 | 0.047429 | 0.045753 | -0.001676 |
| bus | 3335 | 0.034055 | 0.035639 | +0.001585 |
| tricycle | 2668 | 0.015547 | 0.023394 | +0.007847 |
| other_vehicle | 153 | 0.000000 | 0.000000 | +0.000000 |

Frames: 23958; GT raw: 190469.
GT elegible historico: 107308; antes ignorado: 83161.

Comparacion pareada sobre las mismas propuestas y poblacion. No mide significancia
entre semillas ni rendimiento oficial del baseline en test.
El test no se usa para tuning.
La metrica historica v2 conserva su definicion y sus resultados originales.
Los empates siguen orden inverso de indice de entrada; GT equidistante usa el primero.
Orden de entrada fijado por shards/tokens guardados y hashes de COMPARISON.json.

Referencia de matching/interpolacion: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py
Configuracion y aritmetica contrastadas con los agregados del organizador.
