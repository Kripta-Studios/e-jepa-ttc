# SIMPLEX-T: resultados del 3 y 4 de octubre de 2026

H8 conserva su identidad histórica. H16 mejora modestamente la precisión local; las simplificaciones no cumplen el cribado fijado. El trabajo de entrenamiento está completo; el perfilado de la ruta real está parcialmente bloqueado.

Se completaron 24 cabezas hasta update 2.500 (60.000 updates científicos): seis réplicas H16, doce ajustes de dependencia de expertos y seis de agregación. Se contabilizan además 80 updates sintéticos y una cota total física de 60180 updates, incluidos los potencialmente repetidos o inciertos. T6 y sus 72 fits anteriores se reutilizan. Los análisis posteriores y esta síntesis no ejecutan optimizadores.

MiD mide el error ponderado en PHASE17: **menor es mejor**. No son milisegundos de TTC ni de reacción. La evaluación usa 8.192 consultas de nueve secuencias OLD_DEV, con masas y folds registrados; es desarrollo reutilizado, no confirmación independiente.

| Modelo | MiD OLD_DEV seed7 | p95 cabeza preparada, mediana de 3 bloques (ms) |
|---|---:|---:|
| H1_SEED7 | 132.831619 | 9.279 |
| H8_SEED7 | 121.646332 | 19.473 |
| H16_SEED7 | 118.865739 | 31.450 |
| FULL_C0 | 121.404706 | 21.074 |
| A5_ONLY_C0 | 127.785376 | 21.421 |
| C2F_ONLY_C0 | 126.704974 | 19.282 |
| A5_PAIR_C0 | 124.668732 | 19.826 |
| SET_AGE_C0 | 122.442604 | 9.996 |
| SET_NOTIME_C0 | 123.686600 | 9.914 |

**H16 frente a H8.** Media de pérdidas de tres seeds: 119.241356 frente a 121.784022; mejora relativa 2.09 %. Delta -2.542666; CI95 jerárquico [-5.2133978893639155, -0.11124053108043853]. Las tres seeds son favorables. Las dos nuevas tienen delta -2.423702, CI95 [-5.244259454753143, 0.04109148518456987]: aún incluye cero. Seed7 ya era exploratoria. Se replican cabezas, no expertos ni escenas; no se promedian TTC para crear un ensemble. La secuencia OBneIVg4Cw empeora en las tres seeds: la ganancia no es uniforme.

**Quitar expertos.** FULL_C0 controla el cambio de objetivo auxiliar (lambda_cost=0). A5_ONLY, C2F_ONLY y A5_PAIR empeoran frente a ese control; ninguno satisface los límites prospectivos de precisión. Un intervalo que cruza cero no demuestra equivalencia. PAIR comparte el encoder A5; no cuenta como otro encoder independiente.

| Contraste nuevo | Delta MiD | CI95 jerárquico | Cribado de precisión |
|---|---:|---|---|
| FULL_C0_minus_H8@7 | -0.241626 | [-0.6210695932243308, 0.14280577379537274] | Pasa |
| A5_ONLY_C0_minus_FULL_C0 | 6.380670 | [-0.09726888418760105, 13.804899200544122] | No pasa |
| C2F_ONLY_C0_minus_FULL_C0 | 5.300268 | [-0.1472445951258283, 11.033927080586398] | No pasa |
| A5_PAIR_C0_minus_FULL_C0 | 3.264025 | [-2.4680534514039114, 8.889672735990562] | No pasa |
| SET_AGE_C0_minus_FULL_C0 | 1.037898 | [-3.0355556542166293, 4.6344481546170835] | No pasa |
| SET_NOTIME_C0_minus_SET_AGE_C0 | 1.243996 | [-0.6957483995635467, 2.6766788018183627] | No pasa |

**Cambiar la GRU.** SET_AGE y SET_NOTIME cuestan aproximadamente la mitad como cabezas aisladas, pero sus intervalos no cumplen el margen registrado de +2 MiD. Son resultados exploratorios útiles, no sustitutos validados. Quitar tiempos explícitos no elimina necesariamente la información temporal que conservan los expertos.

**Por qué aporta algo frente a C2F.** El productor C2F compilado de esta campaña obtiene 158.942029 MiD; la referencia EWMA registrada 159.278795. H8 seed7 obtiene 121.646332. C2F sigue siendo un componente útil: SIMPLEX-T estudia cómo combinar y refinar sus observaciones con contexto pasado. El C2F oficial V7 y Garl tienen consultas y folds coincidentes tras resolver aliases, pero falta ligar completamente sus productores, TRAIN y disponibilidad del ROI. No se proclama superioridad sobre todos los históricos. La revisión reconciliada sustituye los flags preliminares que atribuían diferencias de población a columnas distintas o redondeo del CSV.

**Coste real y pendiente.** Se completaron 13.500 medidas emparejadas de nueve cabezas en CPU y se conservan 1.114 de 1.728 mediciones raw → contexto → expertos GPU → cabeza → TTC. Hay 41 consultas con las 27 mediciones completas, de dos de las tres familias previstas; los 64 contextos pasaron previamente la admisión numérica. En warm_block2, lectura y ROI/voxel suman 84.9 % del tiempo H8 y 90.0 % del H16. La cabeza supone menos del 1 % en esos dos casos. Son fracciones de la suma de tiempos observados, con GPU compartida y ROI suministrado; no son p95 sumados ni latencia AEB. Faltan 614 mediciones porque dejó de estar accesible E:/eAP_dataset/data/train. La exclusividad GPU dejó de ser un requisito por autorización posterior; los documentos anteriores conservan su estado histórico.

**Ingeniería y comprobaciones.** Se implementaron máscaras y anchors sin expertos excluidos, los dos agregadores, checkpoints completos, pruebas de reanudación, análisis y publicación por fragmentos, inferencia con productores congelados y tiempos por etapa. Se preservaron los fallos de recursos y el intento inicial con un runtime CUDA distinto; se restauró la configuración histórica antes de medir, sin ampliar tolerancias. El análisis posterior reconcilió 98.304 predicciones; el último bundle regeneró 398 entradas de cabeza y 27 filas de coste con diferencia cero. Los manifests verifican bytes y SHA-256; reproducir cachés no equivale a reconstruir datos crudos o entrenar expertos.

**Qué significa.** H8 mejora sustancialmente OLD_DEV y el contexto pasado aporta información. H16 es un candidato posterior de precisión, con ganancia pequeña e incertidumbre entre escenas. Las reducciones probadas no justifican sustituir el sistema. El mecanismo cronológico sigue sin demostrarse; tampoco hay evidencia de AEB más rápida, tracking online persistente o incertidumbre calibrada. LATENT conserva su diagnóstico histórico negativo; no se corrigió ni reentrenó en esta campaña.

**Continuación.** Primero recuperar E: y completar las 614 medidas ya autorizadas, conservando los fragmentos. La única propuesta posterior es optimizar la preparación retrospectiva raw/ROI con pesos congelados y paridad verificada, bajo un protocolo independiente. No se ejecuta esa propuesta ni se abren entrenamientos o holdouts.

[Descargar todos los bundles y sus hashes](https://github.com/Kripta-Studios/e-jepa-ttc/releases/tag/simplex-t-local-results-20261004). El bundle nocturno incluye también el ZIP H16, disponible por separado. T6 es la base histórica del 2 de octubre. Se publican los entregables finales y la última entrega parcial, no copias temporales redundantes. RELEASE_ASSETS.json inventaría el alcance; los raw externos no están incluidos.

Fuentes: [campaña nocturna](../simplex_t_nocturnal_20261003/INFORME_NOCTURNO_SIMPLEX_T.md), [diagnóstico y coste emparejado](../simplex_t_post_campaign_20261003/INFORME_POST_CAMPANA_SIMPLEX_T.md), [ruta GPU parcial](../simplex_t_shared_gpu_route_20261004/partial_1114/INFORME_GPU_COMPARTIDA_SIMPLEX_T.md). RESULTS_SUMMARY.json conserva cifras completas e INPUT_HASHES.json sus fuentes. Regeneración local: `python -B operational/build_simplex_t_results_review.py`.
