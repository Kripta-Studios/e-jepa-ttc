# PHASE-TTC: estado, resultados y artefactos de detección eAP

Fecha de corte: 1 de noviembre de 2026. Base de implementación verificada:
`e6abc0440def083ad858719f9d56e32f7d3d74e9`, rama
`dev/eap-s0-evaluation-20261029`, repositorio `Kripta-Studios/phase-ttc`.

Este documento registra lo ejecutado. El documento compañero
[PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261101.md](PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261101.md)
describe el trabajo futuro y contiene el mensaje completo para otra sesión.
Las propuestas de ese plan no son experimentos terminados.

## 1. Conclusión que debe conservar la siguiente sesión

Disponemos de un detector RGB MonoDGP-R50 entrenado, un ranker aprendido solo en
TRAIN, un paquete reproducible y un envío oficial aceptado. El detector aprende
señal útil, sobre todo en coches. La calidad global de ocho clases es baja y no
justifica presentarlo como modelo de producción ni como SOTA.

La cifra local histórica y la oficial medían poblaciones y definiciones de AP
distintas. Ya añadimos una evaluación local alineada con el contrato observado
del benchmark y reevaluamos las predicciones guardadas. Esa corrección no mejora
los pesos: permite comparar con una métrica más pertinente y elegir el siguiente
experimento con evidencia.

El trabajo aprovechable incluye pesos, datos preparados, entrenamiento
reanudable, predicciones, integración, métricas y resultados negativos. La
continuación debe reutilizarlo. Una buena infraestructura no demuestra por sí
sola buena detección.

## 2. Trabajo completado

| Entrega | Hecho verificado | Límite de la conclusión |
|---|---|---|
| Piloto portable | Entrenamiento e inferencia independiente, bridge hacia PHASE | Demostración funcional inicial |
| R0/R2 y R2-rare | Comparaciones de desarrollo y selección de receta; R2-rare no pasó su regla final | Una semilla; protocolo histórico |
| Dfull | 70.279 frames únicos legibles, 281.116 exposiciones; checkpoint seleccionado en update 34.333 | Train completo de las **24 secuencias asignadas a train**, no las 40 públicas |
| Ranking | Ajuste en 4.096 frames TRAIN; comparación pareada de propuestas sobre selección completa | Cambia confianza/orden, no clase, geometría ni velocidad |
| Temporalidad | Cuatro brazos cortos; ninguno pasó el margen de promoción registrado | No demuestra que la historia causal sea inútil |
| Eventos | Tres brazos locales cortos; ninguno pasó el margen registrado | No demuestra que otras representaciones o recetas fallen |
| Paquete de inferencia | Checkpoint, ranker, productor congelado y verificación de hashes | Candidato de investigación |
| PHASE multiobjeto | Replay real; TTC firmado, límites de fase y abstenciones | No demuestra nueva precisión TTC; exposición TRAIN40 |
| Benchmark | JSON aceptado y resultado oficial obtenido | Un envío RGB; sin comparación oficial baseline/ranker |
| Evaluación alineada | Perfil nuevo, selección raw completa, tres variantes reevaluadas | Matching privado no certificado |

Fuentes: [decisión R2-rare](R2_RARE_FINAL_AND_DFULL_DECISION.md),
[resultados Dfull y posteriores](POST_DFULL_RESULTS.md),
[paquete de inferencia](RELEASE_CANDIDATE_20261101.md) y
[alineación del evaluador](BENCHMARK_LOCAL_ALIGNMENT_20261101.md).
Los documentos de lanzamiento conservan estados históricos; para saber qué
terminó, consultar resultados y recibos, no asumir que un lanzamiento sigue vivo.

## 3. Resultados numéricos

Los bloques siguientes se regeneran desde JSON con
`python scripts/build_detection_research_handoff.py --write`.
Sin `--write`, el script comprueba que las tablas siguen coincidiendo.

<!-- BEGIN GENERATED RESULTS -->
### Resultado oficial recibido (test privado)

| Métrica | Valor |
|---|---:|
| mAP | 0.05129508 |
| ATE / trans_err | 0.92764309 |
| ASE / scale_err | 0.56968339 |
| AOE / orient_err | 0.64073705 |
| eAP compuesto | 0.13980148 |
| Muestras | 35940 |
| Objetos GT | 194735 |
| Predicciones | 1796950 |

Método enviado: `PHASE-TTC MonoDGP-R50 RGB + Quality Ranking`; `uses_event=false`, `uses_image=true`.

### Selección completa con el perfil alineado

| Variante | mAP | ATE | ASE | AOE | eAP compuesto |
|---|---:|---:|---:|---:|---:|
| baseline | 0.05701726 | 0.89031084 | 0.48414809 | 0.51719072 | 0.17417958 |
| scalar_uncertainty | 0.06053089 | 0.87593926 | 0.48331091 | 0.51124909 | 0.17901940 |
| ranked | 0.06204229 | 0.86912914 | 0.48303802 | 0.50785847 | 0.18127323 |

Población: **23958 frames, 190469 GT raw y 1197900 propuestas** por variante.
Antes: 107308 GT elegibles y 83161 ignorados (43.66 %).
Ganancia del ranking: **0.00502503 mAP absoluta; 8.81 % relativa**.

### Diagnóstico por clase del candidato con ranking

AP en escala 0–1. Los recalls son fracciones en selección local raw.

| Clase | GT local | AP local | Recall local 2 m | Recall local 4 m | AP oficial |
|---|---:|---:|---:|---:|---:|
| car | 90884 | 0.347893 | 0.535727 | 0.606069 | 0.322367 |
| bicycle | 789 | 0.000000 | 0.076046 | 0.117871 | 0.000000 |
| truck | 38670 | 0.043176 | 0.201991 | 0.283165 | 0.010160 |
| pedestrian | 14415 | 0.000482 | 0.087617 | 0.192716 | 0.000000 |
| motorcycle | 39555 | 0.045753 | 0.255265 | 0.392517 | 0.049135 |
| bus | 3335 | 0.035639 | 0.187406 | 0.233583 | 0.028698 |
| tricycle | 2668 | 0.023394 | 0.167916 | 0.234258 | 0.000000 |
| other_vehicle | 153 | 0.000000 | 0.026144 | 0.071895 | 0.000000 |

### Separar fórmula y población

| Cálculo | Baseline | Ranking |
|---|---:|---:|
| Histórico v2, dominio antiguo | 0.16474566 | 0.18502078 |
| Control AP101 con envolvente, mismos matches raw nuevos | 0.09713875 | 0.10937358 |
| Perfil alineado, población raw | 0.05701726 | 0.06204229 |

El control de la segunda fila cambia la fórmula sobre los mismos matches de la
tercera. La primera también cambia población, filtros y reglas de empates.
No atribuir toda la diferencia histórica–oficial a una única causa.

Hash del perfil alineado: `797e6cbd1f5e6779c1521c900b7c36108d846cc757d9ec5c582e12bc15dd1ade`.
<!-- END GENERATED RESULTS -->

La columna oficial corresponde al envío con ranking. La columna local corresponde
a selección expuesta a desarrollo: no son las mismas secuencias. El recall local
usa matches greedy de clase correcta en el pool guardado. No es un oracle ni un
recall a un umbral operacional calibrado.

AP cero no implica que no haya ninguna caja acertada. Los mínimos de precisión
y recall pueden anular la AP de una clase que acierta algunos objetos. Aun así,
los recalls bajos de bicicleta, peatón y other_vehicle identifican debilidades
reales; no basta con cambiar cómo se presenta la cifra.

## 4. Qué cambió en la evaluación

El perfil histórico v2 restringía la población por geometría de cámara y trataba
objetos fuera del dominio como ignorados. Su AP101 utilizaba envolvente de
precisión, sin los mínimos del benchmark. Conservamos sus resultados y su hash.

El nuevo perfil `phase_ttc_eap_observed_reference_v1`:

- Incluye todas las anotaciones raw públicas de selección y las propuestas
  guardadas, sin el filtro local de profundidad/imagen.
- Usa ocho clases, distancias de centro XY de 0,5/1/2/4 m, matching greedy por
  score y comparación estricta `< umbral`.
- Interpola en 101 puntos; aplica `min_recall=0.1` y `min_precision=0.1`.
- Calcula ATE, ASE y AOE a 2 m con los defaults de bajo recall observados.
- Incluye las ocho clases en la macro. No evalúa velocidad.
- Valida el límite de 500 propuestas por muestra. Nuestras salidas contienen
  50; cambiar el límite del evaluador no crea propuestas nuevas.

Reproducimos la aritmética de las 32 AP y los 24 errores TP publicados por el
organizador. La implementación de matching/interpolación sigue la referencia
pública nuScenes. **No tenemos el código privado que certifique equivalencia
de todos los filtros, empates y convenciones del servidor eAP.**

Los errores TP globales incluyen defaults de clases con recall insuficiente.
No interpretar ATE como media simple de todas las cajas, ASE como porcentaje de
error de longitud ni el compuesto eAP como mAP. Tampoco usar este benchmark para
afirmar precisión de velocidad o TTC.

El evaluador externo nuevo no cambia la selección histórica de checkpoints.
**Los runners congelados siguen usando su configuración original.** Integrar
el nuevo perfil en la validación periódica y en la selección de checkpoints de
un productor nuevo es una tarea pendiente antes de otro entrenamiento largo.

## 5. Evidencia que no permite una conclusión más fuerte

Solo hay una semilla del detector final. Las secuencias de selección ya se han
consultado en varias decisiones. Falta certificar independencia por adquisición
y disponer de una auditoría final apropiada; no se debe llamar holdout virgen a
una partición reutilizada.

En la auditoría de admisión Dfull, other_vehicle tenía 53 anotaciones admitidas
y seis pares `(secuencia, track_id)` en train. Bicicleta tenía 655 y 69 pares.
Repetir esos pares añade exposiciones, no identidades independientes. Son
recuentos bajo la admisión antigua, no un censo nuevo de todos los objetos raw.
La siguiente fase debe recomputar ambos denominadores.

El muestreo raro probado mejoró algunos indicadores de clases raras pero no pasó
la regla conjunta de promoción. Los screenings temporales y de eventos fueron
cortos, con una semilla y protocolo antiguo. No justifican descartar para siempre
esas ramas ni repetirlas sin cambiar una hipótesis concreta.

El entrenamiento y la evaluación históricos excluyeron parte del rango de
profundidad. Es una hipótesis prioritaria para explicar limitaciones de cobertura,
no una prueba de que ampliar un número de configuración resolverá la detección.
Hay que revisar objetivos, discretización de profundidad, decoder y geometría.

No se ha verificado una posición en un leaderboard comparable. Un resultado
KITTI, una AP de Car o un experimento RGB+eventos con otro scorer no establece
qué puntuación de ocho clases conseguirá nuestro modelo.

## 6. Archivos que implementan y documentan el cierre

| Archivo | Función |
|---|---|
| `src/e_jepa_ttc/detection/benchmark_evaluation.py` | Evaluador local alineado y perfil versionado |
| `scripts/evaluate_detection_benchmark_local.py` | Reevaluación de shards de selección; verifica recibos y geometría |
| `tests/unit/test_detection_benchmark_evaluation.py` | Fronteras, empates, errores, población, referencia y aritmética oficial |
| `scripts/analyze_detection_official_result.py` | Análisis de agregados oficiales sin etiquetas privadas |
| `scripts/build_detection_research_handoff.py` | Regeneración/verificación de tablas de este documento |
| [Evidencia oficial](evidence/official_benchmark_20261101/ANALYSIS.md) | Score recibido, curvas y límites de interpretación |
| [Evidencia alineada](evidence/benchmark_local_alignment_20261101/REPORT.md) | Comparación global y por clase |
| [Recibo de comparación](evidence/benchmark_local_alignment_20261101/COMPARISON.json) | Fuentes, hashes, versiones y denominadores |
| [Validación](evidence/benchmark_local_alignment_20261101/VALIDATION.json) | 36 pruebas; Ruff/Pyright; dos ejecuciones con resultados idénticos |

Los 36 tests corresponden al cierre del evaluador y sus regresiones; no representan
la ejecución de toda la suite del repositorio. Esta reevaluación no necesitó
entrenar, inferir imágenes de nuevo ni utilizar un pod de pago.

## 7. Rutas locales y recuperación

Worktree que contiene este cierre en este PC:

```text
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-s0-evaluation
```

El proyecto abierto originalmente puede apuntar al hermano
`e-jepa-ttc-v18-joint-measurement`. No asumir que tiene el mismo checkout. Revisar
`git worktree list`, rama y commit antes de modificar código. El worktree de
evaluación contiene cambios ajenos al cierre, incluidos perfiles de optimización;
conservarlos y evitar `git reset --hard`, `git clean` o `git add .`.

| Ruta en este PC | Contenido y uso |
|---|---|
| `D:/PHASE_TTC_RELEASE_20261101/dfull-rgb-ranked-seed7` | Paquete de inferencia: `infer.py`, `MANIFEST.json`, `ranker.json`, productor y pesos |
| `D:/PHASE_TTC_DFULL_20261031/campaign/r2_dfull_seed7_batch8_v1/checkpoints/selected.pt` | Checkpoint evaluado y enviado |
| `D:/PHASE_TTC_DFULL_20261031/campaign/r2_dfull_seed7_batch8_v1/checkpoints/last.pt` | Estado final conservado; no asumir que es el seleccionado |
| `D:/PHASE_TTC_DFULL_20261031/campaign` | Estados y evidencias de entrenamiento/reanudación |
| `D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards` | 47 shards, predicciones baseline/ranked/control y GT público elegible+ignorado |
| `D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/restored/ranking_results_v2/ranker.json` | Ranker TRAIN conservado |
| `D:/PHASE_TTC_BENCHMARK_ALIGNED_20261101/verified` | Resultado final de la reevaluación alineada |
| `D:/PHASE_TTC_SUBMISSION_20261101` | Recibos y predicciones del envío test |
| `E:/eAP_dataset` | Datos públicos locales; comprobar disponibilidad y hashes antes de leer |

Hash SHA-256 del checkpoint seleccionado, vuelto a comprobar al redactar:

```text
c58c13297fd15638d89364a25b88245cd8be669ab8bfdbcb741f03df0e9215b3
```

Hash del manifiesto original del paquete:

```text
61fc1023d67cf63aec8409023ddbdc44d02041c7413c3c2f83361e69a9ab2fbc
```

El paquete de inferencia no sustituye los estados completos para reanudar.
No depender del volumen RunPod ni de que sigan vigentes las IP o credenciales de
conversaciones anteriores. Los pesos importantes están en este PC. No se afirma
una segunda copia externa comprobada en este cierre.

En `C:/Users/Álvaro Schwiedop/Downloads` quedaron el ZIP enviado
`PHASE_Dfull_RGB_ranked_eAP_3D_Detection.zip` y los resultados recibidos
`scoring_result.zip` y `prediction_result.zip`. El último solo contenía metadata
de ingestión; el scoring terminó con `exitCode: 0`. El test incluye una muestra
con detecciones vacías por un PNG público truncado cuya procedencia se verificó.
Consultar [la entrega al benchmark](BENCHMARK_SUBMISSION_20261101.md).

## 8. Reproducción mínima de la evaluación

Desde el worktree, el entorno ya usado en este PC es el del hermano
`../phase-ttc-main-integration/.venv`. Comprobar que existe; no reinstalar ni
actualizar dependencias de los productores congelados por rutina.

```powershell
& '../phase-ttc-main-integration/.venv/Scripts/python.exe' `
  scripts/evaluate_detection_benchmark_local.py `
  --shards D:/PHASE_TTC_LOCAL_CONTINUATION_20261101/ranking_local_v1/shards `
  --output D:/PHASE_TTC_BENCHMARK_ALIGNED_20261101/reproduction_new

& '../phase-ttc-main-integration/.venv/Scripts/python.exe' -m pytest `
  tests/unit/test_detection_benchmark_evaluation.py `
  tests/unit/test_detection_official_result_analysis.py `
  tests/unit/test_detection_evaluation_v2.py `
  tests/unit/test_detection_evaluation_ap_fast.py -q
```

El directorio de salida debe ser nuevo. No hace falta repetir esta evaluación
si se verifican los hashes y no cambia el evaluador. Evitar cargar de golpe los
JSON históricos de métricas de cientos de MB: leer los shards acotados.

## 9. Qué queda abierto

El siguiente trabajo es diagnóstico y adaptación al problema completo: geometría,
recall por clase/rango, objetivos entrenables, cobertura, validación del nuevo
trainer y una intervención que pueda atribuirse. La arquitectura RGB actual
sirve de baseline conservado. Su sustitución debe depender de esas pruebas.

Un modelo mejor en eAP tampoco cierra automáticamente TTC, tracking, latencia
operacional o seguridad. PHASE y sus productores TRAIN40 necesitan una evaluación
independiente si se desea reclamar mejora del sistema completo.
