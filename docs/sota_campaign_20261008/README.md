# Campaña TTC 2026-10-08 — evidencia reproducible

Este directorio es una vista sanitizada y regenerable de artefactos medidos. No afirma estado del arte ni convierte ramas en ejecución en resultados.

## Estado de las ramas

| Rama | Estado | Lectura |
|---|---|---|
| `scoring_existing_946` | `COMPLETE_EXPLORATORY` | 946 GT-eligible rows; exploratory frozen historical cohort |
| `expanded_metrics` | `COMPLETE` | Full expanded cohort; no result is inferred while inference/scoring runs |
| `fcwd_assets` | `COMPLETE` | Three public FCWD sequences; small transfer set with uncertified ancestry |
| `fcwd_metrics` | `COMPLETE` | H8/event-only comparison; unavailable comparators remain null |
| `garl_parity` | `PASSED` | Native preprocessing parity and declared EvTTC transfer adaptations |
| `system_cost` | `COMPLETE` | Fixed-eight host observation; warm-cache and architecture contexts differ |
| `baseline_failure_inventory` | `COMPLETE_DIAGNOSTIC` | Development-32 coverage diagnostic; scorable-subset metrics remain conditional |
| `official_contract` | `METADATA_READY_OFFICIAL_REPRODUCTION_BLOCKED_CONTRACT_UNRESOLVED` | Metadata evidence may be complete while official reproduction remains blocked |
| `r1_measurement` | `WAIT_TRANSIENT_BACKOFF` | Resume state is reported literally and never promoted by inference |

## Alcance científico

`scoring_existing_946` es exploratorio y procede de un conjunto histórico de 946 targets elegibles. Los resultados expandidos sólo se publican cuando su manifest de hashes verifica los seis productos del scorer. Los intervalos bootstrap son descriptivos, por secuencia o grupo completo, sin corrección por multiplicidad. Las métricas sobre predicciones puntuables sí se incluyen, etiquetadas como condicionales y acompañadas por cobertura y tasa de fallo; no sustituyen la cohorte completa.

FCWD contiene tres secuencias públicas: sirve como transferencia pequeña y no certifica ascendencia independiente. Los assets están completos, pero eso no hace usable la calibración full: el MAT es MCOS, las cajas son 1280×720 de eventos y RGB es 1920×1200 sin transformación verificada. Event-only tiene 630 consultas viables; full está bloqueado por el mapping de entrada, no por un resultado negativo.
La población FCWD de 630 consultas sólo cubre esas tres secuencias: sus intervalos por clúster tienen poca independencia. La expansión dev32 añade timestamps a las 32 secuencias ya observadas; no constituye un holdout nuevo.

H8 usa sólo eventos durante esta inferencia, pero su entrenamiento no fue SSL JEPA puro: empleó teacher RGB DINO y supervisión geométrica. Las semillas 7/13/23 son tres cabezas H8 sobre productores compartidos, no tres réplicas independientes de los productores. Ningún error Garl se filtra por rendimiento; fallos y cobertura permanecen en la cohorte y las vistas puntuables se etiquetan condicionales.
H8 limita su salida nativa a ±60 s; Garl conserva la división nativa sin clipping. Cuando sus dos alturas predichas son casi iguales, Garl puede emitir TTC finitos extremos. Por eso se muestran conjuntamente MAE y mediana, sin eliminar filas. La mediana no tiene intervalo bootstrap en este análisis; una mediana menor no se describe como una diferencia estadísticamente demostrada. El diagnóstico de extremos y cualquier sensibilidad post hoc se conservan separadamente en `evidence/root_qa/full_outlier_audit/`; no sustituyen las métricas nativas.

El inventario de baselines de desarrollo retiene fallos. Las métricas del subconjunto puntuable se publican sólo como condicionales, con cobertura explícita. Los costes son observaciones del host sobre ocho consultas, con caché caliente y contextos distintos; no demuestran causalidad de arquitectura ni son comparables directamente con latencias publicadas. El protocolo mide batch 1 en float32 sin TF32 y separa preparación, GPU y E2E. H8 y EO se miden con un único worker GPU del proyecto en el escritorio Windows WDDM; los clientes gráficos ambientales y la telemetría quedan registrados. No se afirma exclusividad física ni ausencia de contención del escritorio. H8 y EO comparten la preparación cruda; la preparación/E2E de EO es conservadora porque también construye history8. El piloto de tres consultas no prueba un 2,59× sostenido.

La paridad Garl separa reproducción nativa de adaptaciones de transferencia EvTTC. La Tabla VI publicada no comparte un protocolo demostrado idéntico con nuestro RTE, por lo que no se presenta como comparación directa ni como reproducción oficial. El contrato oficial conserva dependencias bloqueadas y `AUTHORS_DRAFT.txt` es un borrador no enviado. El presupuesto restante no admite el plan inicial de nuevo entrenamiento; esta campaña registra cero updates. R1 registra 486 pares preexistentes preservados en un límite de grupo y registra cero avances de prefijo. Los grupos guardados no releen payloads, aunque CachedEventReader sí abre handles; cachés OS/HDF5 no están controladas y el timing final puede mezclar tramos antiguos y nuevos. No se afirma timing bit-exact ni una medición cold/warm ininterrumpida. Se informa el estado literal del recibo vigente; R1 y el coste del sistema TRAIN40 son mediciones distintas.

Fuentes primarias: [EvTTC](https://nail-hnu.github.io/EventAidedTTC/), [Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC) y [FCWD](https://nail-hnu.github.io/EventAidedTTC/).

Ejecute `python regenerate.py --verify` desde este directorio para verificar hashes y reproducir exactamente las tablas desde la evidencia publicada. Esta verificación comprueba `SHA256SUMS.txt`; el `.sha256` del ZIP es el ancla externa sólo si se conserva u obtiene por separado. Ningún manifiesto autocontenido protege frente a la sustitución coordinada del snapshot y de todos sus hashes.

## Resultados medidos

| Rama | Método | Población | GT elegibles | Cobertura | RTE cohorte completa (%) | MAE (s) | Mediana AE (s) | RTE condicional (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| scoring_existing_946 | H8_seed7 | 1024 | 946 | 1.0 | 62.35138260868976 | 1.3208407080791975 | 0.7660472305174808 | 62.35138260868976 |
| scoring_existing_946 | H8_seed13 | 1024 | 946 | 1.0 | 62.9234239935187 | 1.3528085341054652 | 0.7314128477156492 | 62.9234239935187 |
| scoring_existing_946 | H8_seed23 | 1024 | 946 | 1.0 | 68.86556210052743 | 1.408714142553364 | 0.7384638920614139 | 68.86556210052743 |
| scoring_existing_946 | public_Garl_event_lhr | 1024 | 946 | 1.0 | 293.47190329233473 | 8.048108926636752 | 0.996929195177185 | 293.47190329233473 |
| scoring_existing_946 | public_Garl_rgb_event_full | 1024 | 946 | 1.0 | 76.76860192016733 | 1.8682761924690516 | 0.6915115955657469 | 76.76860192016733 |
| expanded_metrics | H8_seed7 | 3947 | 3640 | 1.0 | 53.70086692017942 | 1.3612839246912378 | 0.7218486548063361 | 53.70086692017942 |
| expanded_metrics | H8_seed13 | 3947 | 3640 | 1.0 | 54.521035404161324 | 1.394969588613207 | 0.6992462247615228 | 54.521035404161324 |
| expanded_metrics | H8_seed23 | 3947 | 3640 | 1.0 | 59.14501214466352 | 1.4564915559295613 | 0.7020513066353757 | 59.14501214466352 |
| expanded_metrics | public_Garl_event_lhr | 3947 | 3640 | 1.0 | 161.12069433648364 | 4.55971165260342 | 0.9578366008360603 | 161.12069433648364 |
| expanded_metrics | public_Garl_rgb_event_full | 3947 | 3640 | 1.0 | 5180.181656628092 | 94.50960188415303 | 0.6614158970779118 | 5180.181656628092 |
| fcwd_metrics | H8_seed7 | 630 | 597 | 1.0 | 30.196258315185233 | 1.77772540445348 | 0.6749509534230853 | 30.196258315185233 |
| fcwd_metrics | H8_seed13 | 630 | 597 | 1.0 | 27.406551092950412 | 1.5973608457606467 | 0.5797485286451121 | 27.406551092950412 |
| fcwd_metrics | H8_seed23 | 630 | 597 | 1.0 | 27.482216621416004 | 1.5692206127233552 | 0.597822276902571 | 27.482216621416004 |
| fcwd_metrics | public_Garl_event_lhr | 630 | 597 | 1.0 | 26.887265328829773 | 2.146172768569033 | 0.6017179625853717 | 26.887265328829773 |

Cobertura se refiere exclusivamente a GT elegibles. Las filas sin GT válido permanecen en la población y su tratamiento sigue la regla predeclarada. Para la rama `expanded_metrics`, los contrastes pareados y sus intervalos por secuencia/grupo completos están en `evidence/expanded_metrics/PAIRED.csv`; no cubren las filas históricas ni FCWD de esta tabla. Describen esos cinco sistemas con sus distintos contratos de salida y no una superioridad intrínseca de arquitectura.

## Coste observado

| Sistema | Etapa | Media (ms) | p50 (ms) | p95 (ms) |
|---|---|---:|---:|---:|
| garl_event_only_shared_preparation | cpu_prepare | 847.4050208326667 | 838.4263999904999 | 1163.3313050099998 |
| garl_event_only_shared_preparation | gpu_inference | 8.227323748893749 | 7.596350013 | 11.692054999199986 |
| garl_event_only_shared_preparation | sequential_end_to_end | 1013.7435333332501 | 981.994950009 | 1837.7643450046985 |
| garl_full_rgb_event | cpu_prepare | 493.9231958305833 | 485.552800004 | 670.9909700093998 |
| garl_full_rgb_event | gpu_inference | 39.3674087486875 | 38.573750003499995 | 44.90349497494998 |
| garl_full_rgb_event | sequential_end_to_end | 639.3396500012917 | 614.5687500105 | 885.0847800075998 |
| h8_system_three_heads | cpu_prepare | 847.4050208326667 | 838.4263999904999 | 1163.3313050099998 |
| h8_system_three_heads | gpu_inference | 140.45620250143125 | 129.2952999935 | 209.76284500784988 |
| h8_system_three_heads | sequential_end_to_end | 1374.570583334125 | 1374.7459500155 | 1631.91426499575 |

## Fallos de baselines en desarrollo-32

Se conservan todos los fallos. Las métricas condicionales, cuando existen, no se presentan como rendimiento de cohorte completa.

| Método | Resultado o causa | Conteo |
|---|---|---:|
| cmax | RuntimeError:cmax_invalid:non_approaching_best_warp | 32 |
| strttc | RuntimeError:Fewer than 12 valid STRTTC normal-flow measurements. | 14 |
| strttc | RuntimeError:insufficient_roi_events:1375 | 1 |
| strttc | RuntimeError:insufficient_roi_events:1583 | 1 |
| strttc | RuntimeError:insufficient_roi_events:1977 | 1 |
| strttc | RuntimeError:non_positive_or_non_finite_endpoint_inverse_ttc | 4 |
| strttc | SUCCESS | 11 |

## Diagnóstico CMax

El adaptador CMax local inicial obtuvo 0/32 porque usaba un warp exponencial referido al último evento. Esa formulación era incorrecta para la expansión afín de inverse-TTC. La adaptación corregida usa tiempo de referencia fijo y obtuvo 15/32. Ambos son diagnósticos locales; no son fallos ni reproducciones del método publicado.

| Variante | Éxitos | Solicitadas | Interpretación |
|---|---:|---:|---|
| initial_local_cmax | 0 | 32 | Incorrect exponential/latest-event warp in the initial local adapter; not a failure of the published CMax method |
| affine_corrected_cmax_reference | 15 | 32 | Corrected affine reference-time local adaptation; diagnostic, not a paper-result reproduction |

## Piloto condicional de baselines geométricos

Los soportes se muestran en la tabla. Son cohortes de éxito distintas: las cifras condicionales describen los casos puntuables y no forman un ranking entre métodos.

| Método | Soporte | MAE condicional (s) | RMSE condicional (s) | No rankeable |
|---|---:|---:|---:|---|
| cmax_reference | 15 | 3.7314863020124296 | 6.243125731077465 | True |
| strttc_adapted | 11 | 6.512758705439506 | 9.53342631595942 | True |
