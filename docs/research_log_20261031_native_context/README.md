# PHASE-TTC: ampliación mediante reinicios del contexto nativo

Informe generado desde una extracción y una comparación completadas. Los datos pertenecen al desarrollo con exposición histórica; los resultados no equivalen a una evaluación oficial ni a una reserva nueva.

## Qué se construyó y comprobó

Se extrajeron 149 consultas: tres cuantiles temporales de cada una de las 24 grabaciones de carretera elegibles y las 77 de Slider-500. La selección no utilizó errores ni targets. Cada consulta conserva observaciones raw17/expert3, relojes, ruta de inferencia y tensores originales del contexto y cabeza.

La reproducción limpia vuelve a leer NPZ y JSON verificados por hash y compara exactamente los tensores con la inferencia instrumentada. Se conservan las rutas de fusión, solo eventos y ausencia; las consultas no disponibles permanecen en los artefactos. El recibo complementario verifica las dependencias numéricas que no figuraban en el inventario inicial de fuentes.

Se generaron 447 variantes: últimas una, dos y tres observaciones aceptadas por modalidad, con relojes contextuales recalculados. La primera observación retenida tiene intervalo desde la anterior igual a cero. Los productores mantienen sus ventanas sensoriales originales: esto es reinicio de contexto, no arranque del sensor ni ampliación por cambio de velocidad.

Rutas conservadas: `{"fused": 148, "unavailable": 1}`. Tiempo instrumentado: 463.508 s. Incluye extracción, guardado y comprobaciones; no es latencia de producción.

## Entrenamiento y criterio de selección

Se ejecutaron 48 ajustes: dos condiciones, ocho familias reservadas y semillas 7,13,21. Se conservaron arquitectura de 688 parámetros, 300 actualizaciones AdamW, lr 0.01, weight decay 0.01 y recorte de gradiente a 1. Cada variante permanece en el grupo de su consulta. Normalización y escala de error usan solo train limpio.

IQ reproduce el control original: mitad de pérdida limpia y mitad con corrupción de un experto en el 25 % de las filas. IQn sustituye ese segundo término por los reinicios nativos, equilibrando familias y consultas y dividiendo el peso de cada consulta entre sus variantes admitidas. Geometría y sus 20 variables de calidad permanecen iguales. No se añaden entradas de soporte nativo a la red.

La selección exige cobertura completa y no empeorar MAE ni error relativo macro por grabación, tanto en carretera como en Slider-500. Las métricas de reinicio reservado son diagnósticas y no seleccionan el candidato.

| Condición | RAE carretera macro (%) | MAE carretera macro (s) | RAE Slider-500 (%) | MAE Slider-500 (s) |
|---|---:|---:|---:|---:|
| IQ | 17.748872 | 0.543418 | 20.851858 | 0.317744 |
| IQn | 17.429901 | 0.547671 | 25.885725 | 0.459413 |

Selección: `None`. Criterios: `{"road_macro_record_mae_seconds": false, "road_macro_record_relative_error_pct": true, "slider_macro_record_mae_seconds": false, "slider_macro_record_relative_error_pct": false}`.

La campaña no promueve automáticamente un modelo. Un candidato admitido requiere refit y comprobación activa de conservación eAP; TRAIN40 es resustitución y Test12 sin targets solo permite medir deriva. Slider-750 sigue necesitando conciliación con el evaluador oficial. Ninguna cifra de este informe demuestra SOTA.

## Reproducibilidad

Los recibos publicados conservan configuración, grupos, semillas, hashes y resultados. El runner `train_phase_native_context_fusion.py` expone los argumentos con `--help`; requiere directorios nuevos y las cachés selladas. Datos crudos y pesos no se incorporan a Git.
