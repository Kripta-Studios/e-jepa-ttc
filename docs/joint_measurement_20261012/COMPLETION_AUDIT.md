# V18: requisitos de cierre y alcance de la evidencia

Corte de revisión: 17-10-2026, tras evaluación A1 y coste completo; verificación de entrega todavía pendiente. Este documento deriva los requisitos del handoff, sus documentos
02–06 y el prompt autorizado. No sustituye la verificación final de los bytes
entregados ni convierte la existencia de un archivo en prueba de su contenido.
Las rutas de evidencia de la tabla son relativas a
`artifacts/joint_measurement_20261012/` salvo que se indique lo contrario.

| Requisito | Evidencia y estado al corte | Falta para el cierre |
|---|---|---|
| Admisión de base, padres P, roles y fuente sensorial | `ADMISSION.json`: ADMITTED; `PRIMARY_ENDPOINTS.json` y admisiones por rama | Conservar bindings y enumerar dependencias externas en la entrega |
| Cuatro fits primarios seed7 de 12.000 updates | Los cuatro `fits/*/CHECKPOINT_RECEIPT.json` registran COMPLETE y 12.000 | Ningún fit primario por repetir |
| Encoder realmente adaptado en DIRECT/CORR; frozen intacto | `verification/PAIR_DIRECT_ENCODER_CHANGE_100.json`, `PAIR_CORR_ENCODER_CHANGE_100.json`, `PAIR_CORR_FROZEN_INVARIANT_100.json` y verificadores de endpoint; witnesses incluidos cuando sus payloads son portables | Mantener la distinción entre prueba ejecutada localmente y prueba repetible desde el ZIP sin checkpoints omitidos |
| Paridad de correlación/salidas/gradientes y reanudación completa | QA CPU/CUDA y registros de runtime admitidos; fallos iniciales retenidos | La comprobación final no debe ocultar ejecuciones rechazadas |
| Endpoints congelados antes de H, P/H completos, scorer y cobertura | `primary_evidence/COMPLETE.json`: PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING | Recalcular tablas incluidas desde la extracción final |
| Selección H y comparación V del candidato primario | `PRIMARY_SELECTION_FINAL.json`, `system_compare/RESULT.json`: comparación completa; NATIVE no cumple gates de sistema | Preservar resultado negativo y contexto P2 frente a historial |
| Coste primario preparado y desde sensores crudos | `cost/primary_raw_e2e/COMPLETE.json`, paridad raw y contratos originales | No confundirlo con la latencia del loader de evaluación optimizado |
| RGB real, tiempos propios y controles | `rgb/evidence/COMPLETE.json`: RGB_READY_FOR_INTERPRETATION; comparación de sistema cierra NO_H_ADMITTED_RGB_CANDIDATE sin abrir V | No promover RGB rechazado mediante otra selección en V |
| C1 geométrico con encoder entrenable y control CORR | `adaptive/C1_geometry/evidence/COMPLETE.json`: C1_READY_FOR_INTERPRETATION | Resultado standalone negativo; uso como productor de fusión declarado |
| D1 capacidad con control y presupuesto igual de updates | `adaptive/D1_predictor_width/evidence/COMPLETE.json`: D1_READY_FOR_INTERPRETATION | Conservar negativo y errata explícita del nombre de scope de coste |
| B1 correspondencia conocida y control TTC | `adaptive/B1_correspondence/evidence/COMPLETE.json`: B1_READY_FOR_INTERPRETATION; cuatro endpoints, 24.000 updates, 144.624 predicciones, recálculos, bootstrap, coste y dos exports verificados. Señal emparejada positiva; NO_CANDIDATE_B1 frente a A5 por banda negativa | Incorporar evidencia y exports al replay final; no abrir V como rescate |
| Continuación adaptativa máxima de cuatro experimentos | C1, D1, B1 y A1 completos. A1: dos endpoints de 12.000, 144.624 filas P/H, bootstrap, diagnóstico, coste y dos exports; NO_CANDIDATE_A1 | Ninguna quinta comparación habilitada; decisiones conservadas |
| Fusión con anchor RGB, ZERO de igual capacidad y fallback | `fusion/evaluation/RESULT.json`, contratos/exports TRUE y ZERO; no promoción de sistema | Coste sensorial completo; aceptación absoluta de 1 ms autorizada después del diagnóstico, con fallo estricto conservado |
| Una familia de réplica completa, seeds13/23 | `replication/fusion_C1/evaluation/RESULT.json`: productores C1, heads TRUE/ZERO, nuevas features y normalizadores; tres seeds con seed7 | Mantener limitación de padres históricos compartidos; no presentar como tres pretrainings independientes |
| Garl completo sin debilitar inputs/modelo | `cost/garl_routes/RESULT.json`: COMPLETE, 64 consultas P, 128 pares y 256 observaciones; inputs/salidas exactos. CPU FP32, carga/primer forward/preparado y rutas raw separados; `GARL_ROUTE_COST.md` regenerable | Incluir recibos por par, resultados y límites de caché/memoria en el ZIP final; no extrapolar CPU a CUDA ni a precisión oficial |
| Coste sensorial de fusión | `fusion/cost/raw_e2e_millisecond_v1/COMPLETE.json`: 64 consultas, 256 salidas, 13 consultas reutilizadas por hash; signo/dominio intactos, diferencia máxima 0,000793457 s; un fallo estricto conservado | Sin mediciones pendientes. Aceptación de 1 ms posterior al diagnóstico declarada |
| Refit final event/multimodal, sólo con receta seleccionada | `DECISION_AFTER_A1.json`: ambos deshabilitados; ninguna receta satisface la admisión/promoción original | No procede entrenar refit ni consumir H/V; no es un fit pendiente |
| Export local test12 condicionado | `DECISION_AFTER_A1.json`: deshabilitado al no existir nueva receta seleccionada/refit; histórico V16 preservado | No procede nuevo export V18; ninguna submission autorizada |
| No ajustar con FCWD/Dev32, no etiquetas privadas | Contratos y coordinación V16 conservan transferencia descriptiva y ausencia de envío | Mantener límites en decisiones finales; no afirmar SOTA |
| Panel solicitado A5/NATIVE/CORR/C1 y auditoría de pérdida | Panel P/H completo con IDs/relojes, controles de población y Parquet; auditoría de pérdida sólo P | Incluir tablas completas y advertencia de que complementariedad condicionada por GT no es selector desplegable |
| Registro de experimentos y negativos | 19 entradas en `EXPERIMENT_REGISTRY.jsonl`; fusión/réplica conservan además sus registros prospectivos propios | Incluir `.jsonl` y registros por fase; no crear fechas prospectivas retroactivas |
| Recursos y contabilidad | `PROGRAM_EVIDENCE.json`: 183.200 científicos, 858 técnicos, 273 recuperación superior, 0 pendientes; 184.331 físicos superiores. Scopes: 196.739,722 s, con siete reservas históricas conservadoras | Dentro de 1.000.000 físicos y 216.000 s autorizados; sin escritores activos al corte |
| Endpoints, pesos, predicciones, curvas y métricas por grupo | Artefactos por fase, exports portables y curvas CSV; disponibilidad local de checkpoints completos | B1 y A1 completos; explicar pesos/optimizer/inputs externos omitidos del ZIP ≤100 MB |
| Reproducción desde otra extracción | Preflight ejecutado: 40 tablas, 1.065.690 filas y 18 exports, con omisiones explícitas | Repetir sobre el ZIP final con B1, A1 y coste final. El preflight no verifica un archivo aún no construido |
| Informes, NEXT_DECISION, SHA-256 y bundle final | `FINAL_SCIENTIFIC_REPORT.md`, `CAMPAIGN_CONCLUSION.json`, `NEXT_DECISION.json`: ciencia NO_CANDIDATE, entrega pendiente | Construir ZIP <=100.000.000 bytes y validar extracción, tablas y exports; recibo externo liga el hash final |

La ampliación de recursos y la autorización posterior de push del usuario están
registradas; no habilitan submissions, merge, pagos ni cambios globales del host.
Los datos y entrenamientos de propietarios históricos permanecen fuera del escritor V18.

La confirmación oficial y una promoción del sistema son conclusiones diferentes
de completar honestamente una campaña. Un cierre negativo sólo podrá emitirse
después de resolver las ramas condicionadas y entregar la evidencia verificable;
el entrenamiento terminado por sí solo no satisface este listado.

La reparación del CI pasa en el commit `27b8d8a` (4.244 pruebas, 23 omisiones explícitas). La comparación de 129 registros confirma cero cambios a 161 rutas vinculadas; las admisiones originales pasan sin nuevos updates. Este control de ingeniería no sustituye la evidencia científica; A1 se completó después y tiene sus propios recibos.
