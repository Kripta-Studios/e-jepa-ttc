# V18: requisitos de cierre y alcance de la evidencia

Corte de revisión: 16-10-2026, B1 completo y diagnóstico de coste activo. **Objetivo completo todavía
no demostrado.** Este documento deriva los requisitos del handoff, sus documentos
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
| Continuación adaptativa máxima de cuatro experimentos | C1, D1 y B1 registrados prospectivamente; tres familias ejecutadas. `ADAPTIVE_SCOPE_AFTER_B1.md` justifica profundidad de adaptación como cuarta comparación, sin atribuir necesidad del encoder a la matriz primaria | Concretar y registrar A antes de updates; QA, dos endpoints, P/H y decisión emparejada. El diseño por sí solo no cuenta como ejecución |
| Fusión con anchor RGB, ZERO de igual capacidad y fallback | `fusion/evaluation/RESULT.json`, contratos/exports TRUE y ZERO; no promoción de sistema | Coste desde sensores aún no admitido por discrepancia numérica |
| Una familia de réplica completa, seeds13/23 | `replication/fusion_C1/evaluation/RESULT.json`: productores C1, heads TRUE/ZERO, nuevas features y normalizadores; tres seeds con seed7 | Mantener limitación de padres históricos compartidos; no presentar como tres pretrainings independientes |
| Garl completo sin debilitar inputs/modelo | `cost/garl_routes/RESULT.json`: COMPLETE, 64 consultas P, 128 pares y 256 observaciones; inputs/salidas exactos. CPU FP32, carga/primer forward/preparado y rutas raw separados; `GARL_ROUTE_COST.md` regenerable | Incluir recibos por par, resultados y límites de caché/memoria en el ZIP final; no extrapolar CPU a CUDA ni a precisión oficial |
| Coste sensorial de fusión | Fallos preservados; diagnóstico completo: las cuatro consultas pasan con batch histórico y cuDNN TF32, sin relajar tolerancia | Comprobar y medir las 64 consultas originales y cuatro rutas con la ejecución restaurada; declarar todo cómputo interno repetido |
| Refit final event/multimodal, sólo con receta seleccionada | No se declara candidato final ni se consumen H/V como refit | Resolver después de la continuación A y gates completos; si no se habilita, registrar decisión y motivo, no entrenar para gastar presupuesto |
| Export local test12, condicionado por congelación/contrato | V16 conserva su export histórico; no se sustituye ni se envía | Decidir aplicabilidad de un export V18 tras selección/refit; ninguna submission autorizada |
| No ajustar con FCWD/Dev32, no etiquetas privadas | Contratos y coordinación V16 conservan transferencia descriptiva y ausencia de envío | Mantener límites en decisiones finales; no afirmar SOTA |
| Panel solicitado A5/NATIVE/CORR/C1 y auditoría de pérdida | Panel P/H completo con IDs/relojes, controles de población y Parquet; auditoría de pérdida sólo P | Incluir tablas completas y advertencia de que complementariedad condicionada por GT no es selector desplegable |
| Registro de experimentos y negativos | 18 entradas en `EXPERIMENT_REGISTRY.jsonl`; fusión/réplica conservan además sus registros prospectivos propios | Incluir `.jsonl` y registros por fase; no crear fechas prospectivas retroactivas |
| Recursos y contabilidad | 160.223 físicos durables: 159.200 científicos, 750 técnicos, 273 recuperación superior; scopes originales preservados | Corte final tras quiescencia, contabilizando evaluaciones/probes; presupuesto ampliado explícitamente a 216.000 s, no saldo borrado |
| Endpoints, pesos, predicciones, curvas y métricas por grupo | Artefactos por fase, exports portables y curvas CSV; disponibilidad local de checkpoints completos | Incorporar B1 y explicar pesos/optimizer/inputs externos omitidos del ZIP ≤100 MB |
| Reproducción desde otra extracción | Preflight ejecutado: 40 tablas, 1.065.690 filas y 18 exports, con omisiones explícitas | Repetir sobre el ZIP final con B1 y evidencia nueva. El preflight no verifica un archivo aún no construido |
| Informes, NEXT_DECISION, SHA-256 y bundle final | Informes parciales publicados; builder con límite 100.000.000 bytes y comprobación CRC/hash | Congelar conclusión final, construir sin escritor vivo, verificar extracción fuera de RUN y entregar ZIP/SHA/recibo |

La ampliación de recursos y la autorización posterior de push del usuario están
registradas; no habilitan submissions, merge, pagos ni cambios globales del host.
Los datos y entrenamientos de propietarios históricos permanecen fuera del escritor V18.

La confirmación oficial y una promoción del sistema son conclusiones diferentes
de completar honestamente una campaña. Un cierre negativo sólo podrá emitirse
después de resolver las ramas condicionadas y entregar la evidencia verificable;
el entrenamiento terminado por sí solo no satisface este listado.
