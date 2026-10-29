# Revisión independiente del plan: código y primera iteración

**Revisor:** agente `pilot_code_audit`, distinto del autor del plan.  
**Fecha:** 2026-10-29.  
**Documento inicialmente revisado:** `SOTA_DEVELOPMENT_PLAN.md`, SHA-256 `431fb23344342fc984022db56c9831baa0bda1c8cd7a789cdd38243c1fdc9c39`.  
**Primera versión corregida revisada:** SHA-256 `4cd86dbeab35a86ab95bfda85921da064f36ae3006578e84d2bcbeb57b003aaf`.  
**Versión final congelada revisada para cierre:** SHA-256 `d63065f341594613554aecb48219b4bbc967aed2fd5628d3c6bd599cce7013b7`.  
**Código contrastado:** `24b8f36f6e4aa17c965f9526d8cf486b512a8046`, MonoDGP upstream `aa059a18214aebf644510e7f0793971b403f9d14` y snapshot publicado del piloto.

## Alcance y conclusión

Se revisaron el plan completo, sus tickets/CLI, implementación y recetas reales del piloto, carga histórica de runtime/checkpoint, perfilador y estados de entrenamiento. No se ejecutaron modelos, entrenamiento ni pruebas GPU. No se modificaron el plan ni el repositorio.

El plan representa correctamente el punto de partida y separa resultados, hipótesis, propuestas y bloqueos. No atribuye SOTA, precisión TTC nueva, scorer oficial ni equivalencia del kernel CUDA al piloto. La prioridad de contrato/diagnóstico/datos antes de eventos/backbone compartido está justificada. Los comandos existentes están separados de las interfaces propuestas.

**Se detectaron y corrigieron dos precisiones materiales en los tickets:** preservar el runtime congelado al añadir el sidecar, y conectar la configuración completa/restauración de velocidad con el builder y perfilador. Se releyeron los cambios y ambos hallazgos quedan cerrados en la especificación corregida. Eran omisiones del plan de continuación; no invalidan las evidencias históricas ni constituyen nuevos fallos demostrados del piloto. El historial siguiente conserva la justificación y aceptación de las correcciones.

## C1 — Material: T03 debe observar el modelo histórico desde fuera del runtime congelado

**Localización en el plan:** S0.2 y T03, aproximadamente líneas 176–196 y495 en la versión revisada.

**Evidencia de código:**

- `scripts/infer_monodgp_eap.py:84–119`, `select_runtime_sources()`, prefiere módulos de `checkpoint_dir/code/` y exige sus hashes registrados; si no hay snapshot, exige que los módulos del checkout coincidan exactamente.
- `:124–149` carga esos módulos bajo sus nombres canónicos.
- `:243–262` obtiene `prepare_sample` y `decode_eap` del adaptador congelado; `:323–330` produce outputs y los decodifica con él.

**Consecuencia:** editar solamente `monodgp_eap_adapter.py` para añadir un sidecar no agrega esa función al snapshot histórico: se ignoraría el cambio o se rechazaría la carga por hash. Modificar la preregistración/snapshot para hacerlo pasar destruiría la propiedad de reproducir la inferencia original y no está autorizado.

**Corrección propuesta:** especificar que T03 añade un observador externo en `infer_monodgp_eap.py` o en un nuevo `diagnostic_export.py` que recibe los outputs ya calculados, sin mutarlos ni sustituir decoder/builder/preparación históricos. Registrar el hash del observador separadamente. El prior geométrico puede reconstruirse de dimensiones, lrtb, fy y tamaño de imagen; el residual puede calcularse de depth menos ese prior, identificando esa derivación. Mantener pesos, preregistración, snapshots y controles de hash intactos. Las extensiones del adaptador para modelos futuros constituyen otro esquema/versionado, no una modificación del histórico.

**Aceptación concreta:** con un snapshot viejo seleccionado explícitamente, salida normal exactamente igual con/sin sidecar; logits/queries completos y descartes presentes en sidecar; cero labels en la ruta de inferencia; archivos históricos inalterados; snapshot corrupto continúa siendo rechazado. La métrica de tiempo del observador debe ir separada del forward/decoding original.

## C2 — Material: T05/T06 necesitan factory/configuración efectiva y restauración de la variante de velocidad

**Localización en el plan:** S1.3–S1.4, T05 y T06.

**Evidencia de código:**

- `scripts/smoke_monodgp_portable.py:200–219`, `build_detector()`, solo acepta `group_num`, `predict_velocity` y `velocity_loss_coef`. Lee el bloque `model` del YAML upstream; no admite directamente queries/capas/pérdidas/backbone del nuevo archivo declarativo.
- `scripts/infer_monodgp_eap.py:279–298` restaura las opciones admitidas y compara estrictamente la configuración construida con la del checkpoint.
- `scripts/profile_monodgp_stages.py:43–45` invoca el builder sin `predict_velocity`, por lo que construye la variante sin velocidad. El checkpoint del piloto contiene cabezas de velocidad y una carga estricta en esa ruta no corresponde a su arquitectura.
- El mismo perfilador usa solo tres imágenes `overfit16`, resolución fija640×384 y targets sin velocidad (`:50–53`); no mide hoy el pipeline ampliado que exige T06.

**Consecuencia:** no basta implementar `--config` en el runner. Campos de arquitectura podrían no propagarse; el perfilador antiguo no puede usarse tal cual con el checkpoint velocity ni con D4096. El perfil publicado del overfit conserva su validez dentro de su variante: no se afirma que hubiera fallado aquella ejecución.

**Corrección propuesta:** añadir explícitamente a T05/T06 una factory versionada o extender `build_detector()` con configuración efectiva validada, preservando el camino legacy. Separar configuración de entrenamiento de `resolved_model_config`, y registrar ambas con hashes/snapshot. La inferencia y el perfilador deben reconstruir la misma arquitectura y cabeza de velocidad, máscaras/coeficientes, input y precisión. El perfilador nuevo debe aceptar población/batch/resolución del experimento y medir lectura/preparación además del update, con salida nueva y sin escribir pesos históricos.

**Aceptación concreta:** roundtrip config→modelo→checkpoint→inferencia/perfilador para variantes con/sin velocidad; carga estricta; igualdad del config efectivo; campos desconocidos o no propagados rechazados; matrices y pérdidas de velocidad activas cuando corresponde; perfil de batches/población solicitados. El config legacy y su snapshot continúan restaurando exactamente el modelo original.

## Observaciones de integración no bloqueantes detectadas inicialmente

1. **T01/T05:** el generador nuevo de GT debe conectarse explícitamente a la validación del runner, no solo al analizador externo. `train_monodgp_portable.py:133–135` sigue tomando hoy `prepared.native_ground_truth`. Añadir una prueba de integración que verifique que selección de checkpoints usa `EvaluationGroundTruth` v2 y que la métrica legacy se calcula por una ruta separada. El plan ya exige esto semánticamente; conviene expresarlo como aceptación del ticket.
2. **T02:** si se adopta24/8/4/4, la admisión v2 debe tener roles nuevos. `pilot_admission.py:15,63–69` solo admite exactamente train/selection/calibration del esquema v1. Conservar compatibilidad por versión; no renombrar auditoría como calibración ni saltarse validación para reutilizar el CLI viejo.
3. **T00/comandos:** `summarize_detection_runs.py --docs` existe y regenera `TRAINING_TABLES.md`; escribe ese archivo, aunque el contenido esperado sea idéntico. Para cumplir literalmente «docs históricos solo lectura», ejecutar sobre una copia temporal de los recibos o comparar inmediatamente bytes/hash y no conservar ninguna diferencia. No ejecutarlo sobre un checkout con cambios preexistentes en ese archivo.
4. Los anexos aún no estaban copiados al directorio final durante esta revisión. Los archivos fuente de auditoría sí existen en scratch y los flags del helper coinciden con el comando documentado. Verificar existencia y hashes del paquete final antes de entregar; no tratar la lista de anexos como comprobación de que ya fueron incluidos.

## Comprobaciones favorables

- Los detalles de AdamW, LR global, clipping, FrozenBatchNorm, capas congeladas, queries y falta de augmentation/scheduler coinciden con código y snapshot.
- La descripción de `obj_region`, prior fy·H/h y diferencia de ranking con upstream es correcta; evita convertir hipótesis en bugs.
- Las curvas recalculadas y la advertencia de macros train7/selection5 son consistentes con los artefactos.
- La primera iteración no depende de ejecutar un modelo externo nuevo, ni exige nube o alterar PHASE.
- Los tests propuestos de geometría, ignorados, causalidad, identidad, finitud y resume verifican riesgos concretos; no se utilizan como sustituto de validación científica.
- Se distinguen outputs del detector, TTC, límites de fase y calibración, y se conserva el bloqueo de generalización por exposición TRAIN40.
- No se inventan flags de CLI existentes: la sección14.3 identifica correctamente sus interfaces como propuestas.

## Estado de revisión

**C1: cerrado en el plan corregido.** S0.2 y T03 especifican ahora un observador externo, hash propio, módulos/decoder/snapshot/preregistración intactos, componentes no disponibles explícitos y prueba de corrupción. La modificación satisface el riesgo de compatibilidad/procedencia detectado.

**C2: cerrado en el plan corregido.** S1.3 y T05 incluyen extensión del builder y configuración efectiva, rechazo de campos ignorados y compatibilidad legacy. S1.3 y T06 incluyen restauración de variante/velocidad/batch/configuración y roundtrip estricto hasta inferencia/perfilador. La modificación satisface el riesgo de construcción y perfilado detectado.

**Observaciones1–3: incorporadas en la versión final.** S0.1/T01 conectan GTv2 a validación y selección del runner; S1.1/T02 versionan el rol `audit` y mantienen v1; la sección14.2 reemplaza la escritura de tablas por `runpy.run_path(...)["render"]` y comparación de contenido. Se ejecutó este último fragmento sobre el checkout inspeccionado, adaptando únicamente la ruta: resultado `PASS: read-only table rendering equality`. No se escribieron archivos del repositorio.

También se releyeron las precisiones finales sobre augmentation: un flip que necesite transformar velocidad queda bloqueado mientras su marco sea desconocido, salvo variante con máscara y control explícitos; no se inventa el signo. La ruta S1→S3 existe tanto en texto como en el diagrama, de modo que una ablation geométrica negativa no obliga a declarar S2 exitoso para investigar historia temporal. Ambas aclaraciones son consistentes con el código y las limitaciones documentadas.

**Veredicto de código y viabilidad de primera iteración: sin hallazgos materiales abiertos en la versión final congelada identificada arriba.** Solo resta comprobar la inclusión/integridad del paquete de anexos cuando termine su ensamblado. No se ejecutó la implementación propuesta ni se declara superado ningún gate científico/GPU por revisar el texto.
