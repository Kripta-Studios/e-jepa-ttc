# Revisión independiente del cierre local

Fecha: 2026-10-29. Revisor: agente `handoff_review`, separado de los agentes
implementadores. Base de implementación inspeccionada:
`3ebac039e56e3e5af98e0e3302084f00923235a1`, más los cambios finales identificados
por los hashes de [final_rgb_verification.json](evidence/final_rgb_verification.json).

**Dictamen: no quedan hallazgos materiales abiertos en el alcance revisado.**
La entrega demuestra entrenamiento local del detector, inferencia independiente
y transporte de una ROI detectada hasta las salidas nativas de PHASE. Es apta
para continuar el desarrollo y conservar estos resultados. No acredita un
detector robusto, una nueva precisión TTC, calibración de sus límites, una
submission oficial ni la superación de todos los gates originales.

## Alcance y método

Se revisaron el adaptador geométrico, backend portable, loader, entrenamiento,
admisión de manifests, inferencia, velocidad, controles, puente PHASE y documentos
de resultados. En este cierre se leyeron las fuentes y recibos, se verificaron
hashes y se recomputaron métricas desde los JSON guardados usando CPU y librería
estándar. No se ejecutaron modelos, entrenamientos ni pruebas GPU adicionales.
Las mediciones GPU son evidencia del runner y sus recibos, contrastada con el
código; no son una segunda reproducción GPU realizada por este revisor.

## Comprobaciones independientes

- Los 49 hashes de fuentes y pruebas de la verificación final coinciden con los
  archivos inspeccionados. El XML enlazado coincide con su SHA-256 y contiene
  **232 casos, cero fallos, errores o skips**. El recibo del ejecutor registra
  Ruff y formato correctos en 49 archivos y Pyright sin errores ni avisos en
  ocho módulos nuevos. Es una suite focalizada, no la suite completa de PHASE.
- La preregistración, reporte y tres archivos de predicciones del piloto
  coinciden con sus hashes archivados. El ZIP del código ejecutado y cada fuente
  declarada coinciden con la preregistración. El runner que ejecutó el piloto
  queda preservado aunque la protección de admisión se añadió después.
- El manifest usado tiene SHA-256
  `048636cd79e8f5160b97a8056b0e2ed6f78846f4bbb365b77efa5884766be159`.
  Su candidato exacto es `data/splits/eap_detection_candidate_v1.json`, SHA-256
  `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903`.
  El hash del manifest de medios local coincide con el declarado. Train y
  selección pertenecen a roles disjuntos del candidato, con dos secuencias y
  256 tokens por rol, sin secuencias de calibración.
- La curva contiene 512 updates y 2.048 exposiciones: los 256 tokens train
  aparecen exactamente ocho veces; ningún token de selección aparece en la
  curva. El [recibo de admisión](evidence/pilot_manifest_admission.json) verifica
  los inputs concretos retrospectivamente y lo identifica expresamente.
- Se recomputaron los controles de imagen y velocidad desde las predicciones
  y GT train/selección guardados. Coinciden con los recibos dentro de la
  tolerancia flotante declarada de `1e-12`, conservando exactos conteos y
  estructuras. AP101 de selección es `0.02064070955899264`, frente a
  `0.004125542906807384` del prior train y `0.0037279336968470966` de la
  permutación. No son métricas oficiales eAP.
- Los archivos de inferencia independiente coinciden con sus hashes. Las
  predicciones de 16 frames train y 256 de selección son exactamente iguales
  a las del runner, incluida la velocidad. Los otros siete frames aportan la
  historia temporal del puente. Las 279 ejecuciones de frames no se presentan
  como 279 muestras independientes ni como evidencia de precisión.
- La [salida integrada](evidence/integrated_prediction_pilot_v1.json) copia
  exactamente la detección seleccionada y los campos nativos del recibo PHASE:
  TTC `-5.234902858734131`, fase `-0.018922388553619385`, q10
  `-0.02246370166540146` y q90 `-0.015581970103085041`. Coinciden el hash del
  puente, sus cinco fuentes históricas y el recibo archivado.
- Las tres parejas de reporte/preregistración archivadas coinciden con los
  originales locales. [TRAINING_TABLES.md](TRAINING_TABLES.md) se regenera
  exactamente desde esos recibos sin GPU ni dataset. Se verificaron los enlaces
  relativos de resultados, tablas, backend, puente e inferencia.

## Hallazgos corregidos durante la revisión

1. **Admisión antes del entrenamiento.** El runner aceptaba manifests sin
   comprobar suficientemente identidad y separación. Ahora exige el candidato,
   valida hashes, roles, tokens, pertenencia y metadatos canónicos antes de
   inicializar el modelo. El recibo final incluye un rechazo real por candidato
   incorrecto, sin checkpoint. Los inputs concretos del piloto eran válidos;
   esta corrección no se atribuye retroactivamente al código ejecutado.
2. **Procedencia en inferencia por tokens.** La ruta de tokens explícitos no
   registraba siempre el manifest de medios/calibración. Ahora registra su hash,
   rechaza discrepancias cuando se proporciona un hash esperado y distingue
   verificación de mera observación. Cada frame registra también hashes de K y T.
3. **Comparación numérica entre intérpretes.** La igualdad exacta de floats
   rechazaba diferencias de redondeo insignificantes. La comparación recursiva
   permite únicamente tolerancia absoluta/relativa `1e-12`; no relaja clases,
   conteos, estructura ni tipos. Los controles del piloto fueron recomputados.

## Límites científicos que se mantienen

La admisión `G0_RGB_BOX_V1` permite el protocolo RGB documentado: las
correspondencias original/HF y reproyecciones reales no certifican todo el
dataset, una rectificación RGB/eventos perfecta ni el contrato del scorer.
`G1_PORTABLE_V1` demuestra la ruta PyTorch ejecutada; no afirma compilación,
paridad con la extensión CUDA externa ni determinismo CUDA bit a bit.

El primer overfit conserva su fallo y la continuación conserva el criterio y
denominador originales. El piloto es otro modelo con inicialización ImageNet y
velocidad, una semilla y soporte incompleto de clases. Su AP es baja. Los
controles muestran asociación útil entre entradas y predicciones en esta
población; no prueban robustez, superioridad SOTA ni generalización amplia.
La separación por secuencia tampoco sustituye validar grupos de adquisición.

La velocidad se evalúa sobre 401 matches, no sobre todos los objetos. Su error
vectorial es menor que el de los controles constantes, pero el segundo componente
empeora. La semántica física completa de los componentes sigue pendiente y no
se usa esa velocidad para calcular TTC.

El puente usa siete detecciones temporales, una ROI común, cuatro observaciones
de eventos y cinco RGB reales. La causalidad demostrada es la del replay al
cutoff sensorial; no se midieron tiempos físicos de llegada ni latencia integral.
Los IDs son locales a la ejecución. El ejemplo pertenece a train del detector
y a la exposición TRAIN40 de PHASE, sin una nueva comparación TTC con GT.
q10/q90 pertenecen al espacio de fase y no acreditan cobertura calibrada ni un
intervalo continuo de TTC. La asociación, la calidad de las ROIs y el desfase
RGB/eventos requieren evaluación posterior.

## Preservación e integridad

La comparación contra `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef` no muestra
archivos históricos modificados, eliminados o renombrados. Los resultados
negativos y snapshots anteriores se conservan. No se inspeccionaron etiquetas
privadas ni se realizó una submission en esta revisión.

La regla local [evidence/.gitattributes](evidence/.gitattributes) desactiva la
normalización de texto para los recibos con hashes de bytes. Se comprobaron
**66 archivos JSON/XML/patch/txt**: los bytes del índice Git coinciden
exactamente con los archivos revisados. Los ZIP preservan los bytes de las
fuentes ejecutadas. Esta revisión no carga ni vuelve a hashear los checkpoints
grandes: su validación y recarga estricta corresponden a los recibos del runner
y de inferencia ya contrastados.

El dictamen cubre este estado de fuentes y evidencias. Los requisitos aún
pendientes para evaluación oficial y validación científica amplia están
correctamente separados de las pruebas locales completadas.
