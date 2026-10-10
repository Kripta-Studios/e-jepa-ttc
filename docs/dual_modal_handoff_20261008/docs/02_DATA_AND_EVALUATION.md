# 2. Datos, información disponible y evaluación

## Dos ámbitos, nunca mezclados

**B / puente TRAIN40**: utiliza H8 y los productores entrenados sobre las 40 secuencias.
Puede entrenar un adaptador RGB usando TRAIN40. Es stacking/fusión entrenada con
features in-sample; evaluar en TRAIN40 no acredita generalización. EvTTC existente es
desarrollo de transferencia ya expuesto, no confirmación. Usar la receta fija y medir
una vez los endpoints B. Estos resultados NO eligen arquitecturas D o umbrales.

**D / arquitectura con exclusión por adquisición**: antes de entrenar, agrupar las
secuencias por adquisición/ruta/continuidad material disponible en metadata. Reservar
aproximadamente 20% de grupos (objetivo 8 de 40 secuencias sólo si las unidades coinciden)
como `MODEL_HELDOUT_DEV`, hash determinista con salt `dual-ttc-v13-dev-20261008`.
Si un grupo tiene varias secuencias, se mueve entero; no forzar exactamente 32/8.
No usar targets para elegir grupos o conseguir una distribución favorable. Publicar
conteos y bucket coverage posteriormente, sin redibujar el split.
Si los metadatos sólo permiten verificar secuencias, usar provisionalmente esas unidades
con `grouping_level=sequence_proxy` y `acquisition_independence_verified=false`; no
inventar grupos independientes ni parar toda la ingeniería por no conocer la ruta física.
Evitar duplicados/solapes conocidos entre conjuntos. Si hay sólo dos o tres adquisiciones,
separarlas enteras aunque el porcentaje difiera; con una sola unidad DEV no publicar un
bootstrap degenerado como incertidumbre cero. Reportar esa limitación y resultados
descriptivos. La confirmación final sigue siendo distinta de este desarrollo.

Estos grupos ya tuvieron exposición histórica: NO son confirmación fresca del proyecto.
Sí pueden estar excluidos del entrenamiento de **estos nuevos modelos**, incluyendo
pretraining de tarea, teacher, normalización y selección de datos. Las métricas describen
ese nuevo pipeline y su desarrollo. Para múltiples variantes, reconocer selección adaptativa.

**Prohibido en D**: inicializar E/R con A5/C2F/H8 TRAIN40 o Garl público entrenado en
esos grupos; producir targets KD con un teacher que los vio; fijar un normalizador con
ellos. Volver al commit anterior NO elimina lo aprendido por pesos entrenados en TRAIN40.
**Permitido**: inicialización genérica ImageNet documentada, sin afinamiento previo en
eAP/EvTTC; los pesos genéricos son los mismos para los brazos comparados.

Garl público se reporta como referencia externa publicada con genealogía no certificada,
no como baseline que sabemos excluyó el split interno. No reentrenarlo entero por defecto.

## Modalidades y dos interfaces

`H8_RGB_BRIDGE` conserva todos los inputs históricos H8, y añade dos imágenes RGB
correspondientes a los endpoints finales. No añadir RGB a los 12 canales de A5.

`PAIR_NATIVE` es la primera interfaz de los modelos nuevos:
- eventos `[B,2,20,128,128]`, mediante representación Garl event40 partida en dos;
- RGB opcional `[B,2,3,128,128]`;
- timestamps enteros y metadatos de disponibilidad, Δt real positivo;
- mismas ventanas finales de 100 ms, orden/polaridad/normalización/crop auditados;
- no valores TTC, depth, velocidades GT, category/sequence IDs como features.
Los nuevos modelos P2 consumen exactamente el tensor nativo auditado del comparador
cuando la representación es la misma. El loader del modelo no normaliza RGB dos veces.

`CONTEXT4`: cuatro endpoints a −300/−200/−100/0 ms, ventanas de 100 ms. Requiere
reconstrucción independiente del índice TTC filtrado. Para tiempos anteriores al par
público, usar sólo el ROI actual como contexto retrospectivo (mismo policy que H8),
nunca cajas futuras ni asociación oracle nueva. Mantener el par final idéntico a P2;
registrar los crops de los dos tiempos adicionales. Es una interfaz ampliada de
información, no una comparación de arquitectura de mismo input con Garl par.
Si las reglas de benchmark prohíben esos eventos adicionales, CONTEXT4 sigue siendo
investigación local; el candidato oficial se elige entre los P2, no se fuerza compatibilidad.

La rama RGB del candidato P2 o CONTEXT4 recibe inicialmente **sólo las dos imágenes
finales**. Así no convierte cada lag de 50 ms de H8 en una nueva foto inexistente a 10 Hz.
Una imagen reutilizada conserva su ID y timestamp, no se presenta como dos adquisiciones.

## Sincronización y geometría

Reutilizar las funciones de `operational/evttc_rgb_transfer/inputs.py` como contrato de
la transferencia anterior, no como loader eAP universal. Seleccionar por timestamps
de las imágenes, no de las cajas; restar enteros int64 antes de pasar a segundos.
Conservar la tolerancia histórica de 1 ms sólo donde el sensor/protocolo la documenta.
La disponibilidad de la predicción incluye un frame posterior si hubo espera admitida;
no ocultar exposición, lectura o latencia no medida.

Cada modalidad mantiene su calibración, ROI y mapa de crop a sensor. No aplicar el
shift eAP de 5 px a EvTTC; no asumir que rotación sola elimina el paralaje; no usar
GT-depth para alinear RGB. La fusión inicial es **tardía entre descriptores temporales**,
no suma pixel a pixel de cámaras con centros distintos.

Ausencia RGB = máscara explícita y fallback exacto a la salida event-only. No rellenar
RGB faltante mediante imagen futura, interpolación entre frames futuros o media global.
Para inputs ausentes reportar cobertura y qué fallback se utilizó en toda la población.

## Métricas y no inferioridad

En eAP, primaria: implementación canónica MiD del benchmark con sus buckets y masas;
reportar también MAE, RMSE, mediana AE, sesgo, signo, q90/q95 AE y crucial `(0,3]`.
Evaluar sobre las filas requeridas por el contrato; valores GT fuera del dominio de
fase no se corrigen ni se esconden: máscara de scoring justificada e inventariada.

En EvTTC: preservar la comparación nativa de 946 filas y sus flags; añadir el sidecar
común ±60, tasas de signo y sobreestimación crucial. No llamar a ese sidecar oficial.
La función audit_saved_predictions.py entrega descriptivos, no entrena ni escoge cap.
Para RTE/MiD oficiales, reutilizar el código y unidades oficiales tras tests; no confundir
RTE relativo con MAE en segundos ni números de OLD_DEV con la tabla de eAP test.

Incertidumbre: bootstrap pareado por adquisición/secuencia; semillas no son escenas.
Reportar por semilla completa y media de pérdidas, no ensemble no autorizado. Cuando
se evalúe EvTTC, los 32 escenarios de ocho familias pueden estar relacionados: acompañar
la lectura por secuencia con estratos; no declarar que sean 32 escenas independientes
sin revisar adquisición. No calcular CIs por ventanas independientes.

Guardrails para promoción D: cobertura completa del endpoint, cero predicciones no
finitas; exceso de error de signo ≤0,5 puntos porcentuales; crucial MiD no peor de +2%;
mediana AE no peor de +2%; q95 AE no peor de +5%. Los porcentajes son criterios
propuestos ahora, no hechos o límites heredados. Si una referencia vale cero, usar
no aumento absoluto para ese guardrail. No redefinir tras ver resultados.

## Evaluación oficial y alcance de autorización

Tras seleccionar receta únicamente en D y refit final TRAIN40, congelar pesos E y ER,
preprocesado, tiempos y schema. Autoriza preparar **localmente** una predicción por
sample_token sobre metadatos/medios de test públicos disponibles (sin GT), después de
admitir las reglas de inputs. Exportar submission.json y ZIP por modalidad. No enviar
submissions, abrir labels privados, Stage76 o puntuaciones selladas automáticamente.
Si falta data/test access, entregar el exportador y la dependencia: no inventar resultados.
Una submission posterior requiere la orden del usuario; no iterar sobre el leaderboard.

## Campos de tiempo y calidad de la nueva interfaz

Tiempo por par E: [separación real de endpoints, duración de ventana anterior, duración
de ventana actual, edad del endpoint final del par respecto al cutoff admitido], segundos.
Tiempo por par RGB: [separación real de frames, edad del anterior, edad del final, espera
de sincronización admitida]. REPEAT tiene separaciónRGBcero; no falsificar sus tiempos.
CalidadRGBparaelgate: media de luminancia, fracción de luminancia fuera de[0,01;0,99],
media de gradiente espacial absoluto de luminancia y edad del frameRGBfinal. Luma =
0,299R+0,587G+0,114B en [0,1]; gradiente = media de las medias |diffx| y |diffy|.
Se calculan sin targets, tras augmentación y antes de ImageNet; faltantes se enmascaran.

Un timestamp puede marcar trigger/inicio de exposición, no disponibilidad de todos los
píxeles. Preservar el contrato nativo de benchmark para comparaciones; si exposición y
readout no están documentados, registrarlos como desconocidos. La tolerancia de1ms entre
timestamps NO prueba latencia física≤1ms. Esa incertidumbre limita claims de causalidad
hardware; no bloquea el entrenamiento offline que respeta los inputs del benchmark.
