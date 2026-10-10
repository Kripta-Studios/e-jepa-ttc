# Port de optimizaciones V12 a V13

Las herramientas de V12 se incorporaron conservando su procedencia (commits
3be588d, 4ecc218 y 4cb5f26). El adaptador `operational.rgb_port_streaming`
ejecuta los productores, PAIR, normalizadores y cabezas propios de V13.
No convierte pesos del estudiante V12 en un estudiante V13.

## Cambios y condiciones

| Ruta | Implementación V13 | Admisión |
|---|---|---|
| Ingestión compacta | PacketRing, offsets uint32, búsqueda sin promoción completa | Exacta, con comprobación de cobertura y límites de memoria |
| Voxel histórico | IncrementalPreparer + EventInputs | Exacta por defecto; reproyección explícita desde originales |
| Características | NativeStream, caché por secuencia/objeto, lotes de misses | Reutilización exacta por defecto; tolerancias explícitas para aproximar |
| RGB | RGBInputs conserva frames uint8 y aplica crop nativo a cada ROI | Frames y deltas reales T2/T3, sin padding temporal |
| Productores | Correlación vectorizada en copias privadas | Paridad CPU/GPU; sin tocar módulos del entrenamiento |
| CUDA Graphs | Grafo del modelo y conversión canónica de fase fuera del grafo | Código/globals independientes por instancia; salidas con memoria propia |
| Precisión selectiva | Encoder FP16/BF16, geometría de inferencia FP32 | Experimental; desactivada por defecto |
| H8/H4/H2 | Selección de observaciones y padding enmascarado de cabezas | Ablation, sin cambiar el entrenamiento de la cabeza |
| Destilación | Estudiante A5, identidad de padres/modalidad/PHASE17, solo H | Requiere features de profesores V13 finales |
| INT8 | Linear dinámico CPU del estudiante | Requiere evaluar TTC y latencia; no activado en GPU |
| Evaluación Garl | Protocolos, scoring MiD, empaquetado test12 y perfiles de V12 importados | No se atribuyen métricas V12 a V13 |

El máximo de ocho observaciones se somete siempre al límite de 650 ms de datos
reales. Con el protocolo Dev32 nativo de eventos (tripletas de 100 ms y
observaciones cada 100 ms) caben cuatro observaciones. No se cambia la cadencia
para rellenar ocho posiciones. La disponibilidad de ROI se conserva: una ROI de
la consulta no pasa a considerarse disponible en una observación anterior.

El lector `EventInputs` recibe un `PacketRing` y observaciones con ventanas
explícitas. `RGBInputs` recibe un lector `(sequence, timestamp) -> uint8 HWC`.
`NativeStream.prepare()` devuelve features/timing/valid/expert_phase para
`temporal_forward()` o `fusion_forward()`. `endpoints.load_stream()` exige
checkpoints finales y normalización nativa del rol H. Mientras no existan, rechaza
la carga en lugar de sustituir checkpoints de V12 o parciales.

`distill.fit()` entrena un piloto de imitación CPU de updates fijos con features
H verificadas por el llamador y separación de secuencias. Debe recibir una fila
por observación única. `distill.load()` rechaza el esquema del estudiante V12.
El piloto no demuestra mejora de TTC. No se han entrenado estudiantes V13 reales.

## Correcciones adicionales

La máscara `known_mask` emitida por el modelo describe el TTC actual, con forma
[B], pero `ProducerObservation.validate()` exigía [B,T]. El adaptador valida la
forma real sin cambiar la máscara ni las características. La cola usa el wrapper
`rgb_port_streaming.infer` para que la extracción nativa posterior también funcione.

Guardar dos veces el mismo update reserializaba el checkpoint y dejaba recibos
inmutables vinculados a los bytes anteriores. `rgb_port_continuity` verifica
igualdad exacta de pesos, AdamW, scheduler, cursor y todos los RNG antes de
reutilizar el checkpoint. Solo cambia el estado de ciclo de vida del recibo.
Un estado distinto en el mismo update provoca error antes de escribir.
Las comprobaciones de memoria, propiedad de procesos y congelaciones originales
siguen vigentes. Una congelación adicional fija estos wrappers y su QA.

La recuperación A5 de 26.338 archivó seis ficheros auxiliares obsoletos y conservó
checkpoint, puntero, recibo nativo, journal y pruebas pendientes. Los validadores
originales admiten reconstruir esos recibos al restaurar. No se afirma paridad
con una serialización anterior cuyos bytes ya no están disponibles. Un primer
intento se revirtió al detectar que PIPELINE_RUNTIME también estaba obsoleto;
la segunda transacción incluyó ese fichero y pasó los validadores.

## Evidencia

Los resultados originales están en `E:/EJEPA_results/v13_streaming_port_20261010`.
Las copias pequeñas publicables se conservan en `evidence/`.

El probe GPU usa los pesos parciales A5=26338 y C2F=17679, entradas de eventos
sintéticas y un PAIR inicializado de forma determinista. No utiliza etiquetas ni
realiza updates. Las medianas de diez repeticiones para batch=2 fueron:

| Núcleo de inferencia | Mediana ms | Máxima diferencia PHASE17 absoluta |
|---|---:|---:|
| Nativo FP32 | 86,072 | 0 |
| Vectorizado FP32 | 71,422 | 0,00003052 |
| Vectorizado + graphs FP32 | 17,490 | 0,00003052 |
| Vectorizado + encoder BF16 | 58,003 | 0,860603 |

Graphs tuvo 9,623 segundos de primera llamada. Estas cifras **no** representan
latencia total, inputs reales, exactitud TTC ni una comparación contra Garl.
La variante BF16 requiere medir el impacto en TTC antes de promoverla.
Los dos primeros intentos de graph fallaron: conversión NumPy dentro del grafo y
colisión de identidad de código entre instancias. Se conservaron los logs y se
corrigieron separando la conversión canónica y clonando el objeto de código.

## Continuidad

La campaña mantiene datos en E:, receta BF16 original, particiones, loss, orden
sampler y final de 30.330 updates. Las nuevas aproximaciones pertenecen a
inferencia. Cambiar ahora la pérdida del entrenamiento rompería la comparación
matched; las pérdidas experimentales importadas permanecen disponibles para
un experimento nuevo. No se reinicia ni se declara terminado un modelo parcial.

Reanudación:

```powershell
python -m operational.rgb_port_continuity.queue resume --run artifacts/rgb_port_20261008
```

Las pruebas GPU de restauración de los wrappers originales se ejecutan antes de
actualizar el optimizador. El estado observado después del lanzamiento se guarda
en `evidence/RESUME_STATUS.json`. Solo los modelos finales permitirán cerrar las
mediciones V13 en Dev32/FCWD y preparar sus predicciones test12.
