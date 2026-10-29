# R_C2F: perfil y optimizaciones probadas del transporte local

**Decisión: no activar las nuevas variantes.** Ambas conservaron la exactitud, incluido el estado completo y la reanudación, pero las comparaciones de rendimiento no justifican sustituir el runtime existente. La caché y la precarga originales se conservan. La cola se relanzó desde el checkpoint 3.116 con el plan original sin cambios.

Se investiga la rotación 0 conservando semilla, datos, batch efectivo, pérdidas, BF16, optimizador, scheduler y contrato de reanudación. La pausa técnica se hizo después del update científico 3.116; las comparaciones usan copias desechables de ese checkpoint.

## Cuello de botella observado

La precarga existente registraba 1,70 segundos de espera por datos en 2.900 updates. Añadir compresión o ampliar indiscriminadamente la caché de entradas no aborda la espera dominante.

El perfil Python de 30 segundos mostró, entre sus muestras del hilo principal, 2,92 segundos en `local_correlation_match`, 2,64 en `deterministic_bilinear_sample`, 1,48 en `_validate_batch` y 1,18 en `_weighted_mean`. Estos tiempos muestreados no son tiempos de kernels CUDA ni una descomposición exhaustiva de la pared. Señalan el transporte local como objetivo de intervención.

## Contraste descartado: validaciones CPU en el trabajador de precarga

Se implementó `ValidatedFoldPrefetchRuntime`, que conserva las comprobaciones originales y utiliza recibos de un solo uso para detectar modificaciones antes del consumo. La prueba completa conservó exactamente el estado y la reanudación, pero empeoró la media emparejada:

| Variante | Media de los bloques ABBA, s/update |
|---|---:|
| Precarga original | 3,1962 |
| Validación en trabajador | 3,5658 |

Es un aumento aproximado del 11,6 % del tiempo por update. Se conserva el resultado negativo y no se activa esa variante. Se ejecutaron 161 updates técnicos fuera de la cadena científica.

## Caché de constantes del transporte

`operational/oof_transport_cache.py` conserva el cálculo diferenciable original. Almacena únicamente máscaras de bordes y desplazamientos constantes, indexados por batch, dimensiones, radio, dispositivo y dtype. Con radio 4 se evita volver a construir las máscaras de los 81 desplazamientos en cada llamada. Las features, correlaciones, probabilidades, salidas y gradientes siguen calculándose para cada batch.

La caché tiene un límite de 64 MiB y 16 firmas; utiliza expulsión de la entrada menos reciente. No guarda targets ni predicciones. Se reconstruye tras una reanudación y no consume números aleatorios. La aplicación propuesta se limita a los productores OOF R_C2F; los productores ya completados conservan sus pesos y admisiones.

Las 11 pruebas específicas incluyen igualdad exacta de forward/backward en CPU y CUDA, caché fría y caliente, rechazo de entradas inválidas, límites y expulsión, y conservación del checkpoint padre durante una migración. Ruff pasa y Pyright pasa con el entorno real del proyecto, declarado en `artifacts/final_refit_20261019_scientific_oof/pyright_transport.json`.

La primera comparación de 16 updates por bloque tampoco mostró aceleración global: controles de 2,1244 y 2,9501 s/update frente a 2,6953 y 2,6344 con caché. El promedio con caché fue aproximadamente un 5,0 % más lento. La variación entre controles motivó una comprobación final con calentamiento de 32 updates y una única fuente reutilizada; se conserva esta primera comparación y no se usa por sí sola para activar la optimización.

En la comprobación con calentamiento completo, los controles dieron 3,5160 y 3,1030 s/update; los bloques con caché, 3,5462 y 3,6912. Las medias fueron **3,3095 frente a 3,6187 s/update**, respectivamente: la caché fue aproximadamente **un 9,3 % más lenta**. Eliminar construcciones constantes no ha producido una mejora global en este ensayo. No se atribuye esta diferencia a una causa hardware específica; las muestras disponibles no la demuestran.

La lectura de hardware durante el ensayo indicó alimentación externa y ausencia de indicadores activos de limitación térmica o de potencia en ese instante. Eso no certifica ausencia de variaciones durante todo el ensayo. No se cambiaron ajustes globales de energía, CPU ni GPU.

La ruta experimental y sus pruebas quedan en el repositorio, sin admisión de ejecución ni sustitución del launcher activo. El script de activación rechaza publicar una admisión si la comparación emparejada no mejora. No se modifica ninguna fuente científica ni el checkpoint original mediante los probes.

## CUDA Graphs, compresión y aproximaciones

La ruta tiene varios tamaños y particiones. La documentación de [PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda) permite capturar regiones parciales, pero exige controlar las restricciones de captura; el proyecto además conserva un fallo de memoria de captura anterior. Esta intervención elimina reconstrucciones constantes sin introducir capturas nuevas. CUDA Graphs no está activado por esta tarea.

Los datos RGB ya están comprimidos sin pérdida y la espera por su preparación es mínima. No se añade compresión ni cuantización. Cambiar el orden de las reducciones, usar otro muestreo o aproximar las correspondencias requeriría otra comprobación numérica y científica. La intervención sigue la recomendación de [PyTorch de medir el cuello de botella y reducir trabajo innecesario](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Evidencias

En `artifacts/final_refit_20261019_scientific_oof/`:

- `RC2F_PROFILE_BEFORE_20261021.json` y `RC2F_PROFILE_BEFORE_SUMMARY.json`.
- `OOF_VALIDATED_PREFETCH_PARITY.json`: resultado negativo de rendimiento con equivalencia exacta.
- `OOF_TRANSPORT_TESTS.xml`: pruebas de la caché y su genealogía.
- `OOF_TRANSPORT_CACHE_PARITY.json`: comparación corta y reanudación de la caché, 81 updates técnicos.
- `OOF_TRANSPORT_STEADY_PARITY.json`: comprobación con 32 updates de calentamiento, fuente compartida y bloques de 32 updates; equivalencia y reanudación PASS, 192 updates técnicos.
- `RC2F_OPTIMIZATION_DECISION_20261021.json`: decisión `NOT_ADMITTED_PERFORMANCE`, hashes y comparación de las tres pruebas.

En total se ejecutaron **434 updates técnicos sobre copias**, sin introducirlos en los contadores científicos. El checkpoint científico original mantuvo su SHA-256 `f0b95de2f3a3de28925d579cd23f14af6672054e7b70750df416a35609b6caf7` durante los probes; el journal confirmó 3.116 updates completados y durables, cero pendientes y cero recuperación adicional antes de reanudar.

El rendimiento de ensayos cortos no se transforma directamente en ETA de la campaña. Los checkpoints siguen siendo cada 100 updates y los márgenes de memoria física y comprometida siguen en 2 GiB.
