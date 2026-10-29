# Prefetch CPU y viabilidad de optimizaciones GPU

## Resultado del perfil actual

El productor R0 permanece congelado. En `r0_1024`, la mediana de preparación
fue 0,8807 s y la del paso forward/backward/update fue 0,6440 s. Ambas fases se
ejecutan en serie, y la copia host–GPU no se mide por separado. Solapar
perfectamente esas dos medianas daría una cota ideal aproximada de 1,73x; no es
una predicción de rendimiento.

El productor usa lectura TAR, verificación, decodificación PIL y preparación de
tensores síncronas. Después usa `.cuda()` bloqueante sobre memoria pageable. El
paso también contiene sincronizaciones explícitas. La pérdida realiza matching
Hungarian mediante SciPy después de copiar costes a CPU; los targets tienen
longitud variable. La atención portable convierte metadata CUDA a listas Python
y comprueba finitud mediante `.item()`.

## Prototipo acotado

`BoundedOrderedPrefetch` es deliberadamente independiente del runner:

- recibe un plan de índices explícito, lo copia a una tupla y requiere que
  cada ítem sea también un valor inmutable (por ejemplo, una tupla de índices);
- usa un único hilo y una capacidad estricta de uno o dos trabajos, incluyendo
  el que esté en curso;
- crea el dataset mediante una factoría dentro del worker y lo cierra siempre;
- conserva orden exacto y propaga excepciones al consumidor;
- no anuncia fin de stream hasta cerrar el loader y propaga también errores de
  cierre sin ocultar un error primario de carga;
- solo avanza `committed_cursor` cuando el consumidor recibe un elemento;
- no importa CUDA, no fija memoria, no mueve ni modifica tensores.

Un sampler aleatorio debe generar y conservar su plan fuera del componente. El
checkpoint guarda ese plan o su recibo reproducible y `committed_cursor`. Los
elementos materializados pero no consumidos pueden descartarse sin alterar el
cursor de reanudación.

Un timeout de `close()` no puede cancelar I/O Python arbitrario. Por eso deja el
cierre pendiente y permite reintentarlo cuando la operación bloqueante retorne;
no afirma que pueda eliminar un hilo atrapado dentro de una librería externa.

Antes de integrarlo, un experimento nuevo debe medir por separado lectura,
decode/preparación, H2D y cómputo. La integración candidata fijaría memoria solo
como opción explícita en el hilo principal y usaría transferencia
`non_blocking=True`; cada worker conservaría sus propios handles TAR. Deben
compararse orden de tokens, hashes de tensores preparados, cursor/resume,
checkpoint y AP. La documentación de PyTorch advierte que fijar memoria sin un
solape real puede no mejorar el tiempo.

Referencias primarias:

- [DataLoader y memoria pinned](https://docs.pytorch.org/docs/main/data.html)
- [Guía de ajuste de rendimiento](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)
- [Carga asíncrona y transferencias](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)

## Orden de las optimizaciones restantes

1. Cachear metadata estática y sustituir sincronizaciones usadas solo para
   medir por CUDA events, manteniendo controles de finitud fail-closed. Esto no
   cambia la formulación matemática, aunque retrasar controles sí cambiaría la
   semántica operativa.
2. Probar BF16 autocast como receta numérica separada, con costes Hungarian,
   reducciones y operaciones sensibles en FP32. BF16 puede cambiar asignaciones
   discretamente, por lo que exige acuerdo de matching, replay, AP y semillas.
3. Probar TF32 por separado. Cambia la aritmética interna y la trayectoria; no
   debe etiquetarse como ejecución exacta ni mezclarse con AMP en el primer A/B.
4. Aplicar `torch.compile` primero solo al forward o a submódulos tensoriales.
   SciPy, `.cpu()`, `.item()`, listas de targets dinámicos y bucles Python rompen
   el grafo del step completo. La fusión también puede cambiar el orden de
   reducciones.
5. No priorizar CUDA Graphs. El step completo actual incumple sus requisitos de
   trabajo exclusivamente CUDA, formas/control estáticos y direcciones de
   memoria estables. Hacerlo capturable requeriría rediseñar matcher, targets,
   optimizer y RNG, con riesgo científico alto.

BF16, TF32, compilación y CUDA Graphs son cambios numéricos o estructurales que
requieren nuevos experimentos. El prefetch conserva los valores de entrada,
pero aun así debe comprobarse la reproducibilidad frente al no determinismo CUDA
ya existente.

Referencias primarias:

- [Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/accelerator/amp.html)
- [AMP, escalado y clipping](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [Rupturas de grafo y formas dinámicas](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html)
- [Semántica CUDA y CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda)
