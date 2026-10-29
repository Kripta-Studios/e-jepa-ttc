# S3 — candidato acotado de fusión temporal RGB

Estado: **primitiva, adaptador, transformador de fuente y batching de campaña
implementados; copia portable materializada y paridad inicial verificada**. El modelo
temporal no se ha entrenado.
No existe una mejora de detección atribuible a historia RGB en este punto.

## Contrato congelado del primer candidato

`CausalTemporalFeatureFusion` recibe las features ya proyectadas de MonoDGP del frame
actual y hasta dos historiales en orden cronológico. Usa los intervalos y máscaras que
produce `temporal_data`; las edades reales en microsegundos se transforman mediante
`log1p(age_us / 100000)`. El módulo solo acredita causalidad en tiempo sensorial porque el
release no publica tiempos de llegada.

La arquitectura tiene un bloque independiente por escala. Cada bloque proyecta el frame
actual y los dos slots históricos a 32 canales, añade un embedding aprendido de edad a
cada slot y mezcla los tres tensores con convoluciones 1×1. La última proyección está
inicializada exactamente a cero y la salida es residual. Así, antes de entrenar, la salida
es bit a bit igual al baseline para cualquier historia. Una fila sin historia conserva esa
igualdad incluso después de entrenar el bloque.

Las features históricas se separan del grafo dentro del módulo. El primer experimento
entrena el backbone sobre el frame actual y el bloque de fusión, sin conservar activaciones
de backbone de toda la historia. Esto acota memoria, pero también limita qué puede aprender
la rama histórica; debe declararse en cada comparación.

El módulo admite presupuestos de uno, dos o tres frames totales. En el presupuesto de dos
frames se usa únicamente el slot histórico más reciente. El control de capacidad
`current_repeated` pasa dos copias del frame actual con edad cero por la misma arquitectura:
no es un bloque distinto y sí puede aprender una transformación del contenido actual. El
baseline de un frame evita el bloque mediante `history_limit=0`, con identidad exacta.

No se aplica warping, BEV, pose ego ni estabilización de cajas. Las features se combinan en
coordenadas de imagen. El sistema no interpreta la posición ego de un objeto como pose del
vehículo y no elimina expansión aparente.

## Correspondencia con MonoDGP portable

El backbone R50 entrega tres niveles con 512/1024/2048 canales y strides 8/16/32. El
detector los proyecta a `hidden_dim=256` y crea un cuarto nivel proyectado cuando usa cuatro
niveles, antes de `region_head`. El spec de integración es por ello
`channels=(256,256,256,256)`, `bottleneck_channels=32`, dos slots y referencia de edad de
100.000 µs. Este spec contiene 116.480 parámetros entrenables. El punto de integración
previsto es entre `input_proj` y `region_head`; las
máscaras espaciales y positional encodings existentes permanecen sin cambios.

No se usarán hooks ni monkeypatches para el experimento. El transformador de fuente creó una
variante portable explícita que separa extracción y decoder. La copia materializada extrae
los tres frames con el mismo backbone, construye el cuarto nivel por frame y permite aplicar
el bloque; la paridad de ingeniería descrita abajo acredita la carga estricta e identidad
inicial, no una receta temporal entrenada.

El prototipo `TemporalDetectorAdapter` fija ese límite de integración. Exige un backend
explícito con dos métodos: extracción hasta los cuatro `input_proj` y decoding desde esas
features. La ruta histórica se recalcula en cada llamada con el mismo backend bajo
`no_grad`; no existe caché mutable durante entrenamiento. El contexto del decoder actual
viaja en un objeto local y nunca se reutiliza para la historia ni para la llamada siguiente.
Clases, geometría, velocidad y demás outputs continúan perteneciendo al decoder original.

La futura copia portable se denomina `monodgp_projected_temporal_v1`. Antes de aplicar el
split debe verificar el `monodgp.py` portable con velocidad congelado por R0, con SHA-256
`ac16913d545b470cf1f741df2dd783f0ca82eaca515cc94a824a7074c4113c6e` y un único marcador
inmediatamente anterior a `region_head`. El transformador implementa esa admisión fail-closed
y se usó para materializar `r0_update512_projected_v1`. Los pesos del productor single-frame se cargan
con `strict=True` antes de adjuntar los 116.480 parámetros residuales.

Cada forward declara `sequence_id`, `frame_id`, generación y versión del productor para el
frame actual y cada slot. Cruces de secuencia/generación, versiones obsoletas, identidades
incompatibles con la máscara y edades mayores que el máximo configurado abortan. El máximo
no se deriva de una frecuencia supuesta; deberá fijarse en cada config. Durante inferencia,
una caché futura tendrá que usar como clave al menos secuencia, generación, frame y versión
del productor, e invalidarse al cambiar pesos.

`prepare_monodgp_temporal_source.py` materializa la copia sin sobrescribir rutas: verifica
el SHA-256 del preregistro R0 y sus 59 hashes del árbol portable, además del hash y estructura
AST de `monodgp.py`; copia el árbol completo, transforma solo ese archivo, compila la fuente
resultante y guarda diff y manifiesto con hashes de árboles, preregistro y checkpoint. Rechaza
cualquier alias o contención entre fuente, destino y recibos para no escribir en el snapshot
congelado ni copiar recursivamente. La transformación convierte `MonoDGP` en un backend explícito sin cambiar los
nombres de sus submódulos: `forward` llama a la extracción y decoding partidos, de modo que
la carga estricta del state dict conserva las mismas claves. El decoder partido contiene el
cuerpo original desde `region_head`, incluidas las cabezas geométricas y de velocidad.

Comando preparado para ejecutar cuando el primer tramo R0 haya cerrado y `last.pt` sea
estable:

```powershell
python scripts/prepare_monodgp_temporal_source.py `
  --source-root output/detection_campaign/r0_d4096_seed7_v2/stage1/code/portable_source `
  --checkpoint output/detection_campaign/r0_d4096_seed7_v2/stage1/checkpoints/last.pt `
  --runtime-preregistration output/detection_campaign/r0_d4096_seed7_v2/stage1/runtime_preregistration.json `
  --destination output/detection_temporal/portable_monodgp_projected_v1 `
  --patch-output output/detection_temporal/portable_monodgp_projected_v1.patch `
  --manifest-output output/detection_temporal/portable_monodgp_projected_v1.manifest.json
```

Ese comando queda como plantilla reproducible; la materialización conservada usó el
checkpoint estable de la actualización 512 y rutas de salida específicas. La carga estricta
y la paridad GPU frame a frame se ejecutaron sobre ese fixture como se detalla a continuación.

## Paridad GPU de ingeniería

Tras el fallo de R0 por un PNG público truncado, se conservó su checkpoint estable de la
actualización 512. No es un candidato seleccionado ni aporta AP. Sobre ese fixture se
materializó la copia `r0_update512_projected_v1` y se ejecutaron cuatro frames válidos de
selección en procesos aislados. La comparación cubrió 116 tensores anidados por cada una de
estas parejas: productor original frente a `forward` partido, partido frente a adaptador sin
historia y partido frente al control de frame actual repetido con residual a cero. Las tres
obtuvieron igualdad exacta y error máximo `0,0`, incluida `pred_velocity`; las predicciones
decodificadas también fueron idénticas.

El reader no abrió etiquetas, `targets=None`, no hubo optimizer ni updates, y los hashes del
estado antes/después coincidieron para detector y fusión. Los picos Torch fueron 311.915.008
bytes para el productor original y 337.362.944 bytes para el proceso temporal. Esto admite
la equivalencia de software inicial; no mide beneficio temporal, AP ni generalización.
La evidencia está en `output/detection_temporal/r0_update512_parity_v2/report.json`; v1 se
conserva como ejecución anterior del mismo resultado antes de añadir el assert explícito de
outputs y el recibo `label_files_read=false`.

## Controles y gate pendientes

`temporal_campaign.py` añade una ruta de datos separada: fija el máximo de historia en
500.000 µs, selecciona primero los dos frames canónicos inmediatamente anteriores dentro de
la misma secuencia y conserva dos slots cronológicos. `current_only`, `current_repeated`,
`correct_past_1` y `correct_past_2` producen la misma firma de llamada de
`TemporalDetectorAdapter`. El control de un historial usa únicamente el slot reciente; nunca
desplaza el slot antiguo para sustituir medios ausentes. Un límite de generación, una edad
excesiva o una decisión explícita de medio ilegible enmascaran el slot sin backfill. Si falta
la decisión explícita de admisión o la lectura/hash no coincide, el batch aborta.

El reader de historia es estructuralmente label-free; en selección también lo es el frame
actual. Solo los targets del frame actual de train entran al batch. Cada batch registra
tokens canónicos, máscaras, edades reales, identidades de secuencia/generación, hashes de
miembros y ausencia de caché de features. `TemporalCampaignSampler` conserva la permutación,
posición, RNG, población ordenada, variante y manifest para una reanudación exacta. La
factoría de campaña admite únicamente el manifest readable D4096-v5 y verifica el hash del
`train.parquet`. Expone tres poblaciones sin mezclarlas: los 4.096 frames train, los 1.024 de
selección monitor del manifest v5 y los 23.958 de selección completa del artefacto separado
`full_selection_v1`, también fijado por hash; los 1.024 son subconjunto de los 23.958 y no se
concatenan de nuevo. El runner consume estas poblaciones mediante una ruta separada; el recibo
de admisión temporal compartido por la unión de 28.054 currents está congelado.

La admisión ya está materializada: `load_temporal_admission_ledger()` exige por hash
la revisión `temporal_d4096_v5_admission_v2` de `temporal_media_admission.json`
(`1e00d240…d38`) y `history_media_receipt.json` (`0804d352…b9c5`), valida 28.054
currents y 32.080 tokens
históricos, y rechaza máscaras que permitan sustitución. La factoría recibe ese ledger,
el root de datos y los artefactos v5/full-selection fijados; no infiere disponibilidad
durante entrenamiento.

La integración crea una factoría por población/variante, obtiene índices con
`TemporalCampaignSampler.next_batch()`, llama al detector con los
cuatro campos posicionales del batch y `**batch.detector_kwargs()`, y guardar/restaurar el
estado del sampler junto a optimizer, scheduler, RNG, detector y fusión. No debe compartir
readers entre procesos ni añadir una caché de features mientras cambian los pesos. Las
evaluaciones monitor y full usan siempre `targets=None`; la full se ejecuta únicamente en
los updates preregistrados del runner.

## Runner de campaña preparado, no lanzado

`train_detection_temporal_campaign.py` y `temporal_training.py` implementan esa integración
como productor nuevo. Su config envuelve, sin copiar ni relajar, un `CampaignConfig` válido y
añade la variante temporal, el checkpoint `selected.pt` padre, su procedencia, la copia
portable proyectada y los dos recibos de admisión. Antes de construir el modelo verifica los
bytes de todos esos artefactos y recalcula el hash del árbol portable completo. Solo carga
con `strict=True` el state dict del modelo padre; optimizer, scheduler, sampler y RNG empiezan
una campaña nueva de 8.192 updates adicionales.

Las cuatro variantes usan la misma clase de fusión y semilla de inicialización. El optimizer
reconoce el prefijo `backend.backbone.` para conservar el multiplicador LR del backbone y
agrupa decoder y fusión sin omitir esta última. Cada checkpoint temporal guarda adapter,
criterion, optimizer, scheduler, sampler, RNG, variante, padre y procedencia completa. La
reanudación rechaza cualquier cambio de config, padre, variante, fuente o admisión.
La procedencia incluye hashes del runner y todos los módulos temporales; cambiar ese código
invalida la reanudación. El preflight no acepta un `PASS` superficial de paridad: verifica
las tres comparaciones exactas, outputs decodificados, estado inmutable de modelo/fusión,
ausencia de labels/targets/updates y hashes actuales de la fuente que produjo la evidencia.
También exige que el `config_sha256` del checkpoint padre sea exactamente el del
`CampaignConfig` heredado. Por ahora se rechaza cualquier augmentation fotométrica no nula;
no se acepta silenciosamente una futura receta R3 con 0,15/0,15/0,10. Los roles quedan fijados
a train/selection, los hashes de split/manifest/full-selection se vuelven a comprobar, el
cache preparado del baseline se declara no aplicable porque no hay feature cache y el límite
de cache de labels sí se pasa a los readers.

Monitor se evalúa en los mismos updates que el baseline y selección completa únicamente en
4.096 y 8.192. El forward de evaluación recibe `targets=None`; un reader independiente abre
después las etiquetas actuales para el scorer. Se conserva el mismo criterio de selección
por métrica del monitor. No se selecciona con full selection.
Los checkpoints de un update con evaluación obligatoria se publican solo después de terminar
monitor y, cuando corresponde, selección completa. Un fallo intermedio conserva el checkpoint
anterior, de modo que resume repite el tramo y no salta evaluación ni selección.
Los seleccionados son archivos inmutables y content-addressed; `last.pt` guarda su ruta y hash.
Un seleccionado huérfano creado antes de un fallo no altera el linaje del último checkpoint
confirmado. Los logs de updates ordinarios se escriben antes de publicar el checkpoint.

La admisión de RAM/VRAM observada se conserva como evidencia separada, pero sus valores
dinámicos no forman parte del linaje de resume. Este solo fija umbrales, FP32/highest, batch 4,
un dispositivo visible, número de threads y presupuesto máximo de segundos. El tiempo
acumulado se guarda en cada checkpoint y continúa al reanudar.

`configs/experiment/temporal_campaign_template.yaml` es solo una plantilla deliberadamente
no ejecutable: contiene placeholders para el padre que resulte seleccionado y para una nueva
materialización de la fuente proyectada desde ese padre. No es una receta aprobada, no declara
ganador y no autoriza lanzar entrenamiento hasta que R0/R1 cierren y esos hashes se congelen.

La ejecución S3 debe comparar con el mismo pool, presupuesto de updates y protocolo:

1. frame actual únicamente;
2. misma arquitectura con frame actual repetido;
3. uno y dos frames pasados correctos;
4. historia permutada y retardada, interpretadas como controles potencialmente OOD;
5. capacidad RGB adicional sin información temporal.

Antes de entrenar faltan el perfil GPU de memoria/latencia, la selección final del padre y
configs materializadas y preregistradas desde ese padre. Una caché de features solo
se estudiará después para inferencia causal y deberá incluir versión del productor; no forma
parte del entrenamiento. El gate exige mejora frente al baseline y los controles,
por clase/rango y con coste de estado documentado. Si no se supera, el detector de un frame
permanece como ruta de producción.
