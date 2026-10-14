# Integración de pares RGB

`operational.joint_measurement.rgb_model` admite el endpoint R_A5 P-only de
30.330 updates, verifica su hash contra el inventario, recibo y procedencia y
carga una instancia CPU privada. `pair_from_parent` copia su CNN de tres canales.
Los módulos nuevos comparten inicialización por semilla, tanto con correlaciones
como sin ellas y con encoder entrenable o congelado.

La comprobación `python -B -m operational.joint_measurement.rgb_preflight` utiliza
cuatro consultas P elegidas por hash del token: dos con historia original T2 y
dos T3. El adaptador conserva los dos últimos frames reales antes de colacionar,
con entrada [4,2,3,128,128] y separación del reloj RGB. En esos ejemplos el
intervalo es aproximadamente 0,100001–0,100002 segundos.

Se comprueba forward/backward CPU, gradientes finitos y no nulos del encoder
entrenable, ausencia de gradientes del congelado, gradiente en la cabeza y
preservación completa del padre. También se comprueba que no haya almacenamiento
de parámetros compartido y que los módulos nuevos tengan pesos iniciales
idénticos. Los pesos no cambian: son **cero updates de optimizador**.

La arquitectura RGB tiene 571.137 parámetros; con encoder congelado son 402.145
entrenables. La diferencia respecto a eventos corresponde a la entrada nativa de
tres canales. No se repiten canales event ni se utiliza su checkpoint como RGB.

Evidencia: `verification/RGB_CPU_PREFLIGHT.json`, que contiene los índices,
identidades, tiempos, normas de gradiente y hashes del código ejecutado. Se
conserva también la comprobación previa a una corrección de formato de una línea.

Esto acredita integración de CPU sobre esos ejemplos, no cobertura completa,
entrenamiento RGB, reanudación CUDA ni precisión TTC. La receta RGB se registra
después del contraste primario event, respetando el máximo de dos fits de 12.000
updates y dos fits de fusión/control de 2.500. La fusión requiere todavía su
integración, control ZERO, genealogía de features nueva y fallback event-only.

## Entrenador preparado, pendiente de admisión científica

`operational.joint_measurement.rgb_train` implementa el par entrenable/congelado,
con 12.000 updates por brazo, las mismas masas P y objetivo de fase de V18,
batch32/microbatch4, FP32 y optimizador registrado. Guarda modelo, optimizador,
scheduler, RNG, sampler y cursor mediante el checkpoint transaccional existente.
Comparte la contabilidad V18, las reservas GPU y los límites de recursos.

El registro exige los cuatro endpoints primarios, sus predicciones P/H completas
y la decisión final con coste. No se ha ejecutado ese registro ni elegido CORR o
DIRECT para RGB. El comando posterior será:

```text
python -B -m operational.joint_measurement.rgb_train register --variant CORR --reason "Observación documentada del contraste P/H"
python -B -m operational.joint_measurement.rgb_train qa
python -B -m operational.joint_measurement.rgb_train fit --fit-id RGB-PAIR-CORR-FT-S7
python -B -m operational.joint_measurement.rgb_train fit --fit-id RGB-PAIR-CORR-FROZEN-S7
```

La variante del ejemplo debe corresponder a la decisión documentada. Cada `fit`
ejecuta un scope y conserva el cursor para la siguiente invocación. `qa` ejecutará
40 updates sintéticos contabilizados: diez continuos frente a cinco más cinco
reanudados para cada brazo, sobre el mismo bucle que utiliza el entrenamiento.
Sus resultados CUDA todavía están pendientes; el fit real los exige.

Las pruebas de admisión verifican que no se puede registrar RGB con endpoints
primarios incompletos o sin la comparación final, y que un cambio de código bloquea
el fit antes de construir el optimizador. Estas pruebas no acreditan todavía
el entrenamiento completo ni su convergencia. La fusión sigue pendiente de
integración y ejecución.

## Evaluación preparada

`operational.joint_measurement.rgb_evaluate` exige ambos endpoints de 12.000
updates con journals durables y sin updates pendientes. Incluye el padre
R_A5-P2-FROZEN como referencia con el mismo contexto de dos frames. Escribe las
predicciones completas y métricas por grupo en `rgb/predictions/{P,H,V}` mediante
fragmentos verificables y reanudables. El modelo recibe únicamente píxeles y
delta RGB; las etiquetas se incorporan después para el scorer.

`select` exige P/H completos y conserva una decisión de desarrollo basada en
métricas. La comparación con R_CTX/F_TRUE permanece pendiente; esa decisión no
afirma promoción del sistema. V exige que antes se haya congelado la
interpretación H con la comparación de coste preparada descrita abajo.
No se ha ejecutado ninguna de estas
evaluaciones sobre datos reales, porque todavía faltan los endpoints RGB.

Seis pruebas adicionales verifican endpoints completos, ausencia de updates
pendientes y rechazo de fragmentos con binding, intervalo o bytes distintos.
El replay del bundle incluye las ramas anidadas de predicciones y exports,
preservando sus nombres para evitar colisiones con los resultados primarios.

`operational.joint_measurement.rgb_cost measure` medirá el padre y los dos
endpoints RGB sobre 128 consultas P elegidas sin etiquetas, batch1, FP32,
20 warmups por modelo y orden intercalado reproducible. El tiempo abarca tensor
preparado residente en GPU hasta salida TTC firmada; excluye decodificación,
transferencia y carga de pesos. Registra p50/p95/p99, throughput y memoria
adicional. Los fragmentos reanudados deben conservar binding, consulta y rutas.

`operational.joint_measurement.rgb_cost select` combinará esa evidencia con H
usando los gates de desarrollo ya registrados. Una vía de coste exige al menos
30 % de ahorro, deterioro agregado máximo del 2 % y los mismos límites de signo,
bandas y cobertura. No constituye comparación E2E ni promoción sobre R_CTX/F_TRUE.
Congela `rgb/SELECTION_FINAL.json`, necesario antes de producir predicciones V.

Las 24 pruebas CPU de admisión, evaluación y coste pasan, incluida la prohibición
de evaluar V con sólo una decisión parcial de métricas. El benchmark de coste
real permanece pendiente hasta que estén completos ambos endpoints RGB.

## Cachés H/V propias

La inspección del 12 de octubre a las 18:15 UTC confirmó que V13 sólo conserva
la caché RGB decodificada P. Las cachés H/V de investigación V16 contienen estados
de refinadores congelados y no sustituyen los frames necesarios para estos modelos.

`python -B -m operational.joint_measurement.rgb_cache both` materializa H y después
V, en un único proceso CPU. Reutiliza `decode_query`, los campos y el formato
lossless del preparador nativo; no cambia los manifests ni bindings de V13.
Los nuevos píxeles se guardan bajo el cache_root de V18, en `rgb/H` y `rgb/V`.
Los bindings propios quedan en `rgb/cache_bindings` dentro del directorio de
artefactos V18. `RGBPairDataset` utiliza esos bindings únicamente cuando falta
una caché histórica del rol correspondiente.

Cada fragmento registra tokens, índices, CRC y SHA-256. El cierre exige cobertura
exacta y paridad de todos los campos contra una segunda decodificación de la
primera y última consulta T2/T3. El manifest sólo se publica después de esos
controles. Un fragmento admitido que cambie se rechaza y conserva para revisión.
La preparación no realiza inferencia, optimización ni selección por TTC.

El proceso se inició el 12 de octubre a las 18:15 UTC. Su estado actual está en
`RGB_PREPARE_H_PROGRESS.json`, `RGB_PREPARE_V_PROGRESS.json` y los manifests del
cache_root. El recibo `verification/V16_COORDINATION_20261012T1816.json` registra
la observación de propietarios, sin utilizar resultados FCWD/Dev32 para recetas.

H cerró a las 18:35:50 UTC: 18.422 consultas, 576 fragmentos y 1.882.328.902 bytes.
La cobertura es exacta y los cuatro casos de segunda decodificación T2/T3
coinciden bit a bit en todos los campos. El adaptador V18 consumió correctamente
esos cuatro casos, con tensor [4,2,3,128,128] y reloj RGB positivo. Recibo:
`verification/H_RGB_INPUT_ADMISSION.json`. No se ejecutó ningún modelo ni métrica.
El manifest H tiene SHA-256
`8cadc01ea182f4bcfc2082790ba942abd22cfca6d1cd7f1d1cbbefc119b17618`.
V cerró a las 18:53:43 UTC: 16.432 consultas, 514 fragmentos y 1.516.027.016 bytes.
También conserva cobertura exacta y paridad bit a bit en los cuatro casos de
segunda decodificación. El adaptador consumió los índices 0, 16.422, 1 y 16.431,
con tensor [4,2,3,128,128] e intervalos RGB positivos. Recibo:
`verification/V_RGB_INPUT_ADMISSION.json`. El manifest V tiene SHA-256
`21975cfe232c6a128376abfde4caa49fa00afa1d9740971785827c7cd278e92d`.
El proceso CPU de preparación terminó con código 0. P/H/V tienen ahora entradas
RGB disponibles; esto no constituye entrenamiento ni evaluación de precisión.

## Export portable RGB

`python -B -m operational.joint_measurement.rgb_export FIT_ID` exige la congelación
de los endpoints RGB. Generará TorchScript batch1, ejemplos sin etiquetas,
normalización, contrato de entrada y recibos con SHA-256 en `rgb/exports/FIT_ID`.
La entrada RGB es [1,2,3,128,128], FP32 en [0,1], y un intervalo positivo [1,1]
del reloj RGB. El NPZ utiliza la clave `rgb`; el replay conserva también la
compatibilidad con los exports event cuya clave es `events`.

Se ha comprobado la ruta de serialización/recarga en CPU con dos pares P reales
(T2/T3), sobre el padre RGB histórico y sobre DIRECT/CORR con sus módulos nuevos
aleatorios. Las tres comprobaciones pasan y el ejemplo publicado en cada
diagnóstico coincide bit a bit. **Cero updates y ninguna medida de precisión TTC**.
Recibos: `diagnostics/RGB_PORTABLE_PREFLIGHT/RESULT.json` y `REPLAY.json` de cada
arquitectura. El código exacto ejecutado se conserva junto al diagnóstico; una
corrección posterior de líneas en blanco entre imports tiene AST idéntico.

El trazado se limita al contrato de forma fija y entradas válidas declarado.
Conserva las advertencias de PyTorch en el log. La preparación de estos artefactos
no acredita los futuros endpoints entrenados: cada export deberá repetirse y
verificarse con sus propios pesos después de completar la comparación RGB.

## Componente de fusión con anchor RGB

`operational.joint_measurement.fusion_model.RGBAnchorComplement` implementa la
parte predictiva de la fusión pendiente. Aprende una corrección en fase sobre
la salida RGB; no promedia TTC entre modalidades. La última capa comienza en
cero, por lo que inicialmente reproduce el anchor salvo el redondeo de la
conversión nativa fase/TTC. TRUE y ZERO tienen idéntica capacidad y admiten el
mismo estado inicial. ZERO anula todo el bloque de features event, incluidos
sus relojes; la predicción event sólo interviene en el fallback cuando falta RGB.

El fallback copia exactamente el TTC event admitido y omite la red de fusión
para esas filas. Tolera campos RGB ausentes/no finitos únicamente en dichas
filas, evitando que contaminen el resto del batch. Nueve pruebas CPU comprueban
el anchor con signos opuestos, fallback completo y mixto, invariancia ZERO,
gradientes de la corrección y rechazo de entradas inválidas. Cero updates.

Esto no registra todavía un experimento de fusión. Siguen pendientes la elección
de padres a partir de P/H, extracción de sus features nuevas, normalización H,
genealogía, receta numérica completa, trainer, QA de reanudación y los dos fits
de 2.500 updates. El componente acepta dimensiones explícitas para permitir el
anchor RGB elegido sin reutilizar por forma una caché de otro productor.

La exportación del componente usa scripting para conservar las ramas dinámicas
de disponibilidad RGB. Doce pruebas CPU pasan, incluidas serialización/recarga
TRUE y ZERO cambiando entre RGB completo, parcial y ausente. El fallback sigue
siendo bit a bit en las filas ausentes después de recargar.

TorchScript no compila los floats globales que captura la función de conversión
compartida. `NativePhaseConversion` conserva sus operaciones y límites como
constantes del módulo; no cambia el archivo nativo usado por la matriz primaria.
La comprobación cubre 1.008 fases, saturaciones, fase cero y derivadas: igualdad
exacta frente a la función nativa en eager, y tolerancia 1e-6 para scripting.
El log fallido por constantes globales se conserva junto al log aprobado en
`verification/FUSION_DYNAMIC_EXPORT_TESTS*.log`. Las advertencias de deprecación
de TorchScript también se conservan. Esto acredita el componente con entradas
sintéticas, no los futuros pesos entrenados ni el pipeline sensorial completo.

## Entrada residente para la futura fusión (14 de octubre)

`fusion_features.py` admite cachés nuevas por hashes del endpoint, esquema y
extractor, y ajusta normalizadores únicamente sobre observaciones H disponibles.
`fusion_data.FusionTensorBank` conecta esas observaciones con
`RGBAnchorComplement`: valida el manifest y orden de consultas compartidos,
normaliza una vez y conserva copias de los tensores en el dispositivo declarado.
Cada batch transfiere sólo sus índices si las features ya residen en GPU; no
decodifica NPZ ni calcula hashes por update. Los relojes de ambas modalidades
siguen siendo propios y no se fuerzan a coincidir.

La entrada no contiene etiquetas, masas ni IDs de grupos. El muestreador y su
estado pertenecen al futuro trainer. Se conservan filas con RGB ausente y la
salida event exacta para su fallback. Cambiar los arrays CPU originales después
de admitirlos no cambia los tensores residentes del entrenamiento.

Catorce pruebas CPU pasaron para alineación, normalización, aislamiento de
memoria, índices repetidos, identidad de productor, estadísticas H aplicadas a V
y fallback TRUE/ZERO. No se realizó ninguna ejecución CUDA ni update. Siguen
pendientes la selección de padres, extracción real, registro de la receta,
trainer reanudable y sus dos fits de 2.500 updates; esta preparación no los cuenta
como completados ni acredita una aceleración medida.

## Motor reanudable de fusión (14 de octubre, 11:54 Madrid)

`fusion_engine.py`, commit `24ee042d05427d0490e71cb06b1709a9940918c7`, implementa
AdamW, scheduler coseno, muestreo uniforme con reemplazo, pérdida de fase emitida
y previa a conversión y checkpoints completos mediante `ProducerCheckpoint`.
La receta numérica no tiene defaults científicos: el futuro registro debe
declarar semilla, TRUE/ZERO, endpoint, batch, capacidad, escala residual, LR,
decay y clipping. Las masas se admiten explícitamente, sin sustituirlas por un
muestreador ponderado que las cuente dos veces.

El motor fija hashes de los tensores normalizados, orden de muestras, etiquetas,
masas, receta y fuentes al iniciar cada segmento. No revalida esos bytes por
update. El caller científico deberá verificar los padres, manifiestos, cachés y
normalizadores y poseer el lock de escritor y scope GPU. No se ha integrado aún
ese registro ni una cola científica de fusión.

En batches sin RGB la pérdida es constante respecto a los parámetros: se
conservan gradientes cero y la semántica ordinaria de AdamW y scheduler. No se
redibujan muestras para encontrar una fila RGB. La salida de fallback sigue
copiando exactamente event TTC; esta política debe constar en la receta final.

Dieciséis pruebas CPU de contratos y gradientes pasaron, sin optimizer updates.
Después se congeló una receta **exclusivamente sintética** antes de ejecutar
40 updates CPU: TRUE y ZERO, cada uno diez continuos frente a cinco más
restauración más cinco. Los nueve estados comparados pasaron exactamente en
ambos brazos, incluidos pesos, AdamW, scheduler, muestreador, RNG y curvas.
CUDA no se inicializó. Estos 40 updates están en los journals `qa` y forman
parte del presupuesto técnico, no de los futuros 5.000 updates científicos.

Recibos bajo `artifacts/joint_measurement_20261012/`:

- `qa/fusion_engine/cpu/REGISTRATION.json`, SHA-256
  `2c4c679a6deb08f738c0e80af1a46aa71d74e135d3fb41d7abb04ca27995ec4e`.
- `qa/fusion_engine/cpu/PASS.json`, con las dieciocho comparaciones aprobadas.
- `verification/FUSION_ENGINE_CPU_QA_20261014.log`.

Faltan padres elegidos tras P/H, extracción real, normalización admitida,
registro científico, QA del dispositivo final, entrenamiento de ambos heads,
evaluación del pipeline y coste. Los resultados anteriores no demuestran una
mejora TTC ni completan la rama RGB/fusión.

## Caché nueva por productor y fragmentos

`fusion_cache.FusionCacheWriter` publica fragmentos NPZ con compresión lossless,
FP32, tiempos int64 y máscaras explícitas. Cada recibo fija identidad del
endpoint, esquema, extractor, fuente sensorial, manifest y orden exacto de IDs.
Una extracción interrumpida reutiliza sólo fragmentos admitidos y verificados;
no sobrescribe un fragmento válido ni repara silenciosamente uno corrupto.
El cierre exige todos los fragmentos y conserva las filas sin RGB.

Diez pruebas CPU pasaron, cubriendo reanudación, igualdad de arrays, IDs Unicode
y bytes, cambios de productor/esquema/fuente/orden, corrupción e interrupción
antes del recibo. No ejecutan modelos ni optimizer updates. El escritor no
selecciona padres, extrae features ni ajusta normalizadores: queda por conectar
al extractor sensorial del pipeline elegido y generar las cachés reales.

## Extractor de descriptores P2 y prueba con consultas reales

`fusion_pair_extract.py` implementa un esquema propuesto de 196 valores:
96 medias espaciales, 96 máximos, intervalo nativo y su logaritmo, fase previa
y fase emitida. Reutiliza el descriptor devuelto por los modelos de par; exige
un endpoint privado congelado en eval y no introduce targets, cajas ni metadata
en el forward. El esquema aún debe ser elegido explícitamente al registrar la
fusión científica. No se aplica automáticamente a R_CTX u otra arquitectura.

La extracción de tiempos contrasta el intervalo FP32 del modelo con los dos
timestamps nativos finales. Event utiliza `producer_times_us` y RGB
`producer_sensor_times_us`. Los tiempos no se sustituyen porque sus duraciones
sean parecidas. Ocho pruebas CPU cubrieron relojes distintos, T3 reducido al par
final, RGB ausente, rechazo de productores entrenables y conversión firmada.

El preflight reproducible `python -B -m
operational.joint_measurement.fusion_extract_preflight` pasó en CPU a las 12:04
del 14 de octubre. Usó el endpoint event cerrado `PAIR-CORR-S7` y únicamente los
índices P 0, 17, 83 y 159, declarados antes de inferir. Verificó predicciones
bit a bit contra el forward original, recarga lossless de todos los campos y
estado del modelo intacto. CUDA no se inicializó; cero optimizer updates.

Evidencia bajo `artifacts/joint_measurement_20261012/`:

- `diagnostics/fusion_features_pair_P_v1/REGISTRATION.json`, SHA-256
  `e8eabbacde9cc4ef169fcc3df4247729b45e693fca72e76a505892c56ffeab4e`.
- `diagnostics/fusion_features_pair_P_v1/PASS.json`.
- `diagnostics/fusion_features_pair_P_v1/event/features.npz`, SHA-256
  `7296d1137fe073c3decafa310930652ef8c4ce884d8ab2a6ff68b39acf66b046`.
- `verification/FUSION_PAIR_FEATURE_PREFLIGHT_20261014.log`.

Son cuatro filas diagnósticas, no una caché completa de entrenamiento ni una
selección del padre event. La prueba no mide precisión, no abre H de RGB y no
convierte PAIR-CORR en candidato. Las cachés completas se crearán después de
registrar los padres y la receta a partir de la evidencia autorizada.

## Export de cabeza con normalización incluida (14 de octubre, 12:17)

`fusion_export.py` incorpora las medias y escalas H como buffers FP32 en una
copia privada de la cabeza. Su entrada son las features sin normalizar y TTC de
los productores, con máscara RGB dinámica. No incluye encoders ni preparación
sensorial. El binding conserva identidades, estadísticas, configuración TRUE/ZERO,
hashes de cada tensor de pesos y ejemplos; rechaza reutilizar un export bajo
pesos o configuración distintos. El fallback event debe coincidir bit a bit.

Cuatro pruebas CPU pasaron para ambos modos, normalizadores no triviales,
disponibilidad completa/mixta/ausente, integridad del artefacto y rechazo de otro
productor. Se preservaron los fallos iniciales: la ruta Unicode de Windows exigió
guardar/cargar TorchScript mediante archivos abiertos desde Python; una tupla
del binding se convirtió a lista para conservar identidad al releer JSON.
Los logs fallidos y el aprobado permanecen en `verification/`.

`fusion_export_preflight.py` produjo exports de los checkpoints **sintéticos**
TRUE/ZERO de diez updates ya existentes, sin entrenar de nuevo. Reconstruyó sus
normalizadores y los comprobó exactamente contra el banco usado en la QA.
Ambos exports pasaron tres casos de disponibilidad con error TTC máximo 0.

Después, `replay_fusion_export.py` recargó cada artefacto en un proceso distinto
usando únicamente NumPy y PyTorch, sin módulos Python del proyecto. Los seis
casos volvieron a coincidir exactamente y no inicializaron CUDA. Los recibos
están en `verification/fusion_normalized_export_v1/{TRUE,ZERO}/REPLAY.json`.
Los TorchScript tienen SHA-256:

- TRUE: `d12fad62ab6a1a9f3b8c00e8813f646761ffefbaff0b28fd06c323d8973b2160`.
- ZERO: `a57cdad19fa692764b807bb4c8a1710a1cdcc5dae0bbc8728bc4aacca9aceee1`.

El replay es repetible, desde el entorno compartido, con:

```powershell
python -B operational/joint_measurement/replay_fusion_export.py artifacts/joint_measurement_20261012/verification/fusion_normalized_export_v1/TRUE
python -B operational/joint_measurement/replay_fusion_export.py artifacts/joint_measurement_20261012/verification/fusion_normalized_export_v1/ZERO
```

Los pesos son de QA sintética y no son candidatos. Cada futuro endpoint
científico necesita su propio export, normalizadores y replay, además de una
verificación sensorial completa y las métricas del pipeline.

## Comparadores históricos completos admitidos

`historical_multimodal.py` verificó R_CTX y F_TRUE para su futura comparación con
el sistema V18. No seleccionó padres nuevos ni leyó columnas de predicción o
target V para calcular scores. Admitió por SHA-256 los seis expertos históricos,
los checkpoints de ambas cabezas, sus normalizadores y cachés de entrenamiento.
Los journals y payloads de las dos cabezas coinciden en 2.500 updates completos
y durables. Los cachés supervisados contienen las 18.422 consultas H exactas.

Las tablas V históricas contienen las 16.432 consultas, en el orden y grupos
correctos, con sus recibos de salida nativa sin filtrado. Sus hashes se
preservan para una comparación posterior con candidatos ya congelados. No se
han recalculado aquí métricas antiguas ni usado V para elegir una receta.

Los checkpoints admitidos son:

- R_CTX: `4838ffcd274254bd104250f116f447d2efca5ce75599d79479256091df8d65fc`.
- F_TRUE: `dba93f9794e852d0e1d7323ac91ed257c59ec8e20a8ea952a93c5742a7f72f1b`.

Recibo: `verification/HISTORICAL_MULTIMODAL_ADMISSION.json`, bajo el root V18 de
artefactos. Incluye genealogía, normalizadores, hashes de tablas y manifiestos.
Las fuentes originales sólo se leyeron. Cero optimizer updates.

Ambas cabezas se entrenaron sobre H. Si la fusión nueva usa R_CTX y después
aprende otro módulo en H, será entrenamiento supervisado secuencial, no OOF ni
evaluación independiente H. Los perfiles históricos de las cabezas tampoco
son costes sensoriales completos. El control P2 y estos comparadores tienen
funciones distintas y deben conservarse ambos al interpretar el sistema final.

## Caché RGB residente: perfil y transición de ejecución

El perfil prospectivo `runtime/rgb_input_cache_profile_v1` comparó 16 batches
P de 32 muestras (8 de calentamiento y 8 medidos), con orden alternado.
Todos los campos de entrada y el RNG CPU coincidieron exactamente. No hubo
trabajo CUDA ni updates de optimizador. La mediana del loader pasó de
1,221072 a 1,028967 segundos (ratio 0,842675), cumpliendo el gate 0,85.
Es una medida del loader con entrenamiento concurrente, no del entrenamiento
completo. El perfil conserva los tiempos individuales, incluidos los lentos.

`rgb_cached_run.py` reutiliza el driver, lock, trainer y contabilidad originales.
Conserva hasta 6 GiB de shards uint8 nativos entre scopes, sin cambiar el decode,
collation, precisión, muestreo, loss, optimizer ni endpoints registrados.
Comprueba cambios de archivo en hits y limita la residencia si hay menos de
4 GiB disponibles. No activa CUDA Graph ni cambia la ruta de evaluación.

La transición debe ocurrir después del guardado final de un scope del driver
original. Un hold operativo temporal en el documento de autorización hace
que ese driver salga antes de comenzar otro scope; no retira autorización del
usuario ni modifica los límites. Se conserva el documento original byte a byte
y se restaura antes de iniciar la nueva ejecución. No se mata el trainer activo.
Con driver anterior terminal, journal completamente durable y checkpoint
preservado, el wrapper restaura el estado original y compara exactamente modelo,
optimizer, scheduler, sampler, RNGs y cursor antes de permitir nuevos updates.

La admisión y los recibos de restauración quedan en `runtime/rgb_resident_v1`.
El perfil y el código por sí solos no demuestran que la transición se haya
realizado; lo prueban el recibo de migración y el avance posterior del journal.

La transición se ejecutó el 14 de octubre a las 12:44 (Madrid). El proceso
original 34964 terminó después de guardar sus 4.657 updates. El checkpoint
preservado tiene SHA-256
`48d1b4bceb2a088ae75f40cf9d98dd4b9129223a5a5b0da02f88664059228f4f`.
El nuevo proceso 16580 pasó los once checks de restauración exacta y avanzó
desde ese punto; no se añadieron updates de recuperación. La autorización
original se restauró byte a byte. El mensaje de salida del driver antiguo
por falta de estado `APPLIED` corresponde al hold operativo deliberado,
documentado en `runtime/rgb_resident_v1/MIGRATION.json`.

Ocho pruebas CPU cubren mutación de shards/binding, presión de memoria,
restauración exacta y corrupta, rechazo de escritor vivo o estado no durable,
y preservación de checkpoint/autorización. Todas pasaron sin optimizer updates.
El observador 39104 vigila el proceso nuevo cada hora en punto, empezando a
las 13:00. `Resume-JointMeasurementRGB.ps1` selecciona la ruta residente
cuando existe su admisión; sigue usando el mismo lock de escritor.

Todavía falta medir el ritmo de scopes completos con la caché caliente. El
15,7 % observado pertenece exclusivamente al perfil previo del loader.

## Corrección detectada en la revisión de las 13:00

El wrapper v1 vaciaba la caché cuando la siguiente vuelta del driver consultaba
el brazo FT ya completo, aunque la caché perteneciera al fit FROZEN activo.
El contador publicado con `fit_id=FT` permitió detectar el fallo. Esto perdió
residencia entre scopes, no updates ni estado numérico del entrenamiento.
Se conservan el código v1 y sus recibos, incluidos esos metadatos mal asociados.

V2 marca si el fit realmente solicitó el dataset; un retorno temprano de un
brazo ya completo no puede liberar ni atribuirse la caché de otro. La prueba
de regresión recorre tres scopes FROZEN intercalados con FT completo y exige
una sola instancia residente hasta el fin del fit. Esa prueba, dos de restore
y tres de frontera durable pasaron. La admisión nueva vive en
`runtime/rgb_resident_v2`; enlaza la anterior por hash. Se usa otra transición
en frontera guardada para aplicar la corrección al proceso que estaba vivo.

La comparación observacional v1, registrada antes de leer las series, tomó
256 updates previos (4402–4657) y 256 posteriores (5045–5300), con al menos
256 de calentamiento. Las medias fueron 4,305123 y 0,912533 s/update;
las medianas, 1,900604 y 0,688124. El tramo reciente equivale a 1,095851
updates/s, excluyendo checkpoints, restauración y pausas entre scopes.
La cola previa incluye una fuerte ralentización del equipo: estos números
no prueban una aceleración causal de 4,7× debida únicamente a la caché.
El primer scope v1 completó 517 updates, y aún perdió la caché al siguiente.
La revisión de las 14:00 debe comprobar counters acumulados entre scopes v2
y actualizar el ritmo de extremo a extremo antes de atribuir una mejora estable.

V2 se reanudó a las 13:14 desde 5.640 updates completos y durables, con SHA-256
`fc1aa6afdf94ad0e1a01aef03dbb587feab98861f3585a9427f96c006dcc8eb7`.
Las once comprobaciones de restauración fueron exactas, sin nuevos updates de
recuperación. El proceso 17536 avanzó a 5.667 en la comprobación de arranque.
El observador 42216 tiene programada la revisión de las 14:00. Los recibos
`runtime/rgb_resident_v2/{DURABLE_ORIGIN,TRANSITION}.json` enlazan la evidencia.

La observación temporal v2 está inscrita antes de leer su serie. Para aplicarla
al checkpoint durable de las 14:00, reutilizar el mismo analizador y ventanas:

```python
from operational.joint_measurement import rgb_runtime_timing as timing
timing.RUNTIME = timing.RUN / "runtime/rgb_resident_v2"
timing.OUT = timing.RUNTIME / "timing"
timing.run()
```

Esta selección de root queda vinculada por el hash del origen y la admisión
de runtime en `runtime/rgb_resident_v2/timing/REGISTRATION.json`. Comparar además
el contador acumulado de muestras atendidas por caché (`hits + misses`) con
32 veces los updates desde 5.640: sin presión de memoria ni reinicio, esos
contadores deben cubrir todos los scopes del mismo proceso, no sólo el último.

## Features RGB comprobadas sobre datos reales P

`fusion_rgb_extract_preflight.py` completó en CPU las mismas cuatro consultas
P de diagnóstico (índices 0, 17, 83, 159) con el endpoint cerrado
RGB-PAIR-CORR-FT-S7. No evaluó H, no calculó métricas de precisión ni eligió
padres de fusión. La receta diagnóstica quedó registrada antes del forward.

Las 196 features y la TTC firmada coinciden exactamente con la salida nativa;
la recarga del archivo conserva todos los arrays, y los pesos quedaron intactos.
Los intervalos RGB verificados son 100.002, 100.001, 100.002 y 100.001 us,
calculados desde timestamps int64 propios de los frames. No se reemplazaron
por el intervalo event de 100.000 us por ser aproximadamente iguales.

Evidencia: `diagnostics/fusion_features_rgb_P_v1/PASS.json`.
Registro SHA-256:
`d9e3103689f6ce10df3c4c41848fec6ba83300c02023ac52c4e46e6c9f9e9194`.
Caché SHA-256:
`19d8c2397d543ce4e7f9335dc0b76045f6692d1aed9b2e3c4ac5cde94c638e8b`.
Cero optimizer updates y CUDA sin inicializar. Esta comprobación de cuatro
filas complementa la de eventos; todavía no es una extracción de población
completa ni una validación de precisión de la fusión.
