# A1: lector paralelo, grafos persistentes y menos sincronizaciones

Las tres mejoras se admitieron prospectivamente para **LAST-BLOCK y FULL**.
La receta científica, sus 12.000 updates por brazo, población, pesos de ejemplos,
precisión FP32, orden de muestreo y gates permanecen registrados en A1. La capa
de runtime conserva las fuentes anteriores y añade su propio registro y pruebas.

## Punto de transición

El coordinador anterior se suspendió mientras su hijo terminaba el scope normal.
El hijo salió por sí mismo con **1.086 updates ejecutados y durables**, cero
updates pendientes y cero recuperación. Después se terminó únicamente el
coordinador y se conservó una copia del checkpoint como testigo de transición.
SHA-256: `80abe18a5b973a83267f6039383247d4772bedbb0a18808709a5f3d6d0339069`.

La nueva ruta restaura pesos, Adam, scheduler, sampler y todos los RNG. No se
reinicializa el optimizador ni se repite el prefijo científico. El primer
checkpoint nuevo genera `ACTIVATION_VERIFIED.json`, contrastando el prefijo
completo de curvas y tiempos, identidad, pasos de Adam, scheduler y los nuevos
IDs producidos por el generador original.

## Cambios de ejecución

1. **Descompresión paralela:** ocho workers leen filas independientes. Las
   vistas de NPY sobre el miembro descomprimido evitan copias intermedias.
   El collator original conserva todos los campos y duplicados del muestreo.
   Se mantienen SHA de admisión, CRC externo y fingerprints de pack/recibo.
2. **Caché y grafos persistentes:** un hijo CUDA por brazo conserva el modelo,
   optimizador, grafo y caché LRU de hasta 8 GiB entre scopes de hasta 900 s.
   Cada scope conserva su propia reserva/liquidación GPU, guardas de recursos
   y checkpoint final. El cambio de brazo utiliza un proceso CUDA nuevo.
3. **Menos sincronizaciones de pérdida:** se agrupan los guards de dominio y
   finitud antes de cada backward y las métricas se transfieren juntas a CPU.
   Se conserva el orden de operaciones FP32 y la suma en double de Python de
   los resultados de cada microbatch. No se añade autocast, TF32, cuantización
   ni una nueva pérdida; tampoco se captura Adam dentro del grafo.

El modelo sigue recomputando sus features con autograd. La caché contiene
sensores, no representaciones neurales detached. La congelación parcial mantiene
exactamente la política de gradientes y dropout del brazo original.

La activación científica quedó verificada a las **19:34:51 de Madrid**, en el
checkpoint **1.100**. Los siete checks de transición pasaron: prefijo íntegro
de 1.086 registros, tiempos anteriores, identidad, sampler, Adam y scheduler
avanzados 14 pasos y cero recuperación. SHA del primer checkpoint optimizado:
`1d6eba149993b7f7667357a53ac83f4646110a75f212a96eb7f7061a44ead51e`.
El primer recibo de runtime confirma ocho decodificadores y una captura CUDA.
El coordinador 23684 programa FULL con esta misma admisión después de LAST-BLOCK.
El observador 38716 vigila al nuevo coordinador a partir de las 20:00 locales.

En el recibo de producción de **1.200 updates**, los últimos 50 promediaron
**0,7491 s/update**, equivalentes a **4.806 updates/h dentro del bucle**.
La espera visible de datos promedió **1,318 ms/update**: la descompresión sigue
costando tiempo, pero el prefetch la solapa con el entrenamiento. Forward,
pérdida y backward ocuparon 0,5687 s/update en los intervalos del host.
Los 211 packs estaban admitidos y la caché retenía 5,78 GB decimales. Había una
captura en el primer scope; la continuidad a través de dos scopes ya se había
comprobado en QA, y se verificará también en los recibos de producción posteriores.

Esta ventana corta excluye los guardados y el arranque. El perfil anterior
observó unos 2.446 updates/h en otra ventana de pared. La mejora observada es
prometedora, pero no se presenta esa comparación como un benchmark emparejado
ni como una ETA definitiva del brazo FULL aún no iniciado.

## Evidencia previa a la activación

Registro: `runtime/A1_parallel_persistent_v1/REGISTRATION.json`, SHA-256
`5f6e5c7d21dc32b1116c596e1cb4dab5821a36410c609b48299924fa47c5fc12`.

| Prueba | Resultado |
| --- | --- |
| Diez tests CPU de transporte, duplicados, RNG, caché y rechazo de corrupción | PASS |
| Nueve tests CPU de términos de fase, gradientes y dominios inválidos | PASS |
| Once tests del observador de procesos | PASS |
| 12 próximos batches P de 32 desde el sampler guardado, más un batch con duplicados, dos pasadas | Todos los campos exactos; RNG global intacto, CUDA sin inicializar |
| Lector serial: mediana de la segunda pasada | 1,246546 s/batch32 |
| Lector paralelo: mediana de la segunda pasada | 0,352757 s/batch32 |
| Mediana de ratios emparejados paralelo/serial | 0,271549; supera el gate prospectivo ≤0,85 |
| Pesos reales del update1086: cuatro contrastes de backward original/nuevo | Pérdidas, gradientes y RNG CPU/CUDA exactos; cero updates |
| Mediana del ratio de forward/backward preparado, excluida captura inicial | Aproximadamente 0,94; supera el gate ≤1,10 |
| LAST-BLOCK: original4 vs persistente2+2 vs nuevo2+restore2 | Nueve comparaciones exactas en cada contraste; una captura en dos scopes |
| FULL: el mismo contraste | Nueve comparaciones exactas en cada contraste; una captura en dos scopes |

Las pruebas de trayectoria consumieron **24 updates técnicos**, contabilizados
en `qa/A1_parallel_persistent_v1`; no incrementan el endpoint científico.
Los nueve estados contrastados son modelo, Adam, scheduler, sampler, RNG CPU,
RNG CUDA, RNG NumPy, RNG Python y la curva con IDs, pérdidas y normas. Sólo se
excluye el tiempo de pared de la comparación exacta.

Se observó el aviso de PyTorch sobre streams de `AccumulateGrad` también en la
ruta original durante los probes. No se suprimió ni se usó para relajar la
paridad. Las pruebas de captura, gradientes y reanudación finalizaron aprobadas.

El benchmark del lector usó caché de filas desactivada en ambos lectores para
medir misses aleatorios; la primera pasada incluyó hashes de admisión, la
segunda usó packs ya admitidos. Alternó el orden de los métodos. La caché del
sistema operativo no se controló. El 72 % de reducción aproximada corresponde
a **lectura de batch**, no a una aceleración demostrada del entrenamiento entero.
La prueba de GPU usa entradas preparadas y tampoco equivale a coste end-to-end.

## Operación y recuperación

```powershell
./scripts/Resume-JointMeasurement.ps1 -Stage A1Training
```

El script elige `operational.joint_measurement.adaptation_runtime` cuando existe
la admisión; el módulo comprueba hashes y pruebas, y el bloqueo exclusivo impide
crear otro escritor. El coordinador ejecuta LAST-BLOCK y después FULL, y congela
ambos endpoints antes de habilitar evaluación H. Ante fallo del hijo se detiene
con causa registrada; no hace reintentos silenciosos.

`PERSISTENT_RUNTIME.json` publica tiempos recientes por etapa, contadores de
caché, scopes y capturas del proceso. Son intervalos del host con posible
solapamiento, sin sincronizaciones CUDA añadidas para medir cada etapa.
Los checkpoints mantienen el formato anterior. `NEXT_DECISION.json` y el
observador horario identifican al coordinador vigente.

Fuentes: `parallel_training_sensor.py`, `phase_transport_loss.py`,
`persistent_adaptation.py`, `adaptation_runtime_qa.py`, `adaptation_runtime.py`.
Pruebas y admisión exportadas en `evidence_A1_persistent_20261016`.

## Continuidad comprobada en producción

A las 19:53:27 del 16 de octubre se verificó el proceso científico 16080,
creado a las 19:33:16, mediante su identidad de proceso y comando activo. El
snapshot del update 2.600 registra dos scopes y una sola captura CUDA.
Los identificadores de los parámetros coinciden con el snapshot del update
1.200, los hits de caché son acumulativos y los 211 packs siguen admitidos.
La caché mantiene filas al cambiar de scope. Los siete controles de continuidad
son PASS, conservados en `PRODUCTION_SCOPE_CONTINUITY.json` junto con los dos
snapshots vinculados por SHA-256.

Esto verifica el comportamiento durante el cambio de tramo de producción,
además de la prueba exacta de continuidad/reanudación previa. No garantiza
retener caché ante un reinicio del sistema: ese caso reconstruye los recursos
efímeros y restaura el estado científico desde el checkpoint durable.

## Perfil pasivo posterior a la optimización

`runtime/A1_optimized_profile_20261016` conserva stacks, muestras de recursos,
errores de muestreo y resumen con hashes. En una ventana de recursos de 44,139 s
se completaron 52 updates: 0,849 s/update, aproximadamente 4.241 updates/h.
Es una observación de producción con profiler, no un benchmark emparejado.

El snapshot de los últimos cien updates de esa ventana mide unos 1,34 ms de
espera expuesta por batch, 110,33 ms de envío de transferencias, 616,81 ms de
forward/pérdida/backward y 86,04 ms de optimizador/journal. Las operaciones
asíncronas impiden interpretar estas categorías como tiempos exclusivos de GPU.

El muestreo del hilo principal atribuye 22,63 % a aritmética de `phase_terms`
y 15,45 % a escalado, guardas, diagnósticos u otro trabajo de la ruta de pérdida.
La suma orienta una posible captura adicional de la pérdida; **no demuestra
que ese 38,08 % sea eliminable**. Los ocho decodificadores se contabilizan por
separado porque sus tiempos se solapan. Se conservaron 77 errores del muestreo
no bloqueante y 21.142 muestras de hilos.

La pérdida aún se ejecuta fuera del grafo. Cualquier captura adicional requiere
paridad exacta de gradientes, estado y reanudación, además de benchmark nativo,
antes de instalarse. Este perfil no cambió el proceso científico ni añadió
updates de optimizador.
