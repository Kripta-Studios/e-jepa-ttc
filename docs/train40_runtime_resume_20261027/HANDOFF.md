# Continuación TRAIN40 — cierre solicitado el 28 de octubre

## Estado y prioridad del usuario

Trabajo pausado por petición expresa del usuario después de completar la
comparación RGB PIL frente a uint8. **50 Hz y menos de 20 ms completos no se han
demostrado.** La prioridad aceptada es conseguirlo primero para un objeto y
después para dos; veinte objetos queda para una fase posterior. No reanudar
automáticamente cargas largas ni entrenamiento a partir de este documento.

Rama: `perf/train40-persistent-50hz`. El cierre anterior era `9f62aebc`; la base
científica publicada permanece en `2ad7a05797799caf1bcdc9202c307d244f9fe31d`.
Consultar `git log` para el commit de este cierre y `git status` antes de editar.
Los cambios están separados en commits de runtime, medición y documentación.

El candidato congelado sigue en
`D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE`. No se escribieron sus pesos,
normalizadores ni las predicciones originales. Tampoco se reentrenó.
Las referencias son `artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json`
y `artifacts/final_train40_transfer_20261027/final_train40/own`.

## Resultado que sí está admitido

`corrected_all554_v1` valida las 554 consultas originales y las tres salidas
con `atol=rtol=1e-5`. E_CTX coincide bit a bit; PHASE y R_CTX pasan la tolerancia.
Las 554 entradas pasan y 553 son bitexact: la diferencia de entrada del índice
321 es diminuta y está conservada, sin ampliar tolerancias. Los 606 archivos
vinculados permanecen intactos. Esta validación usa un objeto por consulta y
reconstrucción de historial antes del reloj; no es una prueba sostenida.

El cambio decisivo fue conservar el batch del encoder de eventos: E reuse OFF,
R reuse ON. Desactivar también R introducía otra discrepancia. Las capturas de
productores usan streams dedicados, con pruebas de primera reproducción y cambio
de cantidad de objetos. Los pilotos multiobjeto conservan el batch RGB mediante
chunks de un objeto y contextos independientes.

La comparación ABBA PIL/uint8 terminó en cuatro procesos, 120 actualizaciones,
dos cantidades de objetos y tres casos. Todos los TTC, inputs, deltas, recibos y
cobertura pasan. Uint8 **no mostró una ventaja robusta de latencia**; la presión
de memoria y la deriva del coste CPU impiden vender esa prueba como mejora causal
estable. Se conservan todas las repeticiones, incluidas las primeras, y tablas
con medias, medianas, p95, p99 y máximos.

Consultar las cifras regeneradas en [RESULTS_GENERATED.md](RESULTS_GENERATED.md)
y [MULTIOBJECT_PAIRED_GENERATED.md](MULTIOBJECT_PAIRED_GENERATED.md). Sus poblaciones
y relojes son distintos: no sumar tramos ni comparar directamente contra los
1,612 segundos de evaluación offline original.

## Configuración predeterminada efectiva

`operational/train40_runtime/defaults.py` define `train40_single_dual_parity_v1`.
El servicio persistente y sus runners utilizan sus valores:

- Productores originales, FP32 y TF32 desactivado en los runners.
- E reuse OFF; R reuse ON; RGB por chunks de un objeto.
- Ramas paralelas y ocho workers RGB; PIL sobre ROI como ruta predeterminada.
- Backend prefix, query graphs y normalización top-k exacta en los runners.
- Servicio con capacidad predeterminada de dos objetos; piloto con uno y dos;
  ensayo sostenido empieza por un objeto. Veinte requiere selección explícita.
- Uint8, caché de características y transporte compilado siguen siendo opt-in.

El servicio recibe su historial explícitamente; al usar su API directamente,
construirlo con las opciones del perfil y respetar FP32/TF32 OFF. Los runners
configuran precisión y primación sintética. No se ha cambiado el runner
científico congelado ni convertido un resultado de paridad en una promesa de
tiempo real. Un consumidor que dependiera del antiguo chunk RGB agrupado debe
seleccionarlo explícitamente y obtener su propia admisión.

## Entorno y reproducción

El `.venv` de este checkout conserva un problema anterior de ruta editable y
codificación. Usar el entorno hermano que se utilizó en las pruebas:

```powershell
$py = '../e-jepa-ttc/.venv/Scripts/python.exe'
& $py scripts/benchmark_train40_multiobject.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --manifest 'artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json' `
  --reference 'artifacts/final_train40_transfer_20261027/final_train40/own' `
  --output 'artifacts/train40_runtime_resume_20261027/REEMPLAZAR_POR_DIRECTORIO_NUEVO'
```

El default ejecuta los casos 0, 137 y 553 para uno y dos objetos. Añadir
`--rgb-uint8` compara la ruta opcional; nunca reutilizar un directorio de salida.
Para un control integral, seleccionar explícitamente `--backend integral
--no-event-query-graphs --event-normalization sort`.

La admisión completa corregida está ligada al worktree detached
`../e-jepa-ttc-train40-admission-corrected`, commit `8c5226dd`. El wrapper reubica
seis rutas de fuentes originales solo después de verificar los hashes de ambas
copias. Comando y wrapper exactos, fuentes ejecutadas, preservación y resumen
están en `corrected_all554_v1` del paquete de evidencia. No volver a ejecutar
sobre su salida existente. El worktree de baseline `9f62aebc` también se conserva.

Hardware de los ensayos: RTX 5070 Ti Laptop de 12 GB, Torch 2.11.0+cu128,
Python 3.11.15, FP32, semilla 7 y dos threads Torch. No se modificó el plan de
energía. Solo un proceso usó la GPU para medir en cada momento. Antes del próximo
contraste, detener pruebas CPU del swarm y registrar RAM, commit, VRAM y carga.
Había numerosos procesos MLflow y backups anteriores, ajenos al ensayo: no se
cerraron. No matar procesos solo por llamarse Python.

## Vías siguientes, en orden

1. **Transporte con log original.** Las variantes GPU compiladas fallaron el gate
   conjunto: default 5/7, sin FMA 4/7, sin FMA y división exacta 5/7. Son más
   rápidas en productores preparados, pero no se admiten. `--native-log` conserva
   las operaciones log sensibles mediante ATen opaco; pasa tests CPU y todavía
   no se ha probado en GPU. El driver incluye atribución por E_A5/E_C2F/R_A5/R_C2F.
   Probar primero los siete casos sensibles y después dos objetos con el mismo
   batch E del servicio. Si pasan, repetir toda la cohorte y la cadena completa.
2. **Identidad RGB inmutable.** Evitar hashear de nuevo todos los frames en cada
   consulta. `rgb_raw_ring.py` y `rgb_frozen_identity.py` sellan frames nuevos y
   conservan bytes inmutables. El ring necesita ambas fases de RGB nativo 20 Hz:
   dieciséis slots evitan releer las dos cadenas alternas. El sellado de frames
   nuevos cuenta desde su llegada; no esconderlo en warmup cuando aún son futuros.
   Los módulos tienen validación CPU; el adapter rápido de servicio quedó como
   diseño y no está activado ni medido.
3. **Servicio sostenido.** `benchmark_train40_paced.py` ofrece llegadas a 50 Hz,
   cola/bytes acotados, todos los TTC en host y paridad posterior separada. Su gate
   distingue uno, dos y veinte objetos y requiere 60 s, cobertura, todas las
   respuestas bajo 20 ms y ausencia de acumulación. No hay aún una ejecución GPU
   sostenida admitida. `--raw-source persistent` es una opción nueva con tests CPU:
   reader exclusivo del productor, warmup causal, caché de 650 ms, cierre en
   `finally` y únicamente lecturas nuevas. El control `disk` sigue separado.
4. **Recorte PIL entero en GPU.** `rgb_pil_exact.py` conserva coeficientes de
   22 bits, redondeo uint8 por eje y LUT de normalización FP32. La matriz CPU,
   incluyendo compilación eager/fullgraph, pasa. Falta medir CUDA y TTC; para
   uno/dos objetos, subir RAW completo puede costar más que subir crops uint8.
5. **Selector event para uno/dos.** El selector compilado ayudó a veinte ROI,
   pero empeoró un objeto. Mantener el query graph original para uno. La variante
   privada de 24 cuartos directos solo tiene prueba entera CPU; no promoverla
   sin medición y paridad. La caché event entre fases casi nunca reutiliza datos
   cuando cambia la ROI: el análisis causal está conservado.

El planificador Windows también importaba: `Event.wait` añadió hasta unos 16 ms
antes de RAW. La opción `sleep` mantiene deadlines absolutos y todas las demoras
dentro del reloj; el runner nuevo la usa por defecto. Dos controles de 60 s con
hilos/cola y resultados ficticios validaron la mejora del temporizador, **no TTC**.
El helper conserva `event` como default de compatibilidad y permite A/B explícito.

## Reparto recomendado para el siguiente swarm

Una sola persona/agente asigna la GPU. Separar transporte numérico, servicio
completo, identidad/ingesta RGB, representación event, medición/preservación y
un consejo de revisión de paridad. Definir propiedad de archivos y congelar
dependencias mientras se miden. No ejecutar compilaciones ni tests CPU pesados
en paralelo con mediciones completas de preprocesamiento. Aumentar población
solo después de admitir el hito anterior.

## Evidencia y limitaciones de archivo

Los artefactos completos locales están en `artifacts/train40_runtime_resume_20261027`.
Los paquetes bajo `evidence/` conservan JSON compactos, casos, comandos y snapshots
sin pesos, RAW, wheels ni cachés del compilador. Cada paquete tiene MANIFEST con
SHA-256; también se verifican los bytes de los blobs staged en Git.
`report_train40_resume.py` regenera tablas desde esos recibos.

```powershell
& $py scripts/report_train40_resume.py `
  --source 'docs/train40_runtime_resume_20261027/evidence/closure_20261028_002'
```

Algunas versiones intermedias cambiaron antes de archivarse. El manifiesto
identifica el hash esperado y la ausencia de la copia actual; muchas están en
snapshots históricos separados. Una versión inicial de `rgb_pil_exact.py` no
importada por el piloto de caché no pudo recuperarse: no se fabricó un snapshot.
La preservación final de aquella corrida sí se comprobó antes del cambio.
Los snapshots exactos de la admisión corregida y de los cuatro runs ABBA están
conservados. Los resultados negativos se mantienen y ninguna tolerancia se amplió.

Al cierre no quedan ensayos GPU propios en marcha. Los nuevos defaults pasan las
pruebas CPU pertinentes; los caminos CPU-only conservan explícitamente ese estado.
Consultar también [CURRENT_FINDINGS.md](CURRENT_FINDINGS.md) y
[MULTIOBJECT_SERVICE_AND_TRANSPORT.md](MULTIOBJECT_SERVICE_AND_TRANSPORT.md).
