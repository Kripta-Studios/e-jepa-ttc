# ADR-0002 — Correcciones directas de RGB-PORT y continuidad de checkpoints

- Fecha: 2026-10-09.
- Decisión: aplicar las correcciones en los módulos originales, por petición explícita del usuario.
- Precisión: conservar la geometría de entrenamiento BF16. No se aplica el cambio FP32 de la auditoría.
- Campaña: `artifacts/rgb_port_20261008`.

## Continuidad y procedencia

Los checkpoints event existentes se conservan sin reescribirlos: E_A5_MATCHED en
15.303 updates y E_C2F_MATCHED en 6.685, con endpoint de 30.330 para cada uno.
No cambia su forward matemático, pérdida, sampler, semilla, límite de updates,
AdamW, scheduler, RNG ni cursor. Cambia la política de caché CPU.

Los cinco manifests históricos de congelación se conservan como evidencia de
sus ejecuciones anteriores. `AUDIT_CODE_MIGRATION.json` enumera cada hash anterior
y actual admitido, conserva los orígenes de los checkpoints y liga los módulos
auxiliares nuevos. Los validadores originales admiten solo esa transición exacta;
cualquier otra modificación sigue provocando un error. No hay un flag para ignorar
hashes. El código anterior está en `audit_fixes_20261009/before`.

Los nuevos payloads de checkpoints registran `code_migration_sha256`; cada fit
tiene también `AUDIT_REVISION_RUNTIME.json`. Los RGB nuevos incorporan la
migración en `source_identity`; su carga rechaza pesos RGB de otra revisión.
La receta histórica queda identificada como ancestro, no como prueba de que los
bytes actuales fueran los ejecutados antes de esta corrección.

## Cambios

1. El router C2F RGB recibe luminancia y `log1p` del gradiente espacial, no los
   canales verde/azul interpretados como count/rate. La rama event no cambia.
2. RGB mantiene T2/T3 reales sin padding. Sus particiones comparten una pérdida
   sobre el batch efectivo: denominadores por posiciones válidas y top-k global
   para el término de cola. Solo hay un backward y una frontera de optimizer por
   batch efectivo. La ruta homogénea event conserva su implementación anterior.
3. La caché event retiene filas aún no consumidas del grupo lógico de 256,
   respetando los límites existentes de 512 MiB/1 GiB. No cambia el sampler.
   `reads` cuenta shards decodificados y `hits` filas reutilizadas; este último
   contador no es comparable directamente con los antiguos hits por shard.
4. Los límites TTC invierten el orden de los extremos de fase en una misma rama.
   Un intervalo que toca/cruza fase cero se marca no disponible; se conservan sus
   extremos en fase. Los nombres heredados `q10_ttc/q90_ttc` son alias de límites
   de intervalo, no una garantía de cuantiles marginales TTC. Se exportan también
   `ttc_interval_low/high/status` y la semántica en el recibo de predicciones.
5. Dev32 utiliza cuatro observaciones event a 100 ms, con todo el input dentro de
   650 ms. RGB selecciona tripletas de frames reales a aproximadamente 100 ms
   (tolerancia 1 ms), con deltas medidos, sin frames futuros ni historia formada
   solo por queries de evaluación. T2 se admite únicamente al inicio real de la
   secuencia seleccionada. No se consulta ground truth para construir inputs.
6. Los costes añaden HDF5→voxels y HDF5→productores/contexto/fusión. Una comparación
   exacta contra la caché en los queries medidos precede al perfilado. Las rutas
   preparadas se identifican como tales. Se registra VRAM allocated/reserved pico
   cuando se mide CUDA; la memoria CPU existente sigue siendo RSS antes/después.
   La primera llamada cronometrada no se presenta como caché del SO vacía.
7. El informe lee mediana y p95 desde `warm_ms`, el esquema realmente medido.

## Evidencia y límites

La evidencia está en `audit_fixes_20261009`:

- `TEST_RESULTS.xml`: suite RGB-PORT, sin optimizer steps técnicos opt-in.
- `RGB_SOURCE_INTEGRATION.json`: batch P real con 3 ejemplos T2 y 29 T3.
- `CACHE_PARITY_BENCHMARK.json`: 256 ejemplos, todos los tensores y metadatos
  idénticos; 131→21 lecturas, 85,7→51,0 segundos en la prueba CPU secuencial.
  No es una medida de throughput GPU concurrente ni una comparación de SO frío.
- `RAW_EVENT_PARITY.json`: HDF5→voxels idéntico a la caché en una muestra real.
- `TRANSFER_INPUT_PARITY.json`: event idéntico a los cuatro slots de 100 ms
  correspondientes del adaptador anterior y RGB causal con deltas reales.
- `QUEUE_COMPATIBILITY.json` y `EVENT_RESUME_VERIFICATION.json`: comprobaciones
  de congelaciones, comandos habituales, restauración y paridad event.

Dev32 continúa siendo un conjunto expuesto y exploratorio. La ROI retrospectiva
se conoce en el query, por lo que su disponibilidad se registra en ese instante;
no se afirma equivalencia con las ROI nativas por observación. La geometría BF16
conserva la limitación numérica descrita en la auditoría. Los costes finales y la
transferencia con todos los endpoints deberán medirse cuando esos pesos existan.
Las comprobaciones negativas de desarrollo se conservan con su causa.

## Reanudación

El entry point sigue siendo el mismo, desde la raíz del repositorio:

```powershell
$env:PYTHONPATH = 'src;.'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -m operational.rgb_port_c2f_graph.queue resume --run artifacts/rgb_port_20261008
```

La pausa preexistente por la revisión V12 permanece: `PAUSE` y los dos
`fits/E_*/STOP_REQUEST`. La corrección de código no la revoca. Cuando corresponda
reanudar, retirar esos tres marcadores tras terminar el otro trabajo y ejecutar
el comando habitual. No borrar checkpoints, journals, cursores ni congelaciones.
Los CUDA Graphs se vuelven a preparar con el gate existente de restauración exacta
sin optimizer steps; los ensayos CPU no sustituyen ese gate CUDA.
