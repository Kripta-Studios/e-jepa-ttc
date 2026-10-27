# Medición del entrenamiento real R_A5 — 17 de octubre de 2026

Se reanudó desde el checkpoint durable 169 y se mantuvo el entrenamiento en
marcha después de medir. No se cambiaron arquitectura, pérdida, BF16, batch
efectivo 32, semilla, población ni frecuencia de checkpoint: cada 100 updates.
La ventana analizada abarca **14:33:00–14:48:00 CEST**, exactamente 900,104 s
entre la primera y última muestra. Avanzó de **316 a 863: 547 updates**.
El monitor terminó cinco segundos después; esos cinco segundos no se incluyen
en el cálculo de velocidad. No hubo recuperación ni reinicio durante la ventana.

**Conclusión:** el tiempo se concentra en forward/backward, con GPU poco ocupada.
La comprobación global de recursos introduce un coste adicional medible. La
lectura anticipada y el checkpoint cada 100 updates no son los cuellos principales.
Esto localiza regiones de ejecución, pero todavía no identifica qué kernels o
esperas del driver explican el coste dentro del backward.

## Consumo y ritmo observados

| Medida | Resultado |
|---|---:|
| Velocidad media | 0,6077 updates/s; **36,46 updates/min** |
| Tiempo medio por update | **1,646 s** |
| Ejemplos procesados/s | 19,45, batch efectivo 32 |
| Ritmo sobre ventanas móviles de 60 s, mediana / p95 | 0,6165 / 0,6666 updates/s |
| Uso GPU, media / mediana / p95 | **14,64 % / 10 % / 49 %** |
| Memoria GPU máxima del dispositivo | 4,169 GiB |
| Potencia GPU, media / p95 | 51,84 / 60,32 W |
| Energía GPU aproximada durante la ventana | **12,96 Wh** |
| Temperatura GPU, media / p95 / máximo | 83,46 / 86 / 89 °C |
| CPU del proceso, media / p95 | 3,72 / 5,17 núcleos equivalentes |
| Hilos del proceso, mediana | 119 |
| RAM residente máxima del proceso | **5,90 GiB** |
| RAM disponible mínima del equipo | 10,74 GiB |
| Margen mínimo de memoria comprometida del sistema | 15,43 GiB |
| Bytes leídos / escritos contabilizados para el proceso | 6,59 GB / 29,82 MB |

El consumo y la utilización GPU son del dispositivo, compartido con aplicaciones
de escritorio; no son contadores exclusivos del entrenamiento. La energía no
incluye CPU, pantalla, pérdidas de alimentación ni consumo en el enchufe. Los
bytes de E/S del proceso tampoco demuestran lecturas físicas del SSD: pueden
incluir accesos servidos por la caché del sistema. No se midió la saturación
física de disco con contadores por unidad.

Las muestras complementarias del driver no indicaron limitación térmica SW/HW
ni límite de potencia activos. La temperatura merece seguimiento, pero no prueba
por sí sola que haya throttling; el muestreo tampoco excluye episodios breves.

Si se mantuviera este ritmo, a partir del update 863 quedarían **22,43 horas para
R_A5** hasta 49.932 updates. Es una extrapolación de la época 1, dentro del warmup
de foreground; no es la ETA de R_C2F, del stacking OOF ni del modelo final, ni
garantiza el mismo ritmo cuando cambia el objetivo después de la época 3.

## Dónde reside el tiempo del hilo principal

Se añadió un perfil externo `py-spy`, a 10 Hz y en modo `--nonblocking --idle`,
durante unos 13 minutos que solapan la ventana de telemetría. Recogió 7.733
muestras del hilo principal y reportó 66 errores de captura. Los porcentajes
son **residencia de la pila Python**, incluyendo las esperas que ocurren dentro
de una llamada: no son tiempos CUDA medidos por kernel ni CPU activa exclusiva.

| Región del bucle | Porcentaje muestreado |
|---|---:|
| Backward | **49,93 %** |
| Forward y objetivo | **36,42 %** |
| Comprobaciones de recursos | **7,46 %** |
| Escritura durable de journal antes/después del optimizer | 1,60 % |
| Comprobación de presupuesto de recuperación | 1,37 % |
| Clipping de gradientes | 1,11 % |
| Optimizer | 0,61 % |
| Transferencia host→GPU | 0,45 % |
| Validación del batch | 0,35 % |
| Conversión de componentes de pérdida a escalares CPU | 0,18 % |
| Espera de prefetch | 0,14 % |
| Checkpoint completo y recibos periódicos | **0,08 %** |
| Comprobación de pérdida finita | 0,05 % |
| Otras ubicaciones | ~0,25 % |

El escaneo global de procesos explica prácticamente toda la comprobación de
recursos: `psutil._pswindows._proc_info` aparece como hoja en el 7,03 % del total.
La ruta ejecuta ese escaneo cada 25 updates, además de guardas baratas por update.

Dentro del forward, `transport_physical_features` aparece en el 12,57 % del
total; el muestreo bilineal determinista en el 5,53 % y la correlación local en
el 5,15 %. **Son porcentajes inclusivos y anidados; no deben sumarse.** La pérdida
`causal_scale_ttc_loss` ocupa un 3,72 % inclusivo del total. El backward queda
oculto dentro del motor C++ de autograd para este instrumento.

Los decodificadores pasan aproximadamente el 96 % de sus muestras esperando
trabajo; el prefetch, el 72 %. Sus tiempos se solapan con entrenamiento y no
se suman al tiempo del hilo principal. Entre los snapshots de los checkpoints
300 y 800, la espera de prefetch aumentó sólo **0,280 s para 500 batches**.
El primer snapshot ya existía al iniciar la ventana: es un intervalo de
checkpoints, no una medida exacta de los 547 updates de la ventana.

## Orden de optimización que justifican los datos

1. **Perfilar CUDA y despacho dentro de forward/backward.** Aquí reside el
   86,35 % del tiempo muestreado. Priorizar correspondencias, lectura geométrica
   y sus gradientes. Hay margen aparente de utilización, pero el porcentaje de
   GPU no permite convertir todo ese tiempo en una aceleración recuperable.
   Un perfil breve con CPU/CUDA debe separar kernels, huecos de lanzamiento,
   sincronizaciones y esperas del driver. Después tiene sentido ensayar CUDA
   Graphs por firma compatible o fusión de operaciones, preservando el sampler
   determinista que resolvió la admisión BF16. No volver a `grid_sample`
   no determinista ni eliminar gradientes para ganar velocidad.
2. **Reducir el coste del inventario global de procesos.** Es la optimización
   administrativa con evidencia más clara. Conservar las guardas de RAM,
   memoria comprometida y disco por update; estudiar un inventario de procesos
   reutilizable que detecte nacimientos, salidas y reutilización de PID, y
   mantenga la cobertura del proyecto. Incluso eliminar hipotéticamente todo
   el 7,46 % daría un techo de aceleración de aproximadamente **1,081×**, no
   multiplicaría la velocidad del entrenamiento.
3. **Comprobar contención de hilos CPU con un contraste controlado.** Los
   119 hilos y los 3,72 núcleos equivalentes justifican medir, no demuestran
   oversubscription dañina. Comparar límites de hilos en procesos descartables
   con batches y estados idénticos, incluyendo preparación concurrente real.
   No cambiar variables globales del equipo. Un perfil aislado de cinco pasos
   precargados dio ~273 ms/paso, pero sus condiciones son distintas y no
   permiten prometer una aceleración 6× del entrenamiento completo.
4. **Mantener por ahora caché, compresión, prefetch y checkpoint cada 100.**
   Reducir sus esperas tiene poco techo de mejora directa. Evitar convertir una
   mejora pequeña de E/S en mayor uso de RAM. El journal durable se conserva:
   su ~1,6 % sostiene la contabilidad de recuperación.

La validación GPU y las conversiones de escalares siguen siendo candidatas
secundarias, pero el muestreo reduce su prioridad respecto a la hipótesis inicial.
No se ha aplicado una optimización científica nueva durante esta medición.
Cualquier cambio de runtime requiere preservar el checkpoint actual, registrar
la nueva identidad, comprobar pérdida/gradientes/optimizer/RNG y medir mejora
de extremo a extremo antes de adoptarlo.

El contraste previo `foreach`/`fused` **no se admite como evidencia de backend**:
el probe carga el estado del optimizer después de construirlo y puede restaurar
las flags de los grupos originales. No demuestra que `foreach` mejore un 3 %
ni que `fused` empeore. Se conserva el artefacto y se registra esta limitación;
ningún backend se adoptó a partir de ese contraste.

## Evidencia y límites de la medición

Los artefactos están en
`artifacts/final_refit_20261017/recovery_optimization_20261017/`:

- `CLEAN15_RUNTIME_TELEMETRY.jsonl`: 181 muestras separadas unos 5 s.
- `CLEAN15_RUNTIME_ANALYSIS.json`: estadísticas, throughput e integración de potencia.
- `CLEAN15_RUNTIME_SUMMARY.json`: recibo del monitor; su duración incluye el sleep final.
- `CLEAN15_GPU_THROTTLE.csv`: muestras complementarias del driver.
- `CLEAN15_RUNTIME_TELEMETRY.png`: series de velocidad y recursos.
- `REAL15_PYSPY.json`: pilas completas en formato Speedscope.
- `REAL15_PYSPY_ANALYSIS.json` y `REAL15_PYSPY_INCLUSIVE.json`: desglose de pilas.
- `analyze_pyspy.py`: regeneración del desglose por regiones de la versión inspeccionada.
- `INITIAL_CONFOUNDED_RUNTIME_TELEMETRY.jsonl`: tramo inicial excluido porque
  coincidió con pruebas CPU ajenas al entrenamiento.

El monitor externo dedicó 56,3 s a recolectar, mayormente esperando `nvidia-smi`:
su campo `monitor_overhead_fraction=0.0622` **no significa una ralentización
medida del 6,22 %**, ni ese porcentaje de CPU del entrenamiento. No hubo un A/B
sin instrumentación que cuantifique la perturbación. El perfil no detuvo
intencionalmente el proceso; se realizó además un único dump diagnóstico.

La comprobación posterior verificó 25 hashes de fuentes sin discrepancias;
las curvas 1–700 estaban completas, sin duplicados y con pérdidas finitas.
El entrenamiento continuó produciendo checkpoints después de esa comprobación.

Las recomendaciones sobre sincronización y lanzamiento se apoyan en la
[guía oficial de rendimiento de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Las restricciones de captura se contrastaron con
[CUDA Graphs en PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
El alcance y los límites del muestreo externo se documentan en
[py-spy](https://github.com/benfred/py-spy).
