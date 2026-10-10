# V13: cuatro decodificadores y dos batches anticipados

Tabla regenerada por `operational/rgb_port_loader_v3/report.py`.

| Fit | Muestras/repetición | Base (s) | Paralelo + prefetch (s) | Reducción |
|---|---:|---:|---:|---:|
| A5 | 512 | 14.520 | 9.977 | 31.29% |
| C2F | 512 | 17.869 | 11.948 | 33.14% |

30 pruebas aprobadas; igualdad exacta de todos los campos de cada batch (bytes, dtype, shape y metadatos), orden y lecturas.

## Alcance

ABBA sequence, two independent CPU consumers; four decoder threads total; two queued batches per fit; distinct fixed orders; two torch threads per fit.

- No optimizer updates or GPU work in these trials.
- Wall time includes SHA256 work overlapping prefetch.
- OS file cache was not flushed; these are warm input-cache trials.
- Only two logical groups per fit, not the entire epoch.
- The live baseline trial was excluded because C2F paused during it.

## Implementación

`operational/rgb_port_loader_v3/loader.py` conserva el lector NPZ original, las comprobaciones de teacher/tokens y el collate completo. Dos hilos por fit descomprimen shards en paralelo; un coordinador por fit anticipa dos batches. Los hilos comparten arrays, sin serialización entre procesos. No se omiten validaciones ni se cambia la precisión, pérdida, sampler u optimizador.

`producer.py` espera a que termine la restauración y el prewarm CUDA antes de activar prefetch. `contracts.py` vincula código, QA y freeze histórico; `queue.py` conserva propiedad de procesos, límites y reintentos anteriores.

El ensayo CPU se realizó con una reserva de commit de Windows de 3 GiB. La revisión posterior solicitada por el usuario usa 1 GiB; véase MEMORY_RESERVE.md. El ensayo inicial con entrenamientos vivos hizo que C2F se pausara de forma segura en 19.081 updates al detectar 2,606 GiB de margen. A5 se pausó después de forma solicitada en 27.633 para aislar las mediciones. RAM libre y margen de commit son distintos.

La evidencia CPU no demuestra todavía una mejora del entrenamiento GPU. Los recibos `fits/<fit>/loader_v3_checkpoints/` registran esa continuación por separado cuando se ejecuta.
