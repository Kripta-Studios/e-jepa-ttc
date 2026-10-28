# Diagnóstico numérico del backend de subpaquetes de 5 ms

El diagnóstico `subpacket_backend_sensitive_v2` terminó `COMPLETE` el 28 de octubre de 2026. Para un objeto y las consultas reales 0, 137 y 553, dividir cada paquete de 20 ms en cuatro paquetes de 5 ms conservó exactamente el estado, las representaciones y los TTC finales comprobados. Los pesos, normalizadores y las tolerancias permanecieron intactos.

Este resultado autoriza conservar el backend como candidato numéricamente validado dentro de esta población. No admite una actualización completa a 50 Hz, no mide latencia y no demuestra un ahorro del 75 %. Tampoco cubre dos objetos, las 554 consultas ni una fuente paced que entregue subpaquetes a 200 Hz.

## Evidencia y procedencia

Los resultados proceden de [summary.json de v2](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/summary.json), su [binding.json](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/binding.json) y las copias ejecutadas en `sources/`. El controlador es [benchmark_train40_subpacket_backend.py](../../scripts/benchmark_train40_subpacket_backend.py). El contrato y las guardas del backend se describen en [SUBPACKET_BACKEND.md](SUBPACKET_BACKEND.md).

| Campo | Valor registrado en v2 |
|---|---|
| Commit de partida | `bfa67f5ad0843f79665f563f146c0f5b5ad01520` |
| Configuración SHA256 | `a4eeafb6f6790b0bdb2ed642ea6060c89fc3d09d434be1c4f99f7894f8a6d532` |
| Binding SHA256 | `a9fa81911ecd677fd907f86e73199b9d204fd360c131f9042676d5c13a2b7e0a` |
| Summary SHA256 | `06ab49e521db725526c3c839d6c7543f47e4e5c4ed75c03b19f5b8faeb29453c` |
| Python / PyTorch / CUDA | 3.11.15 / 2.11.0+cu128 / 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Inicio / fin UTC | `2026-10-28T11:38:34.606677+00:00` / `2026-10-28T11:38:53.257910+00:00` |
| Tolerancia TTC | `atol=1e-5`, `rtol=1e-5` |

El commit por sí solo no sustituye al inventario de fuentes ejecutadas. Los dos ensayos tienen las mismas 49 entradas y hashes en `binding.sources`. Entre v1 y v2 sólo cambiaron `max_raw_bytes` y el directorio de salida. El SHA256 del controlador ejecutado es `00751e079e907b1f8f59947cb9fd5225bb992cb172763f303c80d0831bcbf1ee`; el de `temporal_prefix_subpacket.py` es `d0c6d7dc748b4f9a56eafb9a0f36550a06655ac0bada8608880f948315a935da`.

## Población y protocolo

| Índice | Consulta | Reset, µs | Anchor final, µs | RAW retenido, bytes |
|---|---|---:|---:|---:|
| 0 | `EXTENDED:CCRm-low-0-overlap-0:14350374` | 13 150 374 | 14 350 374 | 324 429 972 |
| 137 | `EXTENDED:CCRs-1-high-50-overlap-50:15002363` | 13 802 363 | 15 002 363 | 160 788 295 |
| 553 | `EXTENDED:CPNAO-medium:20200738` | 19 000 738 | 20 200 738 | 527 817 513 |

Cada consulta usa el mismo reset y anchor final para los dos backends, con 1,2 segundos de RAW real y los mismos intervalos semiabiertos:

1. Ambos historiales inicializan los primeros 600 ms mediante los mismos 30 paquetes de 20 ms. La primera comparación corresponde a 120 buckets de 5 ms.
2. En los 600 ms restantes, el padre ingiere un paquete de 20 ms y el candidato ingiere sus cuatro subpaquetes de 5 ms. La inicialización común no se presenta como una prueba de 240 ingestas nuevas de 5 ms.
3. Se comparan las 31 fronteras completas `120,124,128,…,240`, incluidos los checkpoints explícitos 120, 124 y 240 y todas las posiciones intermedias del ring buffer.
4. Las consultas intermedias usan una ROI diagnóstica fija de sensor completo. Sólo la frontera final usa la ROI real de la consulta y genera TTC; no se aplica esa ROI futura para producir TTC anteriores.
5. Los grafos de consulta se priman antes del reset y antes del RAW real. No se permiten capturas tardías durante las comparaciones.
6. Se liberan ambos historiales antes de cargar los modelos congelados. Se conservan únicamente los tensores finales en CPU para comprobar la entrada canónica y el TTC.
7. El oracle conserva su batching original: E procesa cuatro observaciones juntas y RGB procesa una observación cada vez. Se comprueban las entradas canónicas y las predicciones históricas originales sin cambiar la ruta por opciones de transporte o reutilización de encoder.

Las comparaciones de estado son `torch.equal` directo entre los dos historiales residentes. No se sustituyen por hashes, ni se clonan los buffers completos. Los hashes de las representaciones y de las cuatro columnas RAW quedan como evidencia adicional en cada registro.

## Resultados comprobados

| Comprobación | Resultado de v2 |
|---|---:|
| Cobertura de fronteras completas | 93/93: 31 por consulta |
| Buffers persistentes iguales bit a bit | 558/558: seis por frontera |
| Campos escalares de estado iguales | 465/465: cinco por frontera |
| Representaciones iguales bit a bit y SHA256 coincidente | 93/93 |
| Guardas en fases incompletas | 18/18 recibos; 36 rechazos de consulta |
| Entradas finales iguales al tensor canónico original | 6/6, error absoluto máximo 0 |
| Bundles TTC finales frente al original | 6/6, todos exactos |
| Valores TTC individuales | 18/18, error absoluto 0 |
| Archivos del inventario preservados | 626/626, ninguno cambiado ni ausente |
| Estadísticas de archivos RAW preservadas | 3/3 |

Los seis buffers comparados completos son `_counts`, `_moments`, `_cumulative_count`, `_cumulative_moment`, `_previous_count` y `_previous_moment`. Los cinco campos son el final temporal, el total de buckets, el inicio de la época, los buckets llenos y el contador de paquetes completos ingeridos.

Las guardas se comprueban en los buckets 121, 122, 123, 237, 238 y 239 de cada consulta. En cada caso `query` y `query_many` rechazan la fase parcial; `ready` permanece falso, la fase es la esperada, el inicio de época no avanza y el contador de consultas no cambia.

Cada uno de los seis historiales captura un grafo y realiza tres consultas sintéticas de primación: seis capturas y 18 llamadas de primación en total. El número de grafos permanece constante durante el RAW real. Al terminar, ambos backends tienen 240 buckets, 60 paquetes completos ingeridos, 31 consultas y `ready=true`. El candidato registra 30 ingestas bulk de 20 ms y 120 ingestas de 5 ms, agrupadas en 30 grupos completos, sin eventos pendientes ni invalidación.

Las seis entradas canónicas finales tienen shape `[4,3,12,128,128]`. Los seis bundles TTC son las tres consultas por los dos backends; cada bundle contiene `PHASE_ONLY`, `E_CTX` y `R_CTX`. Los 18 valores son exactos contra el original, además de pasar el gate sin modificar `atol=rtol=1e-5`. También son exactos entre ambos backends.

El recibo declara `weights_unchanged=true`, `normalizers_unchanged=true`, `targets_read=false` y `optimizer_updates=0`. Para RAW, la preservación comprueba tamaño y `mtime_ns`; no implica volver a calcular el hash de los HDF5 completos. Las matrices RAW usadas sí tienen hashes por columna registrados.

## Fallo previo y presupuesto de memoria

[v1](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v1/summary.json) terminó `FAILED` entre `2026-10-28T11:34:29.631939+00:00` y `2026-10-28T11:34:30.959372+00:00` con:

```text
ValueError: RAW archive exceeds explicit 268435456 byte budget
```

La consulta 0 necesita 324 429 972 bytes de archivo RAW retenido y excede el cap de 256 MiB. V1 tiene cero registros numéricos: falló antes de comparar los historiales o ejecutar los modelos. Conservó los 626 archivos inventariados y las estadísticas RAW. Su configuración SHA256 es `b717b5b8fc207e5acd227031cc740e7a1c93b45b0ce020a883a9fe799635dfed`; el fallo queda preservado y no se interpreta como fallo de paridad.

V2 elevó explícitamente ese cap a 512 MiB (536 870 912 bytes). El mayor archivo retenido, consulta 553, ocupa 527 817 513 bytes. El límite conservador archivo más concatenación es 1 073 741 824 bytes según configuración; para el mayor archivo real, el doble es 1 055 635 026 bytes. Son cotas de esas asignaciones, no mediciones de RSS total. El lector mantiene además un presupuesto separado de scratch de 268 435 456 bytes.

| Memoria GPU | Bytes |
|---|---:|
| Buffers declarados por historial | 2 976 513 264 |
| Dos historiales simultáneos | 5 953 026 528 |
| Margen exigido en preflight | 1 073 741 824 |
| Memoria libre mínima requerida | 7 026 768 352 |
| Libre observado, índice 0 | 11 561 598 976 |
| Libre observado, índices 137 y 553 | 11 515 461 632 |
| Pico PyTorch allocated, cada caso | 6 187 959 296 |
| Pico PyTorch reserved, cada caso | 6 287 261 696 |

Los preflight individuales están en [índice 0](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/memory_preflight_0.json), [137](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/memory_preflight_137.json) y [553](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/memory_preflight_553.json). Cada caso informa cero bytes PyTorch allocated después de liberar los historiales; también se registra cero antes de cargar modelos. Estas cifras corresponden al asignador de este proceso, no a toda la memoria del contexto CUDA, escritorio u otros procesos.

Los seis tensores finales retenidos en CPU ocupan 56 623 104 bytes (54 MiB), bajo el límite explícito de 64 MiB. No se guardan volcados de estado de varios GB ni archivos `.pt` con RAW.

## Verificación y reproducción

Las [36 pruebas CPU del controlador](../../tests/unit/test_train40_subpacket_backend_benchmark.py) pasaron en 5,15 s; Ruff y Pyright terminaron sin errores. Los recibos están en [child_tests.log](../../artifacts/train40_runtime_council_20261028/qa/subpacket_backend_v1/child_tests.log) y [script_static_qa.json](../../artifacts/train40_runtime_council_20261028/qa/subpacket_backend_v1/script_static_qa.json). El log conserva un intento inicial con 35 aprobadas y una expectativa de fixture incorrecta sobre el intervalo `[110,115)`; se corrigió esa expectativa para incluir el evento en 110, se añadió el caso vacío `[111,115)` y la repetición terminó con 36 aprobadas. No fue un fallo del backend científico. Las pruebas propias del backend se documentan por separado en [SUBPACKET_BACKEND.md](SUBPACKET_BACKEND.md).

Comando de v2, conservado como reproducción del ensayo realizado; una repetición debe usar un directorio de salida nuevo:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_subpacket_backend.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --reference artifacts/final_train40_transfer_20261027/final_train40/own `
  --output artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2 `
  --indices 0 137 553 `
  --max-raw-bytes 536870912 `
  --gpu-headroom-bytes 1073741824
```

V1 usó el mismo controlador y fuentes con output `subpacket_backend_sensitive_v1` y `max_raw_bytes=268435456`. Durante este cierre sólo se leyeron los recibos y se redactó documentación; no se repitieron pruebas GPU ni se cambiaron fuentes runtime.

La evidencia respalda la equivalencia en los tres casos ejecutados y sus 93 fronteras. El siguiente trabajo, cuando se reanude la investigación, sigue requiriendo integración causal del feeder, cobertura de todas las llamadas y medición completa desde la última observación necesaria hasta todos los TTC, sin acumulación de consultas. La prioridad continúa siendo un objeto antes de dos y, después, hasta 20. El objetivo de 50 Hz sostenidos y menos de 20 ms permanece pendiente.
