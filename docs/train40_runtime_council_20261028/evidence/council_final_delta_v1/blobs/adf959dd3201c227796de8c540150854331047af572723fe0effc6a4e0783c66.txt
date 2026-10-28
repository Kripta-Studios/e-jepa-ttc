# Perfil seleccionado para el cierre

`DEFAULT_PROFILE` usa el perfil del mejor replay verificado de 250 actualizaciones
con un objeto, sin caché de crops. Los pesos, normalizadores y lotes nativos se
conservan. El objetivo de 50 Hz sostenidos con todas las actualizaciones por
debajo de 20 ms sigue pendiente; este perfil no constituye una admisión física.

El servicio usa PIL con ROI primero, `prefix`, normalización `topk`, grafos de
consulta, reutilización E desactivada, reutilización R conservada, chunks RGB de
un objeto, ramas paralelas, solapamiento de preparación RGB con E y transporte
exacto `reverse-summaries`. Las primitivas `ProducerRuntime` y los oráculos
conservan su control `original`.

En `benchmark_train40_paced.py` los defaults son además `--raw-source memory`,
`--prime-native-histories` y `--rgb-feature-cache`. Los frames mantienen el
sellado SHA-256 V1 dentro de la adquisición causal; sólo el RAW privado se
precarga. El límite de memoria del feeder sigue siendo 2 GiB por defecto y
rechaza la reserva si no cabe. Esta fuente mide llegadas CPU, con scope
`PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`; excluye I/O de disco del
reloj y declara la precarga en startup.

Para reproducir el mejor ensayo de 250 actualizaciones hay que declarar
`--max-preloaded-source-bytes 3221225472` (3 GiB) y `--verify-parity`, además de
los paths y la secuencia del handoff. Ese ensayo necesita más que los 2 GiB del
default; el runner no amplía el presupuesto ni recorta el RAW automáticamente.
`--verify-parity` sigue siendo una opción explícita y verifica todas las
consultas después de terminar la medición.

Controles explícitos, además de los argumentos obligatorios de cada runner:

| Control | Opción |
| --- | --- |
| Disco / lector persistente / tokens con disco | `--raw-source disk`, `persistent` o `immutable` |
| Transporte original | `--transport-reuse original` |
| Sin solapamiento de preparación | `--no-overlap-event-with-rgb-preparation` |
| Sin caché de features RGB, en paced | `--no-rgb-feature-cache` |
| Primación sintética anterior, en paced | `--no-prime-native-histories` |
| Verificación posterior de todas las consultas | `--verify-parity` |

La primación anterior puede dejar formas frías durante el replay; el control
conserva ese coste visible. El runner multiobject comparte los defaults de
transporte y solapamiento, pero sus cachés y tokens de admisión siguen siendo
opcionales. `--serial` o `--serial-preprocessing` requieren desactivar también el
solapamiento explícitamente. Los consumidores CPU del servicio pueden pasar
`parallel_branches=False, overlap_event_with_rgb_preparation=False`; los
productores inyectados deben declarar el mismo modo de transporte solicitado.

La caché de crops, el hash V2 BLAKE3 y los subpaquetes de 5 ms permanecen fuera del
perfil seleccionado. Sus resultados y propuestas se conservan en el handoff;
no se inicia otro experimento durante este cierre.
