# RGB exacto GPU: un caso, un objeto

Cuerpo de resultados generado por `artifacts/train40_runtime_council_20261028/rgb_exact_gpu/write_findings.py`;
la adenda de cierre siguiente documenta el estado posterior y no modifica esas cifras.
Fuente: [recibo completo idx553_v2](../../artifacts/train40_runtime_council_20261028/rgb_exact_gpu/idx553_v2/summary.json).

## Adenda de cierre

Código e investigación congelados por solicitud del usuario. La ruta GPU exacta
pasó paridad de píxeles/deltas, pero **no demostró mejora frente al servicio PIL
ni completó una admisión TTC end-to-end**. PIL y sellado V1 SHA siguen por defecto.
La propuesta de integrar el ring/recorte GPU al servicio sigue pendiente.

El [core V2 BLAKE3](FROZEN_RGB_V2_DESIGN.md) quedó implementado en `9e752ae4`,
con 101 pruebas CPU y rerun de 20 tests V2 aprobados. Aún no tiene wiring a las
fuentes/runner ni paridad/latencia TTC. La ventaja de copia+hash del
[micro CPU](FRAME_HASH_CPU_GENERATED.md) no demuestra mejora end-to-end.

Por separado, el [crop companion PIL V1](../../artifacts/train40_runtime_council_20261028/multiobject/crop_rgb_companion_v1/summary.json)
pasó 18/18 actualizaciones con paridad. Su posterior
[ensayo paced con caché de crops](../../artifacts/train40_runtime_council_20261028/paced/memory_crops_reverse1_250_v1/summary.json)
falló: dos drops bloquearon la historia y sólo se produjeron 115/250 salidas,
todas con paridad; 724 archivos intactos. No alcanzó el objetivo y no identifica
la causa de los picos. Una traza GC opt-in y un A/B controlado SHA quedan como
diagnóstico pendiente, sin atribuir por adelantado el fallo a GC o a la caché.

Los resultados aislados siguientes no autorizan avanzar a dos objetos ni
afirmar 50 Hz/<20 ms. Ningún peso o normalizador se cambió en esta investigación.

## Resultados del recorte GPU aislado

Las 7 preparaciones del índice 553 pasan píxeles,
normalización FP32 y deltas exactos. Se conservaron 175
archivos, con 0 recompilaciones durante las muestras
templadas. El primer prepare real tardó 6631.424 ms e incluyó compilación.
Hubo primación sintética explícita del perfil declarado; no cubre ROI arbitrarias.

El reloj va de RAW disponible en CPU a representaciones RGB listas en GPU.
Incluye sellado nuevo, copia pinned, H2D, coeficientes, recorte y sincronización.
Excluye disco, reconstrucción del prefijo, modelos, TTC y paridad posterior.
Son tres repeticiones por modo, no un stream ni una estimación fiable de colas.
El MISS sella siete frames; el prefijo sella el frame nuevo 404 y conserva seis.

| Tramo, ms | MISS completo | Prefijo causal reconstruido |
|---|---:|---:|
| Reloj completo del microbenchmark | 35.600–39.863 | 8.686–10.566 |
| Sellado CPU incluido | 29.357–31.984 | 4.184–4.435 |
| Ring: despacho host | 2.447–3.755 | 0.399–0.543 |
| Coeficientes + metadata host, diferencia de marcas | 1.554–2.551 | 1.716–2.512 |
| Despacho de kernels y tablas host | 1.493–1.962 | 2.118–2.608 |
| Cola final de sincronización + wrapper | 0.131–0.149 | 0.104–0.504 |

Las diferencias se calculan dentro de cada actualización. Los tramos host no
son tiempos GPU exclusivos y los extremos de sus rangos no deben sumarse.
El sellado domina el MISS completo. En el prefijo, el sellado sigue siendo
material y el resto está repartido entre preparación CPU y despacho.

## Transferencias y coeficientes

`ResidentRGBFrameRing` ya usa memoria pinned y copias `non_blocking=True`.
Las esperas al uso anterior son del orden de microsegundos en este recibo.
No hay evidencia de que habilitar otra opción async acelere esta ruta.
Un stream de copia separado necesitaría nuevos fences/leases y admisión; este
microbenchmark no mide solapamiento con productores E/R.

Una caché acotada de tablas exactas por `(input_size, output_size=128)` puede
reutilizar coeficientes cuando cambian offsets pero se conserva el tamaño
efectivo del crop. Cada MISS debe calcularse dentro del reloj; offsets, mapping,
metadata y deltas siguen siendo actuales. También sería causal precomputar
todas las longitudes de la resolución conocida durante startup, registrando
tiempo y memoria, pues no utiliza pixels ni ROI futuros. Ninguna opción elimina
el sellado, el despacho ni los MISS de compilación. La fila coeficientes+metadata
es una cota del tramo potencialmente afectado, no una mejora medida.

## Tokens ya sellados: integración propuesta, sin implementar

El driver standalone comienza con arrays RAW y llama a `seal_frames` dentro de
cada reloj. `ring.bind` no rehashea: valida tokens y copia payload nuevo a pinned.
En cambio, `PreloadedRawSource._release_frames` y el reader immutable ya sellan
cada frame causalmente en `acquire_raw`; su payload contiene `FrozenRGBFrame`
auténticos. Esos tokens son directamente compatibles con el ring GPU.

El adapter debe construir `rgb_pil_exact.FrozenRGBHistory` desde
`payload.rgb.tokens`, conservando query/triplets/reason; no debe volver a llamar
a `freeze_rgb_frame`. Los dos tipos `FrozenRGBHistory` pertenecen a módulos
distintos. Hay que pasar `PreparedRGBBatch(metadata, pixels, delta)` al servicio,
retener el lease durante productores/contexto y liberarlo tras el join, también
en excepciones. La caché de features puede evitar crop en HIT y el ring conserva
RAW entre MISS. El sellado único sigue dentro del deadline del feeder: desplazar
el tramo a adquisición no autoriza restarlo del tiempo completo.

## Corrección del límite de chunks y prioridad

El bucket observado tras clipping es `[1200, 16, 21]`. El bound temporal conservador
de chunk1 es 88,473,600 bytes; chunk2 sería 176,947,200 bytes,
por debajo de 256 MiB para este caso. La estimación previa de 342.835.200 bytes
para chunk2 correspondía al peor bucket horizontal 31, no al crop efectivo
medido. El driver conserva chunk1; chunk2 no se ha medido en GPU.

No se ha demostrado ventaja frente al servicio PIL en un contraste completo
pareado. La reutilización exacta de tripletas RGB, el transporte reverso y la
caché de coeficientes son líneas investigadas o propuestas, sin trabajo activo
tras este cierre. Antes de dos objetos, cualquier integración GPU debe pasar
TTC y reloj completo con uno; este recibo no admite 20 ms ni 50 Hz.
