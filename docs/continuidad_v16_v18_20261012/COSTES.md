# Coste y paridad de la continuación

La nueva caché de lectura RGB y mapas CNN pasa **128 comparaciones exactas**
en 64 consultas: 32 FCWD y 32 Dev32, dos pasadas por consulta, **256 rutas
completas medidas**. Se comparan entradas RGB, features, expertos, bloques
contextuales, fase, cuantiles y TTC. Las consultas son las identidades del perfil
publicado, ordenadas por secuencia y tiempo; incluyen ROI variable y fallback.

No se ha modificado el candidato ni instalado la optimización en V16. Es un
adaptador de inferencia **CPU FP32, un hilo**. No existe medición GPU de este
adaptador, y no se puede trasladar su paridad a otro lote, precisión o dispositivo.

La primera propuesta cacheaba toda la salida del encoder y cambió el TTC en
2,384185791015625e-7 s. Se rechazó con tolerancia cero. Su código y diferencias
permanecen en cost/pilot. La segunda sólo reutiliza mapas CNN y reconstruye el
lote original antes de las cabezas token/foreground. El piloto de seis consultas
y la población de 64 pasan; se conservan las fuentes correspondientes a ambos.

## Medición completa

| Sistema | Cohorte | Pasada | n | RGB disponible | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| PHASE_native | DEV32 | 0 | 32 | 32 | 5429.9 | 11517.9 | 14463.9 |
| PHASE_native | DEV32 | 1 | 32 | 32 | 5191.0 | 9964.9 | 12087.5 |
| PHASE_optimized | DEV32 | 0 | 32 | 32 | 4887.2 | 10782.1 | 13892.4 |
| PHASE_optimized | DEV32 | 1 | 32 | 32 | 4453.2 | 8357.9 | 10796.4 |
| PHASE_native | FCWD | 0 | 32 | 24 | 2287.0 | 5814.7 | 6505.1 |
| PHASE_native | FCWD | 1 | 32 | 24 | 2000.0 | 5598.7 | 6506.0 |
| PHASE_optimized | FCWD | 0 | 32 | 24 | 2104.8 | 5666.8 | 8148.1 |
| PHASE_optimized | FCWD | 1 | 32 | 24 | 1874.6 | 5092.9 | 6244.6 |
| Garl_native_existing_optimized_inputs | DEV32 | 0 | 32 | 32 | 4991.3 | 6078.6 | 7972.9 |
| Garl_native_existing_optimized_inputs | DEV32 | 1 | 32 | 32 | 5248.8 | 7787.4 | 9320.6 |
| Garl_native_existing_optimized_inputs | FCWD | 0 | 32 | 24 | 5335.0 | 9798.3 | 10630.3 |
| Garl_native_existing_optimized_inputs | FCWD | 1 | 32 | 24 | 4551.6 | 8463.4 | 9131.1 |

Las cifras provienen del reloj de la ruta completa; no se suman medianas de
etapas. Pesos residentes: incluye lectura/preparación de sensores, registro
FCWD nuevo, productores event/RGB, contexto y cabeza. Excluye carga inicial,
detector/tracker, comprobaciones de evidencia y escritura de CSV. No representa
un detector en vídeo ni el sistema de percepción entero.

Garl se mide de nuevo sólo para coste, sobre las mismas 64 identidades, con red
nativa full y fallback event y el preprocesado exacto ya optimizado previamente.
Sus 128 forwards residentes de control coinciden bit a bit. La comparación de
precisión multimodal de 630 consultas no se repite. Las historias temporales de
ambos sistemas siguen siendo diferentes. El perfil Garl se ejecuta después del
perfil PHASE: la carga del equipo no está controlada entre ambos.

## Efecto pareado

- DEV32, pasada 0: cociente mediano pareado 0.991; optimizada más rápida en 17/32; diferencia media -207.9 ms.
- DEV32, pasada 1: cociente mediano pareado 0.896; optimizada más rápida en 27/32; diferencia media -503.1 ms.
- FCWD, pasada 0: cociente mediano pareado 1.018; optimizada más rápida en 15/32; diferencia media +103.9 ms.
- FCWD, pasada 1: cociente mediano pareado 0.929; optimizada más rápida en 24/32; diferencia media -278.7 ms.

V18 y el escritorio continuaron activos. Las diferencias son descriptivas;
un tramo sin RGB puede parecer más rápido o lento por ruido aun ejecutando la
misma ruta. Esta medición no acredita un ahorro universal ni autoriza promoción
del sistema. Hay que repetir la aceptación en el dispositivo y régimen objetivo
antes de desplegar, sin sustituir el candidato congelado.

Pasada 0: LRU de frames vacía al inicio de cada consulta. Pasada 1: LRU de frames
caliente para esa consulta. La caché CNN sólo vive dentro de una ruta; nunca
arrastra estados temporales ni mapas entre consultas. No se vacía la caché del
sistema operativo; **no es una prueba de disco frío**. Se alterna el orden
native/optimized por consulta y pasada.

Máximo conservador de memoria explícita de cachés: 52,054,016 bytes.
Las observaciones de RSS y máximo de working set del proceso, incluida carga de
pesos, están en cost/*/MEMORY_OBSERVATION.json cuando se pudieron observar.
El muestreo puede perder un transitorio final; no se presenta como pico exacto
del allocator. No se suman picos de procesos diferentes.

Identidad de mapas: archivo/frame, ROI fijo de consulta, transformación,
checkpoint, código, CPU/FP32/eval y hash de los bytes del tensor normalizado.
La ROI se mantiene durante la historia. La caché cruda actúa antes del crop y
se identifica por archivo, dataset e índice; devuelve copias para impedir
mutación de datos cacheados. El coste del hash está incluido.

La reutilización event anterior permanece terminada. Los 64 tiempos GPU previos
(p50 1.213 ms, p95 3.061 ms) y los 0,523 ms de cabeza CPU pertenecen al perfil
histórico con su propio perímetro. No son resultados de esta caché RGB.
