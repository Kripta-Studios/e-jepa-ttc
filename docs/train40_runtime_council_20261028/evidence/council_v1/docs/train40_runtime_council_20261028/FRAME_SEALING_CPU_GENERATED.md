# Sellado RGB CPU: coste por frame medido

Generado por `artifacts/train40_runtime_council_20261028/frame_sealing_cpu/write_findings.py`.
Fuente: [recibo original](../../artifacts/train40_runtime_council_20261028/frame_sealing_cpu/idx553_v1/summary.json);
[distribuciones y SHA de origen](FRAME_SEALING_CPU_GENERATED.json).

Se conservaron 22 pares de siete frames, incluidos dos warmups.
Todas las rutas pasan bytes, píxeles, SHA del payload, descriptor, IDs y clocks;
172 archivos y el stat RAW permanecen intactos.
La tabla utiliza directamente los **140 `frame_checks` de los 20 pares medidos**,
con órdenes AB/BA alternados. No divide percentiles del historial entre siete.
Cada frame contiene 6.912.000 bytes uint8, shape 1200×1920×3, strides [5760,3,1].

| Tramo por frame, ms | n | Media | Mediana | p95 | p99 | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Copia C-order → bytes | 140 | 0.8363 | 0.8453 | 0.9436 | 1.0899 | 1.3202 |
| SHA-256 del payload | 140 | 3.2145 | 3.1846 | 3.4745 | 4.0612 | 4.1922 |
| Descriptor JSON + su SHA-256 | 140 | 0.0222 | 0.0206 | 0.0395 | 0.0476 | 0.0555 |
| Suma de los tres intervalos instrumentados | 140 | 4.0729 | 4.0407 | 4.5215 | 4.9572 | 5.1099 |
| Intervalo interno informado por freeze original | 140 | 4.0763 | 4.0655 | 4.4277 | 4.6237 | 4.8523 |

El intervalo interno del factory se mide antes de publicar el token en su
registro; no equivale a un reloj externo completo por frame. Los tres intervalos
manuales excluyen validación, registro y publicación del token. El JSON conserva
también las distribuciones por ID de frame y las del historial completo.

## Interpretación y margen potencial

La SHA del payload representa trabajo material: media 3.2145 ms
por frame, frente a 0.8363 ms de copia y 0.0222 ms
de descriptor. En los siete frames completos, las medias fueron
28.6125 ms para el factory original,
22.5014 ms de SHA y
5.8541 ms de copia.

La hipótesis inicial de eliminar el hash del payload mediante identidad local
se ha sustituido por una opción que conserva identidad fuerte de contenido:
[micro SHA-256 frente a BLAKE3-256](FRAME_HASH_CPU_GENERATED.md) y
[diseño V2 propuesto](FROZEN_RGB_V2_DESIGN.md). Los
3.2145 ms medios y p95 3.4745 ms aquí son costes
observados de SHA, **no una mejora V2 medida ni un plazo garantizado**.
Un update sin frames nuevos no paga este SHA;
los bursts y el número real de llegadas deben conservarse en el ensayo completo.
No se puede restar este percentil al p95 de otro ensayo o convertir su inversa
en frecuencia sostenida.

V2 no está implementada. Los tokens V1 y su integridad permanecen intactos.
Una futura V2 debe distinguir algoritmo/namespace y conservar copia inmutable,
procedencia autenticada y validación de los píxeles realmente consumidos, sin
rebajar la paridad TTC ni los snapshots de pesos/normalizadores. El coste de
copia, hash fuerte, metadata, preparación, transferencias y modelos seguirá
dentro del deadline.

## Límites

Es un microbenchmark CPU sobre el mismo historial ya disponible, con lecturas,
validación y escritura de recibos fuera del reloj. No simula llegada física,
colas ni solapamiento GPU. Las 140 muestras comparten 20 pares y siete frames:
no son 140 episodios independientes ni una base para un intervalo estadístico
de producción. No se ha implementado runtime nuevo ni demostrado 50 Hz/20 ms.
