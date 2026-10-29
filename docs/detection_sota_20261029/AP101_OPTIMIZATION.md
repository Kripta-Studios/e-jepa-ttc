# Optimización exacta de AP101

Esta optimización sustituye el barrido completo de la curva para cada uno de los
101 niveles de recall por una envolvente de precisión calculada de derecha a
izquierda y búsquedas sobre el recall no decreciente. No cambia el protocolo, el
matching, los desempates, los 101 niveles ni el orden de la suma final.

## Evidencia interpretable

El perfil temporal atribuible es
`output/evaluation_ap_fast/ap101_exact_v2/receipt.json`. En los perfiles de 1.024
frames, `domain_reason`, que no fue modificado, permaneció estable: 23,647 s en
el control y 23,960 s con el helper. En esa misma comparación, AP101 pasó de
26,014 s a 1,975 s y el evaluator completo de 82,022 s a 57,520 s bajo
`cProfile`. La igualdad profunda y el hash canónico del resultado fueron
idénticos.

`ap101_exact_v3/receipt.json` es la evidencia de paridad y memoria más fuerte:
incluye los cuatro tamaños 16/64/256/1024 y un pico de working set de
342.712.320 bytes frente al gate de 1 GiB. Sus tiempos no se usan para estimar
speedup: entre las dos mitades del perfil cambió también el tiempo de
`domain_reason` de 24,533 s a 4,556 s, señal de disponibilidad CPU distinta.
El speedup wall de v3 queda registrado, pero está contaminado.

## Control congelado y alcance

El profiler de integración extrae `_ap101` por AST de los snapshots congelados
del run v5 y exige antes estos hashes:

- v2: `0264fb73f46ad887cd1048ea2c8cc1e417d52ff7ed369203f6c0d143a61783bc`;
- legacy: `1f53f8fc042d1840d95c9a438d334bbd6ab82e7f56fb9767cefae6b7bcb582c6`.

Así, el control no se convierte en una comparación tautológica después de que
los wrappers mutables importen el helper rápido. Las pruebas unitarias conservan
además copias independientes y pequeñas de ambas fórmulas históricas.

La paridad end-to-end retenida corresponde al evaluator v2. Para legacy, el
payload retenido no serializa su ground truth histórico; su evidencia es la
comparación directa exacta contra la fórmula congelada y la suite funcional del
scorer. Esto no afirma compatibilidad con el scorer privado oficial.

El fallo inicial de instrumentación se conserva en
`ap101_exact_v1/FAILURE.json`. No contiene resultados científicos. Las
evidencias v2 y v3 son directorios posteriores y no sobrescriben ese intento.
