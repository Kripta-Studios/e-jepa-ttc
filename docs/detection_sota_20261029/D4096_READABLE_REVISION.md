# Revisión D4096 v5 legible

La revisión `output/campaign_data/d4096_v5_readable` sustituye exclusivamente
el PNG nativo malformado descubierto durante R0. No altera el manifest v4, el
run negativo, el audit de población ni la selección histórica.

| Campo | Resultado |
|---|---|
| Índice train | 2.063 |
| Token excluido | `OYgB6RGWcq:000000661_000000160` |
| Token admitido | `OYgB6RGWcq:000000660_000900159` |
| Distancia temporal | 100.001 µs, anterior |
| Regla | misma secuencia, no usado, no overflow, distancia mínima; empate anterior |
| SHA del miembro nuevo | `27bd317ae7cfcb8dab3363fb19c229b44d9904c55cc3c960cdf838aaf16e7500` |
| Bytes del miembro nuevo | 688.998 |
| Resolución | 1280×720, Pillow estricto |

La primera candidatura ordenada pasó. Su TAR conserva tamaño y SHA fijados:
`d1f9956ad10e2cbd436625edc734b90de7de0bd7368204bd7e2273208f3d3aa5`.

La población final contiene 4.096 train y mantiene exactamente los 1.024 tokens
de selección. El artefacto de selección completa conserva sus 23.958 tokens. El
ledger multimedia contiene la unión exacta de 28.054 tokens, con owner, TAR,
member, tamaño, SHA y resolución esperados para validar el primer read del
trainer.

La cobertura rara original se conserva: 69 identidades `(secuencia, track_id)`
de bicicleta y seis de `other_vehicle`. El reemplazo aporta tres anotaciones
trainables adicionales de coche; el número de frames raw negativos baja de 55
a 54. Ninguna clase pierde identidades trainables.

## Hashes

| Archivo | SHA-256 |
|---|---|
| `d4096_manifest.json` | `9beeb6a3c5b76bf994808eee0b3f0dfa2f7339f139bd120dbfa6318e414945f1` |
| `media_admission.json` | `c625eec068727b5df15f527483b9bf343f6b566fc19eba20449066de2390eae8` |
| `d4096_coverage.json` | `9084776b9a8f1412630fbae8a414f4f749185aef16b888da840511741503d42c` |
| `population_audit.json` | `b8d5d5fec171e2af73a360fd13e8466757baf801b15c571728e31cb107d42673` |
| `receipt.json` | `4ba97b0b0ec19936c2cdfe9fa718e5f79e30bb3348d67df0026b4267d82f309a` |

`population_audit.json` es una copia byte a byte del audit v4. El manifest v5
usa el mismo schema v1 y añade un bloque `media_admission` relocatable que apunta
a `media_admission.json`; el recibo multimedia liga el manifest anterior y la
nueva población de tokens, sin autorreferencia circular al futuro manifest.

La campaña R0 debe reiniciarse en frío con v5. No se migra el checkpoint 512 ni
se reinterpreta el run fallido bajo una población distinta.
