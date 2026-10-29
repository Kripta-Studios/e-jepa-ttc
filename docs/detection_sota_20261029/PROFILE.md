# Perfil del detector portable con velocidad

Fecha de ejecución: 2026-10-29. El recibo completo está en
`output/s1_profile/pilot_velocity_batch4.json` y tiene SHA-256
`f9a70e1d62c4b41460c5c9be95ca5bac706092a29ada98f5ee7273e3abbc7581`.

El perfil restauró estrictamente el checkpoint del piloto, su optimizer, la
configuración con cabeza de velocidad y las fuentes congeladas. Ejecutó dos
updates de calentamiento y cuatro medidos, con batch cuatro, FP32 y resolución
640×384. Todas las pérdidas fueron finitas y positivas (11,4971–13,2340), y el
backward y el update terminaron correctamente.

| Medida | Resultado |
|---|---:|
| Preparación única del batch, CPU | 0,8569 s |
| Transferencia, mediana | 4,68 ms |
| Forward, mediana | 153,19 ms |
| Criterion, mediana | 81,13 ms |
| Backward, mediana | 183,03 ms |
| Clip de gradiente, mediana | 7,91 ms |
| Update del optimizer, mediana | 12,25 ms |
| Total de fases instrumentadas, mediana | 454,49 ms/update |
| Pico Torch asignado | 4.530.583.552 bytes |
| Pico Torch reservado | 4.961.861.632 bytes |
| Pico de working set del proceso | 3.237.814.272 bytes |
| Memoria total reportada por el dispositivo | 12.820.480.000 bytes |

El total instrumentado incluye comprobaciones de finitud y sincronización CUDA
en cada frontera. No representa latencia de inferencia ni throughput de un
dataloader solapado.

## Población y alcance

Los seis updates reutilizaron el mismo batch preparado de cuatro frames de
`qGsgzl4Q8B`. Esto mide el coste sostenido del modelo y de un optimizer step con
targets reales. No mide lectura aleatoria, decodificación o diversidad a través
de las 24 secuencias de train, ni el throughput de D4096. La preparación CPU se
midió una vez y permanece separada del tiempo por update.

La memoria máxima observada admite el control histórico batch cuatro en esta
GPU. No acredita todavía mayor resolución, historia RGB, eventos o ejecución
simultánea con PHASE.

## Procedencia

| Artefacto | SHA-256 / identidad |
|---|---|
| Checkpoint | `dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338` |
| Manifest del piloto | `048636cd79e8f5160b97a8056b0e2ed6f78846f4bbb365b77efa5884766be159` |
| Split candidato | `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903` |
| Metadata `train.parquet` | `69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314` |
| Profiler ejecutado | `f61b17131a09141d014a96dfd40281f7781f8adcf52b238b556d7dc2b4045542` |
| Variante | `monodgp_pytorch_grid_sample_velocity_v1` |
| Runtime | Torch 2.11.0+cu128, CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |

Después de esta ejecución se sustituyó la suma Python de pérdidas escalares por
`torch.stack(terms).sum()` para satisfacer el type checker. Por ello, el hash de
profiler del recibo identifica los bytes realmente ejecutados; cualquier cifra
generada con la fuente corregida requiere un recibo nuevo y no sobrescribe este.
El parche inverso mínimo para reconstruir y verificar esos bytes se conserva en
`evidence/pilot_velocity_batch4_profiler_executed.patch`.
