# V18: conclusión actual y trabajo pendiente

**El programa científico termina sin candidato admitido; falta verificar la entrega final. No hay promoción del sistema ni confirmación SOTA.** La fusión presenta una señal controlada en las tres semillas, pero no cumple el gate crucial original del10%. Los negativos y las regresiones se conservan.

Corte UTC: `2026-10-16T22:32:18.532479+00:00`. Fuente: `CAMPAIGN_CONCLUSION.json`.

| Etapa | Evidencia completa | Estado |
|---|---|---|
| primary | True | PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING |
| RGB | True | RGB_READY_FOR_INTERPRETATION |
| C1 | True | C1_READY_FOR_INTERPRETATION |
| fusion_seed7 | True | FUSION_METRICS_READY_FOR_INTERPRETATION |
| fusion_three_seeds | True | REPLICA_METRICS_READY_FOR_INTERPRETATION |
| D1 | True | D1_READY_FOR_INTERPRETATION |
| B1 | True | B1_READY_FOR_INTERPRETATION |
| A1 | True | A1_READY_FOR_INTERPRETATION |

| Fusión, media de3 seeds | Agregado V | MiDc V | MAE V, s |
|---|---:|---:|---:|
| TRUE | 87.287 | 126.569 | 1.814 |
| ZERO | 92.469 | 136.601 | 1.838 |

El agregado mejora frente al control ZERO de igual capacidad. Esto no demuestra superioridad frente a V16: el intervalo de la diferencia agregada entre sus medias incluye cero. Véase [el informe de réplica](FUSION_REPLICATION_RESULTS.md).

## Contabilidad del corte

- Updates científicos: 183,200.
- Técnicos: 858; recuperación superior: 273; pendientes superiores: 0.
- Físicos superiores: 184,331.
- Scopes GPU contabilizados/reservados: 196739.7 / 216000 s.

Corte después de cerrar los procesos. Las reservas de fallos históricos siguen contabilizadas como cotas superiores. No son kernels GPU puros; las reservas activas pueden cambiar al liquidarse. No se recontaron los controles históricos reutilizados.

## Pendiente

- Publicar los informes finales y verificar el ZIP <=100000000 bytes desde otra extracción, con recibo externo que identifique su SHA-256

## Ejecución y comprobaciones recientes

A1 mantiene su receta científica; el lector paralelo, la caché y los grafos persistentes tienen admisión y prueba de continuidad. [Detalle del runtime](A1_PERSISTENT_RUNTIME_20261016.md).

La evaluación A1 está registrada antes de los endpoints: [contrato P/H](A1_EVALUATION_CONTRACT_20261016.md). Su implementación y pruebas no equivalen a resultados del modelo.

El coste raw de fusión cubre las 64 consultas y cuatro rutas, bajo la aceptación absoluta de 1 ms autorizada después del diagnóstico. Los fallos estrictos permanecen visibles. [Contrastes y límites](FUSION_COST_DIAGNOSTIC.md).

Las ramas condicionadas se justifican con evidencia; un resultado negativo cierra su experimento. La evaluación oficial requiere autorización separada y no se utiliza como bucle de ajuste.
