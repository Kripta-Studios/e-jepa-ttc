# V18: conclusión actual y trabajo pendiente

**La campaña sigue abierta; no hay promoción del sistema ni confirmación SOTA.** La fusión presenta una señal controlada en las tres semillas, pero no cumple el gate crucial original del10%. Los negativos y las regresiones se conservan.

Corte UTC: `2026-10-16T18:50:24.570069+00:00`. Fuente: `CAMPAIGN_CONCLUSION.json`.

| Etapa | Evidencia completa | Estado |
|---|---|---|
| primary | True | PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING |
| RGB | True | RGB_READY_FOR_INTERPRETATION |
| C1 | True | C1_READY_FOR_INTERPRETATION |
| fusion_seed7 | True | FUSION_METRICS_READY_FOR_INTERPRETATION |
| fusion_three_seeds | True | REPLICA_METRICS_READY_FOR_INTERPRETATION |
| D1 | True | D1_READY_FOR_INTERPRETATION |
| B1 | True | B1_READY_FOR_INTERPRETATION |
| A1 | False | PENDING |

| Fusión, media de3 seeds | Agregado V | MiDc V | MAE V, s |
|---|---:|---:|---:|
| TRUE | 87.287 | 126.569 | 1.814 |
| ZERO | 92.469 | 136.601 | 1.838 |

El agregado mejora frente al control ZERO de igual capacidad. Esto no demuestra superioridad frente a V16: el intervalo de la diferencia agregada entre sus medias incluye cero. Véase [el informe de réplica](FUSION_REPLICATION_RESULTS.md).

## Contabilidad del corte

- Updates científicos: 165,800.
- Técnicos: 858; recuperación superior: 273; pendientes superiores: 0.
- Físicos superiores: 166,931.
- Scopes GPU contabilizados/reservados: 185408.6 / 216000 s.

Es una instantánea no atómica mientras hay trabajo vivo. No son kernels GPU puros; las reservas activas pueden cambiar al liquidarse. No se recontaron los controles históricos reutilizados.

## Pendiente

- Completar las 64 consultas de coste de fusión con aceptación autorizada de 1 ms; conservar en paralelo los fallos del criterio estricto
- Congelar cualquier export test admisible; el envío oficial sigue sin autorización
- Publicar informes y verificar independientemente el ZIP final <=100000000 bytes
- Completar A1 registrada: dos endpoints, P/H, coste y exports
- Resolver refit/export condicionado por A1 y los gates originales

## Ejecución y comprobaciones recientes

A1 mantiene su receta científica; el lector paralelo, la caché y los grafos persistentes tienen admisión y prueba de continuidad. [Detalle del runtime](A1_PERSISTENT_RUNTIME_20261016.md).

La evaluación A1 está registrada antes de los endpoints: [contrato P/H](A1_EVALUATION_CONTRACT_20261016.md). Su implementación y pruebas no equivalen a resultados del modelo.

La cabeza F_TRUE reproduce las 13 consultas diagnósticas en CPU con el batch histórico; el coste raw completo sigue pendiente de paridad. [Contrastes y límites](FUSION_COST_DIAGNOSTIC.md).

Las ramas condicionadas se justifican con evidencia; un resultado negativo cierra su experimento. La evaluación oficial requiere autorización separada y no se utiliza como bucle de ajuste.
