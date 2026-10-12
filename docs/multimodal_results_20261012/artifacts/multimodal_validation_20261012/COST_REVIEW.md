# Coste medido: CPU FP32, un hilo

64 identidades idénticas entre pilotos, 32 Dev32 y 32 FCWD; dos pasadas. V15 siguió activo: no es GPU aislada ni SLA de despliegue. Primera pasada no significa disco sin caché. Cada fila resume duraciones observadas, sin sumar medianas.

| Ruta / etapa, segunda pasada | Cohorte | n | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| shared_E_R_F / E_CTX_ROUTE_END_TO_END | DEV32 | 32 | 1099.843 | 2380.466 | 3089.557 |
| shared_E_R_F / REGISTRATION_SEARCH | DEV32 | 32 | 0.000 | 0.000 | 0.000 |
| shared_E_R_F / TOTAL_SHARED_E_R_F_ROUTES | DEV32 | 32 | 1936.193 | 3281.581 | 4051.708 |
| shared_E_R_F / E_CTX_ROUTE_END_TO_END | FCWD | 32 | 799.090 | 2952.733 | 3125.869 |
| shared_E_R_F / REGISTRATION_SEARCH | FCWD | 32 | 177.413 | 249.909 | 259.178 |
| shared_E_R_F / TOTAL_SHARED_E_R_F_ROUTES | FCWD | 32 | 1833.997 | 4141.281 | 4354.132 |
| Garl_full_fallback_event / total_raw_system_ms | DEV32 | 32 | 1484.192 | 1891.024 | 2340.842 |
| Garl_full_fallback_event / resident_network_ms | DEV32 | 32 | 931.482 | 985.548 | 1066.219 |
| Garl_full_fallback_event / total_raw_system_ms | FCWD | 32 | 1445.067 | 2511.434 | 2670.541 |
| Garl_full_fallback_event / resident_network_ms | FCWD | 32 | 954.517 | 1009.839 | 1036.363 |

TOTAL_SHARED_E_R_F ejecuta productores compartidos y cabezas E_CTX, R_CTX y F_TRUE: no debe presentarse como coste aislado F_TRUE. La ruta E_CTX excluye registro RGB. Garl total incluye registro FCWD, lectura, crop, representación nativa y red full/fallback event; resident_network sólo repite la red sobre el mismo tensor. Se usa la preparación nativa vectorizada exacta ya admitida, no un Garl lento artificial.

El estado temporal se reconstruye desde sensores por consulta; no es un servicio persistente con commit, resets y varios objetos. No se midieron H2D/CUDA graphs. Quedan pendientes el perímetro de servicio GPU aislado y su pasada cronológica completa antes de promover una implementación. Los tiempos por etapa y carga se conservan en COST_SUMMARY.json, sin inferir un ganador de velocidad entre perímetros distintos.
