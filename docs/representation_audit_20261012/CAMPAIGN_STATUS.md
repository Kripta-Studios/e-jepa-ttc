# V17: estado y contabilidad

Estado: **NO_CANDIDATE**. No hay afirmación SOTA.

Generado: 2026-10-12T13:36:40.006701+00:00. Los estados pendientes no son resultados negativos ni tareas completadas.

| Cola | Estado |
|---|---|
| A | NO_CANDIDATE |
| B | NO_CANDIDATE |
| D | NO_CANDIDATE |
| E | NO_CANDIDATE |
| F | NO_CANDIDATE |
| G | NO_CANDIDATE |
| H | NO_CANDIDATE |
| I | NO_CANDIDATE |
| C (diagnóstico OOF) | COMPLETE |

Updates científicos: **106500**; técnicos: **180**; recuperación superior: **0**; pendiente superior: **0**. Total físico superior: **106680 / 1.000.000**.

Ajustes cerrados adicionales: **28**. Se contabilizan como entrenamiento aunque no usan optimizer.step. Las réplicas comparten encoders congelados; no son representaciones entrenadas independientemente.

| Scope GPU | Segundos liquidados |
|---|---:|
| evaluation | 114.267 |
| feature_extraction | 3162.007 |
| scientific_training | 7263.054 |
| technical_or_parity | 116.536 |

Total liquidado: **10655.864 s**. Reserva viva/conservadora: **0.000 s**. Cargo superior: **10655.864 / 64.800 s**.

Este coste incluye carga y guardado dentro de los scopes GPU. No mide energía, tiempo puro de kernels ni latencia completa desde eventos. El equipo es compartido y se preservan los propietarios ajenos.

## Entrenamientos

| Fit | Estado | Updates |
|---|---|---:|
| D-S13 | COMPLETE | 2500 |
| D-S23 | COMPLETE | 2500 |
| D-S7 | COMPLETE | 2500 |
| F1-S7 | COMPLETE | 6000 |
| G-S13 | COMPLETE | 2500 |
| G-S23 | COMPLETE | 2500 |
| G-S7 | COMPLETE | 2500 |
| N0-S7 | COMPLETE | 6000 |
| NL1-S7 | COMPLETE | 6000 |
| NLH-S7 | COMPLETE | 6000 |
| Q0-S7 | COMPLETE | 6000 |
| Q1-S7 | COMPLETE | 6000 |
| SP0-S7 | COMPLETE | 6000 |
| SP1-S7 | COMPLETE | 6000 |
| U0-S13 | COMPLETE | 6000 |
| U0-S23 | COMPLETE | 6000 |
| U0-S7 | COMPLETE | 6000 |
| U1-S13 | COMPLETE | 6000 |
| U1-S23 | COMPLETE | 6000 |
| U1-S7 | COMPLETE | 6000 |
| Z-S13 | COMPLETE | 2500 |
| Z-S23 | COMPLETE | 2500 |
| Z-S7 | COMPLETE | 2500 |

Los controles U0 de B se reutilizan en H; Q1 seed7 de E se reutiliza en F. No se cuentan como nuevos entrenamientos ni como evidencia independiente al aparecer en otra tabla.

## Evidencia y límites

- `RESULTS.md`: comparación A y reproducción de métricas.
- `QUEUES_RESULTS.md`: todas las colas adicionales, incluyendo NA por cobertura inválida.
- `MECHANISMS.md`: hipótesis, alcance de los diagnósticos y resultados negativos.
- `PUBLIC_COMPARABILITY.md`: por qué estas cifras no acreditan SOTA oficial.
- `REPRODUCTION.md`: reanudación real, pesos y dependencias.

V es desarrollo interno expuesto. FCWD/Dev32 no han entrenado ni seleccionado estos fits. No se realizan submissions ni publicaciones automáticas.
