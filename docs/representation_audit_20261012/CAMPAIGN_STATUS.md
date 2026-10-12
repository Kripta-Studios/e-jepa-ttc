# V17: estado y contabilidad

Estado: **RUNNING**. No hay afirmación SOTA.

Generado: 2026-10-12T11:15:37.508574+00:00. Los estados pendientes no son resultados negativos ni tareas completadas.

| Cola | Estado |
|---|---|
| A | NO_CANDIDATE |
| B | NO_CANDIDATE |
| D | NO_CANDIDATE |
| E | NO_CANDIDATE |
| F | NO_CANDIDATE |
| G | PENDING |
| H | PENDING |

Updates científicos: **76500**; técnicos: **100**; recuperación superior: **0**; pendiente superior: **0**. Total físico superior: **76600 / 1.000.000**.

Ajustes cerrados adicionales: **26**. Se contabilizan como entrenamiento aunque no usan optimizer.step. Las réplicas comparten encoders congelados; no son representaciones entrenadas independientemente.

| Scope GPU | Segundos liquidados |
|---|---:|
| evaluation | 79.435 |
| feature_extraction | 847.226 |
| scientific_training | 5213.370 |
| technical_or_parity | 56.158 |

Total liquidado: **6196.190 s**. Reserva viva/conservadora: **120.000 s**. Cargo superior: **6316.190 / 64.800 s**.

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
| Q0-S7 | COMPLETE | 6000 |
| Q1-S7 | COMPLETE | 6000 |
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
