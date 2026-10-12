# V17 — representación, geometría y optimización TTC

Campaña independiente posterior al `NO_CANDIDATE` de V15. Presupuesto autorizado: 18 h GPU y 1 M de updates físicos nuevos. Busca mejorar TTC corto conservando las salvaguardas originales. Un resultado interno no acredita SOTA oficial.

## Informes

1. [Estado, decisiones y contabilidad](CAMPAIGN_STATUS.md): generado desde artefactos; distingue colas pendientes y terminadas.
2. [Resultados de representación A](RESULTS.md): seis cabezas, semillas y contraste H/V.
3. [Resultados de las demás colas](QUEUES_RESULTS.md): tablas completas, incluyendo negativos y valores ausentes.
4. [Mecanismos investigados](MECHANISMS.md): hipótesis, evidencia y límites.
5. [Comparabilidad pública](PUBLIC_COMPARABILITY.md): poblaciones, modalidades y métricas de las referencias.
6. [Reproducción](REPRODUCTION.md) y [portabilidad](PORTABILITY.md): comandos, dependencias, pesos y ejemplos.
7. [Piloto de foreground](FOREGROUND_PILOT.md): sensibilidad local de la ruta nativa, 40 consultas P y cero updates.
8. [Dependencia de recursos](RESOURCE_STATE.md): pausa observada, resolución verificada y comando real de reanudación.
9. [Verificación](VERIFICATION.md): pruebas, continuidad, recarga, reproducción de métricas y límites del bundle.

## Registro prospectivo

| Cola | Pregunta | Protocolo |
|---|---|---|
| A | ¿Ayudan los tokens anteriores a PHASE17? | [A](PROTOCOL.md) |
| B | ¿Influye la ponderación por incertidumbre? | [B](PROTOCOL_B_PRECISION.md) |
| C | ¿Se conserva información al dejar fuera grupos completos? | [C](PROTOCOL_C_DIAGNOSTIC.md) |
| D | ¿Ayudan las diferencias temporales de los tokens? | [D](PROTOCOL_D_DIRECTIONAL.md) |
| E | ¿Se puede calibrar la altura desde tokens congelados? | [E](PROTOCOL_E_GEOMETRY.md) |
| F | ¿Supervisar cambios de altura reduce el ruido temporal? | [F](PROTOCOL_F_TEMPORAL_GEOMETRY.md) |
| G | ¿El mapa espacial conserva información perdida por el pooling global? | [G](PROTOCOL_G_SPATIAL_GEOMETRY.md) |
| H | ¿El clipping permanente limita el scorer? | [H](PROTOCOL_H_UNCLIPPED_READOUT.md) |
| I | Si G/H fallan, ¿ayuda una lectura no lineal de las mismas features espaciales? | [I](PROTOCOL_I_NONLINEAR_HEIGHT.md) |

La [política de réplica](REPLICATION_POLICY.md) admite como máximo una familia condicional entre las colas pendientes. Los contrastes A/B/D de tres semillas se fijaron antes de su evaluación.

## Evidencia

Rutas relativas a `artifacts/representation_audit_20261012`:

| Evidencia | Ruta |
|---|---|
| Decisión global | `CAMPAIGN_NEXT_DECISION.json` |
| Decisión A original | `NEXT_DECISION.json` |
| Ledger | `FIT_LEDGER.csv`, `ACCOUNTING.json` |
| Costes y reservas | `COST_SUMMARY.json`, `GPU_BUDGET.json` |
| Pesos y continuidad | `fits/<fit>/` |
| Pesos portables y ejemplos | `exports/<fit>/` |
| Predicciones de cabezas | `evaluation/` |
| Predicciones de productores | `precision_evaluation/`, `geometry_evaluation/`, `temporal_geometry_evaluation/`, `spatial_evaluation/`, `unclipped_evaluation/`, `nonlinear_evaluation/` |
| Métricas por grupo | `GROUP_METRICS.json` junto a cada tabla |
| Bootstrap | `BOOTSTRAP.json`, `QUEUES_BOOTSTRAP.csv` |
| Clipping | `CLIPPING.csv`, `QUEUES_CLIPPING.csv`, `ALL_FIT_CLIPPING.csv` |
| Gradientes | `head_gradient/`, `directional_gradient/`, `readout_gradient_*/`, `geometry_gradient_I/` |
| Curvas | `fits/<fit>/CURVE.csv`, `figures/TRAINING_CURVES.png` |
| Auditoría | `ENDPOINT_AUDIT.json` |
| Reproducción de métricas | `BUNDLE_METRIC_REPLAY.json` |
| Recarga portable | `PORTABLE_EXPORT_VERIFICATION.json` |
| Continuidad sintética | `qa/*PASS.json` y checkpoints asociados |
| Continuidad de muestras y relojes | `SAMPLING_CONTINUITY_AUDIT.json`, `sampling_audit/` |
| Reanudación terminal sin nuevos updates | `TERMINAL_RESUME_PROOF.json` |
| Perfil CPU sobre entradas preparadas | `PORTABLE_CPU_FORWARD_PROFILE.json` |

Un archivo puede estar ausente mientras su cola siga pendiente. El estado generado es la referencia; este índice no afirma que todas las ramas hayan sido admitidas.

Las pérdidas de productores incluyen términos geométricos congelados cuyo valor constante puede cambiar con la calibración. No equivalen a calidad TTC. En H, una norma mayor que 1 no implica clipping: el recorte está desactivado y se registra por separado.

Los encoders son compartidos entre réplicas. Los ajustes ridge cuentan como entrenamiento aunque no usan optimizer.step. Los controles reutilizados no son nuevos fits ni evidencia independiente. FCWD/Dev32 no entrenan ni seleccionan estas variantes.
