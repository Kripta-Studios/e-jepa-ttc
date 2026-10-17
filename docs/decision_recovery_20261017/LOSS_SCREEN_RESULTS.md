# V19: resultado del contraste de ponderación

Los dos brazos completaron 12.000 updates y su evaluación completa sobre las 53.890 consultas P y 18.422 H. El pipeline cerró esta fase a las 03:33 del 17 de octubre de 2026, hora de Madrid. Esto termina el contraste registrado, no la investigación V19.

## Intervención y control

L0 y LW parten del mismo B1 de correspondencias, con adaptación FULL, semilla 7, sampler uniforme y presupuesto idénticos. L0 asigna masas 50/30/10/10% a crítico/3–6/6–10/negativo. LW usa 44,44/26,67/8,89/20%, conservando la proporción positiva 5:3:1. No se seleccionaron checkpoints con H.

L0 reproduce exactamente los valores numéricos de las métricas P/H del A1-FULL histórico. Los 87 updates de recuperación por la interrupción accidental están contabilizados; ambos journals terminan con 12.000 updates durables y cero pendientes. No constituyen 87 ejemplos independientes ni una nueva réplica científica.

## Resultados H

| Modelo | Agregado MiD | Crítico | 3–6 | 6–10 | Negativo | MAE (s) | Error de signo |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 | 112,285 | 157,267 | 68,257 | 66,231 | 65,520 | 5,637 | 5,694% |
| L0 | 78,920 | 95,846 | 53,485 | 62,426 | 87,087 | 5,446 | 5,607% |
| LW | 79,867 | 100,418 | 54,545 | 62,170 | 70,775 | 4,613 | 4,055% |

LW reduce el error negativo un 18,73%, el MAE un 15,29% y el error de signo en 1,55 puntos porcentuales frente a L0. A cambio, el agregado empeora un 1,20%, el crítico un 4,77% y la banda 3–6 un 1,98%. Es una modificación del compromiso entre objetivos; no domina a L0.

Los intervalos pareados por grupos de LW menos L0 son [0,017; 2,055] para agregado, [1,306; 7,888] para crítico y [−23,859; −9,869] para negativo. Describen esta semilla sobre H reutilizado; no son confirmación independiente ni variabilidad entre entrenamientos.

## Resultados P

| Modelo | Agregado MiD | Crítico | Negativo | MAE (s) | Error de signo |
|---|---:|---:|---:|---:|---:|
| A5 | 100,874 | 141,813 | 55,522 | 4,751 | 3,333% |
| L0 | 59,652 | 59,786 | 93,547 | 5,926 | 6,413% |
| LW | 60,028 | 63,688 | 71,782 | 4,785 | 4,363% |

El mismo patrón aparece en entrenamiento: mejor negativo, MAE y signo, con peor crítico y agregado. No puede explicarse únicamente como falta de generalización hacia H.

## Decisión registrada y continuación

LW satisface la elegibilidad numérica de desarrollo frente a A5: el negativo empeora un 8,02%, dentro del margen del 10%. No satisface la protección más estricta del sistema, que exige conservar el negativo frente a A5. Estos dos controles son distintos y se informan por separado.

El disparador de replicación requería mejorar al menos un 3% el agregado frente a L0. Al empeorarlo un 1,20%, el resultado es `REPLICATION_NOT_ENABLED`. No se cambian retrospectivamente los umbrales ni se ejecutan automáticamente las semillas 13/23.

Continúa el diagnóstico mecánico: gradientes por banda y bloque en un panel P fijo, prueba de recuperabilidad del signo desde características congeladas y medición del coste de ejecución. Estas tareas no modifican los endpoints terminados. Una siguiente intervención de aprendizaje se registrará por separado a partir de esas evidencias, sin un barrido de configuraciones sobre H.

Fuente ejecutada: `artifacts/decision_recovery_20261017/evaluation/REPORT.json`, SHA-256 `1073ab21e307ae266221816fd52422f5117f23b0fb7927429e89970a6d996927`. Incluye recibos de las cuatro evaluaciones completas, comparaciones y disparador. La precisión oficial y SOTA siguen sin establecerse.
