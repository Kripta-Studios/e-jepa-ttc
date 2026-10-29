# Panel de consultas V18: complementariedad y fallos compartidos

Se unen los cuatro endpoints verificados por `sample_token`, sin intersección selectiva: 53.890 consultas P y 18.422 H. P es entrenamiento; H es desarrollo del productor. No se abre V, no se entrena un selector ni se cambia la selección.

## Admisión y relojes

El constructor rechaza identificadores duplicados o nulos, consultas ausentes/adicionales, targets/grupos/secuencias/roles distintos y bindings sensoriales incompatibles. Verifica los SHA-256 originales de las ocho predicciones, manifiestos, filas y TRAIN40_INDEX; contrasta el reloj de consulta contra TRAIN40_ROWS. Los cuatro modelos comparten exactamente el mismo sensor_complete por rol. No se releyeron todos los tensores sensoriales: la garantía se apoya en sus recibos de admisión y verificación previos, conservados en RESULT.json.

Se conservan query_time_us/cutoff_us (reloj de consulta), los inicios/finales de las dos ventanas event (reloj del sensor), su separación real, el delta nativo de entrada y el delta de métrica. No se confunden relojes ni se infiere el delta por restar timestamps de dominios distintos. Las tablas originales de predicciones no contienen esos tiempos: se adjuntan desde su genealogía común verificada. Si cambian las fuentes o un binding, se rechaza el panel.

Se guardan target, fase target original y recalculada, TTC y fases de cada modelo. En A5/NATIVE, la columna histórica raw_phase fue calculada desde TTC emitido; en CORR/C1 sí es la salida anterior a la conversión. El panel distingue phase_saved y phase_emitted; no atribuye al nativo una fase interna no exportada.

## Negativos concretos donde CORR equivoca el signo

Aquí acertar significa únicamente recuperar el signo negativo. Se informa también el error en segundos: un signo correcto no garantiza un TTC cercano.

| Rol | Casos | Modelo | Signo corregido | Corrección % | Delta MAE frente a CORR, s |
|---|---:|---|---:|---:|---:|
| P | 9213 | A5 | 8442 | 91.63 | -9.695 |
| P | 9213 | NATIVE | 8525 | 92.53 | -11.836 |
| P | 9213 | C1 | 5355 | 58.12 | +4.579 |
| H | 2423 | A5 | 2044 | 84.36 | -8.543 |
| H | 2423 | NATIVE | 2056 | 84.85 | -10.372 |
| H | 2423 | C1 | 1317 | 54.35 | +4.331 |

Valores negativos del delta significan menos error que CORR.

- P: A5 y NATIVE corrigen simultáneamente 8312 casos; ambos siguen fallando en 558; los cuatro fallan en 485.
- H: A5 y NATIVE corrigen simultáneamente 2005 casos; ambos siguen fallando en 328; los cuatro fallan en 271.

## Casos críticos donde CORR mejora

Se define mejora por menor MiD individual, sin umbral añadido. La banda crítica es 0,1 < TTC <= 3 s. Se conserva el contraste completo y los subsets separados contra A5, NATIVE y ambos en los CSV.

| Rol | CORR mejor que ambos | Total crítico | CORR peor que ambos |
|---|---:|---:|---:|
| P | 3877 | 6069 | 1356 |
| H | 910 | 1827 | 688 |

En los casos donde CORR gana a ambos:

| Rol | Modelo | Delta MiD medio | Delta MAE, s |
|---|---|---:|---:|
| P | A5 | +144.422 | +2.396 |
| P | NATIVE | +118.954 | +1.557 |
| P | C1 | +36.581 | +0.576 |
| H | A5 | +154.144 | +2.898 |
| H | NATIVE | +134.549 | +2.196 |
| H | C1 | +50.228 | +1.132 |

## Interpretación y reproducción

Existe complementariedad observada por consulta, además de fallos compartidos. Estas particiones usan GT para describir los resultados: no son una regla disponible en inferencia ni prueban que pueda anticiparse qué modelo ganará. No se presenta un selector oráculo como sistema alcanzable. Los promedios anteriores son pooled dentro de cada subset, no el agregado oficial ni un intervalo estadístico. Los resultados por grupo están incluidos; las ventanas de una misma secuencia no se consideran réplicas independientes.

Ejecutar `python -B -m operational.joint_measurement.query_panel` con `PYTHONPATH=src;.`. Requiere las predicciones y metadatos admitidos. Los paneles completos P_QUERY_PANEL.parquet y H_QUERY_PANEL.parquet están en `artifacts/joint_measurement_20261012/diagnostics/query_panel/`, con compresión Zstandard sin pérdida. RESULT.json liga sus tamaños de población, columnas y hashes. No se eliminan predicciones inválidas: se marcan y los promedios de métricas incompletas quedan indefinidos.

[Bindings y paneles](evidence_query_panel_20261016/RESULT.json) · [Contrastes y grupos](evidence_query_panel_20261016/PAIRED_SUMMARIES.csv) · [Contingencia de signo](evidence_query_panel_20261016/NEGATIVE_SIGN_CONTINGENCY.csv) · [Contingencia crítica](evidence_query_panel_20261016/CRUCIAL_MID_CONTINGENCY.csv)
