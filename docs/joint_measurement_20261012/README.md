# V18 JOINT-MEASUREMENT: índice de evidencia

Estado al 17-10-2026: programa científico cerrado sin candidato admitido; entrega independiente pendiente. A1 y las 64 consultas de coste están completos. La decisión de no habilitar refit/export está registrada. Véase [FINAL_SCIENTIFIC_REPORT.md](FINAL_SCIENTIFIC_REPORT.md).

El [estado regenerado](CURRENT_CONCLUSION.md) y la [auditoría de cierre](COMPLETION_AUDIT.md) separan la evidencia científica de la verificación posterior del paquete. No hay promoción del sistema ni confirmación SOTA. Los resultados negativos se conservan.

| Trabajo | Informe | Alcance |
|---|---|---|
| Matriz primaria | [Resultados P/H](evidence_20261014/PRIMARY_RESULTS.json) | Cuatro fits seed7; P/H y selección prospectiva |
| Rama RGB | [RGB_RESULTS.md](RGB_RESULTS.md) | Encoder entrenable frente a congelado |
| Lectura geométrica C1 | [C1_RESULTS.md](C1_RESULTS.md) | Mismo volumen, otra lectura; conserva sus regresiones |
| Capacidad D1 | [D1_RESULTS.md](D1_RESULTS.md) | Predictor más ancho y control reutilizado |
| Correspondencias B1 | [B1_RESULTS.md](B1_RESULTS.md) | Señal frente al control TTC; rechazo frente a A5 por banda negativa |
| Adaptación A1 | [A1_RESULTS.md](A1_RESULTS.md) | LAST-BLOCK/FULL completos; NO_CANDIDATE por banda negativa |
| Fusión y tres semillas | [FUSION_REPLICATION_RESULTS.md](FUSION_REPLICATION_RESULTS.md) | Una familia replicada; no cumple el gate crucial de promoción |
| Complementariedad por consulta | [QUERY_COMPLEMENTARITY.md](QUERY_COMPLEMENTARITY.md) | Panel completo por ID; no es un selector desplegable |
| Objetivo de aprendizaje | [LEARNING_RECIPE_AUDIT.md](LEARNING_RECIPE_AUDIT.md) | Hipótesis sobre fase y ponderación; sin prueba causal de una nueva pérdida |
| Coste de Garl | [GARL_ROUTE_COST.md](GARL_ROUTE_COST.md) | Rutas emparejadas CPU FP32, preprocesado nativo |
| Paridad del coste de fusión | [FINAL_SCIENTIFIC_REPORT.md](FINAL_SCIENTIFIC_REPORT.md) | 64 consultas completas; aceptación de 1 ms y fallo estricto preservado |

## Entrenamiento, reanudación y recursos

[A1_PERSISTENT_RUNTIME_20261016.md](A1_PERSISTENT_RUNTIME_20261016.md) documenta
la reanudación desde el update 1.086 sin reinicio científico, descompresión paralela,
caché persistente, captura CUDA conservada entre scopes y reducción de
sincronizaciones de la pérdida. Incluye QA exacta y observación de producción.

La [captura adicional de la pérdida](A1_LOSS_GRAPH_20261016.md) pasó después los
controles CUDA exactos y de reanudación de ambos brazos. Se instaló desde el
update 6.400; su mejora del 40,7 % corresponde al forward/backward preparado,
no a una garantía de aceleración del entrenamiento completo.

El [perfil de FULL y la reparación de su comprobación al guardar](A1_BOTTLENECK_AND_BOUNDARY_FIX_20261016.md)
confirman los grafos y la caché persistentes, y documentan la reanudación desde
1.425 updates sin pérdida. La investigación y validación de CI se registra
en [ci_repair_20261016.md](../ci_repair_20261016.md).

La ejecución y sus observadores finalizaron. Para repetir la auditoría CPU sin entrenar:

```powershell
$env:PYTHONPATH = 'src;.'
$env:CUDA_VISIBLE_DEVICES = '-1'
../e-jepa-ttc/.venv/Scripts/python.exe -B -m operational.joint_measurement.final_report
```

El comando está probado en este host. No reinicia entrenamientos ni consume updates.
Los comandos históricos de reanudación conservan su función de recuperación de checkpoints.

La ampliación autorizada vigente fija 216.000 segundos acumulados de scopes GPU
propios y 1.000.000 updates físicos. Los scopes incluyen carga y guardado; no son
tiempo exclusivo de kernels. Se conserva la contabilidad anterior y los límites
de experimentos, replicación, recursos y acciones externas del programa.

## Qué falta para la entrega

A1 y el coste están completos. `DECISION_AFTER_A1.json` deshabilita refit y nuevo
export test12 porque ninguna receta cumple el contrato completo. La familia de
réplica autorizada ya se utilizó en C1/fusión. Falta certificar el ZIP final desde
una extracción independiente; un recibo previo al ZIP no puede certificar sus bytes.

La entrega final deberá contener documentación, predicciones y pesos portables,
con replay desde otra extracción, manifiesto y SHA-256, en un ZIP de como máximo
100.000.000 bytes. Los bundles anteriores son cortes históricos. Los datos crudos,
estados completos de optimizador y padres externos permanecen locales; se declara
ese límite de reproducción. La preparación local de test12 es condicional y no
autoriza su envío al servidor.

[RESEARCH_STATUS.md](RESEARCH_STATUS.md) y [RESUME_20261016.md](RESUME_20261016.md)
preservan la cronología: sus cifras antiguas no sustituyen el estado actual. V17
permanece cerrado con `NO_CANDIDATE`, y V sigue siendo desarrollo interno expuesto.
