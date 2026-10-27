# V18 JOINT-MEASUREMENT: índice de evidencia

**Entrega verificada el 17-10-2026:** [ZIP, replay y alcance](DELIVERY_VERIFIED_20261017.md).
**Estado operativo: V18 cerrado, entrega completa, NO_CANDIDATE.** C1, D1, B1,
A1 y las réplicas están terminados. No hay una cola V18 pendiente que reanudar.
La [decisión posterior a V18](../post_v18_20261017/DECISION_POST_V18.md) separa
el candidato V16, la alternativa de fusión y el diagnóstico A1. Su preparación
autoriza cero nuevos optimizer updates y no modifica campañas posteriores.

**Comparación posterior autorizada y ejecutada:**
[Garl, EvTTC y eAP con pesos congelados](../post_v18_transfer_20261017/REPORT.md).
La nueva autorización habilitó inferencia, no entrenamiento ni envío oficial.
V18 continúa cerrado; la comparación conserva todas sus limitaciones de población.

**Preparación del refit final V16:**
[inventario, particiones y exposición propuesta](../v16_train40_refit_20261017/FULL_REFIT_PLAN.json).
TRAIN40 verifica 88.744 consultas y 40 grupos. Es un plan de viabilidad con cero
updates ejecutados; requiere adaptador, elegibilidad y presupuesto medido antes del fit.

**Continuación posterior del refit TRAIN40, autorizada por separado:**
el plan de viabilidad anterior queda como documento histórico. La ejecución
actual de los productores RGB A5 y C2F utiliza el runtime v6 con
[reservas física y de commit de 2 GiB autorizadas](../final_refit_20261017/RAM_RESERVE_2GIB.md);
véase [optimización aplicada y medición concurrente](../final_refit_20261017/RUNTIME_V4_APPLIED.md).
El localizador operativo es `artifacts/final_refit_20261017/ACTIVE_RUNTIME.json`.
El [supervisor de reanudación por recursos](../final_refit_20261017/AUTO_MEMORY_RESUME.md)
reanuda cada productor tras una pausa limpia de memoria o inventario caducado,
cuando los recursos y el inventario firmado vuelven a ser válidos;
conserva las pausas manuales y bloquea los errores de integridad o checkpoint.
Esta ejecución no reabre V18 ni constituye una evaluación oficial.

El corte previo a continuación se conserva como procedencia del paquete.

**Corte histórico anterior a la entrega:** programa científico cerrado sin candidato admitido; en ese corte faltaba verificar la entrega independiente. A1 y las 64 consultas de coste estaban completos. La decisión de no habilitar refit/export permanece registrada. Véase [FINAL_SCIENTIFIC_REPORT.md](FINAL_SCIENTIFIC_REPORT.md).

El [estado regenerado](CURRENT_CONCLUSION.md) y la [auditoría de cierre](COMPLETION_AUDIT.md) conservan el corte previo al paquete. Sus pendientes de entrega quedan resueltos por el recibo posterior enlazado arriba. No hay promoción del sistema ni confirmación SOTA. Los resultados negativos se conservan.

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

La ejecución y sus observadores finalizaron. Referencia histórica para regenerar
el corte científico previo al paquete; no es una tarea pendiente:

```powershell
$env:PYTHONPATH = 'src;.'
$env:CUDA_VISIBLE_DEVICES = '-1'
../e-jepa-ttc/.venv/Scripts/python.exe -B -m operational.joint_measurement.final_report
```

El comando está probado en este host. No reinicia entrenamientos ni consume updates.
Reescribe el corte previo a la entrega; el recibo externo sigue siendo la autoridad
sobre el ZIP verificado. Esta revisión posterior no ejecuta ese comando.
Los comandos históricos de reanudación conservan su función de recuperación de checkpoints.

La ampliación autorizada vigente fija 216.000 segundos acumulados de scopes GPU
propios y 1.000.000 updates físicos. Los scopes incluyen carga y guardado; no son
tiempo exclusivo de kernels. Se conserva la contabilidad anterior y los límites
de experimentos, replicación, recursos y acciones externas del programa.

## Corte histórico anterior al empaquetado

A1 y el coste están completos. `DECISION_AFTER_A1.json` deshabilita refit y nuevo
export test12 porque ninguna receta cumple el contrato completo. La familia de
réplica autorizada ya se utilizó en C1/fusión. En este corte faltaba certificar el
ZIP desde una extracción independiente. El recibo posterior
[DELIVERY_COMPLETE.json](delivery_evidence_20261017/DELIVERY_COMPLETE.json)
certifica esa entrega; no queda pendiente repetirla.

La entrega final contiene documentación, predicciones y pesos portables,
con replay desde otra extracción, manifiesto y SHA-256, en un ZIP de como máximo
100.000.000 bytes. Los bundles anteriores son cortes históricos. Los datos crudos,
estados completos de optimizador y padres externos permanecen locales; se declara
ese límite de reproducción. La preparación local de test12 es condicional y no
autoriza su envío al servidor.

[RESEARCH_STATUS.md](RESEARCH_STATUS.md) y [RESUME_20261016.md](RESUME_20261016.md)
preservan la cronología: sus cifras antiguas no sustituyen el estado actual. V17
permanece cerrado con `NO_CANDIDATE`, y V sigue siendo desarrollo interno expuesto.
