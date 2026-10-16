# V18 JOINT-MEASUREMENT: índice de evidencia

La campaña sigue en ejecución. A1 es la cuarta comparación adaptativa registrada;
sus dos brazos deben terminar antes de evaluar H. No hay promoción del sistema
ni confirmación SOTA. Los resultados negativos se conservan.

El estado regenerado está en [CURRENT_CONCLUSION.md](CURRENT_CONCLUSION.md).
Su fecha identifica una instantánea: durante un entrenamiento los journals,
checkpoints y el observador horario local son la fuente de progreso actual.
La [revisión de cierre](CLOSURE_REVIEW_20261016.md) relaciona los trabajos
pendientes con la evidencia necesaria y sus condiciones de ejecución.

| Trabajo | Informe | Alcance |
|---|---|---|
| Matriz primaria | [Resultados P/H](evidence_20261014/PRIMARY_RESULTS.json) | Cuatro fits seed7; P/H y selección prospectiva |
| Rama RGB | [RGB_RESULTS.md](RGB_RESULTS.md) | Encoder entrenable frente a congelado |
| Lectura geométrica C1 | [C1_RESULTS.md](C1_RESULTS.md) | Mismo volumen, otra lectura; conserva sus regresiones |
| Capacidad D1 | [D1_RESULTS.md](D1_RESULTS.md) | Predictor más ancho y control reutilizado |
| Correspondencias B1 | [B1_RESULTS.md](B1_RESULTS.md) | Señal frente al control TTC; rechazo frente a A5 por banda negativa |
| Adaptación A1 | [A1_EVALUATION_CONTRACT_20261016.md](A1_EVALUATION_CONTRACT_20261016.md) | LAST-BLOCK/FULL; evaluación preparada, resultados pendientes |
| Fusión y tres semillas | [FUSION_REPLICATION_RESULTS.md](FUSION_REPLICATION_RESULTS.md) | Una familia replicada; no cumple el gate crucial de promoción |
| Complementariedad por consulta | [QUERY_COMPLEMENTARITY.md](QUERY_COMPLEMENTARITY.md) | Panel completo por ID; no es un selector desplegable |
| Objetivo de aprendizaje | [LEARNING_RECIPE_AUDIT.md](LEARNING_RECIPE_AUDIT.md) | Hipótesis sobre fase y ponderación; sin prueba causal de una nueva pérdida |
| Coste de Garl | [GARL_ROUTE_COST.md](GARL_ROUTE_COST.md) | Rutas emparejadas CPU FP32, preprocesado nativo |
| Paridad del coste de fusión | [FUSION_COST_DIAGNOSTIC.md](FUSION_COST_DIAGNOSTIC.md) | Fallos y contrastes preservados; coste completo todavía no admitido |

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

```powershell
./scripts/Resume-JointMeasurement.ps1 -ShowCommand
./scripts/Resume-JointMeasurement.ps1
```

No iniciar otro escritor mientras el coordinador vigente siga vivo. El script
comprueba procesos y los módulos mantienen sus bloqueos. Después de A1, el modo
Auto pasa a evaluación sólo con ambos endpoints completos. Los observadores
comprueban el proceso y sus contadores cada hora en punto, sin reiniciarlo.

La ampliación autorizada vigente fija 216.000 segundos acumulados de scopes GPU
propios y 1.000.000 updates físicos. Los scopes incluyen carga y guardado; no son
tiempo exclusivo de kernels. Se conserva la contabilidad anterior y los límites
de experimentos, replicación, recursos y acciones externas del programa.

## Qué falta para la entrega

Finalizar A1, interpretar sus cohortes completas, resolver la paridad pendiente
del coste de fusión y decidir las ramas de refit/export habilitadas por evidencia.
El refit requiere una receta seleccionada y registrada; H/V incorporados a sus
pesos dejan de ser evaluación independiente. La familia de réplica autorizada
ya se utilizó en C1/fusión. No se inventa otra réplica para rescatar una semilla.

La entrega final deberá contener documentación, predicciones y pesos portables,
con replay desde otra extracción, manifiesto y SHA-256, en un ZIP de como máximo
100.000.000 bytes. Los bundles anteriores son cortes históricos. Los datos crudos,
estados completos de optimizador y padres externos permanecen locales; se declara
ese límite de reproducción. La preparación local de test12 es condicional y no
autoriza su envío al servidor.

[RESEARCH_STATUS.md](RESEARCH_STATUS.md) y [RESUME_20261016.md](RESUME_20261016.md)
preservan la cronología: sus cifras antiguas no sustituyen el estado actual. V17
permanece cerrado con `NO_CANDIDATE`, y V sigue siendo desarrollo interno expuesto.
