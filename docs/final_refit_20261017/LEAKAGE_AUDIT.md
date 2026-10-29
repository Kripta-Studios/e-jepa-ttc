# Auditoría de causalidad, aislamiento y genealogía

Revisión del 17 de octubre de 2026 por tres agentes independientes: causalidad,
linaje de modelos y aislamiento de Test12. La revisión combina lectura del código
con inspección de los inventarios reales; no repite entrenamientos ni evalúa
etiquetas privadas. Los cambios correctivos se verifican por separado.

## Resultado sobre los inventarios actuales

Los tres revisores no encontraron evidencia de información futura ni fuga de
Test12 en los artefactos inspeccionados. Esto es una conclusión delimitada por
esas comprobaciones, no una garantía universal sobre toda ejecución posible.

- Las 88.744 filas de P/H/V coinciden con TRAIN40 por ordinal nativo, token y
  secuencia. Hay 40 secuencias, sin tokens ni consultas físicas duplicadas.
- TRAIN40 y Test12 no comparten tokens ni secuencias.
- Los 1.812 identificadores de track de TRAIN40 pertenecen cada uno a una sola
  secuencia. El agrupado histórico sólo por track no mezcló secuencias en estos
  datos; para otra población esa propiedad debe comprobarse de nuevo.
- No aparecieron clocks de anotación/disponibilidad posteriores a la consulta,
  productores de otra secuencia ni historiales RGB de más de 650 ms.
- Test12 contiene 6.762 tokens y 5.805 consultas físicas; las 957 repeticiones
  se representan mediante aliases declarados. El inventario V16 inspeccionado
  no contiene violaciones de propiedad, orden, disponibilidad o ventana.

Los timestamps absolutos del sensor y de anotación están en dominios distintos.
La auditoría compara orden, disponibilidad en el dominio de anotación y
concordancia de intervalos; comparar directamente sus valores absolutos sería
una comprobación incorrecta.

## Hallazgos que requieren corrección o implementación

| Área | Problema encontrado | Alcance |
|---|---|---|
| Cachés de capas superiores | Tamaño, unicidad y número de grupos no bastan para identificar TRAIN40; falta exigir el inventario exacto | Antes de PAIR/CTX/PHASE |
| Normalizadores | El inventario de observaciones debe estar vinculado a una fuente externa comprobada, no sólo a IDs autodeclarados | Antes de CTX |
| Padres y export | Un recibo COMPLETE debe acreditar también el modelo concreto, su receta y toda su procedencia | Antes de reutilización/sellado |
| Test12 | Exigir tokens oficiales, propiedad secuencia/track, ventanas y procedencia de sensores; ampliar rechazo de campos de GT | Antes de inferencia final |
| Stacking OOF | Separar los padres temporales que producen entradas de entrenamiento de los padres finales desplegados | Bloquea la cadena OOF completa |
| Productores por fold | Falta una fuente limitada a los índices del rol; el runner TRAIN40 no puede utilizarse directamente | Bloquea productores OOF |
| Rotación histórica 0 | El recibo previo no probaba todo el cierre transitivo de observaciones y normalizadores | Reauditar antes de reutilizar |

Los productores RGB finales R_A5/R_C2F usan una fuente distinta de las cachés de
las capas superiores. Su fuente sí verifica los ordinales y tokens TRAIN40,
los roles, el teacher y los hashes del plan. Los hallazgos de PAIR/CTX/PHASE no
demuestran contaminación de estos productores ni los bloquean por sí solos.
Su inicio continúa condicionado a la admisión técnica de ejecución y recursos.

## Límites de interpretación

H y V históricos pasan a formar parte del TRAIN final. No son validación
independiente de ese modelo. Las salidas OOF pueden preparar el stacking, pero
no son confirmación ciega: la investigación anterior ya observó H y V.

La metadata pública permite agrupar por secuencia, pero no demuestra que todas
las secuencias procedan de sesiones de adquisición diferentes. Esa independencia
adicional queda como supuesto no verificado, no como fuga demostrada.

El contexto causal de 650 ms no constituye lookahead por su duración. Su
admisibilidad en la competición es una cuestión distinta de la causalidad y
no queda confirmada únicamente porque Garl use una ventana más corta.

## Evidencia regenerable y correcciones

La auditoría de inventarios se regenera sin Torch, GPU, decodificación de medios
ni acceso a HDF5 con:

```powershell
& ../e-jepa-ttc/.venv/Scripts/python.exe -B scripts/audit_final_refit_leakage.py
```

Su recibo es `artifacts/final_refit_20261017/audit/CAUSALITY_AND_SPLITS.json`.
Conserva hashes de las fuentes, conteos y violaciones. El alcance es metadata;
no equivale a volver a comprobar todos los bytes de los medios crudos.

La rotación histórica 0 se reauditó con el cierre completo de los seis endpoints
de productores/PAIR, las observaciones E/R, sus normalizadores ajustados en H y
los tokens/grupos de HISTORY. El recibo nuevo
`artifacts/final_refit_20261017/folds/ROTATION0_REUSE_AUDIT_V2.json` pasa y conserva
el recibo v1 sin sobrescribirlo. Su SHA-256 es
`ed1fe22aa3e53039821840e67ab212419c48c2a27b78fb217633e060392e9443`.
Esto admite esa reutilización concreta; no completa las restantes rotaciones.

El acceso científico a `fit_fold_stage` queda explícitamente bloqueado hasta
implementar y admitir el cierre transitivo OOF. Las pruebas técnicas acotadas
no habilitan por sí solas el entrenamiento científico.

Correcciones implementadas y revisadas de forma independiente:

- Recipe v2 y cachés exigen el hash de TRAIN40 y el orden exacto token/grupo.
  Una población con igual tamaño no satisface este contrato.
- Los padres deben acreditar esquema, nombre de fit, endpoint, bytes del
  checkpoint e identidad canónica; no basta un estado `COMPLETE`.
- El normalizador usa exactamente la unión de observaciones válidas referidas
  por los historiales, con pertenencia de grupo y recibos de origen. No admite
  observaciones adicionales aunque se les genere un recibo coherente.
- La materialización event compacta las observaciones no usadas después de
  fijar la máscara causal. Conserva tiempos, ROI, fases e historiales y registra
  los conteos generados, retenidos y descartados.
- El sellado verifica los cinco recibos de caché de entrenamiento y exige los
  hashes de los normalizadores realmente usados por las cabezas.
- Test12 exige los hashes fijados de INPUTS, Parquet oficial y ambos freezes,
  además del conjunto exacto de tokens y sus invariantes causales y de objeto.

Los artefactos de receta v1 previos a estas correcciones quedan como evidencia
histórica. Deben regenerarse admisiones v2 en una ruta nueva; no se sobrescriben
ni se presentan como compatibles con el contrato actualizado.

Validación conjunta posterior: **93 tests, cero fallos, cero errores y cero
omitidos**, registrados en
`artifacts/final_refit_20261017/LEAKAGE_FIX_TESTS.xml` y
`leakage_fix_tests.log`. Incluye las regresiones negativas y la integración
de la nueva receta, cachés, normalizadores, compactación y sellado. Ruff pasa
para el namespace, sus tests y el auditor. Los únicos avisos de esta suite son
de deprecación de TorchScript.

La auditoría regenerada por el agente principal devuelve `PASS` y cero
violaciones; su hash de contenido es
`808caa8a091b87d6d3f32063a3674f7dd8a0dfb909449fdcd436465565840b3b`.
No se ejecutaron nuevos updates científicos, inferencias Test12 ni submissions
durante esta auditoría. La suite previa de 76 tests se conserva como evidencia
anterior y no se suma al resultado de 93 pruebas.
