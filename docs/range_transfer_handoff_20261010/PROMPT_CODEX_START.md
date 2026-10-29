# Encargo nuevo — RANGE-TRANSFER

Autorizo una campaña nueva y separada, según `execution_policy.json`, para investigar y reducir la sobreestimación TTC corta sin destruir la precisión media/larga del sistema.

## Identidades

Repo `Kripta-Studios/e-jepa-ttc`.
Evidencia V12: `1a686e338a1145c6794bec0704bdae4cda95f06d`, padre `d1f561df7b173464de24e1b8e3ef09170081f7fe`.
Código V13 auditado: `da068ebdeb2d967ee10b2afa348e1317b8720ad0`.
Trabajo nuevo: `scientific-recovery-v14-range-transfer`, en worktree separado.

## Lee primero

README y documentos01–07 del handoff, política JSON y el mapa de integración. Lee la evidencia nueva de V12 y el estado actual del host, no sólo memorias de conversaciones. Verifica el handoff antes de aplicarlo.

No reinicies ni cambies V13. Su loss, pesos, sampler, BF16, datos, splits y admisión Git siguen siendo su contrato. Los padres RGB aún incompletos no se convierten en un fallo científico. No hay autorización para publicar commits en remoto.

## Objetivo operativo

Implementar, probar y ejecutar N0–N2 y el factorial event-only de cabezas cuando sus dependencias reales estén completas. No regresar sólo con documentación o comandos si hay trabajo autorizado viable. Tampoco ejecutar ciencia sobre padres parciales para evitar un bloqueo.

1. Verifica estado, snapshots y los recálculos públicos; conserva resultados negativos y contratos de métricas.
2. Produce trazas por módulos para las consultas diagnósticas ya publicadas y controles H seleccionados por hash, cuando exista acceso real a datos/pesos. No entrenes con FCWD.
3. Audita la masa nativa: V12 usa bandas/secuencia, V13 contiene pesos por grupo. Es una diferencia de receta, no permiso para editar los fits actuales. Mide masa/gradientes/exposición de H.
4. Aplica el patch de referencia sólo en V14, después de `git apply --check`. Integra el runner pequeño reutilizando checkpoints y el modelo nativos; el paquete no trae ese runner implementado.
5. Usa E_A5_MATCHED/E_C2F_MATCHED/PAIR_E_MATCHED completos y caché H nativa congelada. No usar TRAIN40 como productor para simular un holdout interno.
6. Entrena las seis celdas masaM0/M1/M2 × lossL0/L1 con semilla 7 y endpoint 2500. Congela todos los endpoints antes de evaluar la matriz en V. Misma inicialización y sampler por seed; no cambiar normalizador, tiempo, output o arquitectura.
7. Escoge como máximo una receta usando únicamente los gates V del protocolo. Si cumple, replica ella y M0-L0 en seeds 13/23. Si no cumple, entrega el resultado sin rescate.
8. Congela selección antes de los scores de transferencia y conserva todas las variantes. Dev32 y FCWD siguen expuestos; no claims de test fresco.
9. Entrega pesos nuevos, predicciones, análisis por bandas/grupos, pruebas, contabilidad, decisiones y bundle reproducible.

## Autorización y límites

Hasta 25.000 updates científicos nuevos,500 técnicos sintéticos,4.500 de recuperación;30.000 físicos. No es autorización para terminar o ampliar presupuestos antiguos, ni para nuevos encoders, estudiantes, H4/H2, adaptaciones RGB, semillas o sweeps. Un endpoint nativo idéntico puede reutilizarse; el ahorro no se reasigna.

CPU FP32, batch 128, GRU 160, receta AdamW/warmup/cosine nativa y outputs firmados iguales. La rama L1 sólo añade el término de TTC relativo fijado. Registrar el dominio y los gradientes, sin straight-through ni exclusión de targets.

Máximo del proyecto23 GB RSS, reserva host2 GiB y recuperación3 GiB; V14≤4 GiB, hasta 4 threads,10 GB libres tras reservas. No competir con GPU/medición de V13/R1, no tocar configuración global. Leer sus estados sin convertirse en otro supervisor.

No abrir Stage76, labels privados, test sellado ni score CodaBench. No submissions, push, merge ni emails. El contrato oficial o la calibración FCWD RGB pendientes bloquean sólo sus ramas, no este factorial.

Si faltan padres/cachés, termina las tareas CPU viables, identifica los faltantes exactos y entrega comando real de reanudación. No fabricar historia/pesos ni repetir eternamente errores idénticos. Ante fallo técnico preservar checkpoints; ante resultado negativo conservarlo.

## Interpretación obligatoria

MiD de FCWD/Dev32 es mean_MiD; overall incompleto no se rellena. H8 de V12 es mediana de tres cabezas, no tres encoders independientes. Estas nuevas réplicas también son de cabezas. Ganar MAE perdiendo TTC corto no satisface el objetivo. No cambiar el relato seleccionando la métrica favorable.

La finalidad es resolver qué factor merece el siguiente entrenamiento mayor, manteniendo V13. No declarar de antemano que el balance, RGB o la nueva loss serán SOTA.
