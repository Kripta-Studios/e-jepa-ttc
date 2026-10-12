# Encargo nuevo: V15 RANGE-READOUT

Autorizo el protocolo de este paquete como campaña nueva, no como continuación
del presupuesto V14. V14 se mantiene cerrado NO_CANDIDATE.

Lee README.md, docs/01_REVIEW.md, docs/02_PROTOCOL.md, docs/03_IMPLEMENTATION.md
 y execution_policy.json. Base Git 1c251781ac845a5d661523e607e0f7c4a8d3af8a.
Trabaja en worktree/rama separados scientific-recovery-v15-range-readout.

Prioridad: averiguar si el cambio de pesos fracasa por varianza/clipping y si el
readout geométrico de A5 pierde rango antes de H8. No inventar otra arquitectura.

1. Verifica el ZIP fuente V14 y aprovecha los recálculos ya incluidos; no repitas
   su entrenamiento ni toda su arqueología. Comprueba padres/roles/cachés reales.
2. Integra y prueba los operadores de importancia y el residual C05/C20. El código
   de referencia NO es un trainer integrado: reutiliza runners y loss nativa.
3. S: ejecuta S-M0/L0 y S-M1/L0 con p=w y pérdida corregida w/p; dos cabezas de 2.500 updates.
4. R: A5 congelado salvo scorer residual, dos readouts de 6.000 updates C=.05/.20, pendiente
   inicial igual, Linear final reiniciada igual. No cambiar encoder ni C2F/PAIR.
5. Regenera PHASE17/normalizadores coherentes con cada A5 y entrena dos CTX de 2.500 updates.
6. Congela todos los endpoints antes de V, puntúa la misma población y gates.
   Replica como máximo la familia habilitada; entrega negativo si ninguna pasa.
7. Sólo después de una réplica favorable y freeze, evaluar transferencia completa
   según el protocolo; no usar FCWD/Dev32 para adaptar los fits.

Máximos: 56.000 científicos, 500 técnicos, 3.500 de recuperación; 60.000 físicos.
Es techo condicionado, no una obligación de consumirlo. No reasignar sobrante.
La réplica de encoders NO está autorizada y no se afirmará haberla hecho.

No regreses sólo con código o planes si las tareas admitidas pueden ejecutarse.
Si falta una dependencia, termina las tareas independientes, guarda estado y
entrega ruta/hash y comando resume real. No esperes indefinidamente ni fabriques
inputs o pesos. No relances el mismo fallo sin diagnóstico. No montar otra
infraestructura masiva de orquestación ni reformatear todo el repositorio.

Consulta estado vivo de V13 y R1 y conserva sus operaciones. No editar allí ni repetir
admisiones históricas por rutina. Preserva todos los outputs V14 y cambios ajenos.
Usa límites de recursos del paquete y entorno existente. No drivers/paging/energía.
No submissions, pushes, merges, envíos de email, labels privados ni test sellados.

Entrega los resultados completos, negativos, inciertos e incompletos separados,
con pesos, curvas, predicciones, masa/muestreo/clipping, alcance de costes, pruebas,
contabilidad, NEXT_DECISION y bundle con SHA-256. No reclamar SOTA por mejorar sólo un
rango ni por una comparación V interna sin Garl equivalente.
