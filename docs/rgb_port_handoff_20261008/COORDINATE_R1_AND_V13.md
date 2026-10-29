Mensaje para la sesión anterior:

Voy a abrir RGB-PORT en otro worktree. No cambies el modelo ni la receta que ya estén entrenándose. Devuelve un handoff breve con HEAD, run IDs, raíces, checkpoints durables, propietario/lease GPU, estado vivo de R1 y comandos existentes de pausa/reanudación. Distingue V13 meramente preparado de fits efectivamente iniciados.

R1 conserva todos los pares. La nueva sesión no debe competir por GPU mientras se mide: termina R1 o coordina una pausa cooperativa registrada en una frontera recuperable, sin matar el proceso ni borrar caché/fragmentos para simular una medición nueva. La consulta o escritura de documentos CPU no es un conflicto GPU.

La nueva prioridad sustituye sólo fases de arquitectura aún no iniciadas. No promuevas o descartes ningún modelo por este mensaje. No lances otra copia del supervisor ni un entrenamiento adicional.
