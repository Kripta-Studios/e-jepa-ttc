# Mensaje al agente de la sesión V12 existente

Voy a iniciar V13 en un worktree nuevo. No reinicies TRAIN40, WIDE o comparaciones ya
cerradas. No detengas ni reinicies un fit o R1 por recibir este mensaje.

Entrega un handoff de coordinación: HEAD/worktree real, rutas de datasets/caches/pesos
con identidades y autoridad vigente, estado/owner/PID de R1 y Stage70, memoria/slot de
GPU y comando exacto de reanudación existente. No copies contenido sensible innecesario.

Si R1 sigue vivo, conserva su progreso hasta una frontera segura y acuerda slot explícito
con V13 mediante los mecanismos existentes. No compartáis un mismo output root ni
lancéis dos supervisores. Un informe de snapshot no es prueba de proceso vivo.
No hace falta repetir la ciencia cerrada para preparar esta coordinación.
