Continúa V15 desde su último estado durable. Lee execution_policy.json y el ledger
local; resta trabajo consumido del techo propio, nunca del saldoV14. Primero revisa
PIDs/owners actuales y archivos parciales. No regenere endpoints existentes ni
vuelva a evaluar V para escoger checkpoints alternativos. Una fila de progreso
no es checkpoint; restaura pesos/optimizer/scheduler/cursor/RNG. No modifiques
V13. Termina tareas habilitadas y entrega la limitación exacta de las bloqueadas.
