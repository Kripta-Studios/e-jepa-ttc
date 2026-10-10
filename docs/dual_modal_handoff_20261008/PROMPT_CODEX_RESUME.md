Continúa V13 DUAL-TTC desde su último estado durable, con el mismo protocolo y presupuesto.
Lee el prompt de inicio, execution_policy.json y los manifests/status/accounting locales.
No reinicies fits completos ni utilices el texto de un snapshot antiguo como estado vivo.

Antes de escribir: identifica HEAD, source freeze, task owner y output root; si existe
trainer vivo no lances otro. R1/Stage70 tienen dueños diferentes. Preserva cambios ajenos.
Reanuda el fit parcial exacto con modelo/optim/scheduler/scaler/RNG/sampler/acumulación y
contador. Verifica inputs y pad/modalidad sin descartar consultas por GT o por error.

No alteres arquitectura, seeds, datos, gates, pérdida o número de updates. Las dependencias
externas que afecten a una rama no bloquean tareas independientes ya autorizadas. No hagas
TRAIN40/test masquerading, submissions, nuevo cloud, push o cambios globales del equipo.

Continúa hasta cerrar ramas habilitadas o identificar un bloqueo externo real. Devuelve
progressdurable, endpoints completos/parciales y contabilidad actualizada, informe,
NEXT_DECISION, comandos realmente ejecutables y bundle verificado. La evidencia negativa
se entrega sin rescate por nuevas variantes. No basta con escribir un plan.
