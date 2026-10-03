# Recuperación operativa SIMPLEX-T

Estos adaptadores preservan los archivos científicos fijados en el protocolo.
No autorizan fits, seeds, arquitecturas o updates adicionales.

- `run.py` reintenta la misma sustitución atómica ante bloqueos de Windows;
  no repite el cálculo de un update, fragmento o draw.
- `resume_remaining.py` reutiliza H16, N2 y las pruebas técnicas completadas.
- `resume_after_commit_recovery.py` conserva la reserva adicional de startup
  empleada en la ventana original.
- `continue_campaign.py` aplica la nueva autorización operativa por proceso.
  Conserva los bytes de `WINDOW_AUTHORIZATION.json` y del protocolo original;
  la ventana efectiva y el presupuesto propio se registran por separado en
  `CONTINUATION_AUTHORIZATION_20261003.json`. La autorización del usuario
  retiró la reserva EXTRA de compromiso de startup; permanecen las
  comprobaciones del motor de RAM, RSS y memoria comprometida.
- `compact_delivery.py` produce y verifica entregas compactas bajo el
  presupuesto original. No aplica automáticamente el presupuesto ampliado.
- `delivery_delta.py` conserva estados posteriores sobre una entrega base
  identificada por SHA-256; exige una admisión física previa del publicador.
- `finalize_continuation.py` corrige los metadatos operativos de la entrega y
  ejecuta publicación y verificadores en procesos pesados separados. Exige
  propietario único, adopción durable de hijos y alcance exacto de los recibos:
  18 cabezas C0 cuando se completan los 24 endpoints. Una entrega parcial
  distingue exports publicados de endpoints completos aún sin evaluar.

Para la continuación autorizada, desde el worktree y su entorno existente:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_io_recovery.continue_campaign --check-policy
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_io_recovery.continue_campaign
```

El primer comando no entrena. El segundo mantiene la cola fijada y exige
autorización local válida, fuentes verificadas y ausencia de otro propietario.
No debe lanzarse mientras siga vivo un supervisor de esta misma campaña,
incluido un supervisor suspendido. La autorización local no se genera ni se
amplía automáticamente al ejecutar estos comandos.

Los bundles contienen entradas cacheadas para regenerar inferencia y explican
su alcance. No reconstruyen datos raw, fuentes TRAIN ni expertos desde cero.
Los ZIP y fuentes grandes permanecen fuera de Git; las tablas, informes,
recibos y sus identidades pueden publicarse con el código.

Una vez terminado el supervisor de entrenamiento:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_io_recovery.finalize_continuation
```

Si únicamente queda la comprobación serial, usar el mismo comando con
`--resume-verification`: conserva el ZIP y los verificadores ya confirmados.
Estos comandos ejecutan cero updates de optimizador. No hay push automático.
