# V17: dependencia de recursos observada

Instantánea: 2026-10-12T12:58:53.677093+00:00. Consultar CAMPAIGN_STATUS para el estado actual.

SP1 conserva **4850/6000 updates** en un checkpoint durable. La evaluación del contraste G está cerrada hasta completar ambos endpoints. H tiene sus6000updates completos, pero su evaluación espera el orden registrado. I tiene implementación y pruebas CPU; sólo se activa si G y H son rechazadas.

| Medida en el rechazo observado | Bytes |
|---|---:|
| Margen de commit Windows | 944517120 |
| Mínimo del guard durante fit | 2147483648 |
| RAM física disponible | 4219772928 |
| RSS del proceso propio | 2646290432 |

El bloqueo observado fue el margen de memoria comprometida, aunque había RAM física disponible. La consulta al driver devolvió carga diferida de módulos, modo2. [NVIDIA documenta LAZY como predeterminado en estas versiones](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html). No se cambió la configuración global ni la de otros propietarios.

Las pausas preservan pesos, optimizador, scheduler, RNG y diario. El observador sólo registra el valor devuelto por el guard; no cambia su decisión. Su coste entra en los scopes normales de tiempo GPU. La auditoría de muestras reproduce la secuencia completa desde cada semilla y comprueba los prefijos retenidos.

## Reanudación real

```powershell
pwsh -NoProfile -File 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v17-representation-audit\scripts\Resume-ResearchV17.ps1'
```

El comando reutiliza fits completos y shards verificados, sigue la cola registrada y vuelve a detenerse si los recursos son insuficientes. Una reserva de presupuesto de GPU no reserva memoria del sistema. No se reducen cohortes, batch, precisión ni gates para sortear este bloqueo.

Las ramas pendientes no son resultados negativos ni trabajo terminado. No se afirma SOTA. Los resultados positivos parciales y los rechazos anteriores permanecen en los informes de A y de las demás colas.

## Resolución verificada

La pausa anterior es histórica. SP1 terminó sus6000updates tras liberarse memoria compartida, con la receta y el guard sin cambios. La reanudación observada pasó sus46 comprobaciones de recursos, sin rechazos. G y H completaron después sus evaluaciones. `RESOURCE_RESOLVED.json` conserva el hash final y la evidencia; `RESOURCE_DEPENDENCY.json` conserva la instantánea original. El estado actual de toda la campaña se regenera en CAMPAIGN_STATUS.md.
