# RGB-PORT: auditoría y estado de continuidad

Se versiona el código RGB-PORT con las correcciones descritas en
[ADR-0002](../decisions/ADR-0002-rgb-port-audit-corrections.md), sus pruebas y
los wrappers de aceleración, prefetch, concurrencia, CUDA Graphs y recuperación I/O.
La geometría de entrenamiento conserva BF16. Las optimizaciones experimentales
de streaming V12 están en su rama correspondiente, no incorporadas a estos fits.

Los recibos históricos están en `evidence/`; se mantienen sus bytes originales
y se enumeran en `evidence/SHA256SUMS.json`. No se publican datos, pesos ni
configuración local de vinculación de esta máquina. Para reconstruir la campaña
se requieren sus artefactos y raíces de datos; los hashes de rutas absolutas
describen el entorno auditado, no una instalación portable lista para reanudar.

## Estado al publicar

La comprobación CPU `evidence/V13_RESUME_PREFLIGHT.json` encuentra estados completos
en E_A5_MATCHED (26.338 updates) y E_C2F_MATCHED (17.679). Coinciden los hashes
nativos, contadores, identidad y pasos Adam, sin acumulación ni update pendiente.
Se conservan pesos, optimizador, scheduler, cursor y RNG. Esta comprobación no
equivale a restauración CUDA validada ni a entrenamiento terminado.

La reanudación posterior a test12 avanzó y volvió a pausarse. A5 tiene una
inconsistencia entre el checkpoint nativo y el recibo auxiliar de aceleración
tras guardar otra vez el mismo update. El error observado fue
`Immutable acceleration runtime snapshot differs`. La reparación general de ese
guardado y su recuperación auditable siguen pendientes; no se han desactivado
validadores ni se afirma que esta publicación resuelva ese problema.

## Verificación de publicación

Con el entorno Python operativo y `PYTHONPATH=src;.`: 148 pruebas aprobadas y
una prueba técnica opt-in omitida (consume updates). Ruff pasa. Se conserva
también el intento inicial con dos fallos de importación en subprocesos por
falta de PYTHONPATH; el intento con el entorno correcto pasa sin cambiar el código.
La auditoría original conserva sus propios recibos históricos de Ruff/Pyright.
