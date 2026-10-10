# Nueva campaña: contexto eficiente y comparación Garl

Continúa Kripta-Studios/e-jepa-ttc sin reabrir T6 ni repetir los 24 fits nocturnos.
Lee README.md, execution_policy.json y docs/01 a docs/08 de este handoff. Este
mensaje autoriza una campaña nueva; no modifica los freezes históricos.

Referencia: tag `simplex-t-local-results-20261004-complete`, commit
`fd16d8102914537653622d3abf298b753120ea43`.

El objetivo es implementar, probar, ejecutar y entregar tres trabajos:
E1, acelerar raw/ROI con pesos congelados y paridad;
E2, probar H8-WIDE para separar alcance y densidad;
E3, obtener comparadores Garl de procedencia correcta y contexto declarado.
No termines tras escribir documentación o imprimir comandos si hay tareas viables.

E0: consulta el estado real de HEAD, worktree, procesos/roles, Stage70 y fuentes.
Conserva cambios ajenos y el último estado operativo válido. No hagas reset, clean,
push o submissions. Usa el worktree separado preparado o crea uno preservando las
fuentes históricas. Vincula los archivos de release y sus hashes; reutiliza los ya
verificados. Los bundles no contienen todos los datos TRAIN/raw ni productores:
comprueba esos requisitos localmente. No repitas la reconstrucción completa de T6. Registra y calcula el control
EWMA_TRANSPORT_CV_H8 descrito en docs/02, con cero updates y sólo cuando el anchor
de sus observaciones esté verificado. No reescribas la EWMA histórica.

E1: sigue docs/03_RUNTIME.md. La unión raw, los handles persistentes y bincount ya
existen. La deduplicación intraconsulta de ventanas exactas es casi nula. Prioriza
el mapeo ROI una vez, views/preasignación, ingestión acotada, dispatch sin padding
innecesario y transferencias agrupadas. Mide los regímenes separados R0/R1/R2.
No ocultes el coste de ingestión ni la GPU compartida. Todo cambio de backend debe
pasar paridad antes de presentarse como optimización del sistema congelado.

E2: autorizo TPR-D1-H8WIDE-C160 seed 7, folds 0/1/2, 2.500 updates cada uno.
Slots fijos [0,2,4,6,9,11,13,15] del contexto H16. CPU FP32, batch 128, GRU160,
receta original y lambda_cost=0.01. Congela los tres endpoints antes de OLD_DEV.
Si cumple la regla prospectiva de docs/04, ejecuta las seis réplicas seeds 13/23.
Máximo 22.500 updates científicos WIDE. No otras ventanas, backbones o sweeps.

E3: recupera primero baselines/Stage70 manifestados. Si faltan contratos, conserva
los números históricos como descriptivos sin bloquear E1/E2. Autorizo hasta 12
productores Garl event-only nuevos donde falten productores admisibles: 3 outer y
9 inner, seed 7, 50 épocas finales de receta source-style sobre D1 y exclusiones
correctas. Máximo conjunto 200.000 updates. Calcula el trabajo y perfila memoria
antes del primer fit. Cambiar microbatch con BatchNorm no equivale a batch 128:
declara la adaptación, no una réplica exacta. No uses checkpoints eAP preentrenados
que hayan visto los holdouts. Autoriza además hasta 6 cabezas Garl-H1/H8, 2.500
updates cada una, con inputs INNER-OOF:15.000 updates. Sigue docs/05_GARL.md.
No debilites Garl reduciendo resolución, capacidad, canales o épocas para que quepa.
Si la rama no es viable por fuentes, VRAM o presupuesto, bloquea sólo esa rama.

Techo científico 237.500; técnico sintético 200; reserva de recuperación 2.300;
techo físico 240.000. Son límites, no una obligación de consumirlos ni permiso
para nuevos brazos. Mantén 2 GiB de RAM disponible, 4 GiB de RSS del árbol, 10 GB
libres tras reservas, hasta 10 GB de artefactos propios, 4 threads e interop2.
Un único trainer pesado. No interfieras con Stage70. E: y fuentes históricas
permanecen en sólo lectura. No reconfigures globalmente el equipo.

No abras Stage76, public validation, private test, EvTTC test ni CodaBench.
No adaptes el diseño a OLD_DEV. No abras ahora LATENT, MotionJEPA, FAR, PixelUMM,
REACT o un predictor de contacto 3D. Documenta esas hipótesis y realiza únicamente
la auditoría semántica TRAIN acotada descrita, sin convertir GT en input.
No canceles trabajo independiente por metadata irrelevante; tampoco reclasifiques
como metadata una exposición ambigua sin evidencia.

Implementa el runner real, sus pruebas y el protocolo congelado. Ejecuta la cola
en esta sesión mientras el entorno lo permita. Registra progreso en endpoints,
updates y solicitudes persistidas. Checkpoints completos cada 100 updates; análisis
y publicación por fragmentos. Ante E:/RAM transitorios, espera con supervisor ligero
y avanza otra tarea independiente. No relances errores idénticos cada 30 segundos.
Si todo queda bloqueado, entrega la dependencia exacta y el comando de reanudación.

Termina con FINAL_REPORT.md, NEXT_DECISION.json, comparación de precisión/coste,
contabilidad, contratos de comparadores, pruebas, paridad, comandos reales, pesos,
predicciones y ZIP esencial regenerado con SHA256. Distingue utilidad OLD_DEV,
mecanismo temporal, réplicas de encoders/cabezas, replay/servicio y confirmación.
H8 mantiene su identidad histórica; H16 no se borra ni WIDE se promueve por una
cifra aislada. El objetivo es el TTC firmado del benchmark, no contacto ni AEB.
