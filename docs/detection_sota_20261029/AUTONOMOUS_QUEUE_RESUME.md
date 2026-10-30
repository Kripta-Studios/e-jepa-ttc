# Continuación tras la pausa del 30 de octubre

El usuario detuvo la campaña a las 08:55 y posteriormente pidió «reanuda todo».
R0 y R1 habían completado sus 8192 actualizaciones y ambas evaluaciones completas.
R2 conservaba un checkpoint completo en 5376 y el seleccionado en 5120. Los 95
updates registrados después de 5376 se conservan en el historial interrumpido y
se repetirán desde el checkpoint; no son avance acumulado adicional.

`scripts/resume_detection_autonomous_queue.py` prepara y ejecuta una continuación
en un directorio nuevo. No cambia el productor, las configuraciones ni el
supervisor histórico. El plan liga por SHA-256 sus fuentes, el plan anterior y
el manifiesto de pausa. La exclusión de procesos y el OWNER exclusivo impiden
relanzar accidentalmente la misma campaña.

La continuación verifica R0 y R1 y los adopta como terminados. R2 recupera pesos,
optimizador, scheduler, sampler y RNG mediante el loader original. Su evaluación
4096 procede del padre, ligada por hash; la 8192 procede de la ejecución nueva.
Ambas pasan el verificador original y la comparación de poblaciones. No se
fabrica un summary para el segmento interrumpido.

Después se conserva la selección R0/R1/R2 y la confirmación del ganador frente a
R0 con semillas 13 y 23. El alcance sigue siendo la campaña RGB; las fases
posteriores continúan identificadas como pendientes en el plan original.

El supervisor se lanza con OMP_NUM_THREADS, MKL_NUM_THREADS y
OPENBLAS_NUM_THREADS a 1. Conserva los gates de 8 GiB de RAM/VRAM y espera hasta
24 horas si no se cumplen. No consume tokens ni hace llamadas a modelos. Los
procesos se lanzan sin ventanas y el supervisor evita la suspensión por
inactividad mientras está activo.

Validación previa: siete pruebas de cola/continuación aprobadas, Ruff limpio,
Pyright sin errores y revisión independiente sin hallazgos materiales.

El plan de ejecución se prepara con `prepare --pause RESUME.json --output DIR`;
se arranca con `run --plan DIR/plan.json --plan-sha256 SHA`. La preparación no
entrena. El directorio de salida de R2 y el del supervisor deben ser nuevos.
El marcador STOP_AFTER_CURRENT pertenece al nuevo directorio del supervisor;
el marcador histórico se conserva como evidencia de la pausa anterior.
