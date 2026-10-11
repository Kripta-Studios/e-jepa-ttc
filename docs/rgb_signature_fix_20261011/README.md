# V13: reparación de identidad RGB y recuperación de la cola

El 11 de octubre de 2026, `H_R_RAW` falló antes de producir sus bloques de
inferencia. `_observation_signature` entregaba arrays NumPy procedentes de Parquet
a `json.dumps`, que no admite `ndarray`. La comprobación se ejecuta cuando una
observación aparece en varias historias. La cola continuó las tareas independientes
de eventos y terminó con `TERMINAL_WITH_BLOCKERS`.

## Corrección

[signature.py](../../operational/rgb_port_signature_fix/signature.py) convierte
recursivamente arrays y escalares NumPy en listas y escalares nativos, exclusivamente
en los siete campos de identidad que ya utilizaba la función original. A continuación
invoca el hash original. No convierte timestamps enteros a floats, no redondea las
coordenadas, no modifica las filas y no añade etiquetas ni IDs de consulta al hash.
Las entradas nativas conservan sus hashes. NaN, infinito y objetos no serializables
siguen fallando, y los cambios de ROI, frames o relojes siguen detectándose.

[infer.py](../../operational/rgb_port_signature_fix/infer.py) aplica esa conversión
al ejecutar la ruta de inferencia existente, conservando su validador de máscaras.
[queue.py](../../operational/rgb_port_signature_fix/queue.py) dirige las extracciones
H/V por esa ruta y conserva los módulos de entrenamiento. La reparación no requiere
reiniciar productores o PAIR ni cambiar BF16, pérdidas, datos o estados de optimizador.

## Evidencia

[EVIDENCE.json](EVIDENCE.json) contiene la reproducción del error original, los
recuentos de la auditoría real, los hashes de código/evidencia y el resultado QA.
Se recorrieron todos los historiales RGB de H y V sin decodificar imágenes, puntuar
predicciones o utilizar sus etiquetas. Todas las identidades repetidas conservaron
las mismas entradas. Las pruebas cubren equivalencia de hashes, timestamps superiores
a `2**53`, arrays anidados, ausencia de mutaciones, cambios en cada campo, rechazo de
valores inválidos, rutas de ejecución y recuperación selectiva de dependencias.

Los 40 archivos protegidos de los ocho fits completados —checkpoint, puntero, recibo,
journal y freeze final— conservaban sus hashes tras la QA. Los resultados completos
y la copia del estado anterior permanecen en `E:/EJEPA_results/rgb_signature_fix_20261011`.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path + ';.'
python -B -m pytest tests/test_rgb_port_signature_fix.py tests/test_rgb_port_streaming.py tests/test_rgb_port_continuity.py
python -B -m operational.rgb_port_signature_fix.audit --run artifacts/rgb_port_20261008 --output E:/EJEPA_results/rgb_signature_fix_20261011/REAL_OBSERVATION_REPLAY.json
```

## Continuidad y recuperación

`RGB_SIGNATURE_FREEZE.json` registra los archivos añadidos, las pruebas y la evidencia
QA y depende del freeze RGB overlap anterior. Conserva y verifica las congelaciones
históricas. El cargador comprueba que el código admitido coincide con Git.

[recovery.py](../../operational/rgb_port_signature_fix/recovery.py) prepara una copia
del estado: reactiva `H_R_RAW` y únicamente sus descendientes `SKIPPED_DEPENDENCY`.
Conserva intentos, contadores y tareas completas; rechaza una recuperación mientras
exista una tarea marcada como activa. Antes de aplicar la copia se comprueba que no
hay supervisor ni tareas vivos y se conserva el estado anterior y el log del fallo.
La cola continúa con un segundo intento y conserva el primer log.

## Reanudación verificada

[RESUME_VERIFICATION.json](RESUME_VERIFICATION.json) registra el segundo intento de
H_R_RAW a las 20:09:59 CEST: 128 de 18.615 observaciones escritas, proceso de
inferencia y supervisor activos, sin marcador de pausa ni tareas en FAILED o
SKIPPED_DEPENDENCY. Las 23 tareas reactivadas incluyen el fallo y sus descendientes;
las 19 tareas ya completas conservaron su estado. Los ocho fits completados
mantienen sus 40 archivos protegidos exactamente iguales. El primer log del fallo
permanece junto al log del segundo intento.

Esta es una observación registrada, no un monitor que se actualice automáticamente.
El arranque pasó por la ruta corregida y produjo bloques reales. Las fases posteriores
quedan en la cola; su finalización y las métricas TTC todavía están pendientes.
La validación de metadatos no sustituye esas métricas ni demuestra SOTA.
