# Recursos, recuperación y entrega

## Coordinación

Una sesión nueva de código en worktree separado. R1 y las campañas anteriores son fuentes de sólo lectura. No iniciar otra copia del supervisor. Los contadores de snapshots no son telemetría viva.

Si R1 usa GPU: primero terminar su medición o acordar una pausa cooperativa mediante su propietario y mecanismo registrado, guardando todos los pares. No matar ni resetear la medición. Preparación y tests CPU pequeños pueden avanzar con cuota agregada; el entrenamiento necesita slot exclusivo del proyecto. No atribuir al port RGB una supuesta aceleración de R1.

Si V13 ya está entrenando en otra sesión: comunicar este añadido, preservar su endpoint recuperable y el protocolo activo. No sobreescribir sus archivos. Esta propuesta sustituye sólo el orden de trabajo NO iniciado, nunca un resultado ya producido.

## Límites de esta autorización local

Mantener 23 GB decimales de RSS agregado de árboles del proyecto, piso duro del host 2 GiB y reanudar con3 GiB. 10 GB decimales libres tras reservas y memoria comprometida Windows monitorizada. Concurrencia CPU inicialmente4 hilos numéricos, hasta4 workers de preparación y buffers acotados. Escalar workers sólo si se mide y sigue cabiendo. No Runpod, API pagada, cambio de driver, paging o Control Center.

Estimar caché desde una muestra TRAIN y reservar disco antes de materializar. Guardar RGB uint8 admite reconstrucción exacta antes de normalización; los eventos interpolados FP32 no deben cuantizarse silenciosamente. Mantener caché deduplicada por identidad y recuperar fragmentos, no cientos de MB por update.

## Durable significa recuperable

Checkpoint cada100 updates y en pausa cooperativa, con modelo, optimizer, scheduler, RNG, sampler, epoch/batch position, config e identidades. Ledger separa guardado, físico repetido e incierto. No asignar al cierre los updates de otra campaña.

Análisis también se guarda por grupos de draws/productos. Comprobar hashes completos al admitir y en publicación; detector de cambios barato durante el bucle. No `git rev-parse`, escaneo total o rehash de datasets en cada update/draw.

Un hash diferente de la fuente científica detiene la rama afectada. Un nombre de archivo visto o una caída transitoria de disco no es un resultado negativo ni contamina mágicamente pesos. Registrar lo observado; no usar información de evaluación no autorizada para cambiar recetas. Las excepciones de acceso conocidas no se borran.

## Comandos reales de la integración

El agente debe implementar y verificar, no imprimir como si ya existieran:

```powershell
& $Python -m operational.rgb_port.run preflight --config configs/rgb_port/execution.json
& $Python -m operational.rgb_port.run execute --config configs/rgb_port/execution.json
& $Python -m operational.rgb_port.run resume --run artifacts/rgb_port_20261008
& $Python -m operational.rgb_port.run package --run artifacts/rgb_port_20261008
```

Los comandos del preparador y los tests de reference sí existen en este handoff. Los módulos operational anteriores son entregables del agente, no una promesa de que el repo ya los contiene.

## Entregables

- `RGB_PORT_FINAL_REPORT.md`: configuraciones, qué se entrenó, nueva evidencia y límites.
- `NEXT_DECISION_RGB_PORT.json`: estado de cada rama; no convertir blocked/incomplete en negativo.
- `RGB_PORT_DIFF.json`, `SPLIT_MANIFEST.json`, `TRAINING_COMMIT.json`, `FRAME_AVAILABILITY.json`.
- `MODEL_REGISTRY.json`: encoders P, PAIR P, heads H, datos/supervisión y exclusiones.
- Predicciones por query con target_status, availability, anchor y contrato de salida; normalizadores y pesos.
- Curvas completas, contabilidad, tests, fallos y comandos reales de resume.
- Tablas de coste según alcance, con una cabeza canónica y referencia Garl nativa conservada.
- ZIP esencial, SHA256 externo y extracción/reproducción independiente desde el alcance incluido.

Los artefactos finales permanecen locales; no push, merge, email automático ni submission. La réplica completa/refit/confirmación se describe como la siguiente campaña tras datos favorables, no se ejecuta sin nueva autorización.
