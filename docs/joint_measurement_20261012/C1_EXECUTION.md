# Continuación C1: estado al 14 de octubre, 11:46 Madrid

La comparación está registrada en
`artifacts/joint_measurement_20261012/adaptive/C1_geometry/REGISTRATION.json`
(SHA-256 `77d12a005b337039b47f3c153b1341113de8b484a777f9b4ab42e685aa731db6`).
La señal registrada de correspondencias temporales P habilita familia C. No es
una prueba de precisión del movimiento ni un candidato TTC.

Se entrenarán 12.000 updates de `PAIR-CORR-GEOMETRY-S7`, con encoder adaptable
y lectura explícita de escala y traslación. Su control es el endpoint primario
`PAIR-CORR-S7`, ya completo y admitido por hash, configuración y población.
Reutilizarlo no añade otros 12.000 updates a la contabilidad. La diferencia de
capacidad de 517 parámetros está declarada. No se ha ejecutado ningún update
científico de C1 en esta instantánea.

La QA CPU de reanudación completó 20 updates sintéticos y verificó nueve estados
exactos entre diez updates continuos y cinco más restauración más cinco. El
export del checkpoint sintético también pasó; el fallo inicial de trazado está
preservado y el adaptador portable no altera el modelo registrado. Falta la QA
CUDA de 20 updates, que debe pasar antes de entrenar en GPU.

## Cola implementada

El commit `de7907b` incorpora `geometry_run.py`, coste emparejado y comando de
reanudación. La cola ejecuta QA, scopes de entrenamiento con checkpoints,
congelación y verificación independiente del endpoint, H/P completos, recálculo
NumPy, coste, cuatro bootstraps por grupos y export portable. Conserva las cuatro
tablas y verificaciones históricas de los controles. Nunca abre V automáticamente.
El terminal `C1_READY_FOR_INTERPRETATION` deja pendiente la interpretación del
experimento y las demás ramas habilitadas del programa V18.

La ruta usa caché sensorial lossless limitada a 8 GiB, ZIP persistente, prefetch
y transferencia pinned. Recalcula las features con autograd. No tiene CUDA Graph
admitido para esta arquitectura: la prueba nativa anterior no acredita C1.

Las cuatro pruebas de orden de etapas y catorce pruebas compartidas de coste
pasaron antes del commit. Esto valida barreras y lógica; no constituye una
ejecución científica ni una medición de rendimiento de C1.

## Uso cuando RGB haya liberado la GPU

Desde este worktree, con una sola instancia del escritor C1:

```powershell
pwsh -NoProfile -File scripts/Resume-JointMeasurementGeometry.ps1
```

`-Once` permite avanzar un único scope o etapa acotada. La cola conserva el lock
del escritor, usa el propietario GPU compartido y no reinicia otros procesos.
RGB seguía vivo al comprobar su PID 34964 el 14 de octubre a las 11:43; en esta
instantánea no se ha lanzado C1 ni un segundo observador.

Con el PID real del futuro proceso Python C1, el observador admite:

```powershell
python -B -m operational.joint_measurement.hourly_observe --job geometry --pid PID
```

Usar el mismo entorno compartido y `PYTHONPATH=src;.` del script de reanudación.
El observador verifica módulo, directorio y fecha de creación del proceso,
espera a cada hora local en punto y escribe en `adaptive/C1_geometry/hourly`.
No entrena, reinicia ni modifica checkpoints. Su lock evita observadores
duplicados de la misma rama. El valor predeterminado sigue siendo RGB y su
observador ya activo continúa con el código que cargó al arrancar.

Cinco pruebas CPU del observador pasaron: separación de contadores por rama,
rechazo de PID reutilizado, terminal RGB/C1 sin confundirlo con un proceso vivo,
y detección de ausencia de progreso entre horas. Ruff y la ayuda CLI pasaron.
No se han medido todavía coste, métricas o ETA científica de C1.
