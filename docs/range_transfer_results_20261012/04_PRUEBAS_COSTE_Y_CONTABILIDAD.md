# 04. Pruebas, coste, contabilidad y límites

## Verificaciones realizadas

- 41 tests y 17 subtests de los módulos nuevos antes de iniciar los fits: masa, objetivo, paridad L0, dominio, gates, separación de roles, presupuesto y contratos de checkpoint. Conservo el stdout; su configuración quiet reduce el resumen a puntos.
- Ruff y Pyright sobre la integración; Pyright informó cero errores y advertencias del analizador. Su aviso de `.venv` local ausente no desencadenó una instalación: usé el entorno existente del proyecto.
- Prueba real CPU FP32 de continuidad: 10 updates sintéticos continuos frente a 5+restauración+5. Su consumo total fue 20 updates técnicos.
- Prueba independiente del adaptador diagnóstico PAIR: forma known por consulta, 133 features exactas y rechazo de intervalos/formas inválidos. Cero updates.
- Auditoría final de seis endpoints: SHA256, 2.500 entradas de curva, pesos finitos, fuentes científicas sin cambios, cohortes alineadas y hashes de predicciones.
- Inspección visual de curvas y gradientes; verificación SHA256 de las entradas del bundle.

No presento esto como ejecución de la suite completa del repositorio, una prueba de robustez bajo corrupciones, calibración probabilística ni validación de seguridad. No hubo un corte real durante un fit científico: la prueba de resume usa el fixture sintético con la misma maquinaria de entrenamiento/checkpoint.

En la preparación del push también ejecuté Ruff sobre todo el repositorio: informó 1.780 errores heredados y el formateador señaló 370 archivos. Comprobé que las rutas con errores pertenecen a archivos ya versionados que esta entrega no modifica. Los checks dirigidos a V14 y al generador pasan. No afirmo que la CI global esté verde ni reformateo el histórico para publicar estos resultados. [PUBLICATION_CHECKS.json](evidence/qa/PUBLICATION_CHECKS.json) registra la comprobación; los logs completos están en el bundle.

## Resume: comprobaciones exactas

| Estado comparado | Igualdad exacta |
| --- | --- |
| accumulation_index | PASS |
| completed_updates | PASS |
| draws | PASS |
| losses | PASS |
| model_state_dict | PASS |
| numpy_random_state | PASS |
| optimizer_state_dict | PASS |
| python_random_state | PASS |
| sampler_generator_state | PASS |
| scheduler_state_dict | PASS |
| torch_rng_state | PASS |

El modelo, AdamW, scheduler, sampler y RNG coinciden. Los hashes completos de los dos archivos checkpoint pueden diferir por identidad y metadatos de los runs; la prueba compara sus estados relevantes, draws y pérdidas.

## Contabilidad física

| Concepto | Consumido | Máximo |
| --- | --- | --- |
| Científicos | 15000 | 25000 |
| Técnicos sintéticos | 20 | 500 |
| Recuperación superior | 0 | 4500 |
| Pendientes inciertos | 0 | incluidos en el límite |
| Físicos superiores | 15020 | 30000 |

Cada fit tiene UPDATE_JOURNAL y checkpoint durable. La cota física suma científicos, técnicos, recuperación e incertidumbre. Los seis fits completaron su endpoint sin recuperación. La selección negativa cerró la cola; el presupuesto libre no se reasigna.

## Costes con scope

| Fit | CPU fit + persistencia s | Forward de 16432 consultas desde caché s |
| --- | --- | --- |
| RT-M0-L0-S7 | 215.056 | 2.465 |
| RT-M0-L1-S7 | 224.994 | 2.491 |
| RT-M1-L0-S7 | 216.165 | 2.439 |
| RT-M1-L1-S7 | 259.261 | 2.477 |
| RT-M2-L0-S7 | 215.256 | 2.491 |
| RT-M2-L1-S7 | 258.226 | 2.484 |

La suma de los seis tiempos de fit es 1388.957 s (23.149 min). Incluye trabajo de la cabeza en H y persistencia; no es el coste de entrenar productores. El reloj de la invocación principal registró 1462.807 s para diagnósticos y trabajo con cabezas desde caché. No sumo tiempos de procesos concurrentes como si fueran tiempo total de pared.

Los forwards V duran aproximadamente 2,44–2,49 s por 16.432 consultas en batch128. Son medidas de esa ejecución, con conversión de salidas. Excluyen productores, lectura sensorial, construcción del historial, preparación de features y servicios de streaming. Dividir por el número de consultas no produce una latencia E2E ni una latencia batch1 comparable con Garl.

La traza H adaptada registra 32,581 s de CPU para caché sensorial, productores e instrumentación. La traza histórica V12 registra 679,829 s con preparación cruda y captura. Tienen scopes distintos; no se usan para una clasificación de velocidad entre sistemas.

## Recursos y protección de V13

V14 usa CPU FP32 y un thread numérico en el runner. El guard comprueba memoria disponible, memoria comprometida Windows, RSS del proyecto, árbol de procesos propio y disco. La política fija reserva dura de 2 GiB, reentrada con 3 GiB, 23 GB de RSS de proyecto, 4 GiB para V14 y 10 GB libres después de reservas. Los snapshots y fallos previos por recursos quedan en el bundle. No cambié drivers, paging, energía ni afinidades.

## Fallos conservados y alcance de las correcciones

Conservo reintentos históricos por recursos/imports, fallos transitorios de escritura Windows y el rechazo de forma known. Para los JSON de V14 reutilicé las primitivas nativas de lectura compartida y reemplazo atómico con reintentos limitados. La prueba sintética y los journals permiten auditar la continuidad; ninguno de esos diagnósticos agregó updates científicos.

El adaptador H separado resuelve la extracción diagnóstica V14 y deja el error nativo V13 registrado. No afirmo haber reparado el Dev32 de otra campaña. La ausencia de réplicas y transferencia nueva obedece al gate científico, no a ese error.

Fuentes: [contabilidad](evidence/ACCOUNTING.json), [resume](evidence/qa/RESUME_TEST.json), [auditoría final](evidence/SCIENTIFIC_DELIVERY_AUDIT.json), [coste](evidence/COST_SCOPE.json), [comandos ejecutados](evidence/COMMANDS_EXECUTED.md).
