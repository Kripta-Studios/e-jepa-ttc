# Plan de continuación para otro swarm

**Esperar instrucción de reanudación del usuario.** El objetivo está pausado por su salida, no bloqueado ni completado. El cierre añade código y evidencia para continuar sin repetir los entrenamientos.

## Primeros pasos

1. Leer este handoff y los capítulos científicos 03/04/07/08 del historial anterior. Comprobar rama, `git status` y último commit publicado.
2. Verificar hashes del candidato y originales. Confirmar que no hay benchmarks ajenos activos antes de reservar GPU.
3. Ejecutar tests focalizados y verificar el entorno correcto. Los dos xfail de transporte deben seguir visibles hasta que haya una decisión explícita sobre la opción experimental.
4. **Ejecutar la paridad de las 554 consultas** con el runner integral ya disponible. No inferir admisión desde tres pilotos. Mantener todos los fallos, los tres TTC y las cuarenta consultas sin GT.
5. Cuantificar la discrepancia `serial_parallel_exact=false` del caso de dos objetos agrupados. Distinguir variación de ejecución, drift por tamaño de batch y errores reales de buffers/streams.

## Reparto sugerido

| Agente | Propiedad exclusiva | Entrega útil |
|---|---|---|
| Coordinador | Interfaces, runners, protocolo, agenda GPU | Ensayos pareados y decisiones de admisión; nunca dos agentes midiendo GPU a la vez |
| Representaciones | `integral.py` y sus tests | Evaluar prefijos temporales acumulados; reducir tráfico de 120 buckets a fronteras sin overflow ni pérdida de cobertura |
| Productores | `producers.py` y sus tests | Layout/algoritmos de convolución, batch por objeto y coherencia de guards/lifetime |
| Transporte/compilador | `transport.py` y wrappers nuevos | Validar reutilización bidireccional; compilar transporte/core completo con gate TTC intacto |
| Servicio | Nuevo runtime multiobjeto y runner de replay | Compartir RAW, procesar todos los ROI/objetos y emitir todos los TTC sin dividir artificialmente la latencia |
| Auditor independiente | Solo lectura | Revisar fuentes ejecutadas, snapshots, paridad, alcance del reloj y evidencia de cola |

No repartir el mismo archivo entre implementadores. Antes de medir, congelar los módulos del proceso o ejecutar un snapshot aislado. El primer piloto ya falló por una edición simultánea de código. Una etiqueta de “consejo de agentes” no sustituye los resultados medidos.

## Prioridades de rendimiento

### 1. Reducir trabajo real, no solo lanzamientos

Los grafos actuales ya eliminan mucho overhead. Veinte objetos siguen costando unos 180 ms solo en productores preparados. La correlación compilada aislada solo mejoró alrededor de 1,25× frente a su grafo eager. Explorar fusión más amplia y el encoder; medir kernels y transferencias por separado, sin convertir un contador teórico en latencia.

Un probe pendiente es `channels_last` y selección cuDNN con FP32/TF32 off. Usar copias independientes de modelos: las vistas actuales comparten parámetros y cambiar su memory format puede mutar el layout de la referencia. Confirmar igualdad de valores de `state_dict` antes/después. No activar otras precisiones por omisión ni aflojar el gate para admitir una mejora.

### 2. Representación multiobjeto

El ring comparte ingestión, pero el query de veinte ROI escala casi linealmente. La alternativa de prefijos temporales está descrita en el capítulo de hallazgos; aún no hay implementación ni benchmark. Atender overflow durante sesiones largas, objetos que nacen/desaparecen, ROI fuera del canvas, gaps, paquetes vacíos y rollback.

### 3. Servicio completo

Falta contexto y cabeza vectorizados para veinte objetos. Los productores agrupados consumen `[80,3,12,128,128]` event y `[100,3,3,128,128]` RGB en el caso de cuatro/cinco observaciones por objeto. Respetar cada identidad y su disponibilidad; no compartir contexto de objetos distintos. Los datos multiobjeto usados hasta ahora son cargas artificiales declaradas, no veinte tracks reales.

La cohorte EvTTC original solo contiene el objeto primario. Definir explícitamente una población de veinte objetos/ROI distintos para capacidad y otra evidencia de tracks reales si se pretende esa afirmación. Prohibido duplicar una sola predicción veinte veces, reutilizar entradas idénticas y contarlas como veinte inferencias, o omitir los objetos lentos.

### 4. Tiempo sostenido

Primero validar offline la nueva población y después programar llegadas a `t0+k*20 ms`. El productor de observaciones debe ser independiente del consumidor. Registrar la llegada real además del deadline programado, el último TTC, backlog, descartes, errores y cambio de cantidad de objetos. Calentamiento e historial inicial se declaran aparte; no usar la última observación de una consulta para calcular su representación antes de iniciar su reloj.

El mínimo prospectivo del gate actual es 60 s, con veinte objetos por actualización. Si el dataset no permite una secuencia física de esa duración, declarar un ensayo de capacidad/replay y mantener fallado el gate físico. No reciclar clips como si fueran un sensor continuo sin documentarlo.

## Comprobaciones antes de afirmar éxito

- Misma identidad de candidato, pesos y normalizadores.
- Predicciones originales intactas y paridad completa con tolerancias ya fijadas.
- Cobertura de las 554 consultas originales y de todos los IDs del ensayo multiobjeto.
- Tiempo de actualización hasta el **último** de los veinte TTC, desde las observaciones necesarias.
- Media, p50, p95, p99 y máximo; trabajo pendiente sin acumulación sostenida.
- Separación explícita entre cold start, datos residentes, disco, cómputo preparado y servicio real.
- Ensayos negativos y limitaciones visibles.

El hardware actual puede resultar insuficiente para la arquitectura congelada y el objetivo de veinte objetos. Eso debe establecerse con perfiles y pruebas, no darse por hecho ni ocultarse cambiando el alcance. No hay autorización en esta sesión para comprar hardware o lanzar recursos de pago; sí se autorizó investigación local y publicación de este cierre.
