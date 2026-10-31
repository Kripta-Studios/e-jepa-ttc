# 12 — Backlog implementable y primera iteración

## 1. Alcance de ejecución

La primera iteración termina con un diagnóstico y un protocolo sellado suficientemente precisos para escoger un piloto. No termina simplemente al «tener una idea» ni exige lanzar entrenamiento para justificar el trabajo.

Partir del commit revisado `30b28b5c14eb9ed60067ed2f1dc262d7b60f2654`, leer el estado actual del checkout y trabajar en una rama nueva. El nombre sugerido es `research/evttc-transfer-v2`; es una propuesta, no una rama creada por este paquete. Si el repositorio ha avanzado, registrar el SHA real y revisar los cambios relevantes antes de aplicar estas instrucciones.

Las tareas y rutas que siguen son **propuestas**, salvo los lectores, runtimes y scripts señalados expresamente como existentes. Las APIs y comandos completos se especifican en [14](14_CONTRATOS_Y_COMANDOS_PROPUESTOS.md).

## 2. Tickets de la primera iteración

### T00 — Punto de partida y entorno

**Entrada:** checkout, candidato original y recibos disponibles en la máquina de ejecución.

- [ ] Registrar `git status`, SHA, plataforma y lockfile sin alterar trabajo ajeno.
- [ ] Localizar `FINAL_CANDIDATE` a partir de recibos y rutas de configuración; comprobar hashes y archivos esperados.
- [ ] Verificar el ZIP original, ledger y freeze mediante el verificador existente, usando un directorio de salida distinto para cualquier evidencia nueva.
- [ ] Crear un entorno aislado con las dependencias del proyecto. Resolver el problema histórico del `.pth` si se reproduce; no usar el workaround de análisis como prueba de un entorno de entrenamiento sano.
- [ ] Comprobar almacenamiento y una lectura pequeña. Medir GPU solo en la máquina donde se ejecutará el piloto.

**Entregable propuesto:** `run_root/preflight.json` y un apartado de procedencia en el handoff. **Cierre:** todos los recursos necesarios están identificados o las ausencias tienen una consecuencia explícita. No es necesario descargar otra copia completa de EvTTC.

### T01 — Auditoría de identidad reproducible

**Rutas propuestas:** `operational/evttc_transfer_audit/identity.py`, `scripts/audit_evttc_transfer.py`, `tests/unit/test_evttc_transfer_identity.py`.

- [ ] Implementar un matcher que no use TTC y lea frames de forma acotada.
- [ ] Exigir correspondencia única; timestamps periódicos solo generan candidatos.
- [ ] Comparar todos los píxeles y hashes del tramo candidato.
- [ ] Añadir fixtures con offset, imágenes distintas, alteración de un píxel y duplicación de fuentes.
- [ ] Reproducir las ocho correspondencias, los índices iniciales y los 365 hashes del anexo.
- [ ] Producir JSON portable y comparación semántica con el recibo previo, sin exigir igualdad de campos como fecha o ruta.

**Cierre:** G0/G1 tienen una evidencia de solapamiento reproducible; ausencia o ambigüedad devuelve fallo explícito. **Commit sugerido:** `feat(audit): reproduce EvTTC frame overlap`.

### T02 — Genealogía y exposición

**Rutas propuestas:** `operational/evttc_transfer_audit/lineage.py`, `configs/protocol/evttc_transfer_v2.yaml`, `docs/evttc_transfer_v2/LINEAGE.md`.

- [ ] Enumerar los nueve componentes del candidato real, no los del modelo histórico que aparezca primero en un README.
- [ ] Registrar padres, scopes de entrenamiento, selección de checkpoints, normalizadores y cualquier teacher.
- [ ] Para cada grabación/grupo, registrar uso histórico en entrenamiento, elección de arquitectura, consulta de scores, inspección de targets y estado desconocido.
- [ ] No inferir ausencia de exposición a partir de la palabra `frozen`.
- [ ] Resolver qué afirmación permite cada componente: procedencia auditada, limitada o desconocida.

**Cierre:** tabla componente→datos→decisiones completa en su estado de conocimiento. Puede contener `unknown` justificado; no se promueve a «zero-shot limpio» mientras permanezca una incertidumbre material.

### T03 — Grupos y población de desarrollo

**Rutas propuestas:** `operational/evttc_transfer_audit/protocol.py`, `data/splits/evttc_transfer_v2.yaml`, `tests/unit/test_evttc_transfer_protocol.py`.

- [ ] Excluir las ocho grabaciones completas y sus cámaras/aliases; ampliar exclusiones si aparecen adquisiciones relacionadas que invaliden el supuesto de independencia.
- [ ] Revisar las 24 candidatas: no usar `split_group = sequence_id` como certificación automática.
- [ ] Contar grupos efectivos, familias, sensores y soporte de TTC sin elegir por rendimiento de un candidato.
- [ ] Decidir holdout + validación agrupada, CV agrupada con confirmación externa o estudio preliminar, según la población real.
- [ ] Sellar una propuesta de roles. Una modificación posterior exige nueva versión y motivo ajeno a obtener un score más favorable.
- [ ] Abrir en paralelo la búsqueda de Slider independiente y documentar disponibilidad, permiso de uso y protocolo de captura/evaluación.

**Cierre:** G1 pasa para los dominios soportados. La falta de Slider no paraliza carretera; permanece como limitación de la afirmación final.

### T04 — Contrato del evaluador

**Rutas propuestas:** `targets.py`, `scoring.py` en `operational/evttc_transfer_audit/`, y `tests/unit/test_evttc_transfer_targets.py`.

- [ ] Definir reloj, unidad, instante de referencia y geometría de TTC.
- [ ] Implementar cobertura sin extrapolar las 11 consultas de carretera fuera de soporte.
- [ ] Añadir fixtures de interpolación interior, GT duplicado/no monótono, GT cero, signos, no finitos y población parcialmente etiquetada.
- [ ] Probar el ejemplo de expansión que distingue TTC al inicio y al final.
- [ ] Mantener separadas predicción bruta, transformación de scorer documentada y métricas diagnósticas.
- [ ] Obtener, cuando sea posible, especificación del organizador y reproducir sus filas con las predicciones ya guardadas. No hace falta otra inferencia para iniciar esa comprobación.

**Cierre:** G2 pasa para desarrollo. La reproducción exacta oficial se marca pendiente si faltan los targets o la política completa.

### T05 — Baselines con cobertura visible

- [ ] Puntuar primero las predicciones existentes que tengan targets compatibles.
- [ ] Evaluar PHASE original y Garl completo/eventos/RGB en la intersección correcta, además de mostrar cobertura de toda la población.
- [ ] Preparar referencia geométrica y ramas RGB/eventos de PHASE para desarrollo.
- [ ] Registrar modalidad, cámara, ROI, historia causal, referencia temporal y coste de cada método.
- [ ] No seleccionar solo una parte favorable de las secuencias publicadas para afirmar superioridad.

**Entregable:** tabla de baselines y contrato por método. **Cierre:** se puede explicar qué diferencia una comparación controlada de una tabla de literatura.

### T06 — Trazas y fixtures físicos

**Rutas propuestas:** `operational/evttc_transfer_audit/diagnostics.py` y `tests/integration/test_evttc_transfer_diagnostics.py`.

- [ ] Registrar observaciones aceptadas y rechazadas, deltas reales, ROI, normalización, fases, estados, corrección final y saturación.
- [ ] Demostrar que activar trazas no cambia la predicción del candidato.
- [ ] Ejecutar pruebas de expansión conocida, traslado global de clock, prefijo corto, silencio registrado, soporte ausente y modificación de datos futuros.
- [ ] Revisar geometría RGB–eventos con una muestra fijada por índices/soporte, no por error TTC.
- [ ] Repetir la política de análisis en desarrollo; marcar inspecciones de competición como expuestas.

**Cierre:** G3: tabla hipótesis/evidencia/refutación y elección de un piloto o de adquisición de datos.

### T07 — Decisión de piloto

**Entregable propuesto:** `docs/evttc_transfer_v2/PILOT_DECISION.md`.

Debe contener una respuesta concreta:

1. Qué contrato ha fallado o qué componente conserva el sesgo.
2. Qué intervención cambia exactamente ese mecanismo.
3. Qué control la podría refutar.
4. Qué población elegirá la variante y qué población confirmará el resultado.
5. Qué se medirá además de error medio: cobertura, TTC corto, negativos cuando existan, peor grupo, retardo y coste.
6. Cuántos candidatos y cuánto cómputo se permiten tras el piloto de throughput.

Si no se conoce la causa, el resultado válido es «ninguna intervención justificada todavía», con la siguiente observación específica que reduciría esa incertidumbre. No se sustituye esa decisión por una búsqueda general de redes.

## 3. Segunda iteración: solo la rama elegida

| Si el diagnóstico indica… | Ejecutar… | No abrir todavía… |
|---|---|---|
| Error físico verificable | B0/B1 y regresión, documentados en [06](06_INTERVENCIONES_MINIMAS.md) | Fine-tuning para absorber el error. |
| Arranque/innovación temporal | T0 y prueba de retardo, con mismo soporte | Filtros bidireccionales o recortes de warmup del score. |
| Productores útiles pero fusión sesgada | F0/F1; después F2/Fcap si procede, según [07](07_FUSION_Y_FINE_TUNING.md) | Reentrenamiento completo de nueve componentes. |
| Productor sin señal suficiente | P0 y probes; después J0/J1/J2, según [08](08_JEPA_Y_ARQUITECTURAS_FUTURAS.md) | Teacher gigante sin evidencia de utilidad para TTC. |
| Falta de datos Slider independientes | Adquisición/solicitud de datos; fixtures y desarrollo de carretera mientras tanto | Declaración de generalización a Slider. |

## 4. Organización temporal propuesta

Son bloques de trabajo, no promesas de duración ni costes medidos:

| Bloque | Trabajo principal | Puede avanzar en paralelo |
|---|---|---|
| A | T00/T01/T02 | Revisión bibliográfica y lista de preguntas de protocolo. |
| B | T03/T04 | Buscar adquisiciones externas y verificar disponibilidad de eAP. |
| C | T05/T06 | Perfilar lectura y almacenamiento sin alterar el candidato. |
| D | T07 y piloto seleccionado | Preparar plantillas de tablas y reporte. |
| E | Confirmación y evaluación externa | Optimización de runtime que preserve la semántica. |

No estimar días de entrenamiento a partir del tamaño del checkpoint. Medir carga, preparación y throughput antes de reservar un presupuesto. No hay gasto cloud comprometido por este plan.

## 5. Prompt preparado para iniciar la implementación

```text
Continúa PHASE-TTC desde el checkout actual. Lee AGENTS.md, el plan de
2026-10-30 y el paquete Ruta SOTA. Registra SHA y estado antes de editar.
Implementa únicamente T00–T03 de 12_BACKLOG_Y_PRIMERA_ITERACION.md:
preflight, auditoría reproducible de solapamiento, genealogía y protocolo
de grupos/exposición. Reutiliza lectores existentes. No cambies pesos,
normalizadores ni predicciones originales. No entrenes ni envíes resultados.

Primero identifica los recursos reales; no asumas que las rutas históricas
siguen disponibles. Para cada contrato nuevo añade un test que falle por
la conducta incorrecta que se quiere evitar. Reproduce las ocho coincidencias
y los 365 frames si están los datos. Si no están, entrega CLI probado con
fixtures y deja la reproducción real pendiente, sin afirmar que ocurrió.

No certifiques independencia por nombres distintos. Registra la exposición
histórica de las 24 candidatas y de los nueve componentes. Produce una sola
propuesta de roles basada en metadata y soporte, nunca en scores de candidatos.
Conserva los estados desconocidos. Termina con archivos, comandos ejecutados,
pruebas, recibos y la decisión concreta de si puede abrirse T04.
```

Este prompt es texto para una ejecución futura. No significa que este paquete haya ejecutado las tareas.

## 6. Preguntas técnicas que debe resolver el organizador

Preparar una ficha revisable con estas preguntas, usando los IDs de las consultas afectadas. Enviar la consulta cuando el autor lo indique:

- ¿Qué instante y referencia geométrica tiene cada TTC evaluado? ¿Centro óptico, posición del vehículo, superficie del objeto u otra convención?
- ¿Cuáles son el reloj, offset e interpolación exactos para los targets de carretera y Slider?
- ¿Cuál es la política completa de negativos y de valores ausentes, incluidos bordes y queries consecutivas no válidas?
- ¿Las 434 filas constituyen la población requerida? ¿Cómo se calcula la agregación y el runtime reportado?
- ¿Qué uso de datos públicos de EvTTC para entrenamiento/adaptación está permitido y cómo debe declararse?
- ¿Existen adquisiciones Slider adicionales e independientes o una evaluación externa que no haya sido inspeccionada durante el diseño?

Las respuestas faltantes se registran como campos pendientes de contrato. No hace falta detener el desarrollo independiente de carretera mientras llegan.

## 7. Qué debe entregar cada iteración

Una carpeta nueva con configuración resuelta, identidades, resultados generados, registro de fallos y handoff. El resumen debe decir qué se aprendió y qué decisión permite, enlazando el recibo que lo demuestra. Un porcentaje aislado, una captura o una frase «tests pasan» no sustituyen esa evidencia.

Base del backlog: [plan original](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md), [documentación del adaptador](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/evttc_competition.md) y revisión estática del código. No se modificaron esos archivos al preparar el ZIP.
