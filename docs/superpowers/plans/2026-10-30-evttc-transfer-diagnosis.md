# PHASE-TTC: diagnóstico y mejora de transferencia a EvTTC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. La ejecución propuesta es secuencial; este documento no solicita delegación automática.

**Goal:** identificar qué limita la transferencia de PHASE-TTC a EvTTC y demostrar cualquier mejora con grabaciones independientes, conservando la entrega oficial original.

**Architecture:** separar auditoría de identidad, protocolo de datos, evaluación, diagnóstico de entradas y experimentación. Mantener congelados los pesos durante las primeras fases. Introducir adaptación supervisada solo si el diagnóstico y la validación de desarrollo justifican ese paso.

**Tech Stack:** Python 3.11, PyTorch, NumPy, h5py, PyYAML, Pytest, Ruff; lectores y runtimes existentes de `e_jepa_ttc` y `operational/evttc_competition`. No se necesitan nuevas dependencias de entrenamiento para empezar.

**Spec:** el contrato de este plan está en las secciones 1–5 de este mismo documento, junto con [AGENTS.md](../../../AGENTS.md), [ADR-0043](../../decisions/ADR-0043-evttc-competition-sensor-adapter.md) y la evidencia enlazada abajo. Las decisiones propuestas no cambian las políticas de los runs históricos.

**Fecha:** 30 de octubre de 2026. **Base inspeccionada:** `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`.

**Estado:** auditoría inicial ejecutada; protocolo nuevo, herramientas reproducibles de esa auditoría y experimentos pendientes. La petición actual autoriza documentar y publicar este plan. Este commit no ejecuta los experimentos propuestos.

## Global Constraints

- Nombre público: **PHASE-TTC: Causal Event–RGB Fusion for Time-to-Contact Estimation**. Conservar `e_jepa_ttc`, IDs históricos, checkpoints y evidencias firmadas.
- No inventar resultados ni presentar desacuerdo entre modelos como error frente a ground truth.
- No elegir hiperparámetros, checkpoints, ventanas, reglas de rechazo o filtros usando el score de competición.
- No repartir ventanas de una misma grabación entre entrenamiento, selección y evaluación independiente.
- No usar ground truth de TTC, profundidad o velocidad para preparar entradas de inferencia.
- Registrar versiones, semilla, hashes de configuración, datos, modelo y fuentes de cada experimento.
- No incluir datos crudos, pesos, PDF de correspondencia privada ni rutas personales en Git.
- Mantener intactos el ZIP, las predicciones firmadas y el `RUN_FREEZE.json` de la entrega original.
- No extrapolar etiquetas ausentes ni rellenar predicciones faltantes para mejorar la cobertura aparente.
- Mantener causalidad: ninguna entrada, estado del filtro o normalización online puede usar el futuro.
- Un resultado sobre un subconjunto no sustituye al resultado oficial de 434 consultas.
- Cada modelo ajustado en EvTTC debe identificarse como una variante distinta del modelo congelado.

## Review Focus

1. **Recortes con clocks reiniciados:** un hash de archivo distinto no descarta imágenes compartidas; la tarea 1 compara timestamps y contenido.
2. **Otros sensores de una grabación excluida:** cambiar de cámara no produce una grabación independiente; la tarea 2 excluye el grupo completo.
3. **Etiquetas fuera de cobertura o con otra referencia geométrica:** la tarea 3 exige razones explícitas y prohíbe extrapolar.
4. **Arranque, silencio observado y soporte ausente:** la tarea 4 los distingue y prueba que datos futuros no cambian una salida pasada.
5. **TTC negativos y promedios incompatibles:** las tareas 3 y 6 conservan las salidas originales y separan score bruto, procesamiento del organizador y subconjuntos.

## 1. Decisión y alternativas

La vía elegida para este plan es **protocolo independiente → evaluación contrastada → diagnóstico con pesos congelados → intervención mínima → confirmación**.

| Vía considerada | Ventaja | Coste o limitación | Decisión |
|---|---|---|---|
| Diagnóstico y adaptación de entradas con pesos congelados | Aprovecha resultados existentes y permite aislar causas | Puede no resolver un sesgo aprendido | Primera vía |
| Fine-tuning de fusión o productores en EvTTC | Puede adaptar el modelo al dominio | Requiere datos independientes y cambia el alcance de la afirmación científica | Condicionado a fases previas |
| Adaptación durante inferencia | Puede usar datos sin etiquetas | Cambia estado/pesos y añade riesgos de deriva y causalidad | Pospuesta |
| Nueva arquitectura o detector 3D | Amplía el sistema | No responde directamente al error TTC con bbox suministradas | Fuera de este plan |

El objetivo no es reducir una cifra por cualquier medio. Es averiguar si el error procede de preparación de sensores, referencia temporal, productores, fusión o cambio de dominio, y evaluar una modificación que se pueda defender fuera de las consultas ya observadas.

Ya conocemos el resultado oficial. Toda investigación posterior debe declarar esa exposición. Conservar pesos no convierte automáticamente un adaptador elegido tras consultar el test en una evaluación ciega.

## 2. Evidencia disponible y límites

El [anexo JSON de auditoría](evidence/2026-10-30-evttc-audit.json) reúne resultados generados por lecturas locales, hashes por frame y métricas extraídas del PDF recibido. Sustituye rutas absolutas por nombres de archivo y conserva hashes de los recibos locales originales. No contiene imágenes, eventos ni etiquetas TTC por consulta.

Las comprobaciones iniciales se ejecutaron mediante scripts de análisis en la sesión; todavía no existe un CLI versionado que reproduzca toda la auditoría. La tarea 1 cierra esa limitación. El anexo es un registro de resultados, no una nueva firma del organizador ni una certificación independiente.

### 2.1. Recursos comprobados

- 32 HDF5 del manifest `data/manifests/evttc_all32_local.yaml` presentes, junto con 32 `ttc.csv`; aproximadamente 184,20 GiB de HDF5.
- Las diez secuencias de competición están disponibles localmente.
- El paquete congelado `FINAL_CANDIDATE` está presente. Presencia no equivale a revalidar todos sus componentes o ejecutar inferencia.
- GPU visible: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12.227 MiB de memoria total según `nvidia-smi`. No es una medición de memoria libre garantizada para un experimento.
- Las configuraciones históricas de eAP referencian una unidad `E:` que no estaba montada durante la auditoría. No se afirma que los datos eAP estén perdidos ni que se haya inventariado todo el equipo.
- No hay secuencias Slider de entrenamiento independiente en el manifest EvTTC-32 inspeccionado.

### 2.2. Solapamiento demostrado entre archivos locales

Para cada archivo de competición se leyeron `blackflys/right/ts` y `blackflys/right/data`. Se buscó un tramo consecutivo en las 32 grabaciones con un único desplazamiento constante de timestamps. Después se compararon todos los arrays RGB y sus SHA-256.

| Competición | Grabación original | Primer índice original, base cero | Desplazamiento original − competición (s) | Frames idénticos |
|---|---|---:|---:|---:|
| CCRs1-low | CCRs-1-low-100-overlap-100 | 188 | 9,380 | 50/50 |
| CCRs1-medium | CCRs-1-medium-100-overlap-100 | 228 | 11,380 | 43/43 |
| CCRs1-high | CCRs-1-high-100-overlap-100 | 401 | 20,028 | 30/30 |
| CCRs2-low | CCRs-2-low-100-overlap-100 | 193 | 9,630 | 46/46 |
| CCRs2-medium | CCRs-2-medium-100-overlap-100 | 178 | 8,880 | 55/55 |
| CCRs2-high | CCRs-2-high-100-overlap-100 | 289 | 14,430 | 59/59 |
| CCRm-low | CCRm-low-100-overlap-100 | 228 | 11,380 | 42/42 |
| CCRm-medium | CCRm-medium-100-overlap-100 | 406 | 20,280 | 40/40 |
| **Total** | | | | **365/365** |

Ejemplo: frame 0 de CCRs1-low y frame 188 del original comparten SHA-256 `140fa24accde68fd59dd2120310f90c9361154095eb621f4d184d1c2d0454765`. Los timestamps son 0,020266 s y 9,400266 s.

Esto demuestra contenido RGB compartido entre los archivos locales. No demuestra identidad de todos los eventos ni de los HDF5 completos, ni revalida los archivos actualmente alojados en el servidor remoto. Tampoco demuestra exposición del modelo TRAIN40 a esas grabaciones: la genealogía de entrenamiento es una comprobación distinta.

Las ocho grabaciones completas quedan excluidas de cualquier entrenamiento, selección y calibración supervisada de la futura variante. Quedan **24 candidatas**, no un split aprobado. Los splits históricos EvTTC-32 no se reutilizan como si fueran independientes de competición.

### 2.3. Resultado oficial recibido

Fuente: `PHASE-TTC_EvTTC_Evaluation.pdf`, suministrado por el autor tras recibirlo de los organizadores. El hash del PDF figura en el anexo. No se publica el PDF ni el correo privado en este commit.

| Secuencia | Consultas | Error relativo absoluto medio (%) | Tiempo medio informado (s/estimación) |
|---|---:|---:|---:|
| CCRs1-low | 50 | 7,57 | 1,982 |
| CCRs1-medium | 43 | 9,42 | 2,653 |
| CCRs1-high | 30 | 13,12 | 3,586 |
| CCRs2-low | 46 | 15,28 | 2,234 |
| CCRs2-medium | 55 | 9,34 | 2,534 |
| CCRs2-high | 59 | 7,99 | 3,006 |
| CCRm-low | 42 | 21,01 | 2,259 |
| CCRm-medium | 40 | 15,05 | 2,566 |
| Slider-750 | 51 | 132,00 | 6,192 |
| Slider-1000 | 18 | 156,88 | 9,157 |

Medias derivadas ponderando las cifras redondeadas por número de consultas: carretera **11,92 %**, Slider **138,49 %**, conjunto **32,04 %**. Son cálculos derivados, no nuevos scores oficiales. No son MiD.

Los organizadores indicaron que sustituyeron las dos predicciones negativas por interpolación lineal entre predicciones válidas circundantes. Los archivos originales conservan los signos. No se atribuye causalidad a esa interpolación posterior.

Los tiempos proceden del envío y no de una nueva medición del organizador. Coinciden con las medias del ledger dentro del redondeo de 0,001 s. Miden el adaptador offline: lectura, preparación y salida del modelo al host; excluyen carga inicial, indexación, calibración y hashing. No equivalen al runtime streaming optimizado.

### 2.4. Lo que indican los diagnósticos, sin atribuir causas

- **Slider-750:** las primeras ocho consultas contienen salidas extremas. Hay dos predicciones finales negativas, dos consultas con alguna observación de eventos marcada sin soporte y siete observaciones de eventos rechazadas por representabilidad matemática. Las siete observaciones no equivalen a siete consultas omitidas: se conservaron 51 predicciones finales.
- **Slider-1000:** 18 predicciones positivas, mínimo 1,0713758469 s y máximo 1,9406098127 s; cero observaciones de eventos rechazadas y cero consultas con observaciones marcadas sin soporte. Un error del 156,88 % puede existir sin picos enormes.
- No se ha demostrado que calibración, cadencia, arranque o fusión sea la causa dominante.
- No se justifica escoger clipping, exclusión de arranque o suavizado porque reduzcan el score conocido.

### 2.5. Cobertura de etiquetas y comparación con Garl-TTC

Tras aplicar los offsets RGB demostrados, los clocks de los CSV públicos cubren **354/365 consultas de carretera**. En CCRs1-medium cubren 32/43; sus últimas 11 consultas están fuera del intervalo de etiquetas. Se inspeccionó cobertura temporal, sin calcular nuevos errores ni certificar equivalencia geométrica del target.

El run local de Garl-TTC ya existe: variante completa 424/434 predicciones finitas, eventos 434/434 y visual 426/434. Su informe mide desacuerdo con PHASE, no precisión; véase [guía del run](../../garl_evttc_competition.md). No rellenar sus huecos para aparentar igualdad de cobertura.

El paper eAP describe entrenamiento en eAP y evaluación sin fine-tuning en tres secuencias EvTTC: CCRs2-medium 8,31 %, CCRs2-high 10,56 % y CCRm-medium 12,93 %; media 10,60 %. No incluye Slider en esa tabla. Las mismas tres filas del PDF de PHASE promedian aproximadamente 10,79 %, pero faltan contratos idénticos de consultas y procesamiento para afirmar una comparación controlada. [Paper, Tabla VI](https://arxiv.org/html/2603.16303v1#S7.SS3).

## 3. Interpretación de la disponibilidad pública

[EvTTC competición](https://nail-hnu.github.io/EvTTC/competition/) ofrece las entradas HDF5/ROS bag y bbox de sus diez secuencias. La [página general](https://nail-hnu.github.io/EvTTC/) anuncia etiquetas TTC del dataset. No se ha verificado que esos CSV sean las referencias exactas empleadas por el organizador para las 434 consultas.

La afirmación de que las anotaciones test públicas no incluyen TTC aparece en [GarlTTC](https://github.com/NAIL-HNU/Garl-TTC), un benchmark sobre medios eAP. No debe trasladarse automáticamente a EvTTC. Este plan no intenta recuperar etiquetas privadas de GarlTTC.

Se usa la expresión **conjunto local de 32 grabaciones**, no «train oficial EvTTC-32». Descargar datos y definir un manifest propio no acredita un split oficial independiente.

## 4. Hipótesis y pruebas que las distinguen

| Hipótesis | Evidencia que buscar | Control | Qué no permite concluir |
|---|---|---|---|
| Correspondencia RGB–eventos incorrecta | Recortes y contornos no corresponden bajo calibración documentada | Overlay de sensores y fixture geométrico con transformación conocida | Un overlay aproximado no prueba precisión TTC |
| Clock, intervalo o referencia de salida incompatibles | Diferencia reproducible entre timestamps reales, deltas declaradas y tiempo al que se refiere TTC | Traslación global del reloj y movimiento sintético con TTC conocido | Un factor ajustado al test no prueba una corrección física |
| Arranque inestable | Desviación concentrada en historia corta y trazable a un productor | Misma política sobre prefijos de desarrollo, sin eliminar consultas | No explica por sí solo Slider-1000 |
| Fusión amplifica señales inestables | Productores y fusión divergen bajo inputs fijados | Diagnóstico de ramas congeladas y secuencias de desarrollo | Desacuerdo no identifica la rama correcta sin etiquetas |
| Cambio de dominio aprendido | Entradas verificadas pero sesgo persistente en desarrollo independiente | Modelo congelado frente a ajuste limitado bajo el mismo protocolo | No justifica entrenar con las grabaciones de competición |

Primero deben comprobarse los contratos físicos y temporales. No se ejecutará una búsqueda cartesiana de ventanas, recortes, filtros y arquitecturas.

## 5. Protocolo experimental propuesto

1. Reservar las ocho grabaciones identificadas y ambas Slider para evaluación/diagnóstico expuesto; no usarlas como validación de selección.
2. Auditar grupos de adquisición entre las 24 candidatas. Si quedan dudas sobre duplicación, excluir o agrupar, sin certificar independencia por ausencia de un hash coincidente.
3. Fijar un split de desarrollo por grupos antes de puntuar candidatos. Propuesta: asignación determinista con semilla 7 hacia 60/20/20 % de grupos para ajuste/selección/holdout, preservando grupos completos y documentando soporte por familia. No forzar cantidades de secuencias si rompe grupos. Aprobar el manifest concreto tras comprobar viabilidad; no generar splits hasta obtener uno favorable al score.
4. Si la población no permite un holdout útil, usar validación cruzada por grupos solo dentro de desarrollo y declarar la limitación. No usar competición como sustituto.
5. Definir TTC de referencia, horizonte de salida, tratamiento de no-colisiones, negativos y cobertura antes de entrenar o filtrar.
6. Evaluar error relativo por secuencia, agregado por consulta y agregado macro por secuencia por separado; añadir MAE en segundos, cobertura y errores por rango TTC. Los rangos se fijan con datos de ajuste, nunca para ocultar errores del test.
7. Comparar predicciones brutas y política del organizador en tablas distintas. Solo reproducir esta última si se conoce el algoritmo, incluidos bordes, clocks y tratamiento de huecos.
8. Para modelos entrenados, confirmar el candidato seleccionado con semillas 7, 13 y 21. Para filtros deterministas, tres semillas no crean replicaciones: medir variabilidad entre grabaciones y perturbaciones predefinidas.
9. Usar bootstrap por grabación o grupo independiente; no tratar ventanas como observaciones independientes. Con solo dos Slider no sostener inferencia poblacional fuerte ni intervalos artificialmente precisos.
10. Mantener todas las consultas en el ledger. Separar razones de ausencia de entrada, salida no finita y ausencia de ground truth.

## 6. Mapa de implementación futura

Las rutas de esta sección son propuestas; no indican código entregado por este commit.

| Ruta propuesta | Responsabilidad |
|---|---|
| `operational/evttc_transfer_audit/identity.py` | Correspondencias de grabaciones y hashes de frames |
| `operational/evttc_transfer_audit/protocol.py` | Exclusiones y manifest por grupos |
| `operational/evttc_transfer_audit/targets.py` | Alineamiento explícito y cobertura de targets |
| `operational/evttc_transfer_audit/scoring.py` | Métricas y ledger de elegibilidad sin selección por predicción |
| `operational/evttc_transfer_audit/diagnostics.py` | Contratos de sensores, cadencia, productores y fusión |
| `scripts/audit_evttc_transfer.py` | CLI con subcomandos de auditoría, sin entrenamiento implícito |
| `configs/protocol/evttc_transfer_v1.yaml` | Exclusiones, grupos, reglas y métricas versionadas |
| `tests/unit/test_evttc_transfer_identity.py` | Pruebas de identidad |
| `tests/unit/test_evttc_transfer_protocol.py` | Pruebas de grupos y exclusiones |
| `tests/unit/test_evttc_transfer_targets.py` | Pruebas de clocks y scoring |
| `tests/integration/test_evttc_transfer_diagnostics.py` | Causalidad y reproducción sobre fixtures pequeños |

Reutilizar los lectores y contratos existentes; no sustituir módulos históricos para acomodar este plan. Las rutas de datos se reciben como argumentos o variables de entorno, sin rutas personales hardcodeadas.

## 7. Tareas y condiciones de cierre

### Tarea 0. Preservar y publicar la evidencia inicial

- [x] Inventariar los 32 HDF5 y CSV locales.
- [x] Demostrar identidad de los 365 frames RGB de carretera y registrar offsets.
- [x] Registrar cobertura temporal de etiquetas sin extrapolación.
- [x] Extraer las diez filas del PDF y contrastar conteos y runtime con el ledger.
- [x] Verificar hashes originales de ZIP, freeze y ledger.
- [x] Preparar un anexo portable sin datos crudos ni rutas personales.
- [ ] Revalidar genealogía completa de entrenamiento de cada componente antes de cualquier nueva afirmación de ausencia de exposición.

**Aceptación:** evidencia con alcance explícito y procedencia; ninguna nueva afirmación de precisión del modelo.

### Tarea 1. Convertir la auditoría de identidad en un comando reproducible

**Archivos:** crear `identity.py`, el CLI y `test_evttc_transfer_identity.py` del mapa anterior.

**Interfaz:** `match_recording(competition_hdf5: Path, source_hdf5: Path) -> dict[str, object]`. Salida: datasets comparados, inicio, offset entero en microsegundos, número de frames, igualdad por frame, hashes y limitaciones. Exigir un candidato único para certificar una correspondencia; reportar ambigüedad.

- [ ] Escribir pruebas: recorte con clock desplazado debe coincidir; timestamps periódicos con píxeles distintos no deben coincidir; un píxel distinto debe fallar; candidatos duplicados deben quedar ambiguos.
- [ ] Ejecutar `python -m pytest tests/unit/test_evttc_transfer_identity.py -q` y comprobar el fallo inicial por ausencia de la implementación.
- [ ] Implementar lectura acotada por frame con h5py, sin cargar una grabación entera ni usar etiquetas TTC.
- [ ] Repetir las pruebas y ejecutar el CLI propuesto `python scripts/audit_evttc_transfer.py identity --manifest data/manifests/evttc_all32_local.yaml --data-root "$env:EVTTC_DATA_ROOT" --competition-root "$env:EVTTC_COMPETITION_ROOT" --output output/evttc_identity_replay`.
- [ ] Comparar automáticamente las ocho correspondencias, offsets y 365 hashes con el anexo; permitir diferencias solo en campos de ejecución como fecha o rutas.
- [ ] Commit separado de herramienta y pruebas: `feat(audit): reproduce EvTTC recording overlap`.

**Aceptación:** reproducción completa de las correspondencias con exit code cero; datos ausentes o ambiguos producen estado explícito y código distinto de cero.

### Tarea 2. Fijar exclusiones y grupos de desarrollo

**Archivos:** crear `protocol.py`, configuración y `test_evttc_transfer_protocol.py`.

**Interfaz:** `validate_protocol(protocol: Mapping[str, object], identities: Mapping[str, object]) -> list[str]`; lista vacía indica ausencia de violaciones comprobadas. El manifest conserva `sequence_id`, `recording_group`, `role`, evidencia de relación y estado de revisión.

- [ ] Escribir pruebas de rechazo si una de las ocho grabaciones, su otro sensor o un alias entra en ajuste/selección; rechazar intersecciones de grupos y asignaciones incompletas.
- [ ] Ejecutar `python -m pytest tests/unit/test_evttc_transfer_protocol.py -q` antes y después de implementar.
- [ ] Revisar metadatos de adquisición y contenido de las 24 candidatas; registrar grupos conocidos y relaciones sin resolver.
- [ ] Generar una sola propuesta determinista de split conforme a la sección 5; inspeccionar soporte y viabilidad sin usar métricas de modelos.
- [ ] Sellar roles y hashes. Registrar exclusiones adicionales si la auditoría las exige.
- [ ] Commit: `feat(protocol): isolate EvTTC development recordings`.

**Aceptación:** ninguna grabación excluida ni grupo compartido entre roles. Las 24 candidatas no se marcan todas independientes sin evidencia. La implementación no espera una cantidad fija de grupos.

### Tarea 3. Alinear targets y contrastar la evaluación congelada

**Archivos:** crear `targets.py`, `scoring.py` y `test_evttc_transfer_targets.py`.

**Interfaces:** `align_targets(query_us: np.ndarray, target_us: np.ndarray, target_ttc: np.ndarray, offset_us: int) -> dict[str, object]`, con arrays `values`, `valid` y `reasons`; `score_rows(rows: Sequence[Mapping[str, object]], policy: Mapping[str, object]) -> dict[str, object]`, con población, cobertura, errores y agregados separados.

- [ ] Escribir pruebas para traslación de clock conocida, puntos exactos, interpolación interior, consultas fuera de rango, targets duplicados/no monótonos, denominador cero y ausencia de GT.
- [ ] Añadir pruebas de agregación macro frente a ponderación por consultas, negativos conservados, huecos explícitos y ninguna exclusión condicionada al error del modelo.
- [ ] Ejecutar `python -m pytest tests/unit/test_evttc_transfer_targets.py -q` antes y después de implementar.
- [ ] Verificar unidades y referencia geométrica de las etiquetas públicas. La cobertura de clock no basta para aprobar su uso como GT equivalente.
- [ ] Puntuar primero las salidas ya congeladas en el subconjunto elegible, con etiqueta **diagnóstico local**, sin extrapolar las 11 consultas fuera de cobertura.
- [ ] Obtener, mediante el autor, aclaración del organizador sobre targets alineados, referencia temporal y procesamiento de negativos. El agente no envía mensajes sin autorización explícita.
- [ ] Si se obtienen las referencias exactas, contrastar con el PDF al redondeo publicado de dos decimales. Si hay discrepancia, localizarla antes de puntuar modelos nuevos.
- [ ] Evaluar las variantes Garl congeladas solo con targets compatibles; publicar métricas sobre consultas compartidas y cobertura completa en columnas distintas.
- [ ] Commit: `feat(eval): audit EvTTC target alignment and score scope`.

**Aceptación:** ningún score local se llama oficial sin equivalencia verificada. Si faltan referencias, la tarea puede entregar un evaluador diagnóstico, pero la equivalencia oficial sigue pendiente y no bloquea el trabajo sobre desarrollo independiente.

### Tarea 4. Diagnosticar sensores y componentes sin cambiar pesos

**Archivos:** crear `diagnostics.py` y la prueba de integración del mapa. Leer las APIs de `readers.py`, `geometry.py`, `prepare.py` y `runtime.py` existentes.

**Interfaz:** `diagnose_query(sensor: SensorSequence, query: Mapping[str, object], calibration: Calibration) -> dict[str, object]`. Consume identidades selladas; produce timestamps efectivos, regiones de entrada, transformaciones usadas, conteos de eventos y observaciones. El contraste con el ledger agrega salidas de productores y fusión sin modificar el modelo.

- [ ] Escribir fixtures de expansión conocida, traslación de reloj, arranque corto, silencio observado y soporte ausente. Probar que modificar sensores posteriores a la consulta no cambia sus entradas ni su resultado.
- [ ] Ejecutar `python -m pytest tests/integration/test_evttc_transfer_diagnostics.py -q` antes y después de implementar.
- [ ] Inspeccionar overlays y clocks en una selección fijada por índice y soporte, no por error; usar la misma política en desarrollo.
- [ ] Trazar por separado Slider-750 y Slider-1000, manteniendo sus resultados como diagnóstico expuesto, no como selección de hiperparámetros.
- [ ] Documentar una hipótesis por vez y una prueba que pueda refutarla. Si no se identifica causa, mantener ese resultado negativo.
- [ ] Commit: `feat(audit): trace causal EvTTC sensor and model inputs`.

**Aceptación:** poder señalar qué contrato falla o qué hipótesis sigue abierta. Un overlay, un test sintético o una correlación no bastan para declarar una mejora de precisión.

### Tarea 5. Ejecutar un piloto controlado de la intervención justificada

Esta tarea se concreta en un diseño corto tras las anteriores, porque todavía no conocemos la causa. No debe implementarse un filtro o una corrección física basándose únicamente en una posibilidad.

- [ ] Elegir una sola intervención: corrección de un defecto probado; consistencia temporal causal si el error es inestabilidad; o ajuste limitado si persiste un sesgo con inputs correctos.
- [ ] Crear configuración versionada, salida nueva y baseline congelado sobre el mismo manifest de desarrollo. Como presupuesto inicial, comparar el baseline y hasta dos candidatos motivados; no explorar combinaciones cartesianas.
- [ ] Para un bug, escribir y observar primero una regresión fallida; implementar el cambio mínimo y pasar la suite afectada. Para una técnica nueva, probar causalidad, identidades y conservación de cobertura antes de medir precisión.
- [ ] Seleccionar por error relativo macro de secuencias en selección, con cobertura completa para una afirmación de mejora de población completa. Reportar MAE, agregado por consulta, peor secuencia y runtime como diagnósticos obligatorios.
- [ ] Si el candidato requiere entrenar, empezar por fusión con productores congelados; comparar fine-tuning parcial solo si existe una razón y presupuesto registrados. Usar únicamente los grupos de ajuste.
- [ ] Confirmar el ganador aprendido con semillas 7/13/21, sin elegir la mejor semilla por test. Abrir una vez el holdout de desarrollo para baseline y candidato seleccionado.
- [ ] Si no mejora en holdout, registrar fallo y no afirmar éxito. Reutilizar ese holdout para nuevas decisiones lo convierte en desarrollo expuesto; exigir otra evaluación independiente para futuras afirmaciones.
- [ ] Commit por intervención con configuración, pruebas y referencias a artefactos; no subir datasets ni checkpoints al Git del proyecto.

**Aceptación:** mejora reproducible en desarrollo independiente, con diferencias pareadas por grupo y limitaciones estadísticas. Si la cobertura cae, el resultado pertenece a una tarea con abstención y no reemplaza la comparación completa.

### Tarea 6. Reportar la nueva variante y preparar evaluación final

- [ ] Generar tablas desde JSON/CSV; no escribir métricas a mano.
- [ ] Separar cuatro categorías: oficial recibido, reproducción local equivalente si existe, diagnóstico parcial y desarrollo independiente.
- [ ] Registrar cuánto se conocía del test cuando se diseñó cada cambio y todos los intentos negativos.
- [ ] Presentar una variante ajustada como `PHASE-TTC, fine-tuned on EvTTC` o una variante de adaptador con versión explícita, según corresponda.
- [ ] Medir runtime integral con el mismo límite y equipo; no sustituir tiempos offline por kernel timings ni sumar comparaciones incompatibles.
- [ ] Preparar una nueva entrega solo si cumple el protocolo y el organizador admite la configuración. Envío externo requiere instrucción del autor; conservar siempre la entrega inicial.
- [ ] Actualizar la documentación histórica mediante una nota fechada que enlaza nueva evidencia, sin reescribir los hechos del run original.

**Aceptación:** resultados rastreables, denominadores y cobertura visibles, y publicación que distingue qué se ha medido de qué sigue siendo hipótesis.

## 8. Orden, recursos y puntos de parada

| Orden | Trabajo | GPU | Dependencia para avanzar |
|---|---|---|---|
| 0 | Evidencia inicial de este documento | No; solo consulta del dispositivo | Ejecutado dentro del alcance descrito |
| 1 | CLI de identidad y protocolo de grupos | No | Correspondencias reproducibles y roles válidos |
| 2 | Targets y evaluación congelada | No para reutilizar predicciones | Contrato de target explícito; equivalencia oficial puede seguir pendiente |
| 3 | Diagnóstico de entradas y componentes | Solo si hay que reproducir inferencia | Identidades y código congelados |
| 4 | Piloto de una intervención | Según la intervención | Hipótesis, split y presupuesto fijados |
| 5 | Confirmación y reporte | Si hay entrenamiento | Selección cerrada y población preservada |

Antes de reservar GPU: comprobar entorno, espacio libre, integridad de pesos y una consulta pequeña. La auditoría encontró un problema de arranque de la `.venv` actual asociado a lectura de un `.pth`; usó Python sin carga de `site` para leer librerías del entorno existente. Eso permitió analizar HDF5, pero no certifica un entorno de entrenamiento sano. Resolverlo de forma aislada y registrar el entorno antes de ejecutar el modelo.

No se estima una mejora porcentual ni una duración de entrenamiento sin piloto. No descargar otra copia de EvTTC ni iniciar una campaña larga mientras los datos existentes permitan cerrar las primeras tareas.

## 9. Criterios de éxito y límites de la entrega actual

El plan habrá cumplido su objetivo si permite explicar con evidencia las limitaciones observadas y evaluar una intervención fuera de los datos usados para decidirla. También es válido concluir que una intervención no mejora o que faltan datos independientes de Slider.

Este commit entrega el plan y el anexo de evidencia. No entrega un nuevo modelo, no reproduce el scorer oficial, no certifica el origen remoto actual de los archivos locales y no afirma un nuevo ranking.

## 10. Referencias

- [Contrato del proyecto](../../../AGENTS.md).
- [Política original del adaptador EvTTC](../../decisions/ADR-0043-evttc-competition-sensor-adapter.md).
- [Inferencia de competición PHASE](../../evttc_competition.md).
- [Comparación local congelada con Garl-TTC](../../garl_evttc_competition.md).
- [Anexo portable de la auditoría del 30-10-2026](evidence/2026-10-30-evttc-audit.json).
- [EvTTC: competición y archivos de entrada](https://nail-hnu.github.io/EvTTC/competition/).
- [EvTTC: documentación general](https://nail-hnu.github.io/EvTTC/).
- [Garl-TTC/eAP: paper, arXiv:2603.16303v1](https://arxiv.org/html/2603.16303v1).
- [Garl-TTC: código y contrato de etiquetas públicas](https://github.com/NAIL-HNU/Garl-TTC).
