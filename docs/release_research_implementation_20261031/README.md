# Implementación y ejecución del plan de investigación

Estado: **cadencia 2608, comparación nativa 5805 y ambas latencias terminadas y selladas**.
El [informe de campaña cerrado](campaign_results_v1/RESULTS.md) reúne la
evaluación de precisión en desarrollo, la sensibilidad nativa y la latencia
desde **NPZ de eventos ya preparado + RGB**. Su tabla de latencia no incluye
la construcción de la representación desde eventos RAW. Esa frontera está
medida y cerrada en el [suplemento RAW/RGB](raw_latency_results_v1/RESULTS.md):
p95 original 0,963 s, canónico 1,008 s (+4,65%), con todas las salidas exactas.
No quedan procesos de esta campaña activos. El estado actual y los hashes de
los recibos principales se conservan en
`output/release_research_20261031/CURRENT_STATE.json`.
El usuario confirma que no hay adquisiciones reservadas ni calibración fotométrica
adicional y solicita usar los datos existentes. Se mantiene el alcance de desarrollo
expuesto. El [ensayo exploratorio S1](SENSOR_PROXY_PREREGISTRATION.md), su
[control RGB](SENSOR_REFERENCE_CONTROL.md) y la
[recuperación de expansión](PHOTOMETRIC_Q_PREREGISTRATION.md) ya se ejecutaron
con los datos locales, sin convertir una escala RGB aproximada en calibración física.
La continuación ya ejecutó O0 RAW, V0/V1 en RGB real, el inventario fotométrico
de las 25 adquisiciones y un perfil nativo de productores/PAIR/contextos.
Resultados y límites regenerables:
[continued_results_v1/RESULTS.md](continued_results_v1/RESULTS.md).
La localización por modalidad, la auditoría del fallo O0, el control robusto,
los bancos de identificabilidad y maniobras, y la deriva nativa por columna
están ejecutados en [followup_results_v1/RESULTS.md](followup_results_v1/RESULTS.md).
La versión ampliada incorpora los cuatro ensayos adicionales con datos existentes:
[followup_results_v2/RESULTS.md](followup_results_v2/RESULTS.md). Incluye los CI
de RGB observado frente a todos los controles: su ventaja media sobre actividad
no es concluyente por adquisición. La recuperación afín de q empeora TTC y los
filtros tampoco superan al contexto tras corregir PAIR.
La continuación original ya terminó: F0 pasa la ampliación de 2608, pero el perfil
CUDA original falla paridad histórica intermedia en ambas consultas, aunque TTC
cumple su tolerancia. La matriz de cuatro políticas TF32 tampoco recupera esa
paridad; conserva repetibilidad local exacta. No se ha relajado el criterio.
La comparación histórica completa del original ya está sellada: 5805 consultas
fallan la paridad intermedia y 504 fallan la tolerancia de TTC. La diferencia
absoluta mediana de TTC es 0,000185 s, p95 0,001094 s y máximo 0,496792 s;
el extremo está en predicciones de magnitud mayor de 16 s. Es deriva de
predicciones entre plataformas, no error frente a etiquetas. Véase el
[desglose completo](native_parity_distribution_v1/RESULTS.md).
El resultado de la ampliación está en
`output/release_research_20261031/continuation_v1/population_report/README.md`.
La [comparación local distinta](NATIVE_PLATFORM_PAIR_PROTOCOL.md) está en
`output/release_research_20261031/native_platform_followup_v2/STATUS.json`.
Mide sensibilidad original/candidato en el mismo equipo, con todas las entradas;
no demuestra conservación de score histórico ni admite promoción.
La ejecución v1 quedó interrumpida el 1 de noviembre, con 5805 eventos y 1474
RGB completos. La [recuperación registrada](NATIVE_INTERRUPTION_RECOVERY.md)
conserva esa ejecución, sella los 7279 fragmentos válidos y recalcula uno
incompleto. Sus 34 controles de productores originales son exactos. La inferencia
RGB pendiente se reanudó a las 00:33:08; copiar fragmentos no cuenta como nueva
inferencia. Los registros `PROGRESS_AUDIT_20261101_*.json` distinguen ambos avances.
La latencia desde NPZ preparado y la latencia desde RAW se registran por separado:
[protocolo NPZ/RGB](NATIVE_LOCAL_LATENCY_PROTOCOL.md) y
[suplemento RAW/RGB](NATIVE_RAW_LATENCY_PROTOCOL.md). Este último comprueba
igualdad exacta del tensor reconstruido antes de medir la cadena completa con
cajas suministradas; no incluye detección de objetos ni adquisición física.
Este trabajo convierte los primeros contrastes del dossier en código ejecutable y
prepara bloques físicos de los brazos posteriores. Hay una mejora de desarrollo
de la corrección de cadencia, no una mejora oficial demostrada del release.
El [plan original](../release_research_plan_20261031/README.md) se conserva intacto.

Los resultados válidos están en [results_v2/RESULTS.md](results_v2/RESULTS.md),
generados desde predicciones selladas y targets de desarrollo. `results/` está
**sustituido**: intercambiaba las etiquetas RGB/eventos al leer el vector nativo.
Se conserva para auditar el error; no debe utilizarse para decisiones.

## Qué está implementado y qué falta

| Brazo | Implementación comprobada | Pendiente antes de una conclusión sobre R0 |
|---|---|---|
| P0 | Verificación local del sello, nueve checkpoints, dos normalizadores y receta; replay local original exacto y candidato completo en 5805 consultas; hashes y fuentes conservados | Conservación histórica falla; precisión nativa no evaluada |
| C-TIME | Contraste por modalidad, ampliación 2608 y sensibilidad local 5805 terminados; sin pérdida de cobertura ni cambios de signo; p95 NPZ +3,4% y RAW +4,65% | Carretera tiene cambio pequeño con CI que incluye cero; la ganancia de Slider procede de una adquisición y no demuestra precisión oficial |
| C-OBS | Seis indicadores RAW en las ROI del release; controles constante/soporte/físico/permutado con adquisición excluida; el físico no supera los controles | O1/O2 no admitidos por esta prueba; falta una medición con señal incremental |
| C-CLOCK | Persistencia, EWMA, transporte y observador de q; 650 ms y fallback; contraste real repetido con PAIR original/canónico, además del banco de maniobras | No hay ventaja de D3 frente al contexto; D4 requiere una carencia corregible adicional, no solo conocer la ecuación |
| C-SENSOR | S0, inventario y S1 exploratorio RAW en 72 consultas/24 adquisiciones; control RGB observado y recuperación radial/traslación/afín/permutada con escala excluyendo cada adquisición | RGB observado explica mejor eventos, pero q recuperada empeora TTC: el proxy no admite S2/S3 ni sustituye calibración física |
| C-VIS | V0/V1 real: correspondencias RGB, rutas independientes y cobertura; ausencia de cambios de crop en los casos auditables; la medición geométrica empeora TTC | Evidencia incremental antes de experto V2 o auxiliar V3; visibilidad real del objeto sigue sin resolver |
| JEPA condicional | Predictor condicionado por tiempo real y pérdida con stop-gradient y anclaje observable; admisión exige tres condiciones del dossier | No se reabre el piloto cerrado; falta evidencia de pérdida de dinámica en el productor, EMA/entrenamiento y controles J0–J3 integrados |
| Evaluación | Cadencia, sensibilidad nativa y latencias cerradas; métricas micro/macro, colas, signo, urgencia, cobertura, bootstrap por adquisición y recibos verificados | No hay adquisiciones nuevas para confirmación independiente; precisión nativa no evaluada; tres semillas si se admite un bloque aprendido |

No hay un trainer nuevo de productores/contextos ni una reconstrucción OOF nueva.
Las clases auxiliares son puntos de integración comprobados con fixtures, no
modelos entrenados ni una sustitución del runtime oficial. La admisión comprueba
recibos y presupuestos; no reemplaza la revisión científica de sus conclusiones.

## Código y contratos

Todo el código nuevo está en
[`operational/release_research`](../../operational/release_research/) y
[`scripts/release_research.py`](../../scripts/release_research.py). El runtime
oficial no importa estos módulos. No se han modificado pesos, normalizadores,
predicciones oficiales ni fuentes congeladas por el worker de 2608.

- `physics.py`: q firmado antes de saturación; observador reconstruido por consulta;
  SVD que admite q identificable aunque el movimiento afín completo no lo sea.
- `visibility.py`: unidades coherentes de sensor/crop y cobertura explícita.
- `auxiliaries.py`: atenuación con identidad exacta en λ=0/1; anchuras declaradas
  no recalibradas; sensor local con gradiente hacia q; predictor temporal auxiliar.
- `photometric.py`: banco S0 limitado a textura log-afín y velocidad axial constante.
  No representa paralaje, texturas generales ni toda la física de un sensor real.
- `protocol.py`: presupuestos, disjunción de adquisiciones, admisión condicional,
  regla de promoción, hashes y fuentes ejecutadas. Ningún método promociona R0.
- `evaluation.py`: comparaciones pareadas con denominadores y CI por adquisición.
  No calcula un Overall nativo ficticio sobre una cohorte positiva.

Los intervalos temporales del sensor incluyen toda su ventana de soporte: un
anchor reciente no permite introducir eventos anteriores al límite de 650 ms.
Los observadores usan una varianza fija exploratoria, sin asumir calibración ni
independencia de ventanas solapadas. D1–D3 no son un postfiltro de la fusión: usan
PAIR de eventos y se comparan también con EVENT_CONTEXT.

El consumidor conserva el orden nativo **F0/RGB/eventos**, comprueba la identidad
de F0 y contrasta el ancla RGB contra su fase cruda. R0 y RGB conservan los valores
capturados; las diferencias CPU/CUDA de emisión quedan registradas separadamente.
Las predicciones nuevas de estos controles se emiten en CPU FP32.

## Reproducción

Desde la raíz, con las dependencias del proyecto instaladas y el entorno activo:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src;.'
python -m scripts.release_research --help
python -m scripts.release_research mechanisms --output output/research_new/mechanisms
python -m scripts.release_research diagnose --screen output/phase_timewarp/screen149_v1 --output output/research_new/blind
python -m scripts.release_research evaluate --run output/research_new/blind --population output/phase_submission/training/POPULATION.npz --output output/research_new/evaluation
python -m scripts.release_research report --evaluation output/research_new/evaluation --output output/research_new/report
```

Cada destino debe ser nuevo. La extracción no abre targets; `evaluate` exige
predicciones completas, verifica hashes e identidad de población antes de unirlos.
Una carpeta parcial de 2608 no es una entrada admitida. No duplicar su extracción.
Para S1 se comprobó capacidad y se registró una excepción de un hilo CPU, sin CUDA,
antes del perfil: [recursos medidos](evidence/SENSOR_PROXY_RESOURCES_BEFORE.json) y
[protocolo de perfilado](SENSOR_PROXY_PREREGISTRATION.md). La extracción S1 completa
quedó por debajo de 1 GiB de memoria residente máxima. Los controles posteriores
reutilizan sus recuentos, sin volver a extraer eventos.

Artefactos locales de esta ejecución:

- `output/release_research_20261031/R0_IDENTITY.json`: sello y componentes verificados.
- `mechanisms_v2/`: mecanismo geométrico y fotométrico, con fuentes ejecutadas.
- `diagnostics149_v4/`: predicciones ciegas vigentes y trazas, con fuentes ejecutadas.
- `evaluation149_v2/`: evaluación vigente y decisión `diagnose`.
- `diagnostics149_v1/FAILED.json`: comparación exacta CPU/CUDA rechazada.
- `diagnostics149_v2/`, `diagnostics149_v3/`, `evaluation149_v1/`: intentos
  conservados y sustituidos por la corrección de etiquetas de modalidad.

Los primeros intentos registraban hashes de fuentes. Desde `mechanisms_v2` y
`diagnostics149_v3`, también se preservan los bytes de código no versionado en
`executed_source/`; el resultado vigente conserva esas copias.

## Validación y límites

La validación acumulada comprende **126 pruebas focalizadas distintas**, incluidas
las de unidad/integración, PAIR/screen canónico, admisión, recuperación y latencia.
Ruff y Pyright del alcance nuevo pasan. Los recibos conservan cada versión probada;
el último incremento está en [VALIDATION_NATIVE_RECOVERY.json](evidence/VALIDATION_NATIVE_RECOVERY.json).
El Pyright específico apunta a `.venv-transfer`, el entorno local válido; el
entorno `.venv` antiguo resolvía incorrectamente los símbolos de Torch.

```powershell
python -m pytest tests/unit/test_release_research.py tests/integration/test_release_research_runner.py tests/unit/test_phase_canonical_pair.py tests/unit/test_phase_canonical_screen.py tests/unit/test_phase_canonical_screen_metrics.py
python -m ruff check operational/release_research scripts/release_research.py tests/unit/test_release_research.py tests/integration/test_release_research_runner.py
python -m ruff format --check operational/release_research scripts/release_research.py tests/unit/test_release_research.py tests/integration/test_release_research_runner.py
python -m pyright -p operational/release_research/pyrightconfig.json
```

Los comandos anteriores reproducen el núcleo inicial; los recibos enlazados más
abajo identifican las ampliaciones y sus JUnit. No se han usado servicios de pago
ni actualizado pesos de R0. O0 incluye ajustes ridge auxiliares; no son un nuevo
gate entrenado. Los tiempos de extracción y replay no se presentan como latencia
de inferencia. Las latencias NPZ/RGB y RAW/RGB están medidas en ensayos separados.
No se han leído etiquetas privadas Test12 ni obtenido una
confirmación independiente. Los fixtures positivos no autorizan por sí solos
abrir S2, D4, O2 o JEPA.

La documentación de las APIs utilizadas se comprobó en las fuentes primarias de
[SVD de NumPy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
y [Huber de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.huber_loss.html).
Las hipótesis científicas siguen siendo las del dossier, no resultados externos
atribuidos al release.

## Estado de la ejecución

1. La población 2608 y su evaluación están COMPLETE; F0 pasa el criterio de desarrollo.
2. El [diagnóstico de políticas numéricas](NATIVE_BACKEND_DIAGNOSIS.md) está ejecutado.
   Ninguna de las cuatro políticas recuperó la paridad histórica del perfil.
   La comparación nativa posterior cubre 5805 consultas, conserva el fallo histórico
   y demuestra repetibilidad exacta local del original. La intervención canónica
   conserva cobertura y signo, con un cambio máximo de predicción de 0,00650 s.
3. O0 RAW y V0/V1 están ejecutados. No abrir cabezas mayores al fallar sus
   controles; una nueva propuesta debe explicar la cobertura o sesgo observado.
4. S1 ya tiene caché RAW anclada y controles ejecutados. La lectura afín supera al
   control espacial permutado, pero empeora mucho TTC frente a R0; entrenar con
   este proxy podría imponer una expansión sesgada. No confundir la mejora del
   residuo RGB observado con admisión del auxiliar ligado a q.
5. Las latencias NPZ/RGB y RAW/RGB están COMPLETE (cocientes p95 1,03393 y 1,04647).
   La reconstrucción RAW es exacta en las 120 observaciones; todas las salidas
   cronometradas reproducen exactamente sus correspondientes predicciones locales.
   La variante permanece como candidato de desarrollo, con exposición declarada.
   No hay una confirmación independiente pendiente de localizar: el usuario confirma
   que esos datos adicionales no existen. La ausencia limita la afirmación científica,
   pero no bloquea la validación local. Reconstruir genealogía si cambian productores
   y medir cadena completa antes de proponer otro release.

Los runners nuevos son `release_observability.py`, `release_observability_analysis.py`,
`release_surface.py`, `release_surface_analysis.py`, `release_sensor_inventory.py`,
`release_native_pair_capture.py` y `release_native_pair_replay.py`, todos en `scripts/`.
La política de continuación está en `scripts/continue_release_research.py`.
Cada CLI ofrece `--help`; los experimentos exigen un directorio nuevo y conservan
fallos, fuentes ejecutadas y recibos de identidad. La recuperación y el cierre se
orquestan mediante `scripts/resume_release_native_research.py`; el replay completo
está cerrado y no constituye una evaluación de precisión Test12.

Validación de la continuación: 69 tests focalizados (incluyen las rutas de parada
y continuación automática), Ruff check/format y Pyright de todos los módulos y
runners nuevos. Evidencia JUnit:
`output/release_research_20261031/validation_continued_v1/pytest.xml`.

Validación adicional de los ensayos con datos disponibles: 17 tests distintos
(8 sensor/referencia, 6 lectura fotométrica, 3 causalidad del control temporal),
además de las 2 pruebas de identificabilidad y las 69 previas: 88 en total.
Ruff check/format y Pyright pasan en los nueve archivos añadidos o actualizados.
Recibos y hashes: [VALIDATION_AVAILABLE_DATA.json](evidence/VALIDATION_AVAILABLE_DATA.json).
El [inventario local de entorno](evidence/ENVIRONMENT_AVAILABLE_DATA.json) queda
vinculado por hash a los nueve runs de extracción/evaluación con datos disponibles.
Se capturó después de esos ensayos; no demuestra retrospectivamente la política
TF32 del proceso histórico. El diagnóstico de backend añade nueve pruebas de
admisión/autenticidad: [VALIDATION_NATIVE_BACKEND.json](evidence/VALIDATION_NATIVE_BACKEND.json).
