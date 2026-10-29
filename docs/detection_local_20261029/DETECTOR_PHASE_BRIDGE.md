# Inferencias con detecciones eAP y PHASE

Fecha: 2026-10-29. Estado: **PREDICTED_DEVELOPMENT_EXPOSED**.

Se ejecutó una consulta real de la cadena **detecciones 3D → asociación causal →
recorte común → eventos/RGB → PHASE**, primero con el checkpoint del overfit y
después con el checkpoint del piloto de 256 frames y cabeza de velocidad.
Ambas ejecuciones usaron exclusivamente predicciones y medios públicos train,
sin cargar etiquetas de objeto/TTC ni IDs Garl durante la inferencia. Producen
TTC firmado y límites de fase; no miden precisión o generalización conjunta.

## Resultado actual: piloto con velocidad

Se repitió la misma consulta `qGsgzl4Q8B:000000929_000731755` y las mismas reglas
de asociación, umbrales, recorte, pesos PHASE y código del puente. El checkpoint
del detector es `dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.
La preparación y la inferencia terminaron con código 0; no hubo abstención.

La [salida integrada](evidence/integrated_prediction_pilot_v1.json) enlaza la
detección nativa exacta mediante `sample_token` y `prediction_index=0` con las
salidas originales de PHASE:

| Campo | Valor |
|---|---:|
| Clase | car |
| Centro 3D, m | `[29.0970653059, 3.8369083604, 0.2489670917]` |
| Dimensiones largo, ancho, alto, m | `[4.7069950104, 1.8282244205, 1.7362303734]` |
| Yaw nativo, rad | −1,5681608559 |
| Velocidad planar aprendida, m/s | `[2.1009831429, −0.0082943831]` |
| Score de detección | 0,3639818430 |
| `ttc`, segundos con signo | −5,2349028587 |
| `point_phase` / `raw_location`, fase | −0,0189223886 |
| `q10`, fase | −0,0224637017 |
| `q90`, fase | −0,0155819701 |
| Frames asociados | 7 |
| Observaciones eventos / RGB aceptadas | 4 / 5 |
| Eventos retenidos | 408.923 |
| ROI común xyxy, píxeles | `[20,410,369,720]` |

La cabeza de velocidad está presente: estos valores son predicciones aprendidas
del target público HF, no un relleno con ceros. No se afirma que representen
velocidad relativa con compensación de ego-motion. PHASE no obtiene su TTC de
esa velocidad; se mantiene como rama independiente. Los límites de fase no
cruzan cero en esta consulta y no se presentan como un intervalo TTC calibrado.

La política fija de mayor score seleccionó una ROI distinta de la ejecución
overfit conservada abajo. Aunque ambas generen el ID local `detected_roi_0001`,
**ese ID no establece identidad entre checkpoints**. Por tanto, la diferencia
entre ambos TTC no demuestra una mejora ni un cambio del TTC de un mismo objeto.
La consulta está entre los frames de entrenamiento del piloto, y PHASE estuvo
expuesto a TRAIN40. El ejemplo demuestra la interfaz; no es evaluación retenida.

La ROI solicitada superaba el borde inferior de la imagen y se recortó una vez,
de modo idéntico para todas las observaciones. Se conservaron los timestamps de
v1: ancla `929731755` µs y corte sensorial `929733441` µs. El runtime PHASE sobre
observaciones preparadas fue **0,71745 s en CPU, un hilo**, excluyendo detector,
carga de pesos y preparación sensorial. No es latencia de extremo a extremo.

Recibos nuevos, conservando intactos los anteriores:

- [Preparación del piloto](evidence/phase_bridge_prepare_pilot_v1.json), SHA-256
  `ad14ece111c710b4f87596b23073879fa137a1814ba3106123b7827804c35c9d`.
- [Inferencia PHASE del piloto](evidence/phase_bridge_temporal7_pilot_v1.json), SHA-256
  `d56916747531bcd53907dea8ef5964a23e8e8b0098eb6fee095a7274566495f5`.
- [Predicción integrada](evidence/integrated_prediction_pilot_v1.json), SHA-256
  `6396867306506c273ab5aa4823b28eae0625ec7c21c8c5e2e1787bfa8b8f3cd2`.

El JSON temporal de entrada tiene SHA-256
`b6b7d7399b1ba2b3543f10dabecec393070aa9b7f4aa81422be93e8a25390bbb`.
La salida integrada copia la detección identificada por el último elemento de
`detected_history` y las salidas nativas del recibo PHASE. Se verificaron hashes,
token, índice, clase, score y presencia de velocidad; no se ejecutó ajuste nuevo.
Estos tres recibos contienen predicciones y diagnósticos, sin GT ni bytes RGB/eventos.

Reproducción de la inferencia actual, con un nombre de salida nuevo:

```powershell
../phase-ttc-main-integration/.venv/Scripts/python.exe -X utf8 scripts/infer_detected_roi_phase.py --eap-root E:/eAP_dataset --predictions output/portable_experiments/pilot256_seed7_velocity_v1/inference_temporal7.json --sample-token qGsgzl4Q8B:000000929_000731755 --candidate-root D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --output output/portable_experiments/pilot256_seed7_velocity_v1/phase_bridge_reproduction.json --device cpu --cpu-threads 1
```

## Resultado anterior conservado: overfit16

Consulta: `qGsgzl4Q8B:000000929_000731755`. Se había elegido antes de ejecutar
PHASE porque dispone de historia sensorial real; el primer frame del overfit
carece de imágenes anteriores en el manifest público.

| Campo | Valor |
|---|---:|
| Clase detectada | car |
| Score de detección, sin presuponer calibración | 0,3509746492 |
| Frames con asociación causal continuada | 7 |
| Observaciones de eventos aceptadas | 4 |
| Observaciones RGB aceptadas | 5 |
| Eventos retenidos en la ROI | 60.730 |
| ROI común xyxy, píxeles | `[469,460,623,614]` |
| `ttc`, segundos con signo | −15,7692270279 |
| `point_phase` | −0,0063214423 |
| `raw_location`, fase | −0,0063214423 |
| `q10`, fase | −0,0168219022 |
| `q90`, fase | 0,0058432594 |

Los límites **q10/q90 están en fase** y cruzan cero. Se conservaron directamente
de `PhaseOnlyFinal`, sin reconstruirlos desde el TTC escalar ni transformarlos en
un intervalo TTC continuo. Sus nombres no demuestran cobertura probabilística
del 80 %. El TTC negativo es una predicción; no se contrastó con verdad terreno.

Recibos versionados, copiados sin modificar sus bytes:

- [Preparación](evidence/phase_bridge_prepare_v1.json), SHA-256
  `1dbcd938aaed1764b16d7de20da9abb7ec1466d2e1abc6a0ebc8148b65b76319`.
- [Inferencia](evidence/phase_bridge_temporal7_v1.json), SHA-256
  `41c2795b951dec6bf993f38f363f259da92496ce97c30465bfa687be748aa944`.

Contienen diagnósticos, ROIs derivadas de predicciones, timestamps y hashes. No
incluyen etiquetas GT ni bytes RGB/eventos. Los originales y logs permanecen en
`output/portable_experiments/overfit16_seed7_v1/`.

## Contrato causal y procesamiento

La entrada de la primera ejecución es `inference_temporal7.json`, un JSON de
predicciones sin GT generado por el checkpoint
`52ec97774d06c2a7bc0d9320db616f2a50a3ac4dce50c7709fbdf4883fc4907f`.
SHA-256 del JSON: `5d81784ee88c4bb6800675abaea54316bc452b7cf06be8b34d8813e9e198567e`.
El puente rechaza archivos que mezclen predicciones con `ground_truth` o campos
de targets. La lectura Arrow solicita únicamente los siete tokens y columnas
sensoriales del manifest; no abre los archivos de etiquetas.

La selección y asociación quedaron fijadas antes de la inferencia:

- score mínimo 0,1;
- NMS por clase con IoU 0,5, solo para escoger regiones TTC;
- asociación greedy uno a uno por clase, IoU mínimo 0,3 y diferencia mínima
  de 0,1 respecto al segundo candidato;
- gap máximo de 150 ms, sin reparación retrospectiva de identidades;
- objeto actual con mayor score; sin buscar otro objeto más fácil si este falla;
- abstención si faltan al menos dos observaciones asociadas sin ambigüedad.

Los IDs son internos del puente. No se usaron IDs o cajas de GarlTTC. NMS no
reescribe el archivo de detecciones; los duplicados suprimidos y las decisiones
de asociación quedan en el recibo.

La ROI es la unión cuadrada de las cajas predichas asociadas, expandida un 25 %
por lado y recortada una vez a los límites de imagen. Se aplica **el mismo
recorte** a todos los frames y eventos de la consulta: esto conserva cambios de
escala aparentes. No se redimensiona cada caja de objeto individualmente para
simular un tamaño constante ni se inventa una caja histórica copiando la actual.
RGB se redimensiona con PIL bilinear a 128×128 y se divide por 255. Los eventos
usan cinco bins por polaridad y dos canales logarítmicos de conteo/tasa, con la
normalización robusta existente. Se conserva y registra el desplazamiento
`event_x_offset=5` del procesamiento eAP empleado por los productores finales;
no se presenta como una calibración física nueva.

Los siete RGB tienen timestamps de inicio entre `929131746` y `929731755` µs.
La referencia de la consulta es el inicio de exposición actual `929731755` µs;
el corte de disponibilidad sensorial se fija en su final, **`929733441` µs**.
El adaptador excluye entradas posteriores a cualquiera de esos límites. Los intervalos
de eventos son exactos y semiabiertos `[inicio,fin)`; las duraciones y deltas se
derivan de timestamps reales. La historia completa utilizada queda dentro de
650 ms. No se presentan ceros ni frames duplicados como observaciones reales:
las historias de ambas modalidades se seleccionan por separado y el contexto
del modelo utiliza sus máscaras de validez para los slots restantes.

La disponibilidad física de transporte/lectura no se midió. Se trata de replay
offline con corte basado en exposición; el resultado no se anuncia disponible
en el pasado. La adaptación de cadencia queda registrada como
`measured_preprocessing_adaptation`; la conversión final fase↔TTC mantiene el
intervalo fijo de 0,1 s del modelo congelado.

## Pesos, salidas y alcance

La inspección previa verificó **14 artefactos locales** contra `PORTABLE_PATHS`:
nueve checkpoints, dos normalizadores, `MODEL_SEAL.json`, `OOF_RECIPE.json` y la
cabeza TorchScript. Para esta inferencia se cargó la cadena de nueve componentes
con sus normalizadores mediante `CompetitionRuntime.from_candidate`; la cabeza
TorchScript aislada no se utilizó como sustituto de la cadena sensorial.

El sello del modelo tiene SHA-256
`cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.
La cabeza se observó mediante un forward hook externo que conserva exactamente
sus salidas originales. No se modificaron fuentes ni pesos PHASE. Tampoco se
ejecutaron actualizaciones del optimizador.

PHASE fue entrenado con **TRAIN40**, que incluye las secuencias del piloto actual.
La separación train/selección del detector no hace independiente a PHASE. Por eso
el recibo declara desarrollo expuesto y no calcula MAE, cobertura ni mejora TTC.
Una sola asociación de siete frames no valida un tracker general, oclusiones,
cruces, escenas nuevas ni el servicio multiobjeto completo.

El runtime de la primera ejecución sobre observaciones preparadas midió
**0,7733 s en CPU, un hilo**.
Esa cifra excluye detección, decodificación y preparación sensorial, carga de pesos
y parte de la coordinación. **No es latencia de extremo a extremo ni rendimiento
en GPU**. El coste sensorial y la latencia física siguen sin caracterizarse aquí.

## Reproducción y pruebas

El script ejecutado mantiene SHA-256
`0ed624e423c3b5ff05742c4de56e4913f9e5a083c9a4c3485f19a1f06a8e01ad`, coincidente con
los cuatro recibos de preparación/inferencia. El mismo script se reutilizó para
el piloto sin modificarlo.

```powershell
../phase-ttc-main-integration/.venv/Scripts/python.exe -X utf8 scripts/infer_detected_roi_phase.py --eap-root E:/eAP_dataset --predictions output/portable_experiments/overfit16_seed7_v1/inference_temporal7.json --sample-token qGsgzl4Q8B:000000929_000731755 --candidate-root D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --output output/portable_experiments/overfit16_seed7_v1/phase_bridge_reproduction.json --device cpu --cpu-threads 1
.venv-audit/Scripts/python.exe -m unittest tests.unit.test_detected_roi_phase -v
```

La CLI rechaza sobrescribir un recibo existente. Para comprobar solo los inputs,
añadir `--prepare-only` y usar otro nombre de salida.

**Ocho tests correctos**, sin cargar Torch ni usar GPU: asociación inequívoca,
ambigüedad, gaps, separación de clases, NMS, rollback temporal, rechazo de JSON
con GT y exclusión de frames futuros o exposiciones todavía no terminadas.
Ruff y formato correctos. Esta verificación cubre el puente pequeño ejecutado,
sin sustituir la evaluación científica pendiente del sistema conjunto.
