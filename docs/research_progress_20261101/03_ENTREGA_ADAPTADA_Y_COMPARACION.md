# Entrega adaptada: trabajo realizado y comparación con el original

Fecha: 1 de noviembre de 2026. Este documento recoge la revisión posterior a los
informes [01](01_TRABAJO_REALIZADO.md) y [02](02_RESULTADOS_Y_PENDIENTES.md):
identificar el archivo que se quiere enviar a EvTTC, explicar qué modelo lo
generó y comparar sus predicciones con las del primer correo.

## Entregas que hemos distinguido

| Entrega | Identidad y alcance |
|---|---|
| Original del 28 de octubre | `PHASE_TTC_EvTTC_results_2026-10-28.zip`; 434 predicciones, dos negativas. Los organizadores comunicaron que la evaluaron. |
| Adaptada del 31 de octubre | `PHASE-TTC-Adapt-20261031`; 434 predicciones, una negativa. Es el modelo de `output/phase_submission/FINAL_MODEL`. |
| Reajuste canónico posterior | Experimento de 48 ajustes que falló sus controles. No forma parte del ZIP adaptado. |

La entrega adaptada ya existía antes de esta revisión. No hemos entrenado otro
modelo ni regenerado predicciones durante la preparación del correo. Comprobamos
los archivos, su procedencia y las métricas publicadas.

## Modelos que generaron la entrega adaptada

Leímos `FINAL_MODEL/MODEL.json`, `upstream/BUNDLE.json`,
`router/FINAL_FREEZE.json`, `router/CONFIG.json` y el código de carga e inferencia.
El modelo conserva congelados estos nueve componentes de TRAIN40:

| Función | Componentes |
|---|---|
| Productores y PAIR de eventos | `E_A5`, `E_C2F`, `PAIR_E` |
| Productores y PAIR RGB | `R_A5`, `R_C2F`, `PAIR_R` |
| Contexto temporal | `E_CTX_MATCHED`, `R_CTX` |
| Cabeza original | `PHASE_ONLY_EVENT_S7_TRAIN40` |

La adaptación añade tres routers QA, semillas 7, 13 y 21, de 688 parámetros cada
uno. Combinan la salida original, los contextos RGB y de eventos y una estimación
geométrica de TTC a partir de cajas causales. Aprenden los pesos de fusión según
las características de calidad y trabajan en coordenadas de TTC inverso.

El ajuste final usó 2.608 consultas de 24 grabaciones de carretera y Slider-500,
300 pasos de AdamW por semilla, learning rate 0,01 y weight decay 0,01. Excluyó
las ocho grabaciones de competición y Slider-750/1000. La campaña anterior había
inspeccionado resultados de esas grabaciones, por lo que las exclusiones finales
no permiten presentar el experimento como una reserva independiente.

La denominación precisa es adaptación de la fusión. No hicimos fine-tuning de
todos los encoders para producir este ZIP. Tampoco incorporamos la intervención
PAIR canónica ni los reajustes posteriores rechazados.

## Comprobación y cambio de nombre del ZIP

Verificamos el archivo contra `replay/COMPLETE.json`: diez archivos de resultados,
434 filas de predicción finitas, cuatro columnas por fila y una predicción
negativa conservada. Al contar filas excluimos la cabecera de cada archivo.
No corregimos, interpolamos ni recortamos valores en el archivo que se adjuntará.

Por petición del usuario, renombramos:

```text
output/phase_submission/replay/results.zip
→ output/phase_submission/replay/PHASE-TTC-Adapt.zip
```

Mostramos el archivo seleccionado en `explorer.exe`. Su SHA-256 antes y después
del cambio de nombre coincide con el recibo de la entrega:

```text
e7e41820003f18a28f2ef23e634992b05d7789ecf48364b2debe659cbb9a20e7
```

El cambio de nombre es local, fuera de Git, y no cambia los bytes. Los recibos
históricos y comandos antiguos todavía pueden citar `results.zip`; para encontrar
el adjunto actual debe usarse `PHASE-TTC-Adapt.zip`. No reescribimos recibos sellados.

## Comparación con la entrega original

Usamos la auditoría corregida de
`output/phase_safe_fusion/evaluator_audit/AUDIT.json`, publicada también en
[evaluator_audit.json](../research_log_20261031_sign_safe/evidence/evaluator_audit.json).
La corrección de las fuentes de CCRs1-low y CCRs1-medium recuperó los once targets
antes ausentes: la comparación actual contiene las 434 consultas.

Comprobamos los hashes de `TARGETS.npz`, `BASELINE_PREDICTIONS.npz`,
`TARGET_ROWS.json` y ambos ledgers contra la auditoría, además del hash del ZIP
adaptado contra su COMPLETE. Para comparar aplicamos a cada entrega la regla de
interpolación de negativos descrita por los organizadores, sin modificar los ZIP.

La siguiente tabla resume medias por consulta sobre el conjunto indicado. Las
reducciones son `100 × (1 − adaptado/original)`, calculadas antes de redondear.
No son promedios macro por secuencia ni cifras oficiales nuevas.

| Conjunto / métrica | Original | Adaptado | Reducción relativa |
|---|---:|---:|---:|
| 434 consultas: error relativo medio | 49,894242 % | 17,249803 % | 65,43 % |
| 434 consultas: MAE | 0,824976 s | 0,335899 s | 59,28 % |
| Carretera, 365 consultas: error relativo | 11,917132 % | 7,469419 % | 37,32 % |
| Carretera, 365 consultas: MAE | 0,305401 s | 0,185200 s | 39,36 % |
| Sin Slider-750, 383 consultas: error relativo | 18,730071 % | 9,417157 % | 49,72 % |
| Sin Slider-750, 383 consultas: MAE | 0,329643 s | 0,193263 s | 41,37 % |
| Slider-750, 51 consultas: error relativo | 283,931058 % | 76,071442 % | 73,21 % |
| Slider-750, 51 consultas: MAE | 4,544831 s | 1,407064 s | 69,04 % |

El adaptado reduce tanto MAE como error relativo en cada una de las diez
secuencias con este evaluador local. Ese resultado corresponde al modelo del
31 de octubre, no a los experimentos canónicos posteriores.

## Qué pidió el organizador y qué preparamos

El usuario proporcionó la respuesta del equipo EvTTC: evaluaron el original,
alinearon o interpolaron targets a los timestamps de las cajas y sustituyeron
las dos predicciones negativas mediante interpolación entre salidas válidas.
Solicitaron CPU, GPU, lenguaje, cámaras, focales y datos de publicación.

Contrastamos la información técnica con la auditoría histórica:

- CPU AMD Ryzen 9 9955HX; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU;
  Python 3.11.15, PyTorch 2.11.0+cu128 y CUDA 12.8 sobre Windows.
- Carretera: par derecho Prophesee EVK-v4 / FLIR Blackfly-S. La identificación
  nominal de 16 mm se infiere por sus campos de visión calibrados y se pide
  confirmación a los organizadores.
- Slider: DVXplorer / DAHENG MER2, con focales calibradas en píxeles. No asignamos
  una distancia focal nominal en milímetros sin confirmación.
- Autor: Álvaro Schwiedop Souto. Para el borrador usamos la condición de
  estudiante de la Universidade de Santiago de Compostela indicada por el
  usuario; los borradores históricos que dicen «sin afiliación» no se actualizaron.

Redactamos versiones completa y breve del correo, con petición de evaluación
separada del adaptado. Indicamos que el mensaje debe declarar la exposición
histórica del benchmark y la predicción negativa conservada. No enviamos correos
ni realizamos una subida a la plataforma. No tenemos confirmación de que el
usuario haya enviado el adjunto por su cuenta.

## Discrepancia de Slider-750 y límites

Para el original, el PDF comunicado por los organizadores indica 132,00 %, mientras
que el evaluador local corregido da 283,931058 %. Las otras nueve filas coinciden
con el PDF al redondear. Pedimos aclarar los targets alineados o el procedimiento
del evaluador antes de confirmar la publicación del resultado original.

El 76,071442 % del adaptado es local. No puede compararse directamente con el
132 % oficial del original mientras no se resuelva esa diferencia de protocolo.
La mejora local justifica solicitar evaluación oficial; no acredita SOTA ni
conservación de precisión Test12. El [análisis de integración](04_ANALISIS_INTEGRACION_MAIN.md)
separa esas limitaciones científicas de la preparación técnica de la rama.
