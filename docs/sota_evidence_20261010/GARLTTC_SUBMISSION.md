# Alcance actualizado: GarlTTC Benchmark

Estado al cierre local del 10 de octubre: H8 y Garl event-only completaron las
6.762 predicciones cada uno; ambos ZIP están empaquetados y no enviados.
No hay score oficial. Los [recibos publicados](evidence/test12/INFERENCE_RESULT.json)
y el [estado de evaluación](STATUS_20261010.md) actualizan la descripción histórica
de preparación que se conserva abajo. Los pesos y datos permanecen fuera de Git.

Por instrucción del usuario, esta campaña se centra en GarlTTC. REACT queda fuera
del alcance; su reproducción ya no es un requisito de esta comparación. La
auditoría anterior conserva sus resultados históricos.

El objetivo es medir la clasificación oficial de GarlTTC, sin anticipar victoria.
Una comparación limitada a GarlTTC no establece por sí sola superioridad frente
a todos los métodos existentes.

## Comparación con los checkpoints descargados

La comparación recalculada está en
[`REPORT.md`](evidence/garl/REPORT.md).
Incluye los checkpoints públicos `paper_event_only_lhr.pth` y
`paper_ours_full.pth`, verificados por SHA-256 contra los recibos de inferencia.
Las predicciones históricas se reutilizan únicamente después de comprobar sus
hashes. Los CSV conservan cobertura, errores extremos, sesgo por rango, avisos
urgentes y bootstrap por secuencia, además de una sensibilidad separada con
el mismo límite ±60 s para todos.

El [informe MiD](evidence/garl/MID_REPORT.md)
ejecuta los bytes verificados del scorer oficial sobre las etiquetas locales.
Incluye MiD por banda, FR y diferencias pareadas con bootstrap de secuencias
completas. H8 mejora a Garl de eventos en MiD medio de Dev32 y empeora en FCWD;
frente a RGB+eventos en Dev32, el intervalo de la diferencia incluye cero.
Estas cohortes no contienen TTC negativos: MiDn y `overall_MiD` quedan sin
valor, sin redistribuir su peso. `MID_OFFICIAL_SCORER.csv` conserva también
conteos de MiD inválidos y los casos fuera de las cuatro bandas.

El ejecutable `operational.garl_comparison.test_event` aplica el checkpoint
event-only a las mismas 6.762 entradas oficiales, con su representación nativa
de 40 planos y conversión TTC sin clipping. Usa CPU y no consume el presupuesto
GPU reservado para H8. El piloto final de tres muestras está en
`artifacts/garl_checkpoint_comparison_20261010/test12_pilot_final/`.
Garl avanza en CPU sobre las primeras once secuencias verificadas mientras se
completa el último HDF5 en E:. Un bloqueo de archivo impide dos escritores
simultáneos. La cola reutiliza esas predicciones, completa Garl, continúa la
preparación H8, genera sus predicciones y solicita reanudar V13.

La primera reconstrucción del último HDF5 desde una descarga parcial falló
su SHA-256 y se preservó como `.sha256_failed`; no se utilizó para inferencia.
La descarga nueva parte de cero y requiere `CLEAN_RAW_VERIFICATION.json`
con estado `COMPLETE` antes de continuar. El estado vivo del coordinador está
en `artifacts/garlttc_submission_20261010/E_CONTINUATION_STATUS.json`.

La comparación RGB+eventos sobre test12 necesita 36 shards RGB que no están
descargados; su comparación con etiquetas públicas Dev32 sí está incluida.
`COMPARISON_PLAN.json` registra ese inventario. Los ZIP completos se copiarán
a `E:/GarlTTC_dataset` tras comprobar tokens, CRC y SHA-256. Hasta entonces,
los estados de progreso no equivalen a una entrega completada.

Reproducción de la tabla local:

```powershell
python -m operational.garl_comparison.report --output artifacts/garl_checkpoint_comparison_20261010
python -m operational.garl_comparison.mid --output artifacts/garl_checkpoint_comparison_20261010
```

## Contrato confirmado

- Reto enlazado por los autores: https://www.codabench.org/competitions/17289/
- Referencia de formato: https://github.com/NAIL-HNU/Garl-TTC
- Población: 6.762 muestras de 12 secuencias, identificadas por `sample_token`.
- Objetivo primario: `overall_MiD`, menor es mejor.
- Pesos: `0.5 * MiDc + 0.3 * MiDs + 0.1 * MiDl + 0.1 * MiDn`.
- Rangos oficiales: c=(0,3], s=(3,6], l=(6,10], n=(-10,0].
- Publicar también FR por rango. Las etiquetas privadas impiden calcular aquí
  la puntuación oficial a partir de los inputs públicos.
- ZIP de envío: exactamente un `submission.json` en su raíz; sin CSV ni recibos.

La consulta web del reto no devolvió contenido reconocible del benchmark en esta
sesión. Su enlace y formato están confirmados por el repositorio oficial y por
el texto de reglas proporcionado por el usuario. La inscripción, aceptación de
participantes, límites vigentes y disponibilidad del envío requieren comprobarse
en una sesión del usuario; no se ha realizado inscripción ni envío.

## Plantilla recibida

`E:/GarlTTC_dataset/sample_submission.zip` fue inspeccionado sin extraer ni
sobrescribir archivos. Su CRC es válido y contiene solo `submission.json`.
Los 6.762 IDs coinciden exactamente con `data/test_inputs.parquet` descargado del
repositorio oficial. Sus 6.762 valores son 1,0 s: son marcadores de posición.
No son inferencias de H8, Direct ni Garl.

SHA256 del ZIP recibido:
`c83f10f926c9190cd1aa5a0d6f4c6eeb4741cc1a4552b2451c142c42c11414c5`.

## Preparación local

El exportador requiere predicciones reales en un CSV con columnas
`sample_token,prediction` y la identidad del checkpoint. Rechaza IDs duplicados,
ausentes o adicionales, valores no finitos y la sobrescritura de un ZIP existente.

```powershell
python -m operational.sota_evidence.codabench `
  --inputs artifacts/sota_evidence_20261010/sources/test_inputs.parquet `
  --predictions artifacts/garlttc_submission_20261010/TEST_PREDICTIONS.csv `
  --checkpoint RUTA_AL_CHECKPOINT_CONGELADO `
  --output artifacts/garlttc_submission_20261010/model_submission.zip
```

La campaña autorizada está implementada en `operational/sota_evidence/`.
El estado actual se registra en
`artifacts/garlttc_submission_20261010/CAMPAIGN_STATUS.json`; los contadores CPU y
GPU están en `PREPARATION_PROGRESS.json` e `INFERENCE_PROGRESS.json`.
El ZIP definitivo se llama `H8_GarlTTC_submission.zip` y se genera únicamente tras
completar todas las inferencias y comprobar su cobertura. La plantilla recibida
permanece intacta.

El candidato se fijó como H8 TRAIN40, con productores A5/C2F/PAIR compartidos y
mediana TTC de las tres cabezas 7/13/23. `MODEL_FREEZE.json` identifica todos sus
pesos, normalización y código. No hay ajuste de pesos, calibración ni selección
mediante etiquetas de test o resultados del leaderboard.

Los eventos se descargan desde la revisión
`a814a946e60d37f7242358af5f344542177e50c5` de `NAIL-HNU/eAP-dataset`.
`DOWNLOAD_PLAN.json` limita la descarga a los doce HDF5 necesarios; cada archivo
se comprueba contra su SHA-256 publicado antes de leerlo. No se descargan imágenes
RGB para este candidato.

`TRAIN_INPUT_QA.json` comprueba la geometría contra todo el índice TRAIN40 y
tensores reales contra la caché original. `GPU_TRAIN_PARITY.json` compara también
los productores y las tres cabezas con la ruta de entrenamiento. La preparación
paralela añade su comprobación de igualdad en `PARALLEL_INPUT_QA.json`.

## Condiciones del candidato

La inferencia consume eventos y las cajas oficiales suministradas. No recibe
píxeles RGB, pero tampoco incluye un detector de objetos. Usa ocho observaciones
separadas por 50 ms; cada una mantiene las tres ventanas nativas del productor.
La historia total ronda los 650 ms y puede reducirse únicamente al comienzo real
de una grabación, mediante la máscara original de validez.

Los timestamps de anotaciones y eventos pertenecen a relojes distintos. Las
ventanas se construyen en el reloj de eventos y se vinculan a las exposiciones
RGB mediante sus identificadores públicos. El modelo conserva la disponibilidad
de la ROI al final de la exposición, igual que en TRAIN40. Las observaciones
pasadas reutilizan la ROI de la consulta; no representan cajas históricas medidas
en cada instante.

La transformación de fase mantiene el límite nativo de magnitud TTC de 60 s;
no se añade clipping adaptado a test. El cálculo usa FP32, TF32 desactivado y el
batch de productores canónico de 16, seguido de las tres cabezas. No modifica
la geometría BF16 ni los checkpoints de V13.

Esta ruta emplea más historia que las dos ventanas del baseline Garl. Una futura
puntuación oficial debe acompañarse de esa diferencia de contexto y del coste
completo del sistema. Los tiempos del exportador sobre tensores preparados no
son una nueva medición de latencia desde eventos crudos. La latencia H8 del
informe anterior corresponde al sistema de tres cabezas.

`METHOD_CARD.json` recoge estas condiciones y los rangos de duración medidos.
`V13_SAFE_PAUSE.json` comprueba los checkpoints durante la pausa autorizada;
el coordinador retira exclusivamente sus marcadores y reanuda la cola V13 con
recuperación de I/O al finalizar, fallar o agotar el presupuesto de inferencia.

El primer intento de reanudar fue rechazado por recibos auxiliares C2F de un
guardado anterior en el mismo update 17.648. En V13,
`artifacts/rgb_port_20261008/pause_recovery_20261010/RESULT.json` documenta la
recuperación: se preservaron los recibos anteriores y se validaron las tres
pruebas pendientes mediante los validadores originales. Pesos, puntero,
recibo nativo, journal y pruebas pendientes mantienen sus hashes. No se
consumieron updates. El runtime debe completar todavía su restauración GPU
exacta al reanudarse; una solicitud de reanudación no demuestra ejecución.
Este procedimiento recupera la transacción, sin modificar el código congelado
del entrenador ni afirmar igualdad tensorial con la serialización antigua,
cuyos bytes ya no estaban disponibles.

Tras producir y validar un ZIP completo, una cuenta participante podrá enviarlo
y recuperar las métricas oficiales. La plantilla descargada permite comprobar
el formato sin proporcionar acceso a las etiquetas privadas.

## Diagnóstico de preparación CPU

Medición de una consulta de Mu6kaUjIz3 durante la ejecución de cuatro preparadores CPU y la descarga. No es un benchmark aislado ni latencia de inferencia GPU. Los valores se generan desde `CPU_DIAGNOSTIC_MU6.json`.

| Etapa | Segundos |
|---|---:|
| Lectura y recorte | 6.622 |
| Proyección de coordenadas | 1.349 |
| Voxelización de ventanas | 5.770 |

Total de la transformación: 13.792 s. La representación proyectada retiene 89,725,626 bytes en coordenadas/timestamps/polaridad. La comparación con la entrada ya preparada fue exacta. No hubo fallback ni updates de optimizador.

La cantidad de eventos y la repetición de voxelizaciones son costes relevantes del sistema completo. Esta medición no permite atribuir errores TTC: las etiquetas oficiales de test siguen siendo privadas.
