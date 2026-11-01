# Preparación del envío eAP 3D Detection

El usuario aportó los requisitos oficiales y `sample_submission.zip` el
1 de noviembre. Esta continuación prepara predicciones test y el ZIP de
resultados; no sube una participación a CodaBench ni modifica el modelo.

## Contrato y población

La competición [eAP 3D Detection, 17291](https://www.codabench.org/competitions/17291/)
requiere exactamente un `submission.json` en la raíz del ZIP, con formato
`eap_native_detection_result`, las ocho clases publicadas y campos
`class_name`, `translation_m`, `size_lwh_m`, `yaw_rad`, `velocity_mps` y `score`.
Cada token del test debe figurar, incluso cuando no haya detecciones.

La plantilla aportada contiene 35.940 tokens, idénticos al conjunto completo de
`data/test.parquet`: doce secuencias. No contiene cajas ni etiquetas test.
El manifiesto y los 45 TAR RGB, 40.315.494.400 bytes, se verificaron contra los
SHA-256 LFS del release `NAIL-HNU/eAP-dataset` fijado en
`a814a946e60d37f7242358af5f344542177e50c5`. Solo faltaban nueve TAR RGB y se
descargaron con `hf download`; los otros 36 se reutilizaron.

## Productor y admisión nueva

Se utiliza el [candidato Dfull RGB con ranking TRAIN](RELEASE_CANDIDATE_20261101.md),
sin reentrenamiento, nuevos umbrales, NMS adicional o selección con feedback test.
Se conservan todas las propuestas admitidas por su decoder histórico. El
exportador elimina únicamente campos diagnósticos ajenos al formato oficial.

`scripts/infer_detection_benchmark.py` introduce una admisión explícita de medios
test. No modifica el checkpoint, la instantánea de código o el manifiesto train,
ni presenta test como train para eludir el loader. Ejecuta la verificación
histórica completa de pesos y fuentes, restaura estrictamente el mismo modelo
y hereda el lector de imágenes congelado. El acceso a anotaciones devuelve una
lista vacía y la admisión rechaza rutas de etiquetas en test.

La preparación se adelanta un frame en un único hilo CPU, con batch GPU uno.
Los módulos del ranker se cargan en otro proceso para respetar el aislamiento
del productor. La nueva ruta obtuvo paridad exacta de baseline y ranking en los
ocho frames de comprobación TRAIN ya guardados. Esta comprobación acredita
paridad funcional en esa población; no proporciona una métrica sobre test.

## Ejecución y recuperación

```text
python scripts/infer_detection_benchmark.py run --release RELEASE_ROOT --data-root DATA_ROOT --template sample_submission.zip --split test --output OUTPUT_ROOT
```

La ejecución test exige toda la población de la plantilla; no admite subconjuntos
con `--tokens`. El modo train solo sirve para comparar tokens explícitos sin
leer sus etiquetas. Sus recibos se marcan `PARITY_ONLY_NOT_FOR_SUBMISSION`.

El directorio local de esta ejecución es
`D:/PHASE_TTC_SUBMISSION_20261101/test_full`. Contiene:

- `ADMISSION.json`: población, hashes de modelo, ranker, adaptador, manifiesto,
  plantilla y TAR, batch y política de scores fijados antes de inferir.
- `STATUS.json` y `progress.jsonl`: progreso observado y tiempos.
- `chunks/`: baseline, diagnóstico, predicciones ordenadas por el ranker y
  recibo `COMPLETE.json` por bloque de 256 frames.
- `submission.json`, `submission.zip` y `SUBMISSION_RECEIPT.json`: productos
  finales, creados al terminar todos los bloques.

Para reanudar después de una interrupción se ejecuta el mismo comando y la misma
versión del script. Los bloques completos se verifican y reutilizan; el bloque
incompleto se vuelve a ejecutar. Un cambio en la admisión rechaza la reanudación.
No ejecutar dos escritores sobre el mismo directorio simultáneamente.

`READY_TO_UPLOAD_NOT_SUBMITTED` solo se escribe después de verificar cobertura
exacta, valores finitos, dimensiones positivas, clases, vectores de velocidad,
confianzas en rango, conservación de frames vacíos, único miembro raíz y hash
del JSON descomprimido. El proceso no solicita etiquetas privadas ni calcula AP
test. La finalización local no implica que CodaBench haya aceptado el archivo.

Las comprobaciones de esta preparación están en
[evidence/benchmark_submission_20261101](evidence/benchmark_submission_20261101/MANIFEST.json).
Son 17 pruebas aprobadas de integridad del paquete y preparación del benchmark,
además del replay real de paridad y la verificación de inputs públicos.
El estado terminal y el hash del ZIP se consultan en el recibo de la ejecución,
no se infieren de esta documentación.

La AP 0,18502078 corresponde exclusivamente a selección de desarrollo. La
puntuación oficial permanece desconocida hasta evaluar el envío en CodaBench.
