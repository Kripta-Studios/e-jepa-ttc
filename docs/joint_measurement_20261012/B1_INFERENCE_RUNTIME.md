# B1: optimización de inferencia sin cambiar el experimento

Los cuatro endpoints terminaron el 16 de octubre a las 16:40:49 de Madrid.
La evaluación eager empezó después y conserva fragmentos durables de 256 filas.
El corte de las 17:01 medía aproximadamente 11,3 consultas/s sobre P.
El evaluador original no utiliza CUDA Graphs y lee los packs lossless en serie.
La caché LRU de entrenamiento no evita lecturas en una pasada sin repeticiones.

Se prepara una prueba de captura del forward sobre el `CaptureCore` ya usado
por el entrenamiento. No cambia pesos, precisión FP32, TF32, batch4 ni tail2,
muestreo, clocks, scorer, selección, costes preparados o exports. Los controles
de dominio se comprueban tras cada replay. La salida se copia para que el replay
siguiente no sobrescriba una predicción todavía utilizada por el llamador.

La admisión exige igualdad exacta de fase y TTC en los dos endpoints sobre las
primeras 64 consultas P y las dos últimas, además de igualdad con el fragmento
eager ya guardado para las primeras 64 del brazo CORRESPONDENCE. También exige
pesos y RNG intactos, rechazo de un reloj inválido y una razón mediana de tiempo
graph/eager de como máximo 0,9. Son 132 predicciones comparadas, cero updates.
No se examinan métricas H/V para decidir esta optimización.

El código conserva las fuentes científicas originales. Una capa separada cambia
el forward sólo durante la generación de predicciones. Cada cohorte registra
`INFERENCE_RUNTIME.json`, ligado al registro, prueba de admisión y hashes de los
fragmentos previos. El medidor de coste y el export siguen sus rutas originales.
Los checks de paridad no equivalen a una prueba sobre todos los inputs posibles.
La aceleración global debe medirse después en fragmentos nuevos de la misma ruta.

Código: `operational/joint_measurement/correspondence_inference_graph.py`.
Registro y resultados: `runtime/B1_inference_graph_v2` dentro de la campaña.
La implementación por sí sola no acredita admisión ni aceleración.

El primer intento se conserva en `runtime/B1_inference_graph_v1`: su guard de
exclusión identificó el servicio `mlflow.exe mcp run`, cuyo cwd era V17, como
cómputo concurrente. No llegó a inferir ni reservar GPU. La revisión v2 permite
ese entrypoint exacto de control; sigue rechazando entrenamientos MLflow y otros
procesos Python del proyecto, y mantiene la exclusión GPU independiente. No se
detuvo ni modificó el servicio MCP. El evaluador original se detuvo con 66
fragmentos y 16.896 predicciones durables, conservadas y verificadas por hash.

Referencia de la API: [CUDA Graphs en PyTorch](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
Context7 no estaba disponible; se consultó la documentación oficial. La versión
instalada y la ejecución local determinan la compatibilidad efectiva.
