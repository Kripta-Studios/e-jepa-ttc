# Coste de las dos rutas Garl: registro previo

El medidor está implementado y registrado; **la medición real está pendiente**.
B1 conserva prioridad y no se ha cargado otro modelo junto a su entrenamiento.
Este trabajo completa una comparación de implementación, no selecciona pesos ni
recetas con P, H, V, Dev32 o FCWD.

El coste histórico importado de V16 mide el modelo completo Garl con su
adaptador de entradas ya optimizado. Sus dos pasadas son repeticiones de esa
ruta, no dos implementaciones distintas. El nuevo registro compara:

| Ruta | Preparación | Modelo |
|---|---|---|
| UPSTREAM | `preprocess_data` y extracción de eventos del checkout fijado | Checkpoint público full completo |
| LOCAL | Adaptador local de lectura, normalización, ROI y timevolume | Exactamente el mismo modelo residente |

La revisión del árbol Git `256661242b8a7f5e56aa3c1c02348b30f6e89de6`
confirma que **el timevolume original ya está vectorizado**. No se ha construido
un bucle lento para exagerar una aceleración. Las dos rutas conservan RGB0,
RGB1, EVENT0 y EVENT1, 46 canales, entradas de 128×128, normalización ImageNet
sólo para RGB, registro eAP `x+5`, ventanas y tiempos nativos, y emisión firmada
sin clipping `0.1 / (1 - height_t0 / height_t1)`.

## Población y controles

Se registraron 64 consultas de P, repartidas entre sus 24 grupos y 24 secuencias.
La selección recorre grupos ordenados y tokens ordenados de forma alternada;
no abre columnas TTC, alturas anotadas ni resultados de modelos. El join de los
53.890 IDs P con el índice nativo se valida antes de seleccionar: duplicados o
IDs ausentes abortan el registro. Todas las rutas de eventos y archivos RGB de
las 64 consultas seleccionadas existen. Una consulta que falle después no se
sustituye por otra.

Cada consulta se mide por ambas rutas en dos pasadas, invirtiendo el orden.
Son 128 pares y 256 observaciones de ruta. Cada par exige inputs float32
idénticos y alturas/TTC idénticos; un fallo deja evidencia y detiene la admisión.
Una salida TTC no finita queda registrada explícitamente y no se retira del
coste. Los hashes de código local y del checkout nativo quedan fijados.

El registro es
`artifacts/joint_measurement_20261012/cost/garl_routes/REGISTRATION.json`, SHA-256
`63e937fd7e220c4b042a86c8a834d63ad2299c5ffc840db8910e2973d93457a8`.

## Alcance del coste

La configuración es CPU FP32, un hilo PyTorch, batch 1, en este mismo equipo.
Se mide por separado carga del modelo, primera ejecución preparada y 64
ejecuciones preparadas después de 20 calentamientos. El replay real mide lectura,
decodificación, normalización, ROI, timevolume, forward completo y emisión.
Los resúmenes contienen media, p50, p95, p99 y throughput derivado del tiempo
de servicio; también se guardan todas las observaciones individuales.

La caché de disco del sistema operativo no se vacía ni se controla: la primera
ejecución del modelo **no es una medida de disco frío**. El modelo permanece
residente; los lectores se reabren por consulta y no se reutilizan tensores
sensoriales ni predicciones de la otra ruta. RSS se observa después de cada
ruta y se conserva el máximo acumulado del proceso; esos valores no son picos
aislados de memoria de cada implementación.

El runner rechaza otros procesos de cómputo del proyecto y comparte su bloqueo
de exclusión con los escritores GPU. Exige al menos 6 GiB disponibles antes de
cargar Garl. Usa CPU y no inicializa CUDA ni añade updates o consumo a scopes
GPU. Las comprobaciones de recursos existentes siguen vigentes.

## Ejecución pendiente

Después de que terminen B1 y los otros evaluadores activos:

```powershell
$env:PYTHONPATH = "$env:LOCALAPPDATA\Temp\garl_parity_cv2_cp311;src;."
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& ..\e-jepa-ttc\.venv\Scripts\python.exe -B -m operational.joint_measurement.cost_guard_overlay garl
```

Se reutiliza la instalación aislada de OpenCV de la auditoría anterior; no se
modifica el entorno compartido. Los pares completos se conservan para reanudar;
cada nueva sesión registra de nuevo su carga, primera ejecución y calentamiento.
Un par fallido no se sobrescribe automáticamente.

La verificación CPU cubre selección determinista, poblaciones ambiguas,
alternancia de orden, paridad exacta, costes inválidos, cálculos conocidos,
conservación de TTC no finito y rechazo de evidencias incompatibles. Son pruebas
del medidor, no mediciones de Garl ni evidencia de precisión TTC.

Esta medición tampoco resolverá por sí sola la paridad pendiente del pipeline
sensorial de fusión V18. Ese límite se conserva por separado en
[FUSION_COST_DIAGNOSTIC.md](FUSION_COST_DIAGNOSTIC.md).

El 16 de octubre se añadió `cost_guard_overlay`: permite únicamente el servicio
de control `mlflow.exe mcp run`, que el guard histórico confundía con cómputo
por tener cwd dentro del proyecto. Conserva exclusión GPU, comprobaciones de
recursos, población, modelo, tolerancias y fuentes originales. No mata ese
servicio ni permite entrenamientos MLflow concurrentes. Su registro queda en
`runtime/cost_guard_overlay_v1/REGISTRATION.json`; cada medidor guarda además
`GUARD_RUNTIME.json`. Para el diagnóstico de shapes de fusión se usa el mismo
comando con `shape` en lugar de `garl`. Ambos se ejecutan secuencialmente y sólo
cuando no queda otro proceso de cómputo del proyecto.
