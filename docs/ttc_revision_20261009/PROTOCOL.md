# Revisión de precisión, coste y comparación — 9 de octubre de 2026

Esta revisión conserva intacta la campaña `sota_campaign_20261008`. Los resultados
nuevos se generan en `artifacts/ttc_revision_20261009`. No es una certificación SOTA.

## Cambios evaluados

1. `EventPreparer` conserva un lector HDF5 y un buffer de eventos crudos por
   secuencia, con límite de retención de 128 MiB. Invalida el buffer al cambiar
   archivo, identidad o intervalo incompatible. Recalcula la ROI y normalización
   en cada consulta. Comparte las 12 ventanas distintas que antes se voxelizaron
   24 veces. La entrada H8 sigue siendo `[8,3,12,128,128]`.
2. H8 prepara solo su entrada. Garl event-only lee solo sus dos ventanas de
   100 ms; ya no recibe el coste de construir el historial H8.
3. La ejecución compacta elimina ocho observaciones de relleno de los productores
   y mantiene las características en GPU. Cambia el orden de algunas operaciones
   de coma flotante: exige admisión numérica, no se denomina bit-exact.
4. `DirectTTCHead` reemplaza experimentalmente la regresión residual en fase por
   una salida continua en segundos, parametrizada mediante `asinh/sinh`. La salida
   admite signo, cruza cero sin salto y tiene límite arquitectónico ±60 s.
   La pérdida combina residuos en segundos, relativos con denominador mínimo
   0,5 s y en coordenada asinh. La transformación asinh no hace simétricos en
   segundos todos los gradientes alrededor de un target distinto de cero.
5. La cabeza nueva usa antigüedad y separación entre observaciones. Excluye los
   canales de disponibilidad cuya definición difiere entre el cache histórico
   y el adaptador de transferencia. No se modifica silenciosamente H8 original.

## Entrenamiento

Se conserva el encoder/productores A5, C2F y PAIR congelados. Se entrenan tres
cabezas nuevas, semillas 7/13/23, con 2.500 actualizaciones cada una, batch 256,
AdamW y la receta fijada en `configs/experiment/ttc_revision_20261009.json`.
Se utilizan exclusivamente las características verificadas de TRAIN40 y sus
88.744 targets. Media, escala y pesos por secuencia/bucket proceden de TRAIN40.
El checkpoint elegido es el final fijado; no hay selección por Dev32 ni FCWD.
El batch histórico de H8 era 128; también cambia a 256. Esta comparación de
candidatos no es una ablation que aísle únicamente la función de pérdida.
La nueva cabeza usa LR constante de 3e-4; H8 tenía warmup de 100 actualizaciones
y descenso coseno hasta 3e-5. También cambia el tamaño oculto, de 160 a 64.

La supervisión TRAIN40 cubre aproximadamente −10 a +10 s. No hay datos nuevos
que permitan afirmar aprendizaje fiable de los TTC de más de 10 s en EvTTC.
Tres cabezas sobre productores compartidos no equivalen a tres entrenamientos
independientes de todo el sistema. El entrenamiento previo usó teacher RGB y
supervisión geométrica; esta revisión no demuestra una mejora atribuible a JEPA.

Los checkpoints conservan modelo, optimizador, RNG y actualización completa.
El archivo `TRAINING_FREEZE.json` vincula configuración, fuentes, índice y cache.
Los recibos guardan entorno, commit, semilla, fechas, hashes y estado. El límite
autorizado es ocho horas de GPU local; no es una obligación de agotarlo.

## Comparación de precisión

La vista principal conserva las predicciones nativas. Otra vista aplica el mismo
límite de ±60 s a las predicciones finitas de todos los modelos. Las predicciones
ausentes o infinitas siguen siendo fallos, incluso en la vista limitada. La
métrica de cohorte completa queda sin calcular cuando hay un fallo: nunca se
mejora un promedio eliminando el caso difícil.
Además de cada semilla, se conserva la mediana fija de las tres predicciones
(`H8_median3` y `Direct_median3`). Esa agregación no consulta GT ni elige una
semilla; permite evaluar una salida única del sistema que ejecuta tres cabezas.

El criterio de elegibilidad depende solo del GT: finito y distinto de cero.
Se conservan todas las filas de población y se informa cobertura. Se publican
MAE, mediana, RMSE, RTE, sesgo, colas, errores de signo y fallos de aviso urgente
(GT positivo ≤1 s, predicción ≤0 o >1 s). Los intervalos remuestrean secuencias
completas y, como análisis más conservador, familias de escenario; ninguna
ventana se trata como réplica independiente.

Dev32 ya se había observado y FCWD solo tiene tres secuencias. Son diagnósticos
de transferencia, no tests ciegos. Las cajas son oráculo. H8 consume 650 ms de
historia causal, Garl dos ventanas que cubren 200 ms; ambos conservan sus entradas
nativas, no un presupuesto de contexto igualado. Garl full usa además RGB. Las
proyecciones RGB/eventos y el contrato oficial de Garl mantienen las limitaciones
documentadas por la campaña original. El full FCWD sigue sin mapping espacial
certificado; no se improvisa uno para obtener un resultado.

## Latencia

Se detienen V13 A5 y C2F en un checkpoint completo y se reanudan al terminar.
El runner rechaza otro proceso Python que figure en GPU. Registra clientes de
escritorio: no afirma exclusividad física bajo WDDM.

Cada muestra mide en la misma pasada preparación CPU, wrapper sincronizado y
tiempo total. La suma CPU + wrapper coincide con E2E. El intervalo de CUDA events
incluye huecos de envío desde el host y no es una suma de tiempos de kernels.
La carga del modelo queda fuera; la lectura HDF5 queda dentro. La representación
no se precalcula para ocultar preparación. El filesystem está caliente.
La obtención de las ROI oráculo y la espera física de sincronización RGB no se
incluyen: son costes adicionales de una aplicación desplegada. La preparación
full conserva el adaptador RGB publicado, sin una nueva caché de frames.
Tampoco reutiliza eventos crudos entre consultas. Sus tiempos representan esa
implementación, no un límite de rendimiento de la arquitectura. El baseline
`h8_legacy_three` conserva la preparación compartida histórica, incluida la
entrada Garl que H8 no consume. El ahorro de la ruta dedicada incluye eliminar
ese trabajo y no debe interpretarse íntegramente como aceleración de la red.

Se comparan H8 original con tres cabezas, H8 compacto con tres y con una cabeza,
Garl event-only y Garl full, FP32 sin TF32, dos threads CPU y batch externo uno.
Hay dos warmups y cinco repeticiones. El modo independiente usa la primera
consulta de cada familia, sin retención cruda. El cronológico usa las primeras
cinco de la primera secuencia de cada familia, con buffer acotado. Las consultas
se seleccionan por identidad y tiempo, nunca por error o duración medida.
El orden de sistemas se baraja por consulta con semilla 20261009; cada sistema
conserva su propio buffer cronológico. El piloto inicial, detenido por una ruta
de memoria innecesariamente costosa y orden por bloques, se conserva completo
hasta su última pasada terminada en `pilot_full_raw`; no se mezcla con la tabla
final. Sin retención, H8 filtra la ROI mientras lee bloques de eventos.
La prueba inicial de 128 MiB se conserva. Una ablation adicional utiliza 512 MiB
por sistema (`--cache-mib 512`) para cubrir las uniones H8 de mayor tasa de
eventos; se publica por separado, con su propio baseline contemporáneo.
El replay offline permite 512 MiB y precarga una consulta en CPU mientras la GPU
procesa la anterior. Ese tiempo de throughput no se presenta como latencia.
El benchmark de latencia utiliza consultas Dev32. El replay de precisión FCWD
conserva su preparador nativo; esta campaña no mide una aceleración E2E de FCWD.
Los intervalos de latencia remuestrean ocho familias completas, manteniendo
unidas consultas y repeticiones emparejadas. No certifican independencia entre
sesiones ni comportamiento de un despliegue. Los CSV contienen también cada
familia y los tiempos crudos, incluidos los warmups excluidos del resumen.

La admisión compacta usa tolerancia `atol=0.01 s, rtol=0.0001`. Las ocho consultas
del benchmark son una comprobación inicial; el replay completo informa si se
cumple para toda la población. Un fallo no se oculta detrás del promedio.

## Reproducción

Desde la raíz, use un Python 3.11 con las dependencias del proyecto y CUDA. En
este host el entorno funcional está en el repositorio hermano `e-jepa-ttc`;
la `.venv` copiada en V12 está incompleta. No se reinstala ni altera ese entorno.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$py = '../e-jepa-ttc/.venv/Scripts/python.exe'
& $py -m pytest tests/unit/test_ttc_revision_inputs.py tests/unit/test_ttc_revision_head.py tests/unit/test_ttc_revision_score.py
& $py -m operational.ttc_revision.train
& $py -m operational.ttc_revision.benchmark
& $py -m operational.ttc_revision.evaluate replay --manifest artifacts/sota_campaign_20261008/dev32_expanded_rgb/QUERY_MANIFEST.json --output artifacts/ttc_revision_20261009/replay_dev32
& $py -m operational.ttc_revision.evaluate join --replay-dir artifacts/ttc_revision_20261009/replay_dev32 --campaign artifacts/ttc_revision_20261009 --original docs/sota_campaign_20261008/evidence/dev32_expanded_rgb/SCORED_PREDICTIONS.csv --output artifacts/ttc_revision_20261009/DEV32_PREDICTIONS.csv
& $py -m operational.ttc_revision.score --predictions artifacts/ttc_revision_20261009/DEV32_PREDICTIONS.csv --manifest artifacts/sota_campaign_20261008/dev32_expanded_rgb/QUERY_MANIFEST.json --output artifacts/ttc_revision_20261009/dev32 --methods H8_seed7 H8_seed13 H8_seed23 H8_median3 Direct_seed7 Direct_seed13 Direct_seed23 Direct_median3 public_Garl_event_lhr public_Garl_rgb_event_full
& $py -m operational.ttc_revision.evaluate replay --fcwd --manifest artifacts/sota_campaign_20261008/fcwd_inference/QUERY_MANIFEST.json --output artifacts/ttc_revision_20261009/replay_fcwd
& $py -m operational.ttc_revision.evaluate join --replay-dir artifacts/ttc_revision_20261009/replay_fcwd --campaign artifacts/ttc_revision_20261009 --original docs/sota_campaign_20261008/evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv --output artifacts/ttc_revision_20261009/FCWD_PREDICTIONS.csv
& $py -m operational.ttc_revision.score --predictions artifacts/ttc_revision_20261009/FCWD_PREDICTIONS.csv --manifest artifacts/sota_campaign_20261008/fcwd_inference/QUERY_MANIFEST.json --output artifacts/ttc_revision_20261009/fcwd --methods H8_seed7 H8_seed13 H8_seed23 H8_median3 Direct_seed7 Direct_seed13 Direct_seed23 Direct_median3 public_Garl_event_lhr public_Garl_rgb_event_full
& $py -m operational.ttc_revision.report
```

El benchmark se niega a sobrescribir un resultado previo. Para repetirlo, copie
la configuración a un experimento con otro directorio de salida. El replay
reanuda fragmentos por consulta solo si coincide la identidad de código/datos.
Los resultados se publican también cuando el candidato empeora; no se selecciona
una semilla favorable ni se promete superar a Garl en todas las métricas.

La ablation adicional de capacidad se ejecuta con
`python -m operational.ttc_revision.benchmark --config artifacts/ttc_revision_20261009/CACHE512_CONFIG.json --cache-mib 512`.
Su configuración tiene un destino distinto al benchmark principal. No deben
ejecutarse dos mediciones o replays GPU simultáneamente.

## Experimento adicional de ejecución

`python -m operational.ttc_revision.compiled` evalúa el backend `cudagraphs` de
PyTorch con las mismas entradas FP32. Registra por separado las llamadas de
compilación/calentamiento y el wrapper caliente. Conserva validaciones; puede
haber cortes del grafo y no se supone una aceleración por el nombre del backend.
El replay admite `--compiled-backend` solo con el recibo de admisión completo y
las mismas fuentes y pesos. La tolerancia del batch compacto sigue siendo la
declarada; el batch canónico compilado usa la misma tolerancia y registra además
si se conserva igualdad exacta en las consultas de admisión.
El tiempo del wrapper compilado no sustituye una medición end-to-end.
Referencia de API: [torch.compile de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).

## Inferencia sin consultar el TTC de referencia

Las dos rutas reciben una consulta sensorial del manifest, sin abrir el GT TTC.
Sí utilizan las ROI oráculo incluidas en ese manifest: no son entradas libres de
toda anotación. La ruta
H8 optimizada conserva las tres cabezas originales. `DirectRuntime` permite
probar los tres checkpoints nuevos de forma explícita y comprueba sus hashes,
normalización, arquitectura y endpoint de entrenamiento.

```python
import json
from pathlib import Path
from operational.evttc_transfer.models import FrozenModels
from operational.ttc_revision.inputs import EventPreparer
from operational.ttc_revision.runtime import H8Runtime
from operational.ttc_revision.direct_runtime import DirectRuntime

manifest = Path("artifacts/sota_campaign_20261008/dev32_expanded_rgb/QUERY_MANIFEST.json")
row = json.loads(manifest.read_text(encoding="utf-8"))["rows"][0]
frozen = FrozenModels(Path("artifacts/train40_system_20261005"), "cuda")
runtime = H8Runtime(frozen, batch=8, on_device=True)
# Alternativa experimental, no promoción automática del candidato:
# runtime = DirectRuntime(frozen, Path("artifacts/ttc_revision_20261009"))
with EventPreparer(cache_bytes=512 * 1024**2) as preparer:
    predictions = runtime.predict(preparer.prepare(row)["own_events"])
print(dict(zip((7, 13, 23), predictions.tolist(), strict=True)))
```

El benchmark de H8 mide sus cabezas originales, no las de Direct. El informe
separa esa distinción y la admisión de características compactas. La evaluación
principal de Direct usa las cabezas en CPU y el replay completo verifica también
las salidas de su runtime GPU canónico y compacto con la tolerancia declarada.
No se sustituye su coste por el de H8. Direct no proporciona probabilidades de
riesgo ni intervalos calibrados; es un candidato de regresión TTC.

La entrega adicional se regenera con `python -m operational.ttc_revision.package`
tras completar resultados e informe. Requiere el repositorio V12, los productores
congelados y los datos locales: el ZIP de revisión no redistribuye datos crudos.

`python -m operational.ttc_revision.training_diagnostics` evalúa los endpoints
sobre TRAIN40 en CPU sin actualizaciones. Ese error descriptivo sin ponderar
corresponde a datos vistos; no es validación ni sirve para elegir checkpoints.
El generador de informes añade automáticamente los diagnósticos de productores,
cambio de distribución y concentración del error desde los replays completos.
