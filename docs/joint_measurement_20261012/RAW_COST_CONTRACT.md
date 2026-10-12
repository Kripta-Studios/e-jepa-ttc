# Coste desde eventos crudos

La ruta `raw_pair` reutiliza la materialización HDF5 nativa y selecciona después
las dos últimas ventanas del ROI común. Conserva el coste de calcular la tercera
ventana cuando la consulta original era T3; no se presenta como un preprocesado
P2 optimizado. Devuelve exclusivamente tensores sensoriales y dt.

El 12 de octubre se verificó igualdad exacta de inputs y dt en una consulta por
cada uno de los ocho grupos V. El recibo es `verification/raw_pair/V.json`.
La selección usa hashes de tokens, sin examinar TTC. El alcance son esas ocho
consultas; no es una reproducción de toda la población ni una medida de latencia.

El comparador completo conserva E_A5_MATCHED, E_C2F_MATCHED, PAIR_E_MATCHED y
E_CTX_MATCHED, sus pesos, normalizador H y el historial V nativo. Sus operaciones
proceden de `RouteRuntime`; cargar únicamente esos cuatro componentes evita
introducir dependencias RGB ajenas a la ruta event-only.

La primera prueba detectó que el validador genérico esperaba una máscara por
instante, mientras el productor nativo entrega `known[B]` para su TTC actual.
Se reutilizó `NativeObservation`, el adaptador histórico de streaming ya existente.
Este valida una copia expandida y conserva la forma y los valores reales de la
máscara; no altera las features ni los inputs de PAIR. Se retuvo el intento fallido
en `RAW_ECTX_CPU_PARITY.log`. La nueva prueba reproduce E_CTX bit a bit con entradas
crudas y preparadas en las ocho consultas, en CPU y sin updates ni evaluación de
métricas. Recibo: `verification/RAW_ECTX_CPU_PARITY.json`.

## Medición pendiente de endpoints

```powershell
$env:PYTHONPATH = 'src;.'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.raw_cost
```

Este comando requiere los cuatro endpoints y la selección primaria H ya congelados.
Antes de cronometrar verifica paridad completa raw/prepared en las consultas elegidas.
Utiliza ocho consultas por grupo V, seleccionadas por hash de token; aleatoriza el
orden de las rutas y conserva fragmentos reanudables. Incluye HDF5, ROI, representación,
transferencia CPU-GPU y TTC. Todos comparten el mismo lector HDF5 acotado a dos handles.
La caché del sistema operativo no está controlada. Los modelos permanecen cargados;
la carga se excluye de la latencia por consulta pero cuenta en los scopes GPU.

Se comparan los cuatro modelos primarios, A5-P2 congelado y E_CTX completo en el
mismo equipo. E_CTX mantiene más contexto; esa diferencia se declara. Las latencias
no relajan retrospectivamente la selección primaria basada en H y forward preparado.
Tampoco constituyen por sí solas una promoción del sistema ni una comparación oficial.

La comprobación CPU del comparador se reproduce con:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.raw_pair V
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.raw_cost --verify-historical-cpu
```
