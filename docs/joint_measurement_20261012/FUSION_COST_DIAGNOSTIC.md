# Fusión: diagnóstico de la paridad sensorial

**El coste sensorial completo sigue sin admisión.** Se conserva el fallo de paridad frente a la cohorte almacenada y no se amplían las tolerancias. Los pesos, predicciones históricas y resultados de los tres seeds no se modifican.

El primer intento se detuvo en la admisión de fuentes históricas. La siguiente ruta cargó las definiciones nativas originales, pero encontró una mediana CUDA sin implementación determinista. Una política explícita y acotada permitió ejecutar esa operación conservando el warning. La prueba posterior falló en la primera consulta cronometrada: ZERO y F_TRUE no cumplen `rtol=1e-5, atol=1e-4 s`. Sus tiempos parciales no forman una medición completa.

## Localización sobre cuatro consultas fijadas

Se usan las primeras cuatro consultas del orden original de coste, sin selección por etiquetas ni cambio de la población. El diagnóstico no ejecuta optimizador.

Pixels P2 y separación temporal raw/preparados idénticos: **True**. C1 con el batch8 original reproduce features y TTC almacenados exactamente: **True**. Por tanto, en estos casos el cache sensorial P2 conserva los datos y la ruta de cohorte puede reproducirse.

| Índice V | Δ máx. features C1 batch1/cohorte | Δ máx. R_HISTORY | Δ máx. E_HISTORY | Δ anchor RGB, s | Δ F_TRUE raw/cache, s |
|---:|---:|---:|---:|---:|---:|
| 16169 | 8.77380371e-05 | 0.000770509243 | 0.00179815292 | 0.000158786774 | 0.00116825104 |
| 7651 | 3.05175781e-05 | 0.00121116638 | 0.00082886219 | 0.000741958618 | 0.00219297409 |
| 1762 | 3.52859497e-05 | 0.00118476152 | 0.00210994482 | 1.71661377e-05 | 0.000335216522 |
| 1515 | 3.81469727e-05 | 0.00113248825 | 0.00757494569 | 0.000367641449 | 0.00482702255 |

Los relojes y máscaras históricas coinciden; las features y fases de los productores raw presentan diferencias. C1 también cambia ligeramente al pasar de batch8 a batch1. Estas son diferencias numéricas observadas, no una demostración de fuga de datos ni una licencia para aceptar el coste.

El control CPU del anchor RGB reproduce la cohorte con batch128. Cambiar únicamente a batch1 introduce como máximo 9.53674316e-07 s en esas cuatro consultas; por sí solo no explica las diferencias de la ruta raw.

## Hipótesis de backend registrada

El extractor histórico y su constructor fijan determinismo, pero no fijan `cudnn.allow_tf32`; el setup V18 lo desactiva explícitamente. Se registra un diagnóstico de ambos valores sólo en las convoluciones de los productores históricos, manteniendo matmul TF32 desactivado y restaurando la política al salir. No cambia las convoluciones C1 ni los pesos. El flag efectivo histórico no quedó registrado: esta es una hipótesis verificable, no una causa ya demostrada por inspección del código.

| Índice V | cuDNN TF32 productores | Δ R_HISTORY | Δ E_HISTORY | Δ F_TRUE raw/cache, s |
|---:|---|---:|---:|---:|
| 16169 | False | 0.000770509243 | 0.00179815292 | 0.00116825104 |
| 16169 | True | 0.00081127882 | 0.00160688162 | 3.33786011e-05 |
| 7651 | False | 0.00121116638 | 0.00082886219 | 0.00219297409 |
| 7651 | True | 0.00209283829 | 0.00417035818 | 0.00017786026 |
| 1762 | False | 0.00118476152 | 0.00210994482 | 0.000335216522 |
| 1762 | True | 0.00273799896 | 0.00128765404 | 7.15255737e-06 |
| 1515 | False | 0.00113248825 | 0.00757494569 | 0.00482702255 |
| 1515 | True | 0.000707983971 | 0.00543865561 | 0.000246047974 |

Máximo error F_TRUE raw/cache en estas consultas: 0.00482702255 s con TF32 desactivado y 0.000246047974 s al activarlo sólo en los productores. Consultas que todavía fallan la tolerancia original: `[1515]`. El backend influye, pero este único cambio no resuelve toda la paridad. No se promueve la ruta ni se continúa una búsqueda de tolerancias.

Recibo del contraste: `5085c1a88ea18de70685e0a18ab458baa08f14d066676115b9df7003d1bd68a4`. El diagnóstico no admite automáticamente la ruta: una nueva medición necesita su propio registro y pasar la paridad original completa.

Diagnóstico GPU: `66975efa18b0cb61cf2db50e81b8207a2886f6c7ca60502536267b7146ccea97`.
Diagnóstico CPU: `8ebc05adac761f087d3d3ab18c4e994d24a333b596552d644c179ee3b6d4c137`.

No se infiere precisión TTC de estos cuatro casos ni se utilizan para elegir un modelo. La inferencia preparada y los exports siguen teniendo un alcance distinto del coste sensorial completo.

## Tamaños históricos de los lotes

Una auditoría posterior, sólo de metadatos, ha verificado los 514 fragmentos event
y 522 RGB que produjeron los historiales V. Sus hashes coinciden con sus recibos,
la concatenación conserva exactamente las observaciones originales y las cuatro
consultas son las mismas del diagnóstico anterior. No se han leído columnas TTC
ni ejecutado modelos nuevos para este inventario.

Un fragmento RGB de 32 observaciones puede mezclar T2 y T3. El código histórico
separaba ambos tamaños antes del forward: el batch del modelo no siempre era 32.
Estos son los tamaños reconstruidos para cada observación de los historiales
seleccionados, en orden temporal:

| Índice V | Batches de los productores event | Batches de los productores RGB |
|---:|---|---|
| 16169 | 32, 32, 32, 32 | 30, 29, 32, 30, 32 |
| 7651 | 32 | 29, 30, 32 |
| 1762 | 32, 32, 32, 32 | 31, 32, 29, 30, 31 |
| 1515 | 32, 32, 32, 32 | 31, 30, 27 |

Todas esas observaciones concretas son T3. Son los lotes de los productores
históricos A5/C2F y de PAIR; no deben confundirse con batch8 de la extracción C1
ni batch128 de la cabeza R_CTX en CPU, examinados en el diagnóstico anterior.

El recibo completo del inventario es
`fusion/cost/historical_batch_inventory/RESULT.json`, SHA-256
`42239442fecddbd30ef549e2653252e94c48c1f895ce2084fcc77dfb80554d0b`.

Se prepara un contraste de esas formas de batch con los dos valores de cuDNN
TF32 ya examinados. Repetirá exclusivamente los pixels y tiempos de la propia
observación hasta alcanzar el tamaño histórico y conservará su primera salida.
No incorpora inputs de otras consultas ni sus etiquetas. La comparación usa las
mismas cuatro consultas, pesos, normalizadores y tolerancia TTC
`rtol=1e-5, atol=1e-4 s`. Los resultados batch1 anteriores permanecen intactos.

Este contraste aún **no está ejecutado** y no constituye un benchmark de
latencia. Su finalidad es comprobar si las formas de batch explican la
discrepancia numérica. Incluso si funcionase, haría falta admitir y medir la
ruta completa sobre toda la población de coste, declarando cualquier trabajo
interno repetido. No se llamará aceleración a esa repetición de inputs.

El contraste quedó registrado antes de ejecutarse, con SHA-256
`9649f45c264b58e883663ced46a18e28a25ad3bd6235498b9ef8d8f476ed3c76`.
Las [condiciones fijadas](evidence_fusion_batch_shapes/REGISTRATION.json), el
[inventario completo](evidence_fusion_batch_shapes/INVENTORY.json) y la
[validación CPU](evidence_fusion_batch_shapes/VALIDATION.json) se conservan juntos.
Las 11 pruebas CPU, Ruff y Pyright pasan. El runner comparte la exclusión GPU
existente y rechaza otros procesos de cómputo, incluido el coordinador B1 entre
segmentos. Tras liberar esos recursos, el comando es:

```powershell
$env:PYTHONPATH = 'src;.'
& ..\e-jepa-ttc\.venv\Scripts\python.exe -B -m operational.joint_measurement.fusion_cost_shape_probe
```

## Resultado del contraste de batch histórico, 16 de octubre a las 18:10

El contraste registrado está completo: ocho forwards de diagnóstico (cuatro
consultas por dos configuraciones). La tabla se genera desde RESULT.json.

| Índice V | cuDNN TF32 | TTC raw s | TTC histórico s | Tolerancia original |
|---:|---|---:|---:|---|
| 16169 | False | -7.775091171 | -7.773911953 | False |
| 16169 | True | -7.773897648 | -7.773911953 | True |
| 7651 | False | 7.867776871 | 7.865612507 | False |
| 7651 | True | 7.865617275 | 7.865612507 | True |
| 1762 | False | -4.648367882 | -4.648036003 | False |
| 1762 | True | -4.648028374 | -4.648036003 | True |
| 1515 | False | 4.487482071 | 4.492309570 | False |
| 1515 | True | 4.492302418 | 4.492309570 | True |

Con la forma histórica y cuDNN TF32 activado pasan las cuatro consultas. Con
TF32 desactivado fallan las cuatro. Esta evidencia apoya una discrepancia
de ejecución numérica, sin modificar pesos, entradas, relojes ni tolerancias.
No prueba todavía la paridad de toda la población de coste.

Se prepara la ruta raw_e2e_historical_shape_v1: las consultas y orden emparejado
son los del contrato original; el batch externo sigue siendo uno. Los productores
repiten cada observación hasta su tamaño histórico, conservando sólo la primera
salida. Todo ese trabajo interno se contará y declarará. No es una optimización
de velocidad ni un cambio de modelo. C1 y las cabezas mantienen su configuración
original; el cambio cuDNN se limita a los productores históricos.

SHA-256 de RESULT.json: 448800f85798e95904c3adbaccb9799e35db7d0c243098f06a4e130bf79e0a98.
