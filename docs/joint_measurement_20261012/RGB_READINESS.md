# Integración de pares RGB

`operational.joint_measurement.rgb_model` admite el endpoint R_A5 P-only de
30.330 updates, verifica su hash contra el inventario, recibo y procedencia y
carga una instancia CPU privada. `pair_from_parent` copia su CNN de tres canales.
Los módulos nuevos comparten inicialización por semilla, tanto con correlaciones
como sin ellas y con encoder entrenable o congelado.

La comprobación `python -B -m operational.joint_measurement.rgb_preflight` utiliza
cuatro consultas P elegidas por hash del token: dos con historia original T2 y
dos T3. El adaptador conserva los dos últimos frames reales antes de colacionar,
con entrada [4,2,3,128,128] y separación del reloj RGB. En esos ejemplos el
intervalo es aproximadamente 0,100001–0,100002 segundos.

Se comprueba forward/backward CPU, gradientes finitos y no nulos del encoder
entrenable, ausencia de gradientes del congelado, gradiente en la cabeza y
preservación completa del padre. También se comprueba que no haya almacenamiento
de parámetros compartido y que los módulos nuevos tengan pesos iniciales
idénticos. Los pesos no cambian: son **cero updates de optimizador**.

La arquitectura RGB tiene 571.137 parámetros; con encoder congelado son 402.145
entrenables. La diferencia respecto a eventos corresponde a la entrada nativa de
tres canales. No se repiten canales event ni se utiliza su checkpoint como RGB.

Evidencia: `verification/RGB_CPU_PREFLIGHT.json`, que contiene los índices,
identidades, tiempos, normas de gradiente y hashes del código ejecutado. Se
conserva también la comprobación previa a una corrección de formato de una línea.

Esto acredita integración de CPU sobre esos ejemplos, no cobertura completa,
entrenamiento RGB, reanudación CUDA ni precisión TTC. La receta RGB se registra
después del contraste primario event, respetando el máximo de dos fits de 12.000
updates y dos fits de fusión/control de 2.500. La fusión requiere todavía su
integración, control ZERO, genealogía de features nueva y fallback event-only.

## Entrenador preparado, pendiente de admisión científica

`operational.joint_measurement.rgb_train` implementa el par entrenable/congelado,
con 12.000 updates por brazo, las mismas masas P y objetivo de fase de V18,
batch32/microbatch4, FP32 y optimizador registrado. Guarda modelo, optimizador,
scheduler, RNG, sampler y cursor mediante el checkpoint transaccional existente.
Comparte la contabilidad V18, las reservas GPU y los límites de recursos.

El registro exige los cuatro endpoints primarios, sus predicciones P/H completas
y la decisión final con coste. No se ha ejecutado ese registro ni elegido CORR o
DIRECT para RGB. El comando posterior será:

```text
python -B -m operational.joint_measurement.rgb_train register --variant CORR --reason "Observación documentada del contraste P/H"
python -B -m operational.joint_measurement.rgb_train qa
python -B -m operational.joint_measurement.rgb_train fit --fit-id RGB-PAIR-CORR-FT-S7
python -B -m operational.joint_measurement.rgb_train fit --fit-id RGB-PAIR-CORR-FROZEN-S7
```

La variante del ejemplo debe corresponder a la decisión documentada. Cada `fit`
ejecuta un scope y conserva el cursor para la siguiente invocación. `qa` ejecutará
40 updates sintéticos contabilizados: diez continuos frente a cinco más cinco
reanudados para cada brazo, sobre el mismo bucle que utiliza el entrenamiento.
Sus resultados CUDA todavía están pendientes; el fit real los exige.

Las pruebas de admisión verifican que no se puede registrar RGB con endpoints
primarios incompletos o sin la comparación final, y que un cambio de código bloquea
el fit antes de construir el optimizador. Estas pruebas no acreditan todavía
el entrenamiento completo ni su convergencia. La fusión sigue pendiente de
integración y ejecución.

## Evaluación preparada

`operational.joint_measurement.rgb_evaluate` exige ambos endpoints de 12.000
updates con journals durables y sin updates pendientes. Incluye el padre
R_A5-P2-FROZEN como referencia con el mismo contexto de dos frames. Escribe las
predicciones completas y métricas por grupo en `rgb/predictions/{P,H,V}` mediante
fragmentos verificables y reanudables. El modelo recibe únicamente píxeles y
delta RGB; las etiquetas se incorporan después para el scorer.

`select` exige P/H completos y conserva una decisión de desarrollo basada en
métricas. La vía de coste y la comparación con R_CTX/F_TRUE quedan separadas y
pendientes; esa decisión no afirma promoción del sistema. V exige que antes se
haya congelado la interpretación H. No se ha ejecutado ninguna de estas
evaluaciones sobre datos reales, porque todavía faltan los endpoints RGB.

Seis pruebas adicionales verifican endpoints completos, ausencia de updates
pendientes y rechazo de fragmentos con binding, intervalo o bytes distintos.
El replay del bundle incluye las ramas anidadas de predicciones y exports,
preservando sus nombres para evitar colisiones con los resultados primarios.

## Cachés H/V propias

La inspección del 12 de octubre a las 18:15 UTC confirmó que V13 sólo conserva
la caché RGB decodificada P. Las cachés H/V de investigación V16 contienen estados
de refinadores congelados y no sustituyen los frames necesarios para estos modelos.

`python -B -m operational.joint_measurement.rgb_cache both` materializa H y después
V, en un único proceso CPU. Reutiliza `decode_query`, los campos y el formato
lossless del preparador nativo; no cambia los manifests ni bindings de V13.
Los nuevos píxeles se guardan bajo el cache_root de V18, en `rgb/H` y `rgb/V`.
Los bindings propios quedan en `rgb/cache_bindings` dentro del directorio de
artefactos V18. `RGBPairDataset` utiliza esos bindings únicamente cuando falta
una caché histórica del rol correspondiente.

Cada fragmento registra tokens, índices, CRC y SHA-256. El cierre exige cobertura
exacta y paridad de todos los campos contra una segunda decodificación de la
primera y última consulta T2/T3. El manifest sólo se publica después de esos
controles. Un fragmento admitido que cambie se rechaza y conserva para revisión.
La preparación no realiza inferencia, optimización ni selección por TTC.

El proceso se inició el 12 de octubre a las 18:15 UTC. Su estado actual está en
`RGB_PREPARE_H_PROGRESS.json`, `RGB_PREPARE_V_PROGRESS.json` y los manifests del
cache_root. El recibo `verification/V16_COORDINATION_20261012T1816.json` registra
la observación de propietarios, sin utilizar resultados FCWD/Dev32 para recetas.

H cerró a las 18:35:50 UTC: 18.422 consultas, 576 fragmentos y 1.882.328.902 bytes.
La cobertura es exacta y los cuatro casos de segunda decodificación T2/T3
coinciden bit a bit en todos los campos. El adaptador V18 consumió correctamente
esos cuatro casos, con tensor [4,2,3,128,128] y reloj RGB positivo. Recibo:
`verification/H_RGB_INPUT_ADMISSION.json`. No se ejecutó ningún modelo ni métrica.
El manifest H tiene SHA-256
`8cadc01ea182f4bcfc2082790ba942abd22cfca6d1cd7f1d1cbbefc119b17618`.
V cerró a las 18:53:43 UTC: 16.432 consultas, 514 fragmentos y 1.516.027.016 bytes.
También conserva cobertura exacta y paridad bit a bit en los cuatro casos de
segunda decodificación. El adaptador consumió los índices 0, 16.422, 1 y 16.431,
con tensor [4,2,3,128,128] e intervalos RGB positivos. Recibo:
`verification/V_RGB_INPUT_ADMISSION.json`. El manifest V tiene SHA-256
`21975cfe232c6a128376abfde4caa49fa00afa1d9740971785827c7cd278e92d`.
El proceso CPU de preparación terminó con código 0. P/H/V tienen ahora entradas
RGB disponibles; esto no constituye entrenamiento ni evaluación de precisión.

## Export portable RGB

`python -B -m operational.joint_measurement.rgb_export FIT_ID` exige la congelación
de los endpoints RGB. Generará TorchScript batch1, ejemplos sin etiquetas,
normalización, contrato de entrada y recibos con SHA-256 en `rgb/exports/FIT_ID`.
La entrada RGB es [1,2,3,128,128], FP32 en [0,1], y un intervalo positivo [1,1]
del reloj RGB. El NPZ utiliza la clave `rgb`; el replay conserva también la
compatibilidad con los exports event cuya clave es `events`.

Se ha comprobado la ruta de serialización/recarga en CPU con dos pares P reales
(T2/T3), sobre el padre RGB histórico y sobre DIRECT/CORR con sus módulos nuevos
aleatorios. Las tres comprobaciones pasan y el ejemplo publicado en cada
diagnóstico coincide bit a bit. **Cero updates y ninguna medida de precisión TTC**.
Recibos: `diagnostics/RGB_PORTABLE_PREFLIGHT/RESULT.json` y `REPLAY.json` de cada
arquitectura. El código exacto ejecutado se conserva junto al diagnóstico; una
corrección posterior de líneas en blanco entre imports tiene AST idéntico.

El trazado se limita al contrato de forma fija y entradas válidas declarado.
Conserva las advertencias de PyTorch en el log. La preparación de estos artefactos
no acredita los futuros endpoints entrenados: cada export deberá repetirse y
verificarse con sus propios pesos después de completar la comparación RGB.
