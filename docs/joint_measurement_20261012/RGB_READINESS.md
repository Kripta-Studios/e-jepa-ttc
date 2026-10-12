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
el entrenamiento completo ni su convergencia. La evaluación RGB y la fusión
siguen pendientes de integración y ejecución.
