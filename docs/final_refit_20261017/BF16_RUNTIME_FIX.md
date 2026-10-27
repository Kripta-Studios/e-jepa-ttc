# Corrección del runtime BF16 del refit

Fecha: 2026-10-17. Alcance: productores RGB finales TRAIN40; no modifica los resultados históricos V16/V18/V19.

## Causa y evidencia

La comparación inicial falló al reproducir gradientes de R_A5 con el objetivo completo. Las entradas, pesos iniciales, RNG y pérdidas coincidían. También divergían dos ejecuciones de la misma referencia, incluso reutilizando la misma instancia y restaurando exactamente su estado. Por tanto, un único control A/A no era una cota numérica válida y se conserva la admisión controlada como `FAILED`.

La traza `RGB_EAGER_BACKWARD_TRACE.json` localizó la primera diferencia observada en el gradiente de las características espaciales del encoder. El muestreo bilineal del flujo inverso de `local_transport.py` usaba `grid_sample` con `deterministic_transport_sampling=False`. En el contraste acotado, activar el muestreador determinista existente eliminó la diferencia: `RGB_EAGER_DETERMINISTIC_TRANSPORT_DIAGNOSTIC.json` registra `SAME_INSTANCE_EXACT`, incluida diferencia máxima cero en gradientes.

Otro defecto del diagnóstico era activar el modo determinista estricto antes de construir los modelos: `build_producer` vuelve a inicializar las semillas con `warn_only=True`. Se corrigió el diagnóstico para registrar los flags efectivos inmediatamente antes del cálculo. El modo estricto, por sí solo, no había eliminado la divergencia.

## Cambio prospectivo

`build_rgb_recipe` fija explícitamente `deterministic_transport_sampling=True` en la receta derivada del refit. El muestreador conserva interpolación bilineal, padding cero, `align_corners=True` y derivadas respecto al campo y a las coordenadas. No se desconectan gradientes. Los archivos de configuración históricos permanecen intactos.

Se conservan semilla 7, BF16, arquitectura, pérdidas, sampler, batch efectivo, exposición, scheduler y endpoints. El cambio numérico queda identificado en la receta. No se promete identidad bit a bit con la ejecución histórica no determinista.

La admisión nueva vuelve al criterio estricto original (`atol=2e-5`, `rtol=2e-4`) tanto para candidato/referencia como para referencia/referencia. Se elimina la alternativa de aceptar una desviación porque sea menor que un control A/A. Los validadores rechazan valores no finitos, casos incompletos, estados incompatibles y contradicciones entre las métricas y los indicadores de aceptación.

Los hashes del plan y de la admisión cubren también modelos, pérdidas, muestreo espacial y reproducción de semillas, evitando que un cambio local en esos archivos quede fuera de la identidad del entrenamiento.

Las 15,39 y 28,42 horas de `BUDGET_PROFILE.json` proceden del runtime histórico. El muestreador determinista puede cambiar el coste; esas cifras no constituyen una ETA medida de esta implementación. Se debe registrar el ritmo real al arrancar, distinguiendo warmup y objetivo completo.

## Verificación y ejecución

Los recibos de ejecución están en `artifacts/final_refit_20261017/technical/`. Los ensayos usan modelos descartables y no constituyen entrenamiento científico. La equivalencia matemática del muestreador se verifica en `tests/test_joint_measurement_sampling.py`, con salidas y ambas derivadas en FP64, incluidos límites y dimensiones unitarias.

Verificación completada antes del arranque científico:

- `RGB_EAGER_ADMISSION_STRICT_FINAL.json`: PASS, 12/12 casos, 96 updates técnicos descartados, cero científicos. SHA-256 `a5c1a4bf3ec1369f32d91e667f1bb3f12936a5d037701b68fc2d5eabea5d8031`.
- `RGB_EAGER_B32_BACKWARD.json`: PASS en ambos productores con objetivo completo y entrada T3 `[32,3,3,128,128]`; dos updates técnicos descartados. Pico de memoria CUDA reservada de aproximadamente 1,54 GB (no incluye la memoria ajena al allocator de PyTorch). SHA-256 `38109dcdd0627d4d504a88453f4a1ed1094eb10c39f062c987e9972745f314fd`.
- 30 pruebas focalizadas aprobadas, incluidos los cuatro casos de equivalencia matemática del muestreador. Ruff y Pyright correctos.

La primera ejecución completa numéricamente correcta fue rechazada por una etiqueta obsoleta en el recibo. Queda conservada como `FAILED`; tras corregir esa etiqueta y añadir su comprobación de regresión, se ejecutó nuevamente la suite que produjo el PASS anterior. No se reetiquetó ningún fallo histórico.

El plan de los dos productores quedó congelado con identidad `fea580230aa513235fd94dd8d177788c640ec8f954b219d4a16d322e47b168d4`. R_A5 inició entrenamiento científico el 17 de octubre a las 14:19, hora local. El estado efectivo, checkpoints y tiempos se conservan en `artifacts/final_refit_20261017/fits/`; este documento no sustituye sus recibos. El arranque de estos productores no acredita que el runner OOF superior esté terminado.

La comprobación inicial `fits/R_A5/FIRST100_VERIFICATION.json` registra 100 updates científicos durables, todas las pérdidas finitas, cero recuperación y 179,88 segundos incluyendo checkpoint: 0,556 updates/s, o 17,79 consultas/s. Es una medida corta del warmup, no una ETA del objetivo completo ni del sistema final. El checkpoint inicial tiene SHA-256 `8e57007199e6f7479f9e034c14be416e4600e31ea6e63cbcf43d51494c0de436`.

Después se lanzó la reanudación hasta el endpoint fijo de 49.932 updates. La secuencia de ejecución prepara R_C2F a continuación únicamente si R_A5 termina con código cero; no hay reintentos automáticos ni lanzamiento de las etapas OOF superiores.
