# Controles de señal de imagen en el piloto RGB

Método `train_constant_prior_and_seed7_frame_derangement_v1`, declarado el
**2026-10-29 a las 13:22:15 UTC**, antes de leer los resultados del piloto.
La declaración se comunicó al coordinador y queda fijada en el script.

Los controles responden a una pregunta limitada: si las detecciones del piloto
contienen información asociada a cada imagen que supera un prior constante
aprendido exclusivamente de train. No demuestran SOTA, causalidad, seguridad ni
equivalencia con el scorer oficial.

## A: prior constante ajustado solo con train

Para cada clase presente en las etiquetas train se ajusta **una** caja:

- Centro y dimensiones: mediana por componente sobre todas sus cajas train.
- Yaw: media circular `atan2(mean(sin(yaw)), mean(cos(yaw)))`. Si la longitud
  resultante es como máximo `1e-12`, se usa cero y se registra el caso indefinido.
- Score: fracción de frames train que contienen al menos un objeto de esa clase.

Se repite ese conjunto de cajas en todos los frames de selección. No se ajusta
ningún parámetro usando sus etiquetas. No se añaden cajas de clases ausentes en
train ni se adapta la cantidad de cajas a cada imagen. Es un baseline deliberadamente
simple: fallarlo resulta informativo, pero superarlo no demuestra generalización.

## B: permutación fija de predicciones entre frames

Se ordenan los tokens de selección lexicográficamente y se instancia
`random.Random(7)` de Python. Se generan permutaciones por `shuffle` hasta obtener
la **primera sin puntos fijos**; hay un máximo de 10.000 intentos y el script falla
si no la obtiene. Nunca se elige una permutación en función de su AP.

Las listas completas de detecciones se transfieren entre frames según esa
biyección. Se conservan globalmente todas las cajas, dimensiones, yaws, velocidades,
scores y cantidades por lista. Solo cambia el frame al que se asignan. Se evalúan
contra las etiquetas originales de los frames destino. Se necesita un mínimo de
dos frames. El recibo conserva semilla, número de intentos y hash de la asignación,
pero no copia ground truth ni listas de predicciones.

El intercambio puede conservar priors útiles entre imágenes parecidas de la misma
carretera; la caída de AP es un diagnóstico de correspondencia, no una prueba causal
ni una estimación estadística con múltiples permutaciones. La ausencia de caída
tampoco demuestra por sí sola que el modelo ignore todos los píxeles.

## Evaluación y trazabilidad

Los tres resultados —modelo observado, prior constante y permutación— se
recalculan con el [perfil de desarrollo](DEVELOPMENT_METRICS.md) sin modificar
matching, umbrales, clases soportadas ni agregación. Se conservan ambas macros
AP, los detalles por clase y ATE/ASE/AOE. No se calcula AVE ni se afirma una
evaluación oficial de velocidad. Las diferencias AP se expresan como
`observado - control`.

El script exige tokens train/selección disjuntos y la misma población GT/predicción
en cada entrada. Recalcula los campos del scorer y los contrasta con las métricas
almacenadas; campos adicionales de loss o tiempo son admisibles. La disjunción
de tokens no certifica por sí sola separación de grupos de adquisición.

Se leen únicamente las dos rutas JSON explícitas. No se usan Torch, GPU, pesos,
imágenes o etiquetas test. El recibo guarda hashes de ambas entradas, del script
y del scorer, versión Python, hashes de poblaciones y parámetros agregados del
prior. Se rechazan claves JSON duplicadas y la sobrescritura del recibo de salida.

```powershell
.venv-audit/Scripts/python.exe -X utf8 scripts/check_detection_image_signal.py `
  --train-results <run>/final_train.json `
  --selection-results <run>/selection.json `
  --output <run>/image_signal_controls.json
```

Los 11 casos analíticos de
[test_detection_image_signal.py](../../tests/unit/test_detection_image_signal.py)
verifican ajuste independiente de selección, medianas y presencia por frame,
media circular, permutación determinista sin puntos fijos, preservación de las
listas, pérdida de matches en un ejemplo conocido, poblaciones y métricas
inconsistentes, y JSON ambiguo. No se modifican los módulos del detector.
