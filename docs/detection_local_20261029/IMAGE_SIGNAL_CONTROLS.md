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

La revisión previa a leer los resultados finales detectó que Python 3.11 y 3.12
pueden diferir en redondeos de aproximadamente `1e-16`. El 2026-10-29 a las
13:56 UTC se corrigió la validación de ambos controles para comparar estructuras
recursivamente: floats con tolerancias absoluta y relativa `1e-12`; claves,
listas, strings, booleanos, enteros y `null` exactos, incluidos sus tipos.
El recibo declara esas tolerancias. No cambia el matching, los controles, la
permutación ni la selección del modelo. Los tests admiten una perturbación
`1e-15` y rechazan `1e-5`, un conteo cambiado o su conversión de entero a float.

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

Los 14 casos analíticos de
[test_detection_image_signal.py](../../tests/unit/test_detection_image_signal.py)
verifican ajuste independiente de selección, medianas y presencia por frame,
media circular, permutación determinista sin puntos fijos, preservación de las
listas, pérdida de matches en un ejemplo conocido, poblaciones y métricas
inconsistentes, y JSON ambiguo. No se modifican los módulos del detector.

## Control adicional de velocidad nativa

El método separado `native_velocity_shared_match_zero_train_mean_v1` se declaró
el **2026-10-29 a las 13:49:51 UTC**, durante el piloto, antes de leer sus resultados
finales. No se presenta como una condición registrada antes de iniciar ese run.
Se implementa en
[check_detection_velocity_signal.py](../../scripts/check_detection_velocity_signal.py)
sin modificar el detector ni `velocity_diagnostics.py` durante el entrenamiento.

Se compara la cabeza aprendida con dos controles: velocidad cero y media global
de cada componente calculada sobre todas las cajas **train** con velocidad válida.
No se ajustan medias por clase ni se utiliza ninguna etiqueta de selección para
ajustar esos valores. Si no hay velocidades train válidas, el control de media
falla en vez de inventar un valor por defecto.

Se reutiliza exactamente `evaluate_velocity_labels` con umbral XY estricto de dos
metros. Las copias de las predicciones solo cambian `velocity_mps`: centros,
dimensiones, yaw, clases, score, orden y tokens permanecen iguales. Por ello las
asociaciones son las mismas, y se comprueban además sus conteos totales y por clase.
Un GT sin velocidad sigue consumiendo el match geométrico y se excluye únicamente
del error de velocidad. Una predicción sin velocidad produce error incluso si no
se asocia; los controles no reparan esas omisiones.

Se registran MAE por componente, suma de MAE, media de norma L2, número de matches
geométricos, matches con velocidad válida y clases con soporte. Las diferencias
L2 se expresan como `head - baseline`, por lo que un valor negativo favorece a la
cabeza. Si no hay matches válidos, los errores y diferencias son `null`.

La comparación mide afinidad a los componentes publicados de la etiqueta sobre
los objetos detectados. No resuelve su semántica física, no produce TTC y no es
AVE oficial. Su población condicionada a detecciones tampoco representa todos los
objetos GT. El recibo contiene hashes de entradas, script, validación y diagnóstico
reutilizado; no copia etiquetas individuales.

```powershell
.venv-audit/Scripts/python.exe -X utf8 scripts/check_detection_velocity_signal.py `
  --train-results <run>/final_train.json `
  --selection-results <run>/selection.json `
  --output <run>/velocity_signal_controls.json
```

Los ocho casos analíticos de
[test_detection_velocity_signal.py](../../tests/unit/test_detection_velocity_signal.py)
comprueban errores conocidos, ajuste independiente de selección, consumo de
matches con GT incompleto, predicciones inválidas, ausencia de train válido,
umbral estricto, preservación de entradas y discrepancias del diagnóstico guardado.

## Resultados del piloto completado

Se ejecutaron ambos controles sobre
`output/portable_experiments/pilot256_seed7_velocity_v1`, después de comprobar
`report.json` con estado `COMPLETED`. Se usaron sus archivos cerrados
`final_train.json` y `selection.json`: 256 frames de train y 256 de selección.
Los dos comandos terminaron con código cero en `.venv-audit`, sin Torch ni GPU.
Los hashes de entrada y código quedan en los recibos
[imagen](evidence/pilot256_image_signal_controls.json) y
[velocidad](evidence/pilot256_velocity_signal_controls.json).

### Detección e información asociada al frame

| Predicciones evaluadas en selección | AP101 macro de cinco clases con GT | AP101 de ocho clases, ausentes a cero |
|---|---:|---:|
| Modelo observado | 0.02064071 | 0.01290044 |
| Prior constante ajustado con train | 0.00412554 | 0.00257846 |
| Predicciones permutadas entre frames | 0.00372793 | 0.00232996 |

La AP101 soportada observada supera al prior constante en **0.01651517** y a la
permutación en **0.01691278**. La primera permutación generada por la semilla 7
ya era una biyección sin puntos fijos; se preservaron las **12.800** detecciones.
No se buscaron otras permutaciones ni se eligió un modelo con estos controles.

El descenso al intercambiar imágenes es compatible con predicciones que contienen
información específica del frame, además de priors constantes. Es un resultado
positivo del diagnóstico, pero la AP absoluta sigue siendo baja y el ensayo no
constituye una prueba causal o de significación estadística.

| Clase de selección | Objetos GT | AP101 media de umbrales | Matches a menos de 2 m |
|---|---:|---:|---:|
| car | 721 | 0.08338670 | 342 |
| truck | 205 | 0.01973244 | 58 |
| pedestrian | 2 | 0 | 0 |
| motorcycle | 6 | 0.00008441 | 1 |
| bus | 4 | 0 | 0 |

Las otras tres clases carecen de GT en esta selección. Car y truck concentran
926 de los 938 objetos; no hay evidencia suficiente de funcionamiento para las
ocho clases. Los numerosos falsos positivos y el soporte escaso de clases
minoritarias limitan la conclusión. ATE/ASE/AOE de los controles están en el
recibo, pero se calculan sobre asociaciones diferentes y no deben interpretarse
como una comparación sobre los mismos objetos.

### Velocidad sobre asociaciones idénticas

El prior de media usa **829** objetos train con velocidad válida y resulta en
`[0.6944943899, 0.0020785578]` en los componentes publicados. La comparación de
selección utiliza exactamente **401** asociaciones geométricas y 401 velocidades
válidas para cada alternativa: 342 car, 58 truck y una motorcycle.

| Alternativa | MAE componente 1 (m/s) | MAE componente 2 (m/s) | Media L2 vectorial (m/s) |
|---|---:|---:|---:|
| Cabeza aprendida | 0.59208865 | 0.03088006 | 0.59490115 |
| Cero | 1.38074415 | 0.02085950 | 1.38115021 |
| Media ajustada con train | 0.75762220 | 0.02062332 | 0.75912775 |

La cabeza reduce L2 en **0.78624905 m/s** frente a cero y en **0.16422659 m/s**
frente a la media train. La mejora está en el primer componente: **el segundo
empeora frente a ambos controles**. No se oculta esta diferencia mediante el
agregado vectorial.

Este resultado acredita aprendizaje útil respecto a dos controles simples en los
componentes de etiqueta de los objetos que se lograron asociar. No evalúa los
otros 537 objetos de selección, no demuestra velocidad física relativa, no es
AVE oficial y no permite inferir TTC firmado. Las métricas de detección y estas
limitaciones deben acompañar cualquier presentación del resultado.
