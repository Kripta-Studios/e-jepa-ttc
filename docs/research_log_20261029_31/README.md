# PHASE-TTC: memoria técnica del 29 al 31 de octubre de 2026

Este dossier reconstruye el trabajo que llevó desde el plan de transferencia
EvTTC hasta el candidato `PHASE-TTC-Adapt-20261031`. Cubre implementaciones,
entrenamientos, auditorías, resultados negativos, optimización, pruebas y entrega.
Las fechas proceden de Git y los números de recibos de ejecución. El autor del
proyecto es **Álvaro Schwiedop Souto, sin afiliación institucional**.

El resultado operativo es un modelo reproducible, tres semillas entrenadas, un
ZIP de 434 predicciones y una entrega con código, pesos y evidencia. El resultado
científico es más acotado: hay mejoras locales de fusión, fallos de arranque
reducidos y límites de protocolo que todavía impiden afirmar superioridad oficial
o generalización independiente. Las dos afirmaciones deben leerse juntas.

## Recorrido de lectura

| Documento | Preguntas que responde |
|---|---|
| [01. Cronología y alcance](01_CRONOLOGIA_Y_ALCANCE.md) | Qué existía, qué se hizo cada día y por qué cambiaron las decisiones |
| [02. Datos, evaluador y física](02_DATOS_EVALUADOR_Y_FISICA.md) | Solapamiento, relojes, cámaras, targets, cobertura y pruebas geométricas |
| [03. Modelos y entrenamiento](03_MODELOS_Y_ENTRENAMIENTOS.md) | Qué bloques se entrenaron, con qué datos, recetas y controles |
| [04. Resultados y comparaciones](04_RESULTADOS_Y_COMPARACIONES.md) | Qué mejoró, qué empeoró, sobre qué población y con qué incertidumbre |
| [05. Rendimiento y recursos](05_RENDIMIENTO_Y_RECURSOS.md) | CUDA Graphs, cachés, preloads, compresión y cuellos de botella medidos |
| [06. Verificación y fallos](06_VERIFICACION_Y_FALLOS.md) | Pruebas ejecutadas, fallos conservados, revisiones y reproducibilidad |
| [07. Artefactos y reproducción](07_ARTEFACTOS_Y_REPRODUCCION.md) | Qué archivos usar, cómo reconstruir el modelo y cómo preparar el envío |
| [08. Límites y siguientes pasos](08_LIMITES_Y_SIGUIENTES_PASOS.md) | Qué sigue sin demostrar y qué experimentos no se han ejecutado |
| [09. Publicación e inventario](09_PUBLICACION_E_INVENTARIO.md) | Qué se sube a Git, qué va como asset y qué permanece local |

## Resultado principal, con su denominador

La comparación final usa 423 targets locales disponibles de las 434 consultas
oficiales. Tras aplicar a ambas versiones la interpolación de negativos indicada
por los organizadores, el error relativo medio pasa de **58,9781 % a 25,8452 %**
y el MAE de **1,057825 s a 0,561545 s**. Es una reducción relativa local del
**56,18 %**. El ZIP contiene las 434 predicciones, incluida una negativa.

Esas cifras no sustituyen al PDF oficial recibido para el modelo del 28 de
octubre. Once targets de carretera siguen sin cobertura local; además, la
correspondencia local no reproduce el PDF en CCRs1-low, CCRs1-medium y Slider-750.
La [tabla completa](04_RESULTADOS_Y_COMPARACIONES.md) conserva estas diferencias.

En rendimiento, la ejecución final mide **1,740290 s por consulta**, frente a
**3,268250 s** en el registro original. La comparación entre ejecuciones equivale
a 1,88 veces el throughput por consulta; no es una ablación controlada de caché.
La preparación de sensores consume la mayor parte del tiempo restante.

## Cómo interpretar la evidencia

Cada cifra pertenece a uno de estos roles:

1. **Oficial recibido:** el PDF o recibo de un organizador, con su población y fecha.
2. **Auditoría local:** recálculo sobre predicciones conservadas y targets locales.
3. **Validación agrupada:** predicciones fuera del grupo usado para ese ajuste,
   pero dentro de un historial de datos ya expuestos.
4. **Reserva de iteración:** grupos no usados por el ajuste/selección actual;
   su exposición histórica sigue vigente.
5. **Ajuste o resubstitution:** evaluación de los pesos sobre las etiquetas que
   acaban de utilizar para entrenar. No mide generalización.
6. **Prueba de ingeniería:** paridad, exportación, causalidad, hashes, instalación
   o formato. Puede pasar aunque el modelo sea científicamente insuficiente.

No se ensamblan cifras de roles distintos en una supuesta mejora oficial. Tampoco
se interpreta `frozen`, una semilla nueva o una nueva carpeta como independencia.

## Fuentes principales y corte documental

El punto de partida del plan EvTTC es el commit `30b28b5c`; la entrega entrenada
final corresponde a `e482c220d6326c6939c5264c1b0f3aefa918c083`. La publicación de
este dossier incorpora esos commits y agrega documentación sin modificar pesos.
El [registro de commits](evidence/COMMITS.txt) conserva los tiempos originales.

Se enlazan los informes anteriores en lugar de reescribirlos: la
[auditoría inicial](../evttc_transfer_v2/HANDOFF.md), la
[campaña F1](../evttc_transfer_v2/CAMPAIGN_RESULTS.md), la
[fusión de calidad](../evttc_transfer_v2/evidence/phase_robust/report/REPORT.md), el
[router de arranque](../evttc_transfer_v2/evidence/phase_startup/REPORT.md) y la
[entrega de competición](../evttc_submission_20261031/REPORT.md).
Sus estados se refieren al momento de cada iteración. Por ejemplo, «no entrenado»
en el piloto inicial no describe el estado de la entrega posterior.

El código mantiene el namespace `e_jepa_ttc` por compatibilidad. El nombre público
es PHASE-TTC. Estos trabajos de adaptación no demuestran una mejora atribuible a
JEPA, ni añaden un preentrenamiento JEPA nuevo al candidato enviado.
