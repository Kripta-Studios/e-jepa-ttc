# Plan de investigación para mejorar el release PHASE-TTC

Fecha: 31 de octubre de 2026. Estado: plan propuesto; no se ejecutan entrenamientos ni se modifica el release. Referencia documental inicial: `e750e38130ae02ba2a362bee3270b4fc3cb8d063`.

Actualización al cierre: se incorporan las auditorías concurrentes hasta `eb49b4cf7df367fc9394fde5e028b747d3f40d6c` sobre reloj nominal de eventos y disponibilidad de entradas Test12. No contienen una evaluación completa del candidato; la ampliación poblacional seguía sin recibo de finalización al comprobarla.

El objetivo es mejorar la cadena TRAIN40 que obtuvo Overall MiD **35,80145158082389** y el primer puesto registrado en GarlTTC, manteniendo su identidad como referencia. La investigación se concentra en precisión, transferencia, fiabilidad de las correcciones y coste completo. JEPA ocupa una vía opcional y posterior.

La apuesta principal es comprobar si el release pierde información física entre **medición por pares, historial y corrección de RGB**. Tiene sentido empezar ahí: ya existen productores útiles, una interfaz física compacta y un problema de cadencia observado. Reconstruir toda la representación antes de localizar ese problema encarecería la atribución del resultado.

## Qué probar y cómo mejoraría el release

| Orden | Hipótesis | Cambio concreto | Beneficio esperado, todavía por demostrar |
|---|---|---|---|
| 1 | PAIR mezcla cadencia real y fase canónica | Conversión explícita antes de construir PHASE17, con pesos congelados | Menos dependencia de frecuencia de muestreo y mejor transferencia |
| 2 | Parte de la corrección de eventos carece de evidencia suficiente | Observabilidad de expansión e innovación para modular la corrección anclada en RGB | Menos correcciones perjudiciales sin perder las útiles |
| 3 | El contexto mezcla mediciones de distinta edad sin una dinámica suficiente | Observador causal de q; después, solo si aporta, contexto aprendido con dinámica física | Mejor seguimiento con soporte irregular y menor error temporal |
| 4 | Los productores confunden movimiento con apariencia/sensor | Auxiliar local de formación de eventos, retirado al inferir | Mejor rama de eventos bajo iluminación y umbrales distintos |
| 5 | Oclusión y recorte contaminan la expansión | Correspondencias persistentes, visibilidad y transformación de coordenadas | Menos cambios de TTC inducidos por silueta/ROI |
| Opcional | La representación no conserva la dinámica recuperable del sensor | Predicción latente condicionada por tiempo y anclada a observaciones | Mejor productor, si supera entrenamiento directo de igual coste |

El orden 2–3 puede intercambiarse tras el diagnóstico: si predomina el desacuerdo temporal con medidas bien soportadas, probar primero el observador; si las medidas de eventos son ambiguas, estudiar primero la fusión. El auxiliar sensor requiere datos/calibración adicionales y no bloquea estos dos brazos.

## Evidencia disponible y límites

El screen canónico de 149 consultas ya terminó en otra ejecución: F0 pasa el criterio fijado; el router adaptado no. El registro de continuación describe una ampliación a 2.608 consultas. Este plan no repite ni controla esa ejecución y no la considera terminada sin recibo de finalización. La conservación de las historias eAP completas sigue por comprobar. [Informe y continuación](../research_log_20261031_canonical_screen149/README.md).

El release usa cuatro productores, dos PAIR, dos contextos y una cabeza de fusión. Los resultados de V8, Stage68 y V18 sirven como antecedentes. No son ablations del release ganador. El router `PHASE-TTC-Adapt-20261031` tampoco es el release del resultado oficial. El primer documento fija estas identidades.

## Documentos de trabajo

| Documento | Decisión que permite tomar |
|---|---|
| [00 · Release y puntos de integración](00_RELEASE_Y_PUNTOS_DE_INTEGRACION.md) | Qué se congela, qué cambia y qué hay que recalcular |
| [01 · Datos y protocolo](01_DATOS_Y_PROTOCOLO.md) | Qué evidencia permite diagnóstico, ajuste o confirmación |
| [02 · Cadencia y mecanismos](02_CADENCIA_Y_DIAGNOSTICOS.md) | Si el primer cambio físico mejora la cadena real |
| [03 · Reloj y estado causal](03_RELOJ_Y_ESTADO_CAUSAL.md) | Si una dinámica explícita supera contexto y suavizado |
| [04 · Observabilidad y fusión](04_OBSERVABILIDAD_Y_FUSION.md) | Cuándo los eventos justifican corregir RGB |
| [05 · Formación de eventos](05_AUXILIAR_DEL_SENSOR.md) | Si una pérdida física mejora un productor del release |
| [06 · Coordenadas, superficies y visibilidad](06_COORDENADAS_Y_VISIBILIDAD.md) | Si el error nace de recortes u oclusiones |
| [07 · Secuencia y presupuesto](07_SECUENCIA_Y_PRESUPUESTO.md) | Qué ejecutar, en qué orden y cuándo parar |
| [08 · Evaluación y promoción](08_EVALUACION_Y_PROMOCION.md) | Qué cuenta como mejora utilizable frente al release |
| [09 · Vía JEPA opcional](09_JEPA_CONDICIONAL.md) | Cuándo tendría sentido investigar representación predictiva |
| [10 · Guía de ejecución](10_GUIA_DE_EJECUCION.md) | Contratos, archivos esperados y checklist reproducible |
| [11 · Fuentes y razonamiento](11_FUENTES_Y_RAZONAMIENTO.md) | Qué respalda cada idea y qué sigue siendo hipótesis |

Cada brazo se compara primero con el mismo release congelado y su control de coste/capacidad. Solo se combinan brazos que aporten por separado. Una mejora de consistencia sin mejora de TTC queda como diagnóstico; una mejora de transferencia que perjudique el dominio nativo no sustituye al release global.

La revisión histórica previa permanece en [physics_review_20261031](../physics_review_20261031/README.md). Este plan responde a la aclaración del usuario: el objeto de mejora es el modelo final ganador.
