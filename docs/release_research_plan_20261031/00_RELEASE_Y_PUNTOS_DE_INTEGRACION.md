# Release de referencia y puntos de integración

## Identidad de la referencia R0

En este plan, **R0** significa el PHASE-TTC TRAIN40 derivado de V16, semilla 7, RGB+eventos y hasta 650 ms de historia causal. No significa V18 ni el adaptador posterior de EvTTC. Su resultado archivado corresponde a la submission **1016512**, GarlTTC Test: 35,80145158082389 Overall MiD, 6.762 tokens y 0% de fallos según el scorer. La clasificación archivada es del 28 de octubre. [Identidad y métricas](../../paper/data/official_benchmark.json).

La [descripción de entrenamiento y sellado](../engineering_history_20261027/02_TRAIN40_OOF_AND_FINAL_TRAINING.md) contiene los nueve SHA-256 completos. La referencia mínima para verificar el paquete es:

| Artefacto | Identidad registrada |
|---|---|
| MODEL_SEAL.json | `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4` |
| Cabeza PHASE_ONLY_EVENT_S7_TRAIN40 | `a24bfc72ca7f53a93d53db806b6866c6a5a4b6abde074b5dfec50436b7005408` |
| ZIP PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF | `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc` |
| Manifest TRAIN40 | `0e59ce2629f512f68dff6ab77beef68090816adac7adf21994cb29fa6954859c` |

Son identidades documentadas, no un nuevo hash de los pesos ejecutado en esta tarea. La preejecución debe verificar los bytes locales. El JSON oficial del score no incluye el digest de las predicciones; el encadenamiento de identidad debe conservar además los recibos de preparación/envío existentes, sin inventar una firma que no contiene el scorer.

Usar aliases de configuración R0, C-TIME, C-OBS, C-CLOCK, C-SENSOR para los experimentos; no renombrar checkpoints históricos. El adaptador de EvTTC se llama A0 en este plan y se evalúa aparte. Un resultado favorable sobre A0 no sustituye un contraste sobre R0.

## Grafo real sobre el que se investiga

```text
Eventos → E_A5 + E_C2F + PAIR_E → PHASE17_E → E_CTX ┐
                                                   ├→ PHASE_ONLY → fase → TTC firmado
RGB     → R_A5 + R_C2F + PAIR_R → PHASE17_R → R_CTX ┘
```

PAIR usa características del A5 correspondiente. Los refinadores consumen historia, tiempos y normalizadores. La fusión selecciona 20 coordenadas de un contenedor de 340, no concatena directamente los 160 estados ocultos de cada modalidad. El punto se calcula como `location = R_CTX.raw_location + 0.03 * correction[0]`. Otras dos salidas corrigen anchuras. [Implementación de PhaseOnlyFinal](../../operational/final_refit/training.py), [especificación del artículo](../../paper/sections/final_model.tex).

Este anclaje abre una intervención pequeña: cambiar cuánto de la corrección original se acepta cuando la evidencia de eventos es mala. No demuestra que la corrección sea exclusivamente información de eventos: también depende de los campos RGB y puede compensar sesgos del ancla. Hay que medir esa dependencia.

La fase `φ=-log(1-0.1/T)` es distinta del cambio logarítmico de tamaño durante un intervalo real. La emisión recorta fase y limita |T| a 60 s. Las pruebas físicas deben registrar estado previo a recortes y salida publicada por separado. [Conversión desplegada](../../src/e_jepa_ttc/simplex_t/phase.py).

## Qué puede cambiar cada brazo

| Brazo | Inserción mínima | Pesos congelados al empezar | Dependencias que deben recalcularse |
|---|---|---|---|
| C-TIME | Salida cruda de PAIR y sus tiempos | Los nueve componentes | PHASE17 completo, expertos, desacuerdos, ambos contextos y fusión |
| C-OBS | Modulación de la corrección final | Productores, PAIR, contextos y cabeza original | Indicadores de calidad y nueva salida/intervalos declarados |
| C-CLOCK | Observador sobre mediciones actuales e historial | Todos al primer contraste | Estado por objeto, predicción causal y emisión |
| C-CLOCK-L | Un contexto con estado dinámico | Productores y PAIR | Contexto cambiado, features superiores y adaptación controlada de fusión |
| C-SENSOR | Auxiliar de un productor de eventos | Rama RGB y componentes fuera del brazo | Productor cambiado y todos sus descendientes compatibles |
| C-VIS | Evidencia de correspondencia/visibilidad | Backbone al primer diagnóstico | Geometría, contexto y fusión si cambia su entrada |

No sustituir un número de TTC en una caché dejando intactos desacuerdos y estados calculados a partir de su valor anterior. Tampoco conectar tokens de un productor nuevo a PAIR antiguo suponiendo que su base latente no cambió.

## Dos preguntas distintas al cambiar un bloque

**Inserción sin reentrenar:** ¿el cambio mejora la cadena existente con sus dependencias recalculadas? Es la prueba más barata y detecta compatibilidad, pero puede fallar por desplazamiento de distribución.

**Integración con ajuste de descendientes:** ¿mejora al adaptar contexto/cabeza bajo la misma receta y presupuesto que un control sin la idea nueva? Es necesaria cuando cambian features. El control debe recibir los mismos updates; no comparar un candidato reajustado con R0 sin medir cuánto aporta reajustar R0.

Los resultados deben nombrar ambas condiciones. Un fallo de inserción no demuestra que la representación sea peor; una mejora tras reajustar no demuestra que la física sea la causa si el control reajustado mejora igual.

## Compatibilidad de producción

Conservar timestamp del target, historia máxima, contrato de localización, formato TTC firmado, reglas de ausencias y frontera de latencia. Añadir diagnósticos internos sin alterar el formato oficial. Una salida extra de confianza necesita su propia validación: los q10/q90 actuales están en fase y no acreditan calibración en segundos.

El sistema evaluado recibe regiones de objeto. Si se añade detección/tracking, registrar un experimento de sistema distinto, con su coste y fallos; no atribuir esos cambios a la loss física.
