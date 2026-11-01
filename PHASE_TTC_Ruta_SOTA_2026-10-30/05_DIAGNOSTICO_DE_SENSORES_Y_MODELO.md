# 05 — Diagnóstico de sensores y del modelo

## Objetivo y entrada

Cerrar **G3, diagnóstico**, con una causa reproducible o una lista de hipótesis refutadas. Antes de ejecutar un cambio científico deben estar fijadas las identidades G0, los roles de datos G1 y el contrato de evaluación G2. La falta de equivalencia con el scorer oficial permite trabajar en desarrollo independiente, pero obliga a etiquetar sus métricas como locales. Este documento no da por ejecutado ningún experimento.

La pregunta útil es dónde deja de ser fiable la cadena: sensor → ROI → representación → productor → contexto → fusión → conversión a TTC. Las inspecciones sobre competición son diagnóstico expuesto. La elección de una modificación se realiza con desarrollo elegible según [03](03_PROTOCOLO_Y_DATOS.md).

## 1. Instrumentar la cadena exacta: A00

La ruta existente pasa por `SensorSequence`, `prepare_query`, `CompetitionRuntime` y `FinalV16Train40Predictor`. La cabeza que produce la entrega es `PhaseOnlyFinal`. No sustituirla por otra arquitectura que comparta nombres históricos. [Código del runtime](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/runtime.py), [predictor final](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/test12.py).

Crear una traza opcional que conserve **todas las observaciones aceptadas y rechazadas**. Actualmente `_prepare_values` guarda detalle de los rechazos y resúmenes de soporte, pero no el recorrido completo de cada observación aceptada. El modo de traza debe demostrar que no altera la predicción del baseline.

| Bloque | Campos mínimos por consulta |
|---|---|
| Identidad | Grabación, fila, frame, timestamp, hash de entrada y política |
| Sensor | Ventanas, frames reales, deltas, origen del reloj, cobertura |
| Geometría | Plano bbox, ROI en cada sensor, clipping, calibración |
| Productores | TTC y fase A5/C2F/PAIR, soporte, `known`, varianza |
| Normalización | Hash, resumen de cada feature estandarizada, referencia de train |
| Contextos | Fase, cuantiles, ancla, edad y longitud de historia |
| Fusión | Ancla RGB, residual, fase antes/después de límites, TTC firmado |
| Coste | Lectura, representación, productores y contexto/fusión |

Unir GT **después** mediante el evaluador, por identidad exacta. La instrumentación de inferencia no debe recibir etiquetas. Seleccionar consultas de inspección por índices, cadencia y soporte fijados previamente; si se muestran los peores errores como exploración, declararlo y no usarlos para escoger hiperparámetros.

## 2. Geometría y plano de las cajas: A01

El adaptador interpreta las cajas como coordenadas del raster de eventos y proyecta hacia RGB con intrínsecas, distorsión y rotación. Omite traslación. La rotación Slider procede de un consenso de poses y queda marcada `candidate_only_no_authoritative_intercamera_extrinsic`. Son limitaciones documentadas, no causas demostradas del score. [Geometría](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/geometry.py), [ADR-0043](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/decisions/ADR-0043-evttc-competition-sensor-adapter.md).

Pasos:

1. Confirmar con especificación del proveedor el sensor de las bbox, orden `xyxy`, origen de índices y límites inclusivos/exclusivos.
2. Registrar transformaciones y convenciones de cada cámara; no deducir dirección solamente del nombre de una matriz.
3. Generar overlays de índices fijados y medir residuos de correspondencia, recorte del objeto y contaminación de fondo.
4. Construir un fixture de dos cámaras con extrínsecas conocidas, objeto en expansión y varias profundidades.
5. Comparar proyección completa con la aproximación sin traslación. El término omitido depende de `t/Z`; el efecto puede crecer a corta distancia.
6. Si hace falta profundidad para una corrección, definir una fuente permitida en inferencia. Nunca utilizar profundidad, distancia o TTC GT para preparar la entrada.

**Cierre:** demostrar una violación concreta de contrato o cuantificar la limitación de la aproximación. Un overlay visualmente razonable no prueba precisión TTC; uno deficiente tampoco identifica por sí solo la extrínseca responsable.

## 3. Relojes y extremo temporal: A02

El lector ya distingue timestamps de eventos, cabeceras RGB y tiempo de reproducción ROS. En Slider usa la primera cabecera RGB como origen y conserva eventos anteriores con tiempo negativo. La preparación selecciona frames reales y pasa deltas medidos. No empezar reimplementando contratos ya presentes. [Lectores](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/readers.py), [preparación](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/prepare.py).

Pruebas imprescindibles:

- Trasladar todos los relojes por la misma constante: entradas relativas y predicción deben conservarse.
- Modificar sensores/bbox posteriores a la consulta: su salida pasada debe conservarse.
- Representar la misma expansión física a distintas cadencias reales: comprobar fórmulas analíticas y medir, por separado, degradación del modelo aprendido.
- Comprobar si el TTC corresponde al extremo anterior, al actual o a otra referencia geométrica.

Para `h=k/Z`, cierre a velocidad constante y `r=log(h_actual/h_anterior)`:

```text
T_actual   = Δ / expm1(r)
T_anterior = Δ / (1-exp(-r)) = T_actual + Δ
```

**Fixture mínimo:** `Δ=0,1 s`, `h_actual/h_anterior=1,1`. El TTC actual es **1 s** y el anterior **1,1 s**. El productor PHASE usa `expm1(r)/Δ` para inverse TTC y su target usa `log1p(Δ/T)`, coherentes con el extremo actual bajo esas hipótesis. [Productor](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/models/causal_scale_ttc.py).

El adaptador Garl ejecuta `Δ/(1-h0/h1)`. Si sus alturas representan literalmente esos dos extremos, sería la referencia anterior. **No es un bug de Garl demostrado**: falta verificar semántica aprendida y contrato del target. Preservar el baseline histórico y comprobar la referencia antes de comparar. Restar Δ porque mejora el score sería una elección inválida. [Conversión Garl](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/garl_evttc_competition/model.py).

La fase final PHASE de 100 ms es una coordenada canónica distinta de `r`. No cambiarla a 80 ms por la cadencia Slider: el productor ya usa el delta real.

## 4. Arranque y soporte: A03

La política existente considera cuatro anclas a −300/−200/−100/0 ms. Con poca historia divide el prefijo grabado en tres ventanas positivas; mantiene silencio observado y distingue falta de grabación. Una salida finita no demuestra que ese prefijo aporte señal útil. [Política y código](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/prepare.py).

Ejecutar reinicios artificiales en puntos predefinidos de desarrollo independiente. Conservar todas las consultas y comparar historia disponible, duración por ventana, event rate, rechazo matemático, contexto y error. Medir curvas respecto al tiempo desde reinicio; informar también el agregado completo. No seleccionar un warm-up que elimina precisamente los errores.

Analizar Slider-750 y Slider-1000 por separado: el primero sugiere investigar arranque y dominio de fase; el segundo tiene error alto sin exigir esos síntomas. Ningún resultado en carretera demuestra por sí solo mejora en Slider.

## 5. Fusión y sensibilidad de fase

La salida cumple aproximadamente `T=0.1/(1-exp(-phi))`, con límites nativos y saturación de magnitud en 60 s. Cerca de `phi=0` la sensibilidad es grande y un pequeño residual puede cambiar signo o magnitud de TTC. Conservar fase, residual, límites y signo; no diagnosticar solo curvas en segundos. [Fase canónica](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/simplex_t/phase.py).

Comparar productores, contexto de eventos, contexto RGB y fusión con targets compatibles. Desacuerdo entre ramas no identifica cuál tiene razón. Revisar también features estandarizadas fuera del rango de entrenamiento: no recalcular normalizadores con competición para hacerlas parecer normales.

## Entrega de G3

Producir `DIAGNOSIS_DECISION.md`, trazas y manifiesto de fuentes. Cada hipótesis lleva estado `supported`, `refuted` o `unresolved`, evidencia y siguiente prueba. La decisión remite a [06](06_INTERVENCIONES_MINIMAS.md) si hay defecto o inestabilidad, a [07](07_FUSION_Y_FINE_TUNING.md) si persiste un sesgo aprendido, o a nueva adquisición si faltan datos discriminantes.
