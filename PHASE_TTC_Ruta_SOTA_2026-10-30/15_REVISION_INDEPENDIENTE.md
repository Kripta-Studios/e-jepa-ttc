# 15 — Revisión crítica antes de declarar un avance

Este documento contiene preguntas concretas para una segunda persona o agente que no haya diseñado el candidato. Se aplica al cierre de cada fase pertinente; no obliga a repetir toda la campaña. Una revisión debe dejar hallazgos verificables, no una aprobación genérica.

## 1. Protocolo y exposición

| Pregunta de revisión | Evidencia que exigir | Si falla |
|---|---|---|
| ¿Coinciden el código descrito y el candidato ejecutado? | SHA, hashes y recibo de carga estricta | Corregir identidad antes de comparar. |
| ¿El splitter conoce grabaciones, cámaras y adquisiciones relacionadas? | Tabla de grupos con evidencia, aliases y casos ambiguos | Agrupar/excluir; no certificar independencia por nombre. |
| ¿Se mantuvieron fuera las ocho grabaciones solapadas completas? | Prueba contra identidad de contenido y relaciones, no solo rutas | Rehacer protocolo y cualquier ajuste afectado. |
| ¿Algún productor, teacher, contexto o normalizador vio esos datos? | Genealogía de nueve componentes y dependencias | Limitar la afirmación de transferencia; conservar incertidumbre. |
| ¿El holdout ya había influido en decisiones históricas? | Registro de exposición a labels/scores/diseño | Describir reserva prospectiva dentro de datos expuestos; añadir evaluación nueva. |
| ¿Hay datos independientes para Slider? | Grupos nuevos, adquisición distinta, roles sellados | No convertir el resultado de carretera en evidencia Slider. |
| ¿La elección del split se hizo mirando scores? | Fecha, registro de propuestas y criterios de asignación | Considerar la población usada en selección y buscar confirmación nueva. |

## 2. Targets y métricas

1. Tomar un ejemplo sintético de TTC conocido y reconstruir a mano su instante de referencia. ¿El evaluador puntúa el TTC actual, anterior o al centro de la ventana?
2. Verificar que cambiar el origen absoluto del reloj no altera la estimación ni la etiqueta alineada.
3. Recalcular macro y micro con un fixture donde sean distintos. Si coinciden por implementación, hay un error.
4. Revisar las 11 consultas fuera de cobertura: deben seguir ausentes en la tabla de targets, aunque el modelo produzca valores finitos.
5. Buscar máscaras de evaluación condicionadas a salida, error o soporte aprendido. La población primaria debe fijarse por el protocolo, no por los casos en que el candidato acierta.
6. Comprobar conservación del signo y distinguir datos negativos de salidas inválidas por una regla concreta del benchmark.
7. Verificar que la interpolación de negativos del organizador se describe como transformación externa del score. No atribuirle causalidad de inferencia.
8. Auditar fórmulas MiD: el mismo nombre en dos papers no garantiza mismo logaritmo, escala, bandas o fallos.
9. Para Garl, comprobar que la tabla pareada no oculta sus 10/8 huecos en las variantes full/RGB. Incluir cobertura de la población completa.
10. Si se afirma reproducción oficial, comprobar targets, filas, transformación y redondeo. Si falta un elemento, denominarla diagnóstico local.

El caso de referencia temporal y la diferencia entre coordenadas `r`, `q` y `phi` están desarrollados en [04](04_EVALUADOR_Y_BASELINES.md) y [05](05_DIAGNOSTICO_DE_SENSORES_Y_MODELO.md).

## 3. Sensor y causalidad

Revisar una consulta de arranque, una madura y una con eventos escasos bajo la misma política. Verificar timestamps máximos de sensores, cajas, estado previo, estadísticas de normalización, teacher si existe y buffers. Modificar artificialmente todo lo posterior a la consulta: ninguna salida pasada debe cambiar.

La ausencia de eventos dentro de una ventana observada no es la misma situación que una ventana que precede al comienzo de grabación. El ledger debe distinguir ambas. Duplicar imágenes, inventar eventos o repetir un estado como si fuera una nueva medición no soluciona el soporte.

La ROI común puede ser aproximadamente correcta a gran distancia y fallar de cerca por paralaje. Exigir pruebas de dirección de transformación, distorsión, resolución y convenciones de caja. Un overlay visual es evidencia de correspondencia aproximada; no acredita por sí solo la exactitud del TTC.

Si se cachea, inspeccionar la clave: una observación histórica recortada con la ROI actual cambia cuando cambia la consulta. Cachear solo por timestamp no demuestra paridad. [Política que origina este requisito](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/decisions/ADR-0043-evttc-competition-sensor-adapter.md).

## 4. Atribución del avance

| Afirmación propuesta | Control que debe existir |
|---|---|
| «La calibración arregla la transferencia» | Fixture físico y evaluación fuera de los datos usados para elegir la corrección. |
| «La fusión aprende fiabilidad» | F1 con capacidad fija y Fcap sin nuevas variables; métricas de calidad/calibración cuando se afirmen. |
| «La consistencia temporal mejora» | Comparación con el transporte ya existente, retardo, cambios de velocidad y resets. |
| «JEPA conserva movimiento» | J2 frente a J1, probes de expansión/signo y TTC en grupos separados. |
| «El backbone preentrenado es mejor» | Misma cabeza, datos, horizonte y selección; coste y preentrenamiento declarados. |
| «El modelo es más rápido» | Misma frontera de tiempo, cobertura y semántica; p95/p99 y acumulación de trabajo. |
| «Generaliza a nuevos sensores» | Sensor/adquisición no usados para diseño, no solo otra cámara del mismo episodio. |

No exigir un control irrelevante para toda contribución. Una corrección de lectura no necesita entrenar tres semillas. Una mejora aprendida no queda validada por repetir un fixture determinista.

## 5. Estadística y elección

Pedir la lista completa de intentos. Si el ganador procede de muchos candidatos, describir el presupuesto de búsqueda. Preguntar si se eligió una semilla por test, si se abrió varias veces el holdout o si se recalibraron umbrales después de ver el resultado reservado.

Examinar las diferencias por grupo y su distribución. Muchas ventanas de una misma grabación no proporcionan muchos episodios independientes. Si el intervalo es inestable por falta de grupos, reportarlo; generar más remuestras bootstrap no crea información.

Separar variación de entrenamiento y variación entre adquisiciones. Una mejora media puede depender de una sola secuencia o degradar TTC corto/negativos. Esa estructura importa más que imprimir muchos decimales.

## 6. Revisión del claim final

Antes de aceptar «SOTA», completar una frase de alcance:

> El método [ID y versión] obtiene [métrica y cobertura] en [benchmark, población y fecha], con [modalidades, cajas e historia], tras entrenar o adaptar en [datos]. La comparación incluye [métodos] bajo [contrato]. La generalización adicional se comprobó en [datos nuevos o limitación explícita].

Si la frase solo puede completarse con «en nuestro subset», el claim es de ese subset. Si una evaluación oficial mejora pero el test influyó en diseño, la posición oficial y la exposición se comunican simultáneamente. Si no hay reproducción de un baseline, se distingue resultado publicado de resultado ejecutado localmente.

## 7. Formato de la revisión

```markdown
# Revisión de [experimento]

Revisor:
Commit y artefactos:
Fecha:
Alcance leído/ejecutado:

## Hallazgos materiales
| ID | Evidencia | Consecuencia | Acción mínima |
|---|---|---|---|

## Preguntas que siguen abiertas

## Decisión
GO / GO_CON_ALCANCE_LIMITADO / NO_GO / INCONCLUSO

## Claim permitido

## Evidencia que permitiría ampliar el claim
```

Un cierre `NO_GO` bien explicado también es un resultado útil. Reduce el riesgo de invertir el siguiente bloque de trabajo en una mejora aparente.
