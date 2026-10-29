# 02 — Atribución antes del nuevo aprendizaje

## N0: revisión pública y autoridad

Reutilizar los recálculos entregados. Verificar las identidades del ZIP/snapshots y los resultados de FCWD, sin ejecutar otra vez una evaluación cruda completa. Leer estados vivos V13/R1 y registrar procesos propietarios. No escribir en la rama, admisiones o directorios de entrenamiento activos.

Construir `ACTIVE_STATE.json` (snapshot fechado), `SOURCE_BINDINGS.json` y `PENDING_DEPENDENCIES.json`. No confundir ausencia de un archivo en Git con ausencia local. Los paths en recibos son referencias que el agente debe resolver en el host.

## N1: trazas discriminantes de productores y readout

Con pesos finales disponibles, generar un CSV por consulta/observación con:

- tiempos anchor/event_end/available, deltas, lag efectivo, ROI y referencia espacial;
- altura anterior/actual, log-ratio analítico, residual, log-ratio final y soporte;
- flujo/transporte fino-grueso, confianza, log-varianza y peso del selector interno C2F cuando exista;
- fases TTC individuales A5/C2F/PAIR, mediana actual y residual de cada cabeza;
- fase antes/después de emisión, salida de TTC, saturación/cap y disponibilidad;
- módulos, pesos, normalizadores, IDs y roles utilizados.

No inventar «pesos de router H8»: H8 es un corrector residual, no un argmin entre expertos. Distinguir la compuerta interna de transporte de C2F, su cabeza auxiliar de costes y el residual principal.

Usar dos conjuntos: las 20 consultas FCWD ya publicadas más sus vecinos para diagnóstico post hoc, y 32 consultas H/P seleccionadas por hash y cubriendo las bandas disponibles para la comprobación técnica. Las primeras no se convierten en TRAIN. No abrir nuevos ficheros sellados. Comparar ruta cruda exacta y ruta admitida, sin warp como referencia.

Para cada rango, separar si el error ya existe en alturas/expansión, en el productor o si aparece/se reduce tras H8. Intervenir suprimiendo el residual sólo para diagnóstico sin entrenamiento. No promocionar una intervención evaluada sobre FCWD como nueva receta.

## N2: masa, exposición y gradientes

Antes de entrenar las nuevas cabezas, auditar en H:

- número de filas, grupos/adquisiciones y episodios con TTC≤1, (1,2], (2,3], (3,6], (6,8], (8,10] y negativos;
- masa total nativa por grupo/banda/subbanda, no sólo conteo de filas;
- aparición esperada en 2.500×128 draws uniformes, ejemplos repetidos y concentración por secuencia;
- norma de gradiente por término y banda en batches H fijos, diagnóstico separado del optimizador;
- predicciones en la región plana del decodificador y saturación de output. No introducir estimadores straight-through clandestinos para «arreglarlas».

336 targets TRAIN40 ≤1 s no son 336 episodios independientes. FCWD falla en 1–3 s y no contiene ≤1; por tanto una acción enfocada sólo a ≤1 no responde directamente al fallo.

## Relojes y sensibilidad

Bajo cierre constante, el TTC cambia con el instante. Antes de restar una latencia o transportar una estimación, demostrar el anchor mediante código y anotación. No buscar un offset que minimice FCWD.

Para fase canónica `phi(T)=-log(1-0.1/T)`, la magnitud de la derivada es `0.1/|T*(T-0.1)|`. Esto explica diferencias MAE/MiD, no prueba la pérdida óptima. La discontinuidad entre TTC negativo y positivo cerca de fase cero representa una inversión a través de infinito; no se debe equiparar automáticamente a un artefacto que se arregla cruzando TTC=0.

## Fallos de integridad y dependencia

Una discrepancia real en input/GT/anchor bloquea sólo el experimento que depende de ella hasta documentar su corrección. Conservar históricos y recalcular todos los comparadores afectados. Un log legible de un resultado excluido no justifica inventar un bloqueo de todo el programa: documentar exposición, separar población y no utilizarla para fit/selección.

No esperar respuesta de autores, FCWD RGB o score oficial para escribir pruebas, auditar H y completar tareas independientes. Tampoco llamar experimento negativo a una dependencia ausente.
