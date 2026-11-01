# Comparación fija de reinicio de contexto nativo

Estado: protocolo previo; extracción ampliada y entrenamiento aún pendientes.
No modifica ni promueve la entrega congelada.

## Hipótesis y puerta de entrada

La fusión puede depender de productores/contextos que fallan juntos cuando hay
poca historia. Las perturbaciones arbitrarias de predicciones ya fracasaron.
Ahora se prueban estados recomputados por los contextos originales a partir de
observaciones reales y causales, manteniendo productores y sus pesos congelados.
Esto es reinicio de contexto, no reinicio del sensor ni aumento por velocidad.

Requerir caché ampliada de 149 identidades: 72 consultas de carretera, tres
cuantiles temporales por cada uno de los 24 grupos elegibles, y las 77 Slider500.
Selección por reloj/identidad, no por error. El piloto ya pasó en dos consultas,
pero cada consulta ampliada debe conservar resultado original y paridad limpia
desde NPZ+JSON verificados. Conservar ausencia/event-only/fusión; nunca inventar
F0 cuando el runtime se abstiene ni reutilizar observaciones de otra consulta.
No abrir Slider750/1000 ni los ocho grupos excluidos. Registrar fallos por separado.

Generar últimas 1,2,3 observaciones aceptadas de cada modalidad. Recalcular clocks
y extraer F0, RGB y eventos de sus salidas contextuales. Conservar geometría y
sus variables de calidad exactamente, pues sus observaciones no se han cambiado.
Las variables derivadas de discrepancia de expertos se recalculan dentro de IQ.
Guardar las matrices y sellar sus hashes antes de unir los targets de desarrollo.
Si una ruta no tiene un experto disponible, conservar NaN y la causa; usar la
misma admisión de IQ. No describir este cambio como aumento de sensores crudos.

## Comparación y selección

Una sola variante IQn frente al control IQ original: arquitectura de 688
parámetros, fusión inversa, semillas 7,13,21, ocho folds por familia, 300 updates
AdamW, lr 0.01, weight_decay 0.01, clip 1. Normalizadores, pesos por familia y
escala de MAE proceden exclusivamente de filas limpias del fold de ajuste.

IQn usa pérdida 0.5 limpia + 0.5 reinicio. Las filas limpias son las 2608 elegibles
del fold; las aumentadas son únicamente las consultas seleccionadas del mismo
fold, con variantes 1/2/3. Equilibrar familias en cada término y dividir el peso
de cada consulta entre sus variantes admitidas. Conservar el mismo objetivo
RAE protegido en 0.2 s + MAE dividido por mediana absoluta de targets de train.
El control IQ reproduce exactamente `fit_router(..., "QA", ..., "inverse")`:
mitad de pérdida limpia y mitad con corrupción de un experto en el 25 % de
filas, factores originales -3,-1,0.25,4. IQn sustituye ese término artificial
por reinicios nativos, manteniendo su peso 0.5. No llamar al control original
"sin perturbación": una revisión del código detectó esa descripción incorrecta
antes de implementar o ejecutar la campaña. No se modifica el IQ histórico.
Se esperan 48 ajustes completos. No cambiar este protocolo según sus resultados.

Las 20 variables de calidad conservadas describen geometría, no soporte de los
contextos nativos. IQn prueba adaptación a sus salidas perturbadas; no incorpora
un mecanismo explícito nuevo que observe cuántos eventos o frames las sustentan.

Evaluación primaria sobre las consultas limpias OOF de los 25 grupos, con
promedio de las tres semillas conforme al control original. Cuatro requisitos:
no empeorar MAE ni RAE macro por grabación en carretera y Slider500, cobertura
completa. Reportar además cada semilla y la población de reinicios OOF como
diagnóstico secundario; no seleccionar usando esta métrica secundaria.

Solo si supera todos los requisitos, refit con desarrollo permitido y evaluar
conservación eAP TRAIN40 activa frente a la versión congelada (MAE y RAE),
aclarando que es resustitución y no generalización. Test12 disponible no tiene
etiquetas: permite deriva de predicciones, no certificar precisión. No promover
ni presentar SOTA con estos datos; una comparación oficial equivalente sigue
requiriendo resolver el evaluador Slider750 y confirmar la nueva entrega.

Preservar código ejecutado, fuentes, hashes, tokens, grupos de ajuste por fold,
checkpoints, pérdidas, predicciones por semilla, intentos fallidos y selección.
Si no mejora los cuatro criterios, conservar el resultado y no ajustar otro
hiperparámetro dentro de esta campaña.
