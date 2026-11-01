# Reajuste emparejado del router a estados canónicos

Protocolo fijado el 1 de noviembre de 2026 antes de ejecutar este entrenamiento.
El usuario ha pedido resolver los tres requisitos científicos, no cerrar únicamente
un informe de pendientes. El cierre acotado preparado anteriormente no se ejecuta.

## Hipótesis y alcance

PAIR canónico cambia las entradas del router congelado, que fue entrenado con
estados anteriores. Se contrasta una única hipótesis: mantener la receta QA inversa
y entrenarla con los estados canónicos puede corregir esa incompatibilidad.
El fallo del router congelado en screen149 permanece como resultado negativo.
Este ensayo no reabre las 312 variantes anteriores ni busca hiperparámetros.

Dos brazos: QA con estados originales recapturados y QA con estados canónicos.
Ambos usan exactamente los 2608 tokens sellados, geometría y calidad originales,
8 folds que excluyen familias completas, semillas 7/13/21, 300 pasos de AdamW,
lr=0,01, weight_decay=0,01, pérdida y aumentos de `fit_router` sin cambios.
Normalización y pesos de familias se calculan dentro del conjunto de ajuste.
Las ocho grabaciones del benchmark y Slider-750/1000 permanecen excluidas.
El control original se reajusta porque los estados recapturados no se presumen
idénticos a los de la antigua caché de entrenamiento. No se escoge una semilla.

Se conservan los tres vectores OOF por brazo. La emisión usa `RouterEnsemble`,
incluido su retorno exacto a F0 cuando no hay historia admisible. Una consulta
ausente se conserva como tal. No se filtran filas para ocultar pérdida de cobertura.
Los checkpoints y predicciones se sellan antes de calcular las métricas OOF.
El entrenamiento usa targets exclusivamente en las familias de ajuste de cada fold;
no es un experimento enteramente sin etiquetas. No consulta targets de Test12.

## Decisión previa

El nuevo brazo debe conservar cobertura y no empeorar MAE ni error relativo
macro por grabación en carretera y Slider-500, tanto en toda la población como
en los 149 tokens del screen fijo. Debe mejorar estrictamente al menos una métrica
en cada una de las dos vistas. Los signos y métricas por semilla se publican.
No se ajustan tolerancias, mezcla, presupuesto ni arquitectura tras abrir resultados.
Si falla, se descarta esta hipótesis sin ajuste final ni exportación.

Superar este filtro solo admite los siguientes controles: ajuste de las tres
semillas sobre desarrollo, conservación de eAP TRAIN40 con cadena coherente,
evaluación Test12 autorizada y comparación EvTTC equivalente. No admite promoción
automática ni acredita SOTA. La población tiene exposición histórica; ni los folds
ni el screen se presentan como confirmación nueva. La regresión temporal del
contexto de eventos sigue documentada y no se elimina mediante este reajuste.

## Dependencias externas

La revisión actual del dataset público GarlTTC
`53fbaade17f3b76aa56dc7df2b527ce279062f59` especifica que los targets Test12 son
privados y la evaluación oficial se realiza en CodaBench. No sustituirlos por
predicciones o etiquetas reconstruidas. Slider-750 sigue necesitando el vector
alineado/evaluador autorizado para explicar la discrepancia del resultado previo.
No enviar correos ni sustituir el modelo entregable sin pasar los controles.
