# JEPA como vía opcional para mejorar un productor del release

## Cuándo tendría sentido

La pregunta no sería recuperar el nombre original del proyecto. Sería comprobar si un objetivo predictivo conserva dinámica útil que el productor actual pierde, y si al sustituir ese productor mejora R0 con un coste aceptable.

Abrir esta vía solo cuando se cumplan tres condiciones: una medición sobre datos observados recupera q con soporte suficiente; el productor actual falla bajo intervenciones que preservan la apariencia y cambian la dinámica; y un cambio de lectura, cadencia o contexto no explica por sí solo el fallo. Una sonda lineal mala no demuestra ausencia de información: contrastar también una lectura no lineal de capacidad limitada y controles igualados.

El [piloto del 31 de octubre](../jepa_closure_20261031/README.md) quedó cerrado sin superar sus controles. La petición actual autoriza estudiar un plan potencial; no exige relanzar aquel piloto ni escalarlo. Aquí se propone un experimento distinto, condicionado a evidencia nueva sobre el productor del release.

## Hipótesis nueva y punto de inserción

Adaptar un único productor de eventos con predicción temporal de representaciones que expliquen cambios de escala/correspondencia y, si se admite C-SENSOR, mediciones de eventos. Mantener RGB y su papel de ancla como referencia. No introducir un nuevo world model en toda la cadena de inferencia antes de demostrar utilidad local.

La representación visual puede conservar riqueza espacial y reservar un subestado físico. No forzar toda la escena a dos números ni suponer que un embedding predictivo guarda q por necesidad. El predictor usa tiempos reales y futuro solo como objetivo de entrenamiento.

## Comparación mínima

| Brazo | Entrenamiento adicional |
|---|---|
| J0 | Reajuste supervisado del productor con receta actual |
| J1 | Auxiliar físico observable, sin predicción de embeddings |
| J2 | Predicción latente temporal más los mismos anclajes físicos |
| J3 | Igual que J2 con futuro permutado compatible |

Todos parten de la misma inicialización admitida y reciben los mismos datos, updates y acceso a etiquetas. Igualar o informar FLOPs y parámetros auxiliares. Si J2 solo gana frente al productor congelado y no frente a J0/J1, no atribuir la mejora a JEPA.

Si el preentrenamiento se declara autosupervisado, no utilizar TTC GT, fase derivada de TTC ni targets filtrados por su valor. Un teacher ya entrenado con TTC aporta conocimiento supervisado aunque sus pesos estén congelados: llamarlo destilación/objetivo mixto y declarar su genealogía. Las cajas anotadas también son información privilegiada que debe igualarse entre brazos.

## Qué medir

Primero recuperación de q y respuesta a intervenciones de tiempo, luz y textura; después TTC de la cadena con el productor sustituido. Retirar el predictor auxiliar y comparar lecturas congeladas con igual capacidad. Finalmente reajustar descendientes con el mismo protocolo para todos los brazos, porque cambiar E_A5 puede invalidar PAIR_E y cambiar E_C2F cambia PHASE17.

Separar «la información puede leerse» de «la salida la usa correctamente». Intervenir en la dinámica del input y comprobar la respuesta de la predicción; no concluir uso causal de q porque una sonda lo decodifique. Esta distinción aparece en [What Can Latent World Models Know?, v4](https://arxiv.org/html/2607.27017v4).

Un rollout con composición puede servir como diseño auxiliar, pero no imponer una ley constante donde hay aceleración ni añadir una identidad que ya cumple la capa. [Semigroup-JEPA](https://arxiv.org/html/2609.10464v1) estudia rollouts condicionados por parámetros físicos conocidos en simulación; no demuestra recuperación espontánea de TTC ni ventaja sobre este release.

## Continuación o cierre

J2 debe superar J0/J1/J3 en el objetivo físico fijado y después mejorar TTC de la cadena sin deterioro nativo. Repetir tres entrenamientos del productor modificado, no solo tres cabezas sobre un encoder común. Si solo mejora la predicción de apariencia futura o la loss latente, cerrar la vía con ese resultado.

La utilidad de JEPA aquí sería una mejora comprobada del productor del release. No es un requisito para que C-TIME, C-OBS, C-CLOCK o C-SENSOR funcionen.
