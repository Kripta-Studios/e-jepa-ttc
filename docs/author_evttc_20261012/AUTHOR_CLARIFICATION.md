# Aclaración recibida y consecuencias para las comparaciones

Fuente: correo firmado por Hongxin y pegado por el usuario en esta conversación.
No se dispone de fecha/cabeceras del correo ni se atribuye su contenido a una
publicación revisada. Aquí se resumen los aspectos técnicos, no se redistribuye
la correspondencia personal completa.

## Confirmaciones aportadas

Los datos/medios/anotaciones publicados fueron revisados y alineados. Se
recomienda usar los recursos públicos de Hugging Face. Esto **no proporciona
la lista exacta de secuencias de entrenamiento del checkpoint**: esa pregunta
sigue abierta.

Test12 conserva etiquetas privadas y se puntúa mediante CodaBench:

https://www.codabench.org/competitions/17289/

No se ofrecen labels, paquete de scoring privado offline ni evaluación
individual por correo. Una submission preparada no es un score oficial. Esta
intervención no envía mensajes ni acredita un envío a CodaBench. La extracción
web automática consultada devolvió contenido incongruente para esa URL; no se
usó como evidencia del estado de la competición. El enlace procede del correo.

Garl se entrenó con eAP y se evaluó en EvTTC sin fine-tuning, según los autores.
El checkpoint público completo es `paper_ours_full.pth`. El correo precisa que
el paper no documenta explícitamente el nombre del checkpoint usado para
EvTTC; no convertir esa identificación en confirmación de equivalencia exacta.

## Configuración EvTTC descrita

Escenarios CCRs2-medium, CCRs2-high y CCRm-medium; intervalos locales respectivos
[9,4;11,6], [15,5;17,4] y [21,4;22,2] segundos. Los autores los describen como
configuración disponible, **no una lista exhaustiva verificada de consultas**.

Cámaras RGB/event izquierda. Cajas de `leftlabel/*.json`; ante varios objetos,
selección por proximidad al centro anterior. Mapping RGB→event mediante
calibración, proyección de cuatro esquinas y rectángulo envolvente; crop cuadrado
con dimensión máxima y factor 1,0. Muestreo RGB ~100 ms, tripletas deslizantes,
dos últimas imágenes y dos segmentos de eventos. TTC leído de `ttc.csv`, vecino
temporal más cercano con distancia máxima 50 ms. Se omiten muestras sin
anotación/GT/eventos, con desajuste temporal o fallo de preparación.

La precisión de esa tabla se mide mediante RTE=100·|pred−GT|/|GT|. El paper
publica Garl RTE 8,31%, 10,56%, 12,93% y promedio 10,60%; runtimes 13, 13 y
12 ms, promedio redondeado 13 ms. **Son resultados de los autores**, no medidos
en nuestro ordenador. Fuente primaria:

https://arxiv.org/html/2603.16303v1#S7.T6

## Información que no proporciona el correo

Faltan IDs de consultas, filenames completos, fase exacta de muestreo, regla
inicial del tracker, detalles algebraicos completos de la proyección entre
cámaras, exclusiones exactas y equivalencia del checkpoint. Tampoco define
con suficiente detalle el perímetro de latencia para afirmar que una medida
local de extremo a extremo es comparable a los milisegundos de la tabla.

Por tanto el experimento siguiente es una **reconstrucción explícita** con
decisiones fijadas de antemano, no un intento de variar crops/tiempos/labels
hasta aproximarse al 10,60%. Dev32 continúa siendo una evaluación diferente.
