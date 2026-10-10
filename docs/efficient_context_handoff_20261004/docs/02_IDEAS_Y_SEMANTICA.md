# 2. Primeros principios y valoración de las ideas

## Cuatro preguntas distintas
Una modificación puede aportar información nueva, mejorar la estimación con la
misma información, cambiar la tarea o abaratar una implementación. No conviene
confundir esas cuatro cosas. El objetivo inmediato sigue siendo el TTC firmado
de Garl/eAP, con las observaciones permitidas por su contrato local.

A5, C2F y PAIR son productores; los routers son selectores posteriores. PAIR
comparte A5. SIMPLEX-T no vuelve a entrenar los encoders ni constituye un nuevo
preentrenamiento JEPA. Aprende una corrección condicionada por las features; los
pesos quedan fijos en inferencia, pero la corrección cambia para cada consulta.
La corrección opera en fase, no sumando una constante de segundos. Puede salir
del intervalo de los expertos. El 76,3% histórico de targets fuera del intervalo
no significa que todos los TTC se sobreestimaran: separar errores de signo,
sobreestimación positiva, infraestimación positiva y capping.

## Contacto físico: otra variable objetivo
El TTC de profundidad del benchmark y el primer contacto entre cuerpos no son
intercambiables. El segundo necesita además posición lateral, orientación,
dimensiones y movimientos de ambos participantes. PHASE17 no contiene ese estado
métrico completo. No basta con aplicar una fórmula de rectángulos al TTC de H16.

Ejemplo analítico: superficie a 20 m del plano de cámara y cierre de 10 m/s dan
2 s de TTC de profundidad. Si el frontal ego está 2 m adelantado y las trayectorias
se intersectan, el contacto longitudinal sería a 1,8 s. Con desplazamiento lateral
suficiente podría no haber contacto. Sustituir 2 por 1,8 introduce unos 58,65 MiD
respecto a la etiqueta 2, aunque 1,8 sea correcto para la otra pregunta.

Aprobar una auditoría acotada de metadatos TRAIN, no reemplazar GT ni incorporar
campos 3D al forward. Encontrar `box3d_Fcam` en un esquema no demuestra cobertura,
velocidades en un marco común ni disponibilidad en test. Prohibido reconstruir
velocidad como Z/TTC_GT y presentar luego esa velocidad como validación independiente.
La extensión de contacto necesitaría estados CONTACT, NO_CONTACT_IN_HORIZON y UNKNOWN;
+60 s o un TTC negativo no codifican esas tres situaciones. Esa implementación
completa queda fuera de esta campaña y no bloquea la investigación del benchmark.

## Dos fórmulas y un posible problema de referencia temporal
A5 utiliza `r = log(h_b/h_a)` y `q_b = expm1(r)/dt`. El evaluador usa
`psi(T) = -log(1 - 0.1/T)`. Bajo cierre constante, `T_a = T_b + dt` y
`r = psi(T_a; dt)`, no `psi(T_b; dt)`. Si la altura pasa de 50 a 55 píxeles en
0,1 s, los TTC son 1,1 s en el primer instante y 1 s en el segundo.

Esta identidad NO demuestra un bug en las etiquetas o el código. Hay que seguir
el timestamp de cada target, productor y conversión. El paper y la implementación
son fuentes que contrastar, no una justificación para elegir retrospectivamente
el desplazamiento temporal que mejora OLD_DEV. Si se prueba un defecto real,
preservar los resultados originales y abrir una enmienda separada.

La revisión de índices incluida aquí acredita que la última ventana actual acaba
en el anchor registrado. No acredita la semántica ancestral de todas las etiquetas.
La disponibilidad del ROI es posterior; esa diferencia tampoco es una medición de
la latencia de un detector desplegado.

## Control analítico adicional, sin entrenamiento: EWMA_TRANSPORT_CV
La EWMA histórica combina estimaciones de distintos instantes sin transportarlas
al anchor actual. Bajo cierre longitudinal uniforme, una predicción perfecta de
2 s tomada 200 ms antes corresponde a 1,8 s ahora. Promediar ambas directamente
introduce un retardo incluso cuando no existía error perceptivo. Eso no demuestra
que esta sea la causa del resultado observado; motiva un control adicional.

Registrar antes de evaluarlo `EWMA_TRANSPORT_CV_H8`, sin modificar la EWMA original.
Usar los mismos ocho slots, mediana de fases expertas por observación y pesos
`exp(-edad/0.3)`. Primero llevar cada mediana al anchor actual bajo velocidad de
cierre constante: `T_actual = T_observado - edad`. En inverse-TTC, la transformación
es `q_actual = q_observado / (1 - edad*q_observado)`. Después convertir con el
contrato canónico de fase y agregar. No promediar TTC firmados directamente.

Usar timestamps y fases originales de los productores, no valores inventados a
partir de los targets. Si no puede acreditarse el anchor de las observaciones,
marcar sólo este control como no admitido. Si un transporte positivo atraviesa el
contacto o queda fuera del dominio TTC > 0,1 s, excluir únicamente ese término
histórico y reportarlo; nunca excluir la consulta. El presente válido es obligatorio
y sirve como fallback. Una fase exactamente cero (infinito según el productor)
permanece cero hasta la emisión canónica. No reinterpretar una salida capada de
60 s como una medición física exacta; conservar sus flags y reportar sensibilidad.

El ROI retrospectivo no garantiza seguir el mismo objeto. Por ello este control
supone consistencia de objeto y movimiento, no acredita que se cumplan en todos
los casos. Mantener todas las consultas y reportar términos rechazados, cobertura,
capping, signos y resultados por secuencia. No calibrar la constante 0,3 ni elegir
una regla de rechazo mirando OLD_DEV. Si mejora, es evidencia descriptiva sobre
alineamiento temporal; si falla, no invalida la utilidad del contexto aprendido.

Este control usa cero actualizaciones, no es una reparación de los resultados
históricos y no altera la regla de réplica H8-WIDE. Forma parte de E0/E1, no abre
una búsqueda de filtros físicos. Su función de referencia ilustra transporte
analítico; la integración debe reutilizar la conversión canónica del repositorio.

## Decisiones sobre las ideas
| Idea | Decisión |
|---|---|
| Optimizar raw/ROI | Prioridad principal: ahí domina el coste observado. |
| Deduplicar ventanas exactas | Correcto como caché; beneficio intraconsulta casi nulo en los índices revisados. |
| Leer una unión / sustituir add.at por bincount | Ya está implementado; no es una propuesta nueva. |
| Mantener GRU entre consultas sin reentrenar | No equivalente: cambia reset, edades y ROI. Requiere otro experimento. |
| Más alcance con ocho observaciones | Probar H8-WIDE con slots fijos. Menos slots no implica menor span raw. |
| Más secuencias | Respaldado localmente; los 22 grupos adicionales inspeccionados ya pertenecen a D1. |
| Router más grande | No ataca el cuello de botella actualmente demostrado. |
| Contacto como GT «mejor» | Rechazar como sustitución de las etiquetas Garl. |
| Auditoría geométrica | Útil para interpretar; no debe volverse un requisito universal. |
| Reparar LATENT sin otra prueba | Rechazar. El fallo se conserva; su causa no está identificada. |

## Literatura y posibles líneas posteriores
El trabajo de 2022 sobre slow features demuestra en un entorno sintético que
preservar variación global puede coexistir con perder movimiento. No diagnostica
por sí mismo una cabeza supervisada sobre A5 congelado.

MotionJEPA/DISReg puede orientar un encoder entrenable que conserve cambios. Su
rama dinámica predice embeddings de diferencias visuales a partir de pares de
embeddings; el modelo completo publicado conserva entradas de acción. Adaptarlo
a eventos exige definir qué cambio se quiere conservar, y controlar duración,
contraste, iluminación y ROI. Comparar supervisado desde cero, SSL y más updates
supervisados; no adjudicar automáticamente a JEPA cualquier mejora geométrica.

FAR orienta la selección de recuerdos útiles. Pero seleccionar cuatro ventanas
después de ejecutar los expertos sobre dieciséis no ahorra el coste dominante.
La selección tendría que usar señales baratas o descriptores obtenidos legítimamente
de antemano, sin consultar el futuro ni GT en inferencia.

PixelUMM aporta una pregunta general sobre interfaces de representación; no es una
dependencia razonable del siguiente experimento de TTC en esta máquina. REACT sí
es una referencia de percepción streaming específica, pero usa otra tarea y métrica;
no se introduce su arquitectura en esta campaña ni se compara su RTE con MiD.

Finalmente, «cronología no demostrada» no significa «no hay información temporal».
Las features ya incluyen dinámica de tres ventanas expertas. SET_NOTIME elimina
cuatro canales, no toda esa información. Una inversión fija durante fit puede ser
reaprendida. Mantener separadas utilidad, mecanismo y novedad del entrenamiento.
