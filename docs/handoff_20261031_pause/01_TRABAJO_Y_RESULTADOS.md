# Trabajo realizado, resultados y límites

## Alcance de esta reconstrucción

Este documento responde al cierre de las últimas cuatro horas solicitado por el
usuario el 31 de octubre, alrededor de las 18:40 CET. Usa commits, informes y
recibos conservados. Las horas de la tabla son **horas de publicación en Git**,
no se presentan como el inicio exacto de cada entrenamiento. Los 312 ajustes son
el total acumulado verificado de la campaña de fusión; no se atribuyen sin más
todos a una ventana de cuatro horas ni se cuentan como entrenamientos completos.

| Hora CET | Commit | Entrega registrada |
|---|---|---|
| 16:24:29 | `d0d93df8` | Cierre JEPA de profesor fijo y conservación de evidencia del pod |
| 16:38:58 | `1fdda033` | Evaluador, eAP activo, Test12 y campañas de fusión descartadas |
| 16:42:11 | `82d6c992` | Estabilidad de hashes de evidencia publicada frente a finales de línea |
| 17:25:47 | `444e3cdd` | Aumentos correlacionados de expertos; acumulado de 168 ajustes |
| 17:31:35 | `e8ec1584` | Experto de expansión de eventos y ruta híbrida CUDA |
| 17:47:22 | `fdca738d` | Caché de observaciones nativas y aumentos de reinicio |
| 18:00:30 | `93d58c6f` | Indicadores de soporte frente a control de capacidad |
| 18:09:39 | `095e1ab9` | Soporte más reinicios; acumulado de 312 ajustes |
| 18:30:26 | `4926b765` | Diagnóstico temporal de productores y sensibilidad de PAIR |

La intervención canónica y la auditoría eAP terminaron después del último
commit de esta tabla y se publican con este handoff. La solicitud de pausa
impidió abrir la siguiente extracción poblacional; no dejó a medias un run GPU.

## Evaluador y significado de las cifras

Se comprobó que las fuentes públicas de ground truth de CCRs1-low y
CCRs1-medium estaban asignadas al revés localmente. Se corrigió la asignación
por identidad de los archivos oficiales, manteniendo los offsets demostrados;
no se ajustaron relojes para favorecer puntuaciones. Se recuperaron las once
etiquetas que faltaban y quedaron 434 targets públicos alineados.

Nueve resultados por secuencia reproducen la precisión mostrada en el PDF de
los organizadores. Slider-750 sigue discrepando: el original procesado da
283,931058 % en el CSV público y 132 % en el PDF. La diferencia exige el vector
alineado o el evaluador/versionado de los organizadores. Existe un borrador de
consulta, pero no se ha enviado correo.

La tabla completa está en
[el informe del evaluador](../research_log_20261031_sign_safe/README.md).
La reevaluación local de la versión congelada arroja 17,249803 % sobre 434
consultas públicas y 7,469419 % sobre las 365 de carretera. **No son resultados
de la nueva corrección PAIR ni una evaluación oficial nueva.** No comparar el
17,25 % local con el antiguo agregado oficial como si compartieran un scorer
demostrado. La discrepancia Slider-750 permanece visible.

## Entrenamientos de fusión y aumentos

La población fue de 2.608 consultas: 2.531 de carretera, distribuidas en 24
grabaciones, y 77 de Slider-500. Se excluyeron las ocho grabaciones de carretera
del benchmark solapadas y Slider-750/1000. Ocho folds reservaron familias, con
semillas 7, 13 y 21; cada ajuste realizó 300 actualizaciones. Normalizadores,
escalas y selección se calcularon sin usar el grupo reservado para ese ajuste.
Existe exposición histórica: no se presenta como un holdout nuevo.

Se ejecutaron seis bloques: 96 ajustes iniciales, 48 con supervisión de coste,
24 con perturbaciones correlacionadas, 48 con reinicios reales, 48 con soporte
y control de capacidad, y 48 combinando soporte y reinicios. Total: 312.
Se preservaron además seis ajustes de un lanzamiento fallido anterior, que no
se suman como campaña completa.

Los criterios exigían cobertura completa y no empeorar MAE ni error relativo
macro por grabación en carretera y Slider frente al control IQ. Ninguna nueva
variante cumplió los cuatro criterios. Mejorar el error relativo empeorando el
MAE no se aceptó como promoción. La última campaña completó sus 48 ajustes en
79,546 s; la comparación completa y regenerable está en
[soporte y reinicios](../research_log_20261031_joint_support/README.md).

Se probaron aumentos geométricos de últimas cajas, ruido de posición y pérdida
de observaciones; perturbaciones de expertos individuales y correlacionados;
historial nativo reducido a 1/2/3 observaciones; y entradas causales de calidad
con un control de capacidad semejante. No se ha demostrado que augmentation
mejore este modelo simplemente por añadir más perturbaciones.

## Cachés y conocimiento del arranque

Se construyó una caché de 149 consultas elegidas de antemano: tres cuantiles
temporales de cada una de las 24 grabaciones de carretera y las 77 de Slider.
Se verificó la reproducción nativa y se generaron 447 reinicios de historial.
Hay una consulta Slider sin salida nativa; su ausencia permanece explícita.
La extracción tardó 463,508 s. La caché guarda raw17, expert3, clocks, contextos
y referencias, no los píxeles completos ni todos los tokens PAIR133.

La caché de soporte añade diez variables de tiempo/recuento a las 2.608 filas,
las variantes de reinicio y las poblaciones eAP/Test12. No incluye una duración
real de historial que no se haya medido: son campos concretos de los estados
340D. Los controles distinguieron información adicional de capacidad extra.
Informes: [contexto](../research_log_20261031_native_context/README.md) y
[soporte](../research_log_20261031_native_support/README.md).

## Expansión de eventos y optimización real

Se implementó y probó un experto causal de expansión sobre eventos, sin
presentarlo como reproducción de CMax. Pasó fixtures sintéticos de expansión;
las pruebas con fondo estático mostraron fallos y abstenciones que se conservaron.
En las 77 consultas de Slider-500 no justificó sustituir el sistema vigente.

Se midió la función objetivo y se implementó una ruta CUDA Graphs en float64.
Float32 se descartó por discrepancias de redondeo en bordes. La ruta híbrida
repite en CPU los casos que GPU declara inválidos: 15 fallbacks, 62 resultados
válidos y 34,272 s para las 77 consultas, con paridad verificada según el contrato.
No se atribuye esa velocidad a toda la cadena PHASE ni se convierte una comparación
no contemporánea en un benchmark definitivo. Véase
[el informe de expansión](../research_log_20261031_event_expansion/README.md).

Los entrenamientos pequeños ya duran segundos o pocos minutos. La preparación,
las extracciones, la reconstrucción de historia y las verificaciones de paridad
explican buena parte del tiempo transcurrido. No se implantó indiscriminadamente
prefill, compresión o preload en todo el sistema. La caché, los lotes y CUDA Graphs
se aplicaron donde había una operación identificada y una prueba de equivalencia.

## Hallazgo nuevo: sensibilidad temporal de PAIR

Se capturaron dos consultas fijas: `evttc:EXTENDED:CCRm-low-0-overlap-0:17000441`
y `Slider-500:38`. El reloj de observaciones y ventanas se transformó alrededor
de la consulta con velocidades 0,5 y 2. Se mantuvo la fase canónica de 100 ms;
en eventos solo cambió el canal de tasa según la duración real. Es una
transformación de representaciones preparadas, no re-encoding de un stream real.

El caso a=1 reprodujo exactamente 120/220 arrays de productores y 30 arrays de
contexto/cabeza por consulta. Los controles reducidos vuelven a ejecutar los
mismos lotes; no recortan salidas de lotes anteriores. El piloto completo tardó
14,312 s. El primer intento se conserva fallido por comparar tuplas del lector
con listas JSON de las cajas; el fix normaliza su representación sin modificar
valores, y una regresión sigue rechazando una caja alterada.

RGB A5/C2F mantiene el log-ratio y responde al delta físico. PAIR mantiene un
TTC parecido pese al cambio de cadencia. Este es el primer desvío grande en las
dos trazas, no una explicación causal demostrada de todo Slider. Informe:
[diagnóstico temporal](../research_log_20261031_producer_timewarp/README.md).

## Intervención canónica ejecutada

Se creó `CanonicalPair`: el PAIR congelado recibe dt/log(dt)/1/dt a 100 ms;
su fase cruda se convierte a inverso TTC, se escala por 0,1/dt_real y vuelve a
fase antes de la emisión original. Una rama conserva bytes cuando dt es exactamente
FP32(0,1). No se usa `phase_to_ttc` antes de escalar porque introduciría clipping
y el suelo de inverso prematuramente. No se han cambiado pesos.

El piloto, reutilizando los tensores, terminó en 9,793 s. Los cuatro contrastes
de la cabeza final se acercan al cociente temporal ideal de 1. Sin embargo,
el contexto de eventos de carretera a velocidad 0,5 empeora de 0,673790 a
1,891432. La predicción limpia de Slider cambia unos -0,0291 s; sin targets
esto es una diferencia, no una mejora. Todas las tablas, incluida la regresión,
están en [PAIR canónico](../research_log_20261031_canonical_pair/README.md).

## eAP: tres evidencias distintas

1. La auditoría activa anterior de TRAIN40 evalúa el adaptador actual con historia
   y features upstream OOF. Es resustitución/conservación, no generalización nueva.
2. Test12 contiene 6.762 consultas lógicas/5.805 físicas y estados de despliegue,
   pero no se encontraron targets. Permite deriva, no precisión.
3. La auditoría canónica nueva usa 88.744 entradas PAIR133 **actuales** por modalidad
   procedentes de productores de despliegue TRAIN40. No son las cachés OOF del
   punto 1 y no cubren toda la historia de cada consulta.

En la tercera auditoría, eventos es exactamente identidad en todas las filas.
RGB tiene 24 deltas exactos de 100 ms; el resto está entre 99,945 y 100,058 ms.
No aparecen nuevos rechazos ni cambios de signo. El cambio medio absoluto de
TTC RGB es aproximadamente 0,00009167 s; el máximo, 0,006958 s. El CLI definitivo
tardó 1,667 s después de cargar modelos y reproduce los arrays de la auditoría
inicial. Se corrigió un contador que comparaba TTC == 60: en FP32 la salida es
59,999996; los dos recibos se preservan. Informe regenerable:
[auditoría eAP actual](../research_log_20261031_canonical_pair/EAP_CURRENT_QUERY_AUDIT.md).

## JEPA y límites de lo conseguido

La publicación de las últimas horas también cerró el piloto de profesor fijo:
nueve entrenamientos, tres semillas y controles estático/futuro permutado.
No pasó el criterio principal de utilidad representacional para expansión.
Su cierre no invalida toda investigación JEPA; tampoco autoriza reabrirlo sin
hipótesis nueva. [Cierre completo](../jepa_closure_20261031/README.md).

No se entrenó un encoder nuevo durante los pilotos temporales. No se promovió
un modelo nuevo, no se generó otra entrega oficial, no se probó conservación
completa de eAP con PAIR canónico y no se alcanzó SOTA. Se consiguió una
intervención concreta respaldada por un diagnóstico reproducible; su utilidad
predictiva poblacional es el próximo trabajo, todavía sin ejecutar.
