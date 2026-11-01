# Pendientes, hipótesis y siguiente acción

## Objetivo que permanece intacto

Continuar investigando, entrenando y validando PHASE-TTC hasta obtener evidencia
reproducible de SOTA en EvTTC bajo un protocolo comparable, preservando eAP y sin
usar consultas de evaluación como datos de ajuste ni presentar métricas locales
como oficiales. El usuario ha pedido una pausa, no reducir ese objetivo a hacer
pasar un piloto ni declarar final una versión que todavía no lo acredita.

## Primera acción al reanudar: screen fijo de 149 consultas

No empezar por otra búsqueda de cabezas ni por repetir los dos pilotos. El
siguiente experimento recomendado es comprobar **precisión y cobertura** de
PAIR canónico sobre las 149 consultas ya elegidas, como filtro de rechazo previo
a la población completa. Todavía no existe un runner completo de esta nueva
evaluación. No hay que buscar un supuesto job en marcha ni reiniciarlo.

1. Leer y verificar los recibos de `expanded149`, `pilot_v2`, `canonical_v1` y
   la auditoría eAP. Identificar la ruta exacta que produce el modelo entregable:
   el piloto actual evalúa F0 de nueve componentes, no el router adaptado final.
2. Fijar antes de abrir targets una única intervención: `CanonicalPair` tal como
   está probado, d0=0,1, sin hiperparámetros escogidos con el benchmark.
3. Instrumentar una extracción que añada PAIR133 y el mapeo estable de
   observaciones/lotes a la caché fija de 149 consultas. Reproducir primero el
   sistema original. Evitar conservar píxeles de toda la población si solo hacen
   falta los tokens y los raw17 ya disponibles.
4. Guardar una vez esos inputs y reutilizarlos para reconstruir PAIR, raw17,
   contextos y cabeza. Conservar relojes físicos, fase canónica, normalizadores,
   orden y tamaño de lote. No reemplazar únicamente un TTC final guardado.
5. Sellar predicciones, rechazos, ausencias y fuentes antes de unir ground truth.
   Medir por separado F0 original y el sistema entregable con su router vigente
   consumiendo estados corregidos. No confundir sus métricas.
6. Evaluar la población seleccionada completa y por grabación. Conservar la
   consulta nativa ausente Slider-500:76 y los posibles rechazos nuevos; no
   presentar una intersección de filas válidas como mejora con cobertura total.

Usar el screen para rechazar daños claros y decidir si merece la evaluación
completa, no para certificar superioridad. Solo hay 72 consultas de carretera,
frente a las 2.531 de desarrollo, y las 149 ya tienen exposición histórica. Antes
de ejecutarlo hay que dejar por escrito el control y los criterios de cobertura,
MAE y error relativo macro. No cambiar la corrección después de ver este screen
sin registrar una nueva hipótesis/experimento y reconocer la exposición.

## Después del screen, si lo justifica

- Ejecutar la población completa de 2.608 consultas con cobertura y métricas
  por grupo. Mantener las exclusiones de las ocho grabaciones del benchmark y
  Slider-750/1000. Cualquier adaptación aprendida requiere el protocolo agrupado
  y semillas 7/13/21; no elegir una semilla por resultados de test.
- Recalcular las historias eAP afectadas. Los PAIR actuales de eventos son
  identidad exacta, pero eso no demuestra por sí solo identidad en cada ruta
  histórica. RGB tiene deltas casi canónicos, pero sus pequeñas diferencias
  pueden amplificarse en contexto/cabeza. Probarlo con fuentes y genealogía
  consistentes; no mezclar inputs OOF y de despliegue como si fueran iguales.
- Si se usa distilación, replay de eAP o regularización para preservar el dominio,
  fijar datos y presupuesto en train. Medir contra la referencia adaptada activa,
  no solo contra un fallback que devuelve F0 al no recibir historia.
- La ausencia de targets Test12 limita lo que puede afirmarse: deriva pequeña no
  prueba ausencia de pérdida de precisión. Hace falta una evaluación etiquetada
  permitida o una confirmación externa para una afirmación fuerte.

## Qué vías están abiertas y cuáles no merecen repetirse sin evidencia

**Prioridad actual: PAIR y condicionamiento temporal.** El piloto encontró una
sensibilidad concreta y la intervención la corrige en las observaciones
diagnosticadas. Falta saber si ayuda a TTC real. Una mejora de consistencia puede
seguir empeorando accuracy. La regresión del contexto de eventos ralentizado es
una pista adicional; no autoriza corregirlo con una fórmula escogida para ese caso.

**Contextos y arranque.** Las entradas físicas y la historia importan, pero los
reinicios y variables de soporte ensayados no dieron una promoción general.
Si PAIR mejora y el error aparece después, estudiar ese bloque con controles
causales y retardo, sin repetir ciegamente la misma matriz de fusión.

**Geometría/representación de eventos.** La expansión sobre eventos conserva
valor diagnóstico, pero el experto actual falló ante fondo estático y no ganó
en Slider-500. Reabrir con un mecanismo concreto de aislamiento del objeto o
correspondencia, conservando latencia y cobertura. El paralaje RGB–eventos y
la calibración provisional de Slider siguen siendo hipótesis, no bugs demostrados
por las cifras actuales. No introducir profundidad real del target en inferencia.

**Fine-tuning/augmentation de productores.** Sigue siendo posible, pero debe
responder a un déficit comprobado. Un warp RGB que escala dt y TTC por igual
no cambia el target de log-ratio: por sí solo no crea nueva supervisión geométrica
del encoder. No anunciar una campaña de encoders cuando solo se entrenan cabezas.

**JEPA dinámico.** La receta de profesor fijo quedó cerrada por su criterio
fallido. Una pérdida latente menor no bastó para la expansión. Reabrir únicamente
con evidencia nueva, controles estático/permutado y presupuesto previo, según
el documento de cierre. No descargar otro foundation model como sustituto de
diagnosticar el pipeline existente.

## Qué falta para una entrega y una afirmación SOTA

Falta un candidato que supere controles de precisión/cobertura y conservación,
una configuración congelada antes de confirmar, trazabilidad de código/pesos,
evaluación equivalente con baselines aprendidos y geométricos, y resolución del
scorer de Slider-750. Los resultados publicados de CMax/STRTTC no se pueden
copiar como una reproducción local. El vector de 434 targets públicos no ha
demostrado equivalencia completa con el vector privado de los organizadores.

El borrador para los organizadores está en
`docs/research_log_20261031_sign_safe/organizer_evaluator_question.txt`. No se ha
enviado; no enviar mensajes externos sin instrucción explícita. El formato de
entrega sigue siendo `Index timestamp(s) ttc(s) cost_time(s)` en todas las
consultas. Autor: Álvaro Schwiedop Souto, sin afiliación y sin otros autores.

## Estado de ejecución al parar

Los pilotos lanzados y la auditoría CPU están completos. Los agentes de
implementación/revisión terminaron; no hay una extracción canónica de 149
consultas pendiente en segundo plano. Pueden existir monitores o backups
históricos de otros trabajos en esta máquina: no matarlos por compartir el
nombre python.exe. Verificar comando y pertenencia antes de cualquier acción.

Al reanudar, comunicar el siguiente entregable concreto y su ETA basada en
mediciones. No prometer una fecha de SOTA. La estimación de 8–10 min corresponde
solo a una futura extracción de 149 consultas, sin incluir implementación,
verificación, evaluación completa o entrenamiento.
