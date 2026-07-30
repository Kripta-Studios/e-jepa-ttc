# Metodología

La metodología completa está en [PLAN.md](../PLAN.md). Este documento resume
las decisiones implementadas.

## Representación

Eventos causales se convierten en voxel grids de cinco bins y polaridad
separada. BASE añade once canales auxiliares causales.

## Backbone

El EventTubeletTransformer histórico expone ahora tokens densos e intermedios.
La comparación matched no sustituye sus pesos: cambia el lugar del pooling y el
mezclador posterior.

## Temporalidad

- block-causal preserva interacción completa dentro de cada frame;
- Object-KDA actualiza memoria solo entre tiempos;
- ningún target o acción futura entra al encoder de contexto.

## Geometría

Height, area, affine y event contrast estiman inverse-TTC por objeto. La
confianza depende de validez del track, soporte de eventos y condición del
solver.

Se distinguen dos familias:

- escala bbox causal: regresión local sobre hasta 21 detecciones pasadas,
  calibrada solo en train;
- STRTTC event-only: NLTS, gradientes de contorno, normal flow por planos
  locales, RANSAC y solver de tres parámetros.

Los fallos geométricos se cuentan como falta de cobertura. No se eliminan para
crear una métrica aparentemente mejor.

## Navegación

La señal GNSS se transforma mediante extrínsecas calibradas. No se usa
`velocity_x` como closing speed directo.

## Selección

Toda promoción se decide por métricas macro de secuencia y gates predeclarados.
Módulos complejos permanecen apagados hasta que el oracle anterior los
justifique.

La confirmación larga cambió la decisión del screen: A1 Dense se promueve
porque su mejor época fue la 20, mientras A2 y K1 se detuvieron en 10 y 7 sin
superarlo. Más épocas son útiles cuando la curva sigue mejorando, no como
presupuesto uniforme ciego.

## Transferencia eAP event-only

El eAP público aporta eventos, calibración y tracks 3D, pero no TTC oficial. Se
usa un piloto fijo de 12 secuencias: nueve para optimización y tres para elegir
el checkpoint. El acceso HDF5 es causal y bajo demanda mediante `ms_to_idx`.
No se decodifican los TAR RGB ni se materializa un voxel cache global.

`eAP-SSL` predice embeddings densos futuros a 100/250/500 ms. `eAP-Geo` añade
seis auxiliares débiles normalizados —centro y tamaño de bbox, cierre radial y
expansión de altura— más una máscara objectness por patch. Las cajas 3D se
proyectan con la calibración de la cámara de eventos. La cabeza geométrica es
desechable: solo el encoder EventTubelet compatible de 21 canales se transfiere
a A0/A1.

La utilidad no se mide por la loss eAP, sino por una comparación pareada contra
inicialización aleatoria en EvTTC. Control y transferencia comparten fold,
seed, ventanas, cabeza TTC, optimizador, máximo de épocas y early stopping. Se
reportan RTE y MAE macro, victorias por pareja y bootstrap OOF por secuencia.

El piloto de tres épocas se amplió a folds 0/1 con seed 7 después de que fold 0
fuera favorable. eAP-SSL fue inconsistente. eAP-Geo mejoró A0 en RTE y MAE en
2/2 folds (+3,66 % y +4,30 % agregados) y A1 en RTE en 2/2 (+6,57 %), aunque el
MAE de A1 quedó 1/2. Esto abre el gate de datos, no el de SOTA: el siguiente
entrenamiento usa las 40 secuencias con split 32/8 y la decisión final exige
cinco folds × tres seeds.
