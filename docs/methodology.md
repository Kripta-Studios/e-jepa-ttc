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

## Navegación

La señal GNSS se transforma mediante extrínsecas calibradas. No se usa
`velocity_x` como closing speed directo.

## Selección

Toda promoción se decide por métricas macro de secuencia y gates predeclarados.
Módulos complejos permanecen apagados hasta que el oracle anterior los
justifique.
