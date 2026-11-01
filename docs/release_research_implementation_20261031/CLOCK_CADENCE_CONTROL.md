# Control del reloj tras corregir PAIR

El contraste D1–D3 inicial usa PAIR original. Antes de atribuir su resultado a
la dinámica, repetir los mismos parámetros con las fases canónicas ya selladas
en screen149. No leer sensores ni ejecutar productores. No cambiar el candidato
de la ampliación 2608 ni seleccionar una combinación con sus targets.

Matriz fija: PAIR original/canónico por persistencia/EWMA/transporte/observador.
Usar el mismo soporte causal completo de 650 ms y la misma disponibilidad.
Conservar el rechazo de un lote completo que indique el replay canónico.
Persistencia es la última q sin transporte; EWMA, transporte y observador usan
ClockConfig existente sin ajuste de varianzas. Registrar observaciones, motivos,
innovaciones, fases crudas y fallback. La referencia de cada fila es su propio
contexto de eventos, no la fusión RGB. Si el reloj no puede emitir, conservar
exactamente ese contexto; si el contexto falta, mantener su ausencia.

Sellar todas las lecturas antes de unir los targets expuestos de desarrollo.
Publicar el error del contexto, los cuatro errores, cobertura y CI pareado por
adquisición para ambas cadencias. Un filtro sobre PAIR no equivale a insertar
un contexto nuevo en la cabeza final. Un control negativo no demuestra que D4
sea imposible; tampoco autoriza entrenarlo sin una carencia corregible concreta.
