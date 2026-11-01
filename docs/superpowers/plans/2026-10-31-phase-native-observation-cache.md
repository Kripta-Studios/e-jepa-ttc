# Cache de observaciones nativas y reinicio causal de contexto

La continuación autorizada busca resolver fallos de soporte nativo que no se
reproducen perturbando solamente la geometría o las predicciones finales.
Tres campañas (168fits completos) no promovieron un candidato nuevo.

## Evidencia y contrato

Las2531cachés de carretera solo conservan el estado final340ywidth_logits2.
El hidden160 de cada contexto no permite reconstruir observaciones descartadas.
En memoria sí existen raw17+expert3 por observación: final_train40_transfer.py,
_BoundTrain40Native/event_values/RGB equivalent; y competencia runtime._prepare_values.
La inferencia contextual final usa estas observaciones y sus relojes.

Primero capturar sin modificar pesos ni preparación una muestra piloto de
consultas elegibles: carretera por grupos y Slider500. No abrir los ocho grupos
excluidos ni750/1000. Preservar por query×modalidad×observación raw17,expert3,
anchor/available/first_sensor/actualtimestamps,accepted/rejected y, si disponible,
knownmask/support. La ROI pertenece a la consulta: no reutilizar solo poranchor.
El orden ynormalizadores originales quedan congelados y se guardan hashes.

Reproducir contexto limpio y cabeza antes de permitir truncamientos. Comparar
rawphase/state/timing ypredicción con la misma extracción instrumentada; esta
comparación debe ser exacta o dentro del contrato determinista del runtime.
La comparación con cachés históricas se registra separada, con su precisión
original, y no se presenta como bitwise si difiere. No ajustar pesos para pasar.

## Primer piloto y ampliación

- Primera entrega: capturar yreproducir al menos una consulta elegible de
  carretera yuna deSlider500, con tests analíticos previos yrecibos deparidad.
- Después:72consultas de carretera (tres cuantiles temporales por cada uno de
  los24grupos) y77Slider500. Selección solamente poridentidad/tiempo, nuncaerror.
  Todas pertenecen a desarrollo; mantenergrupo en cualquieraugmentation/fold.
- Replay de últimas1,2,3observaciones aceptadas: recalcular intervalodesdeanterior;
  la primera observación retenida tiene interarrival0. No copiar clocks ya
  codificados que contengan información de observaciones eliminadas.
- Esto es reinicio de CONTEXTO. Los productores retenidos aún pueden utilizar
  sus propias ventanas sensoriales anteriores. Para llamarlo arranque de SENSOR,
  aplicar cutoff a first_sensor_ts_us o recomputar prefijos con el adaptador
  auditado. Registrar ambas semánticas yno equipararlas.
- Featurecache final depende dequery,ROI,calibración,modelo,preparaciónyobservaciones;
  guardarrecibos porconsulta para reanudarsin repetir productores verificados.
- Ningún fit adicional autorizado por este documento hasta paridadlimpia. Después
  fijar una sola comparación deaugmentaciónnativa/control antes de entrenar.

CPU/GPU/decodificación ycontexto se medirán separadamente. Referencia histórica:
65minutos para toda carretera, no ETAactual. El piloto estima el coste nuevo.
No repetir extraccióncompleta2608si149consultas bastan para decidir lahipótesis.
