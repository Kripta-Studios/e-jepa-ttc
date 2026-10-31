# Piloto de cadencia variable a nivel de productores

Estado: protocolo previo. Continuación tras 312 ajustes de fusión sin promoción.
No se entrenan pesos, no se usan targets y no se modifica la entrega congelada.

## Pregunta

¿En qué bloque se pierde la respuesta física a una transformación temporal
coherente: A5/C2F, PAIR, contextos o cabeza? La ampliación de cadencia de frames
RGB no cambia el target de log-ratio si dt y TTC se dividen por el mismo factor;
no se presentará este piloto como nueva supervisión geométrica del encoder.

## Población y captura

Solo las dos consultas previamente elegidas por identidad y tiempo:
`evttc:EXTENDED:CCRm-low-0-overlap-0:17000441` y `Slider-500:38`.
Verificar los recibos del piloto/expanded149 y el cierre de fuentes. Capturar
los tensores exactos usados en `_expert_block`, dt y correspondencia explícita
entre llamada, índice de batch y observación. Reutilizar preparadores, calibración,
ROIs, pesos y normalizadores. Conservar rechazos y observaciones admitidas.
Ningún archivo de targets entra en inferencia. Guardar tensores y fuentes una vez.

Primero reproducir a=1 con las mismas agrupaciones de batch: igualdad exacta de
inputs, raw17/expert3 y tensores contextuales/cabeza de la misma extracción.
Comparaciones con archivos históricos se registran separadas. Si hay diferencias
de batch entre rutas, no afirmar paridad exacta por una tolerancia más laxa.

## Transformación fija

Factores a={0.5,1,2}. Consulta q invariable; t'=q+round((t-q)/a), redondeo
nearest-even documentado. Endpoints, ventanas, anchors y disponibilidad usan la
misma función. Deltas se vuelven a calcular desde endpoints enteros transformados.
No cambiar la constante de fase de 100 ms. El target físico sería TTC/a, pero no se
lee ningún target real durante el piloto.

RGB conserva píxeles. Eventos conserva canales 0..10 y transforma el canal 11 de
log1p(rate) a log1p(rate*old_duration/new_duration) por ventana. Son tensores de
representación transformados, no una afirmación de re-encoding del stream real.
Comprobar previamente con fixtures de eventos crudos usando el encoder original:
casos temporalmente alineados sin error de cuantización y un caso con redondeo.
Conservar errores medidos por canal; no cambiar tolerancias según resultados.

La selección inicial de observaciones queda fija. Filtrar las transformadas por
causalidad y presupuesto query-first_sensor<=650000us, máximo 8 por modalidad.
No rellenar historia nueva. Esto NO reproduce la política de selección de
ventanas físicas de 100 ms sobre un stream acelerado. Registrar todos los descartes.
Si cambia el conjunto de identidades retenidas, ejecutar un control limpio con
esas mismas identidades y relojes originales, además del control limpio completo.

La ruta de identidad conserva directamente los píxeles y deltas FP32 capturados.
La comparación vincula cada fila del lote a una identidad de observación estable.
El control de un subconjunto vuelve a ejecutar el mismo lote reducido; no recorta
salidas del lote completo. Se separan observaciones solicitadas, retenidas por
reloj y admitidas por productores. Si los conjuntos admitidos difieren, la
comparación de contexto/cabeza se marca como no aislada. Si estas dos consultas
contienen originalmente observaciones rechazadas o no seleccionadas, se conserva
la captura y se detiene el piloto para examinar la diferencia de ejecución.

## Ejecución y trazas

Reejecutar A5/C2F, PAIR, raw17, contextos y cabeza. Capturar por observación
pair_log_height_ratio, inverse_ttc_mean, TTC, known_mask, soporte y salidas PAIR;
por consulta, fases y TTC de contextos y cabeza. Preservar rechazos por dominio,
ausencias y saturación. No dividir simplemente una predicción guardada.

Comparar q_warp/(a*q_control) y a*T_warp/T_control cuando sean finitos y no cero;
no confundir equivariancia con precisión. La expectativa analítica de invariancia
del log-ratio RGB se comprueba sobre inputs iguales; PAIR y cabeza son diagnósticos,
no puertas que se deban forzar a pasar. No se ajustan parámetros ni umbrales.
Registrar también el cociente real entre deltas de cada par, para distinguir
cuantización de timestamps de una respuesta del modelo que no escala con el tiempo.

La captura y replay limpio deben pasar antes de interpretar el piloto. Guardar
configuración, código ejecutado, hashes, grupos de batch, identidad de modelos,
todos los resultados y tiempos separados. Tras el piloto se elegirá la siguiente
intervención según el bloque señalado; sin entrenamiento amplio anticipado.
