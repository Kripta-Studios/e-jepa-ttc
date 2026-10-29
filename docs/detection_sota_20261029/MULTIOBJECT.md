# Servicio causal multiobjeto de detección a PHASE

Estado: implementación de ingeniería S5, sin afirmación nueva de precisión.

`CausalMultiObjectTracker` asocia detecciones por asignación global de coste IoU,
con gates de clase e IoU. Una asignación se rechaza cuando el margen es ambiguo
en cualquiera de las dos direcciones: detección hacia tracks o track hacia
detecciones. Los timestamps deben crecer estrictamente. El lifecycle expuesto es
`tentative`, `confirmed`, `lost` y `closed`.

Un track perdido no puede reactivarse. Una detección posterior crea otro
`track_id:generation`, de modo que los buffers PHASE no heredan observaciones de
una identidad dudosa. Una ROI extrapolada se conserva como detección, pero no se
envía a PHASE.

`DetectionPhaseService` usa la firma real de
`MultiObjectTrain40Service.update(raw_packet, start_us, end_us, objects,
rgb_history)`. Cada paquete se ingiere exactamente una vez. Cuando no hay tracks
confirmados, el servicio TRAIN40 no admite una lista vacía, por lo que el adapter
ingiere el paquete mediante el contrato de `EventHistory`; así mantiene el ring
causal para objetos confirmados después. Si hay más tracks elegibles que la
capacidad lógica configurada, selecciona por score y conserva los restantes con
`ttc_reason=runtime_object_capacity`.

La extensión aditiva `ChunkedTrain40Runtime`, versión
`train40_ingest_once_chunked_v1`, permite que esa capacidad lógica sea superior
a veinte sin cambiar el runtime ni los productores sellados. Ingiere el paquete
una sola vez en el `EventHistory` compartido y, después, ejecuta en orden
`query_many`, preparación RGB, productores congelados y contexto para cada chunk
acotado. Cada tensor de eventos prestado se consume antes de la siguiente
consulta. Se comprueba que cada salida conserva el mismo `object_id` y orden;
un error posterior a la ingestión deja el wrapper cerrado a reintentos del mismo
paquete.

Esta primera versión rechaza expresamente feature cache RGB, leases uint8,
ramas paralelas y overlap. Esos modos tienen lifetimes ligados al `update`
monolítico histórico y no se emulan. Los frames con cero objetos también pasan
por el wrapper para mantener su reloj de ingestión. La preparación CPU real
`temporal8_prepare_logical30_v2.json` admite capacidad sellada ocho y capacidad
lógica treinta. La ejecución real GPU con más de veinte objetos sigue pendiente
de una población label-free que produzca al menos veintiún tracks maduros y de
admisión de recursos; no se simula con duplicados.

La salida conserva clase, score, track, generación, procedencia de ROI, centro
3D, dimensiones, yaw, velocidad con frame/semántica y la salida nativa del
runtime. `ttc_signed_s` no se recorta ni toma en valor absoluto. Solo
`PHASE_ONLY_TRAIN40_S7` se admite como TTC final; una salida que contiene solo
ramas RGB/eventos produce abstención.
La ruta opt-in `BatchedPhaseContextRuntime` transporta `point_phase`, `q10` y
`q90` del mismo head congelado y width logits R_CTX. Permanecen en espacio de
fase y no se convierten en un intervalo TTC continuo. Sin historia RGB se marca
su indisponibilidad y no se fabrican límites de la rama de eventos.

Por defecto se exigen 600 ms de antigüedad del track antes de consultar PHASE.
La generación forma parte de `object_id`, que también forma parte de las claves de
caché RGB exactas; una reaparición no puede reutilizar una entrada de la generación
anterior. La implementación no usa IDs anotados, targets TTC ni datos futuros. Las pruebas
CPU cubren ambigüedad inversa, lifecycle, reaparición, timestamps tardíos,
capacidad, ingestión única, cero/uno/varios/más del chunk/más de veinte objetos,
fallo cerrado, signo TTC e intervalos de fase nativos. La verificación pendiente
incluye replay real con más de veinte tracks maduros, latencia integral por
número de objetos y una evaluación TTC con población e
independencia admisibles. TRAIN40 ya estuvo expuesto a las cuarenta secuencias;
esta integración no acredita generalización TTC.
