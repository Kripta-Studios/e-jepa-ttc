# Objetivo de comparación — 2026-10-29

La campaña compara detectores sobre el release público fijado de eAP. Su resultado
primario es de desarrollo local. La identificación de un líder oficial y la equivalencia
con el scorer del organizador siguen pendientes; ninguna AP de esta campaña se etiqueta
como puntuación oficial o SOTA.

| Contrato | Valor fijado |
|---|---|
| Release | `NAIL-HNU/eAP-dataset`, revisión `a814a946e60d37f7242358af5f344542177e50c5` |
| Roles | 24 secuencias train, 8 selección y 8 calibración; sin nuevo holdout independiente |
| Taxonomía | car, bicycle, truck, pedestrian, motorcycle, bus, tricycle, other_vehicle |
| Métrica primaria | AP101 local por distancia de centros ego XY, umbrales estrictos 0,5/1/2/4 m |
| Agregación | Media de clases con soporte; clases ausentes explícitas, nunca inventadas como observaciones |
| Dominio | Protocolo v2 independiente del entrenador, con reglas simétricas para GT/predicciones e ignorados |
| Propuestas | 50 por frame antes del filtro de dominio |
| Selección durante R0/R1/R2 | Monitor fijo de 1.024 frames cada 1.024 updates |
| Comparación completa | 23.958 frames de las ocho secuencias de selección en updates 4.096 y 8.192 |
| Presupuesto inicial | D4096, batch 4, 8.192 updates y 32.768 exposiciones por candidato |
| Confirmación | Semillas 7, 13 y 23, grupos completos y bootstrap por secuencia |

El protocolo completo tiene SHA-256
`10da30cf8f08247d48d975f3650cb02f1b43375fe0e841a5389dfb3b5d7b4a5e` y está
conservado en [la evidencia v2](evidence/current_v1/s0/evaluation_protocol_v2.json).
Los controles constante train-only y permutación pertenecen al mismo protocolo.
La referencia histórica permanece reproducible con su GT original y se reporta aparte.

R0 usa RGB actual, R50 inicializado con ImageNet1K V1, backend portable, FP32 y ocho
salidas de clase. R1 modifica únicamente warmup/scheduler; R2 añade LR diferencial de
backbone. Las futuras intervenciones de ranking, temporalidad y eventos conservarán
población y controles declarados. Una ganancia de una semilla es screening, no confirmación.

Para temporalidad, el frame actual está disponible al finalizar su exposición. La historia
RGB debe haber terminado antes del comienzo de la exposición actual. Los eventos usan
`[fin_exposición − ventana, fin_exposición)`. Solo se acredita causalidad sensorial:
el release no aporta timestamps de llegada que permitan medir latencia física completa.

El repositorio oficial enlaza detección eAP con la competición 17291 y TTC con la 17289.
También documenta que el test público TTC carece de etiquetas. Esto identifica las rutas
oficiales, pero no valida nuestro scorer de detección ni acredita una posición en sus
leaderboards. [Release oficial Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC),
[benchmark eAP](https://www.codabench.org/competitions/17291/).

La [auditoría externa](EXTERNAL_PROTOCOL.md) conserva la comprobación del release,
los seis IDs ausentes respecto a las 46 secuencias descritas y los límites de acceso al
scorer. No se usarán etiquetas test, se enviarán resultados ni se contactará a organizadores
como parte de esta campaña local. La publicación previa del paper eAP sobre Car tampoco
es directamente comparable con la macro local de ocho clases.

El servicio PHASE tiene su gate propio: funcionamiento, tracking y calidad TTC se evalúan
por separado. TRAIN40 expuso sus productores a las 40 secuencias; reservar ocho para el
detector no crea independencia para una conclusión conjunta de generalización TTC.
