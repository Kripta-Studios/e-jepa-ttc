# Latencia local desde RAW: protocolo suplementario

Registro anterior a medir el ensayo. El benchmark `native-local-latency-v1`
incluye lectura de NPZ, decodificación RGB y predicción; **excluye** construir
la representación de eventos desde HDF5. Este suplemento mide esa frontera.

Se conservan las primeras diez consultas físicas de cada secuencia, ordenadas
por timestamp y token, del mismo manifest público. Se incluyen todos sus
historiales admitidos. No se leen targets privados ni se seleccionan consultas
por coste o predicción. Las cajas suministradas siguen siendo entradas: no se
mide detección de objetos, adquisición del sensor ni transporte físico.

Primero se reconstruye cada observación de eventos necesaria desde los HDF5
locales. Se reutilizan el lector y las funciones exactas del materializador:
ventanas semiabiertas, desplazamiento horizontal de cinco píxeles, ROI común,
proyección float64, voxelización y escalares de 12 canales. Se exige igualdad
elemento a elemento con el último slot del NPZ congelado. Se leen únicamente
las tres ventanas de la observación actual; cada miembro del historial se
construye cuando el predictor lo solicita. No se inventan ventanas previas.

La medición GPU espera a que terminen captura, replay y benchmark NPZ. Requiere
COMPLETE de esos ensayos y de la paridad RAW. Compara original y PAIR canónico,
con pesos, normalizadores y contexto idénticos. Dos consultas de calentamiento
por brazo, luego reinicio de lectores y cachés por secuencia. Alternancia de
orden por consulta/secuencia. Se conserva el caché de observaciones calculadas
dentro de cada secuencia, separado por brazo. Un lector HDF5 por brazo, acotado
y persistente; no se conservan tensores RAW ya preparados entre consultas.

Tiempo principal: sincronización CUDA, reloj de pared alrededor de `predict`,
sin instrumentar etapas, sincronización CUDA final. Incluye abrir/leer HDF5,
recortar, voxelizar, transferir, productores, RGB, contexto y fusión. Los hashes
de medios se verifican fuera del cronómetro, al comienzo y al final. No se
ejecuta hashing del HDF5 completo por consulta. El caché del SO puede estar
caliente: esto es replay local, no una medida de almacenamiento en frío.

Toda salida debe ser exactamente igual a la correspondiente predicción del
replay local completo. Se guardan todas las duraciones, p50/p95/p99, memoria,
cociente p95 canónico/original y cola de un servidor con timestamps reales.
No se suman percentiles de ensayos distintos. El límite descriptivo de +10%
no concede promoción; el resultado no resuelve la paridad histórica fallida
ni aporta evaluación de precisión nativa o adquisiciones independientes.
