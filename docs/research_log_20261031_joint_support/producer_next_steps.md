# Siguiente frontera: productores, antes de otra búsqueda de fusión

Estado al cerrar la comparación conjunta: auditoría de viabilidad, no piloto
de velocidad ejecutado ni entrenamiento de encoders. Los 312 ajustes completos
de esta continuación son ajustes de cabezas/fusión, no 312 entrenamientos
completos de encoders. La versión entregable anterior permanece congelada.

## Inventario comprobado sin abrir targets

- 88.744 observaciones TRAIN40 de eventos están representadas por 2.774 shards
  en `E:/EJEPA_TRAIN40_CACHE_20261005/producer_inputs/`. La cabecera contiene
  `events[N,3,12,128,128]` y `delta[N]`.
- Hay 2.775 shards RGB enumerados en manifests locales y presentes: P53.890 en
  `E:/E_JEPA_TTC_RGB_PORT/prepared/P/`, H18.422 y V16.432 en
  `E:/E_JEPA_TTC_FINAL_REFIT_20261017/rgb_cache/{H,V}/`.
  Suman88.744; los manifests declaran decode-parity completada. Las cabeceras
  inspeccionadas contienen `rgb_uint8[N,T,3,128,128]`, conT2/T3, clocks,
  `delta_t_s`, validez y ROI. No se ejecutó otra validación de decodificación.
- Para EvTTC existen HDF5/bag y preparación original. Las149cachés nuevas guardan
  raw17/expert3 y relojes, pero NO píxeles ni voxels de productores. No se ha
  establecido una caché sensorial completa reutilizable de las2608consultas.

Las identidades de eventos TRAIN40 figuran en
`../e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005/TRAIN40_INDEX.npz`.
Disponibilidad local y cabeceras no equivalen a una nueva validación de todos los
valores de los tensors, ni garantizan que cualquier trayectoria de entrenamiento
pueda consumirlos sin verificar normalizadores, grupos y checkpoints.

## Qué significa una ampliación temporal coherente

Para factor de velocidad a y consulta q: `t'=q+(t-q)/a`, `dt'=dt/a`, `TTC'=TTC/a`.
También cambian ventanas, anchors, disponibilidad y relojes contextuales. La
coordenada canónica de fase de100ms permanece igual; no es el delta entre frames.

Los canales de eventos definidos en `src/e_jepa_ttc/data/eap_representation.py`
incluyen diez voxels temporales, `log1p(count)` y `log1p(count/duration_seconds)`.
Cambiar solo dt deja incoherente el último canal. La invariancia de bins bajo
transformación afín es continua; una implementación con microsegundos enteros
debe verificar el efecto de cuantización contra re-encoding de eventos cuando
afirme equivalencia. Transformar una representación no demuestra que se haya
reprocesado el stream crudo.

En RGB, los frames intactos con dt/a yTTC/a conservan el target
`log1p(dt/TTC)`. Esta ampliación, por sí sola, añade poca supervisión geométrica
al encoder RGB. Puede diagnosticar el condicionamiento temporal dePAIR/contextos;
el cambio de tasa de eventos puede afectar también al encoder de eventos.

## Rutas que habría que ejecutar

`operational/rgb_port/transfer.py::_expert_block` ejecuta A5/C2F, construyePAIR
y reconstruye raw17/expert3. A5/C2F usan el delta real para convertir el log-ratio
a inverseTTC y para mezcla temporal e incertidumbre. PAIR incluye dt,log(dt),1/dt
y soporte entre sus133entradas. Las fases y diferencias de raw17 se regeneran;
no basta con dividir la predicción final por a.

El adapter congelado no sirve para backpropagation de extremo a extremo:
`producer_features` pasa TTC por detach/NumPy y `frozen_context_state` usa
inference_mode. Un ajuste de productores debe usar el modelo y pérdida directos,
por ejemplo `causal_scale_ttc_loss`, y recomputar los estados eAP afectados.
Los raw17/phase340 antiguos no demostrarían conservación tras cambiar encoders.

## Alcance mínimo del siguiente piloto

Dos consultas elegidas previamente por identidad/tiempo (medianas elegibles
de carretera ySlider500), preparación limpia reproducida y tensores retenidos.
Medir primero la respuesta a un warp coherente de observaciones ya seleccionadas.
Eso es un experimento de cadencia variable: el preparador original de carretera
exige ventanas100ms. Mantener esa política sobre un stream transformado requiere
reselección con lector temporal inverso; a=2 puede exigir1,3s originales para
650ms transformados. No confundir ambos protocolos.

El piloto estimado ocupa unos25MiB de tensores preparados, excluyendo buffers de
eventos crudos. La estimación10–60s de ejecución proviene del coste limpio de
dos consultas, no de una medición de warp ya realizada; implementación y pruebas
necesitan tiempo adicional. No se iniciará un entrenamiento amplio ni se declarará
mejora hasta que este diagnóstico justifique qué bloque cambiar.
