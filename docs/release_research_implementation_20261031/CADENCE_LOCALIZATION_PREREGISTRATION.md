# Localización de cadencia T1: eventos, RGB y ambas ramas

Contraste descriptivo solicitado en el dossier, separado de la ampliación F0.
Población: las 149 consultas ya expuestas, sin seleccionar casos ni umbrales.
Se reutiliza PAIR133 sellado; no se repiten productores ni lectura RAW.

Cuatro rutas prefijadas: original, PAIR_E canónico, PAIR_R canónico y ambos.
Mismos pesos, lotes, relojes, historias, normalizadores y política de emisión.
Cada ruta reconstruye PHASE17 completo, contextos y cabeza. Ejecutar todas en
CPU FP32, un hilo y algoritmos deterministas, dejando libre la GPU del worker.

Guardar primero predicciones sin abrir targets. Comparar el original CPU con
el original GPU archivado, incluyendo máscaras de disponibilidad. Exigir
rtol=1e-4/atol=1e-4 para TTC antes de interpretar las métricas; informar deriva
de fase/estado por separado. El contraste principal usa el original CPU
emparejado, nunca atribuye la deriva de plataforma a la corrección física.

Comprobar identidad exacta cuando todos los deltas de eventos son float32(0.1):
la ruta E y el original deben coincidir en toda la salida; ambas ramas deben
coincidir con la ruta RGB. Un fallo detiene el análisis de mecanismo.

Después del sello, unir los targets de desarrollo previamente verificados.
Informar MAE/RAE por dominio y adquisición, cobertura, signo e interacción
en predicción y error. No escoger la combinación ganadora ni modificar el
candidato de population2608: este ensayo localiza qué rama causa el cambio.
No admite un nuevo entrenamiento ni constituye confirmación independiente.
