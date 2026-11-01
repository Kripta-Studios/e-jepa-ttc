# Comprobación de plataforma tras el perfil CPU

El perfil CPU original capturó dos observaciones por modalidad y pasó el replay
exacto PAIR/PHASE17 local. Ambas consultas fallan la tolerancia histórica en
algunos tensores intermedios; la salida TTC cumple su tolerancia en ambas.
Las discrepancias aparecen ya a la entrada normalizada de los contextos,
antes de aplicar el candidato. No prueban un fallo de la corrección canónica.

La siguiente comprobación técnica conserva consultas, lotes, pesos, medios y
tolerancias del [protocolo original](NATIVE_PAIR_PREREGISTRATION.md), y cambia
solo el dispositivo a CUDA. Fijar TF32 desactivado, cuDNN benchmark desactivado,
algoritmos deterministas y CUBLAS_WORKSPACE_CONFIG=:4096:8, como en el screen.
Ejecutar únicamente después del cierre de population2608 y si F0 queda admitido;
no competir por su GPU ni abrir el candidato para diagnosticar el original.

Si el perfil CUDA pasa todas las comparaciones históricas, continuar con captura
nativa completa y exigir de nuevo paridad completa antes del candidato. Si
falla, conservar el resultado y localizar la diferencia por productor/columna;
no ampliar tolerancias ni tratar la salida TTC cercana como paridad interna.

La captura CPU fallida en paridad y los dos intentos de integración anteriores
se conservan. Este documento no modifica su protocolo ni convierte sus fallos
en éxitos retrospectivos.
