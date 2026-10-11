# ADR-0003: serializar metadatos NumPy antes del hash de observaciones RGB

Fecha: 2026-10-11. Estado: aceptado para corregir H_R_RAW y la misma ruta V.

Las filas Parquet y `history_observations` contienen arrays NumPy anidados. La
comparación de observaciones repetidas fallaba al serializarlos como JSON. La
identidad y sus campos son correctos; falta una conversión de representación.

Se añade `operational/rgb_port_signature_fix`, que convierte únicamente metadatos
antes de llamar a la función original. Conserva el hash de las representaciones
JSON nativas, la exactitud de enteros, el rechazo de valores no finitos y las
comprobaciones de identidad. No se usa `str(array)`, no se omite la comparación y
no se normalizan píxeles ni tensores del modelo.

El código previo está congelado por la campaña en ejecución. Una capa adicional
admitida mediante hashes permite mantener esas congelaciones y los checkpoints.
La cola reintentará sólo el fallo y sus dependencias omitidas, conservando registros
y tareas terminadas. La evidencia y los comandos están en
[el informe de reparación](../rgb_signature_fix_20261011/README.md).
