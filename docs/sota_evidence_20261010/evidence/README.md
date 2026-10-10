# Evidencia publicada

Copias de resultados CSV/JSON y journals; no incluye datos crudos ni checkpoints.
`SHA256SUMS.json` conserva los hashes de estas copias. `.gitattributes` evita que
Git cambie los bytes mediante conversiones de fin de línea.

Los paths absolutos y hashes de fuentes de los recibos describen el entorno
original; no implican que esos paths existan en otro equipo. Las fuentes ejecutadas
y los checkpoints completos permanecen en los archivos locales de campaña.

Regenerar el informe desde esta copia (con el paquete instalado):

```text
python -m operational.streaming_revision.report --root docs/sota_evidence_20261010/evidence/streaming --document docs/sota_evidence_20261010/STREAMING_OPTIMIZATION.md
```

FCWD es exploratorio y ya expuesto. Test12 contiene solo recibos de preparación
y empaquetado: no hay etiquetas ni score privado. El piloto de latencia usa doce
consultas TRAIN40, no toda FCWD. Se conservan también pilotos fallidos/parciales.

La prueba amplia inicial de publicación V12 tuvo un fallo en un test antiguo que
usaba la fecha límite real. La corrección aísla esa fecha solo en el test; el runtime
congelado no cambia. Se incluyen el resultado inicial, la suite de cambios nuevos
y la repetición completa del archivo de tests R1 tras corregirlo.
