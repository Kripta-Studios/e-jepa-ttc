# Comprobaciones posteriores a la publicación

El ZIP final contiene 2.793 miembros y tiene SHA-256
`f7d915b3d203453696d7006b08dbde0c731c3d1f92e652d616edfad60e0e2ff1`.
Todos los miembros pasaron SHA-256, CRC y extracción. Los 24 checkpoints
completos conservan su receta y suman 60.000 updates científicos.

Las 18 cabezas C0 reprodujeron exactamente 49.152 predicciones, en 396
fragmentos: error TTC máximo cero. La diferencia máxima en agregados fue
1,8474111129762605e-13 MiD, debida al orden de suma. La comprobación original
H16, vinculada al ZIP interno idéntico, conserva seis cabezas y 16.384
predicciones con error TTC cero. Ninguna verificación ejecutó optimizador.

También se ejecutó el perfilador incluido, desde la extracción independiente:
nueve cabezas, 500 mediciones batch1 cada una, 25 warmups y 50 mediciones
batch128 por cabeza. Usó las mismas 64 consultas TRAIN seleccionadas por hash,
sin targets, raw ni expertos. Los datos crudos de tiempos están en
CACHED_HEAD_PROFILE_REPLAY.json y la comparación con los perfiles primarios en
HEAD_COST_REPLAY.csv. Esta repetición confirma que el consumidor del bundle
funciona; sus tiempos dependen del estado del host y no sustituyen los perfiles
primarios ni los criterios de decisión. No mide el coste integral.

La variación es material: C2F_ONLY_C0 tiene p95 primario 2,502355 ms y
20,89574 ms en esta repetición; FULL_C0 pasa de 40,654565 a 22,49892 ms.
Las cabezas GRU conservan la misma capacidad. Los ratios de un bloque con
distinta carga del host no identifican un ahorro causal por excluir expertos.
Los criterios de precisión siguen fallando y no se decide sustitución.

Los recibos finales son externos al ZIP que verifican y se entregan en Git.
ARTIFACT_MANIFEST.json relaciona ambos alcances. La dependencia pendiente es
el slot exclusivo GPU/lectura pesada del propietario, para la ruta completa.
No quedan fits autorizados pendientes; la única propuesta futura permanece
sin ejecutar y no concede updates adicionales.
