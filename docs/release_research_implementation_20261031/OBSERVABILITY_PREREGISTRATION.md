# O0 RAW: diagnóstico incremental de observabilidad

Registro anterior a abrir los targets en este nuevo brazo. Población fija:
screen149, históricamente expuesta, no confirmación. No se añaden ni eliminan
consultas según errores, soporte o disponibilidad de planos.

La extracción usa las dos últimas ventanas de la última observación de eventos
solicitada, en el cuadrado común exacto del release. Conserva cámara, reloj,
calibración y eventos a resolución original. El frame de consulta suministra el
recorte, como en el release; no se presenta el histórico como disponible antes.
Se conservan también consultas donde los productores rechazaron una observación.

Seis indicadores físicos prefijados: fuga al núcleo de c, sensibilidad de q,
residuo normalizado, innovación transportada entre dos endpoints reales,
desacuerdo entre el interior/exterior de la caja y fracción de parches sin
solapamiento. Los dos últimos son proxies, no visibilidad verdadera ni una prueba
de independencia estadística. Seis controles de soporte: log-recuento, fracción
de planos admitidos, soporte PAIR, desacuerdo entre expertos, número de
observaciones y fracción del recorte conservada dentro del sensor.

Primero se sellan features sin targets. Después se calcula gain = error RGB -
error R0, usando las predicciones y targets ya sellados en evaluation149_v2.
No se calcula gain invirtiendo TTC saturado a fase: fase usa las ubicaciones
crudas guardadas. Se reportan errores, cobertura y correlaciones dentro de
adquisición; ninguna variable derivada de target entra como feature.

Prueba incremental: regresión ridge con intercepto, lambda=1, sin búsqueda,
cuatro brazos (constante, soporte, físico, físico permutado). Cada vector tiene
seis campos más sus seis máscaras; mediana, media y escala se calculan solo con
grupos de entrenamiento. Validación leave-one-acquisition-out; ponderación igual
por adquisición. Son sondas diagnósticas ajustadas sobre padres R0 expuestos, no
una validación OOF independiente de la cadena. No hay actualizaciones de R0.

La permutación usa semilla 7 y estratos sin targets: dominio carretera/Slider,
terciles de recuento y patrón de valores ausentes. Conserva todas las máscaras;
se registra cuántas filas cambian. Es un control aleatorizado, no un candidato
causal de despliegue. No se incluyen IDs de secuencia/cámara como regresores.

Se abre un piloto O1 aprendido solo si el brazo físico reduce al menos un 2% el
MSE macro de gain frente a constante, soporte y permutación, y si el límite
superior del CI95 pareado de MSE frente a soporte es <0. Se exige cambiar al
menos 25% de filas en el control permutado; si no, se declara insuficiente.
El bootstrap usa 2.000 remuestreos de adquisiciones y semilla 7. Esta regla admite
un piloto, no un release. Un resultado negativo no descarta modelos de calidad
más ricos: localiza el límite de estos planos/indicadores con esta preparación.

Si se admite O1: cuatro brazos igualados, máximo 1.000 updates por brazo sumando
cinco folds de desarrollo (200 por fold), una semilla, pérdida de tarea en fase,
pesos de productores/contextos/cabeza original congelados. Se conserva RGB/R0
como control constante y se exige la regla del dossier antes de una confirmación.
No se abre D4, S2 o JEPA por el solo hecho de que este brazo falle.

V0 usa las mismas trazas para comprobar que los históricos comparten recorte,
que los tiempos no exceden consulta y que recorte/resolución no se confunden con
expansión física. Una pérdida de área de recorte se informa; no se resta de la
ecuación como si fuese un zoom puro.
