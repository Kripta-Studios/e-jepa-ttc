# V17: qué mecanismos se han investigado

La investigación distingue representación, ponderación de la pérdida y medición geométrica. Una mejora de ajuste en P o H no basta para promover un sistema. Las cifras de cada cola se regeneran en `QUEUES_RESULTS.md`; los gates siguen siendo los de V14.

## Información anterior a PHASE17

A compara el H8 nativo con una rama adicional que recibe los tokens originales de geometría y transporte de A5. Los seis endpoints se congelaron antes de abrir V. La rama mejora el TTC crucial, pero no alcanza la mejora mínima ni conserva todas las salvaguardas en todas las semillas.

La mejora de A es mayor en H que en V. Eso identifica una brecha de generalización; no demuestra por sí mismo que el modelo memorice apariencia. La cola C usa validación dejando fuera grupos completos de H para explorar esa posibilidad con modelos lineales. Son 24 ajustes cerrados adicionales, aunque no realizan pasos de optimizador. Sus resultados no se presentan como resultados de H8 ni como modelos candidatos.

D recibe diferencias de tokens entre instantes y diferencias entre transporte directo e inverso. Elimina el nivel estático de esos vectores y también reduce su dimensionalidad efectiva. Por tanto, no aísla exclusivamente apariencia. Los tres endpoints D se congelaron antes de V. No superan el gate crucial ni mejoran consistentemente a G; este resultado negativo se conserva.

## Ponderación por incertidumbre

La NLL original contiene una precisión por consulta calculada por la incertidumbre congelada de A5. Los ejemplos de TTC muy corto reciben una precisión menor, en promedio, que la población completa. Esa observación no equivale a una medida del gradiente ni demuestra causalidad.

B compara el readout C20 con la NLL nativa y con una precisión constante fijada a la media de P. Conserva el muestreo uniforme y el resto de términos. Los seis readouts se congelaron antes del cribado P/H. La intervención mejora MiDc frente a su control en las tres semillas, pero no alcanza el 10 % requerido. Los intervalos por grupos y los fallos de salvaguardas se conservan, sin convertir significación estadística en admisión automática.

`readout_gradient_B/` mide gradientes del componente NLL sobre muestras estratificadas de P. Excluye expresamente el término top-k, riesgo y demás componentes. Las normas de gradiente no son masas aditivas ni gradientes exactos de toda P.

## Sensibilidad de la altura aprendida

`diagnostics/P_GEOMETRY_CONSISTENCY_*` contrasta el log-ratio requerido por TTC, el derivado de cajas anotadas P y el producido por la altura aprendida. Las alturas anotadas son targets de entrenamiento y diagnóstico, nunca inputs de inferencia. Los ejemplos anotados tampoco constituyen un techo alcanzable garantizado: sus errores geométricos y temporales permanecen visibles.

La geometría aprendida tiene poca variación respecto a las cajas. E comprueba si los tokens congelados contienen información para calibrarla mediante una regresión de altura. Conserva los tokens originales que alimentan PAIR y vuelve a propagar TTC, incertidumbre y riesgo mediante las fórmulas nativas. Los pesos del encoder, foreground, transporte, incertidumbre y PAIR permanecen congelados.

El ajuste de altura recupera parte del sesgo mediano de los cambios temporales en P, pero aumenta su error medio. Esto motivó F antes de conocer el cribado H de E. F incorpora supervisión del cambio entre alturas consecutivas, usando sólo cajas de P y un objetivo registrado con normalización por varianzas P. No busca coeficientes sobre H/V. Ambos ajustes cerrados cuentan como entrenamiento, aunque no usan `optimizer.step`.

E y F siguen un cribado de productores en H. Sólo una familia admitida puede reconstruir features, normalizadores H y refinadores bajo un addendum de integración. No se conectan productores modificados a features antiguas para atribuirles una mejora del sistema completo.

## Límites de las conclusiones

P/H/V pertenecen al mismo programa de desarrollo; V ya estaba expuesto. No se han usado FCWD o Dev32 para ajustar o seleccionar estas variantes. Las comparaciones públicas tienen otras poblaciones, modalidades o métricas, descritas en `PUBLIC_COMPARABILITY.md`. Ningún resultado interno de este documento acredita SOTA universal.

Los exports verifican recarga exacta en CPU sobre vectores o historiales preparados. No sustituyen una prueba de latencia desde eventos crudos. Las curvas de objetivos distintos, sus normas de gradiente y sus costes tienen scopes explícitos y no se mezclan en una única cifra de calidad.
