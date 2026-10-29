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

E queda rechazado: la calibración Q1 produce una predicción H y cinco P fuera del dominio positivo de MiD. Esas filas permanecen y el agregado correspondiente es NA; no se recortan ni se eliminan para recuperar cobertura. Además, las bandas admisibles muestran regresiones grandes. F recupera cobertura H completa, pero falla los gates crucial, agregado, MAE y otras salvaguardas frente al A5 histórico. Su control emparejado es Q1 reutilizado, con cobertura insuficiente. Mejorar el ajuste de altura o su derivada en P no garantizó mejorar TTC en H.

## Posición espacial y clipping

G conserva ocho posiciones verticales del mapa congelado de A5 y las contrasta con promedios globales. Sigue el mismo objetivo de altura y cambio temporal de F. La extracción no usa targets; los normalizadores y el ridge se ajustan sólo con P. Antes de su primer ajuste, la acumulación se implementó por bloques para reducir memoria, y se verificó numéricamente contra la fórmula densa en ambos brazos. Esto no cambia el objetivo, lambda, población ni gates.

H cambia exclusivamente el clipping del scorer C20 respecto al control U0 ya congelado de B. Registra norma y finitud, sin volver a ponderar pérdidas. Una norma superior a1 no significa clipping aplicado en este brazo. La evaluación y las réplicas condicionales mantienen el orden G→H aunque una pausa de recursos permita preparar el fit independiente de H antes de cerrar G.

Ambos contrastes terminaron con NO_CANDIDATE. En G, conservar posición vertical mejoró el ajuste P de altura frente al pooling global, pero empeoró TTC en H. En H, quitar el clipping dejó las métricas muy próximas a las del control y no cumplió los gates. Este resultado descarta esa intervención concreta; no demuestra que el clipping sea irrelevante bajo cualquier objetivo o sampler. Las tablas exactas y sus intervalos están en `QUEUES_RESULTS.md` y `QUEUES_BOOTSTRAP.csv`.

## Lectura no lineal de la misma geometría

I se registró antes de los cribados de G/H y sólo se activó tras su rechazo. Añadió un MLP512→64→1 a la calibración espacial congelada de SP1, con última capa cero y las mismas pérdidas P de altura y cambio temporal. Tras6000updates se congeló el MLP y se ajustó el scorer nativo otros6000updates. No se pasaron targets TTC a la pérdida geométrica. La prueba agrega capacidad y una etapa de entrenamiento; no aísla exclusivamente no linealidad ni elimina todas las diferencias de regularización frente al ridge.

El candidato también empeoró TTC frente a SP1 y al A5 histórico. No se activaron sus réplicas ni la reconstrucción de PHASE17/H8 condicionada a admisión. Sus pesos intermedios NLH son un endpoint de geometría con scorer reiniciado y sin entrenar; el endpoint final de productor es NL1. Ambos se conservan, con esa distinción en sus metadatos portables.

`geometry_gradient_I/` describe el gradiente del objetivo geométrico sobre cinco rangos TTC de P. Los rangos sólo estratifican el diagnóstico; las etiquetas TTC no entran en el objetivo. `readout_gradient_G/`, `readout_gradient_H/` y `readout_gradient_I/` mantienen el scope NLL y las limitaciones del diagnóstico B. No se suman normas como si fueran masas de probabilidad.

## Qué justifica la decisión siguiente

Las nueve líneas registradas quedan cerradas sin candidato promovible. Los resultados no prueban que los mapas congelados carezcan de toda información TTC ni que la arquitectura haya alcanzado un techo matemático. Sí muestran que las lecturas ensayadas, las calibraciones geométricas y las dos intervenciones de optimización no superan conjuntamente los criterios fijados.

No se elige un nuevo coeficiente o epoch a partir de V, ni se gastan las réplicas condicionales en candidatos rechazados. Una continuación científica necesita una hipótesis distinta y un contraste prospectivo; por ejemplo, comprobar una medición geométrica temporal que conserve correspondencias espaciales, o conseguir evaluación independiente bajo el protocolo público. Ninguna de esas propuestas se presenta como un experimento ejecutado ni como una garantía de mejora.

## Límites de las conclusiones

P/H/V pertenecen al mismo programa de desarrollo; V ya estaba expuesto. No se han usado FCWD o Dev32 para ajustar o seleccionar estas variantes. Las comparaciones públicas tienen otras poblaciones, modalidades o métricas, descritas en `PUBLIC_COMPARABILITY.md`. Ningún resultado interno de este documento acredita SOTA universal.

Los exports verifican recarga exacta en CPU sobre vectores o historiales preparados. No sustituyen una prueba de latencia desde eventos crudos. Las curvas de objetivos distintos, sus normas de gradiente y sus costes tienen scopes explícitos y no se mezclan en una única cifra de calidad.
