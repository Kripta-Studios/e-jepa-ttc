# Cola F: calibración geométrica que supervisa cambios temporales

Registro prospectivo antes de evaluar E en H. Evidencia exclusivamente P: la calibración de altura de E recupera la mediana del log-ratio ≤1s (0,1091 frente a target0,1089), pero empeora su MAE (0,1085 frente a0,1019) y el de las demás bandas. Ajustar alturas aisladas no garantiza derivadas temporales estables. Se conserva este resultado negativo.

## Intervención única

Se mantiene el mismo readout lineal de altura sobre los128 tokens originales A5, las mismas medias/std P, intercepto P y clip prospectivo[-2,2]. Se modifica sólo el objetivo del ajuste cerrado de sus coeficientes:

- F0: ridge de altura aislada ya ajustado por E (control Q1).
- F1: suma de MSE de altura y MSE de cambio temporal de altura, cada término dividido por la varianza de su propio target de corrección en P. Pesos unitarios fijos, sin búsqueda. Ridge lambda1 sobre coeficientes; suma de residuos normalizada al número de observaciones de altura.

Target de altura: log(bbox_height_P)−log(height_original). Target temporal: log(bbox_height_actual/bbox_height_previa)−analytic_original de la última pareja, con ambas cajas válidas. No se usa TTC para este ajuste cerrado. Ambos objetivos usan exclusivamente P; H/V no ajustan coeficientes, escalas ni intercepto. Son dos fits cerrados en total contando el control E, no cero entrenamiento.

Con X estandarizado por P y centrado por observación, D=X_actual−X_previa, y centrado sólo para altura e intercepto fijo b=mean(y), resolver:

`[XᵀX/var(y) + (N_height/N_pair)DᵀD/var(d) + I] beta = Xᵀ(y-b)/var(y) + (N_height/N_pair)Dᵀd/var(d)`.

El target d no se centra: un offset temporal no es un intercepto de altura. Se registra la identidad del ajuste y ambos términos. No se selecciona la normalización de términos después de ver resultados.

## Scorer, control y admisión

Igual que E: encoder, foreground, geometría original usada por PAIR, transporte, incertidumbre y calibración congelados; sólo se reajusta el scorer C05 reiniciado,6000updates, P uniforme, FP32, batch128, misma pérdida y scheduler nativos. PAIR133 debe permanecer idéntico.

Primero F1 seed7; reutilizar Q1 seed7 de E como F0, pues todos sus parámetros, datos, muestreo, inicialización y presupuesto son los del control registrado. No volver a entrenarlo bajo otro nombre. Congelar ambos antes de H. F1 debe pasar todos los gates originales contra F0 y A5 histórico.

Si pasa, entrenar F0/F1 seeds13/23 en esta cola independiente (incluso si E había rechazado su propia comparación Q1/Q0). No se promociona E ni se reabre su decisión. Si alguno falla, terminar F sin otro peso. Sólo si todos pasan, reconstruir las features H/V y normalizadores H, registrar integración y entrenar seis H8 de2500updates, congelándolos antes de V; gates originales pareados y frente a V14M0, con coste E2E antes de promoción.

Techo local nuevo:30000updates de scorer (control seed7 reutilizado) y15000condicionales de cabezas,40técnicos. Contadores globales compartidos:1M físicos y18h GPU. No nuevos encoders, PAIR, RGB, datos externos de selección ni modificación de V13/R1/V16. No afirmación SOTA por resultados internos.
