# Cola E: calibrar altura antes de aplicar el residual TTC

Hipótesis registrada desde diagnósticos P, sin resultados nuevos de E ni selección externa. Se mantiene el techo global V17:1M updates físicos/18h GPU y la protección de V13/R1/V16.

## Evidencia

P≤1s: mediana del log-ratio requerido0,108897; alturas bbox anotadas0,104967; geometría aprendida0,008431. Sobre toda P, std de altura aprendida0,005154 frente a0,090570 anotada. El gap anotación/física existe en algunas consultas y se conserva; no se usan cajas como inputs en inferencia.

## Intervención y control

Se ajusta una calibración lineal ridge de delta_log_height desde los128 geometry_tokens originales A5 por observación. Target: log(height_bbox_P)−log(height_original). Usar sólo endpoints P cuyo target_valid sea verdadero; medias/std e intercepto P, lambda1 fijo, sin sweep. Guardar pesos, normalizador, cantidad de observaciones y hashes. Es un ajuste cerrado adicional, no cero entrenamiento por carecer de optimizer.step.

Q0: corrección constante igual al intercepto ajustado P. Cancela al formar diferencias temporales y controla la calibración de escala global.

Q1: intercepto más regresión de tokens normalizados. La corrección se acota a[−2,2] por seguridad numérica prospectiva; no se ajusta ese límite sobre H/V. Nuevo analytic=analytic_original+delta_current−delta_previous. visible_height se multiplica por exp(delta); se presenta como altura calibrada de la rama analítica, no una nueva máscara segmentada.

En ambos brazos los geometry_tokens originales, transport_tokens, sensor_support y pair133 se conservan exactamente. Las incertidumbres quedan congeladas y se propagan nuevamente por la fórmula nativa. La ruta PAIR continúa usando sus inputs originales, no tokens recalculados con la altura calibrada. Se exige prueba de identidad de pair133. No hay otro encoder, otra CNN ni otro PAIR.

Después se reajusta únicamente el scorer residual original con capC05 y operador original/equal-slope0,05. Se reinicia a cero la última capa en ambos brazos. Mismo P uniforme, batch128, NLL y demás pérdidas nativas, AdamW3e-4→3e-5 con warmup100 y cosine6000, wd1e-4, clip1, FP32 GPU y TF32off. La calibración ridge permanece congelada durante esos6000 updates.

## Cola y gates

Primero Q0/Q1 seed7,6000updates cada uno. Congelar ambos antes del cribado H. Comparar Q1/Q0 y Q1/A5 original con todos los gates V14 sin relajar; H no ajusta ridge ni scorer. Si ambos pasan, repetir Q0/Q1 seeds13/23 (24.000updates). Deben pasar todos los contrastes H antes de entrenar cabezas. Si falla una condición, conservar el negativo y terminar E sin nuevos coeficientes.

Sólo con padres admitidos se reconstruyen PHASE17 H/V y normalizadores sólo H, se entrenan los seis refinadores H8 M0/L0 de2500updates y se congelan juntos antes de V. Las cabezas deben pasar gates pareados por seed y frente a V14M0 para considerarse candidatas; falta además coste E2E. Esta extensión debe ligarse a un addendum de código antes de sus fits.

Máximo de esta cola:36.000 updates de scorer y15.000 de cabezas condicionales, más60 técnicos como techo local. Todos pertenecen al contador V17. No se da por hecho que la calibración generalice. V es desarrollo expuesto; FCWD/Dev32 no ajustan ni seleccionan estos modelos.
