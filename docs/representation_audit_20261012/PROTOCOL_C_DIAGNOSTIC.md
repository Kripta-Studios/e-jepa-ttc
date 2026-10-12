# Diagnóstico C: transferencia entre grupos H

Sin GPU ni updates de optimizador. Se registran24 ajustes cerrados ridge:8 folds leave-one-group-out ×3 representaciones, lambda=1 fijo, sin búsqueda de hiperparámetros.

R0: PHASE17 actual sin bypass. RG: PHASE17 actual más los896 vectores geométricos/transportados originales del bypass. RD: PHASE17 actual más diferencias temporales de geometry_tokens y diferencias forward-minus-reverse del transporte (512 dimensiones). RD excluye el nivel absoluto de los tokens extra; no elimina toda información de apariencia de la ruta PHASE17.

Objetivo: residual de phase respecto a mediana actual de expertos, error cuadrático con masas M0 renormalizadas en cada fold, penalización ridge1 por coeficiente e intercepto sin penalizar. Se ajustan mean/std e intercepto sólo con grupos de entrenamiento de cada fold. No se usa V, FCWD, Dev32 ni P para ajustar estos probes; los productores son los padres originales congelados. Los grupos H no son réplicas de encoders.

La señal principal es MAE en phase fuera de grupo. Se conservan TTC/MiD, signos y bandas, sin inventar overall si falta soporte. Son probes H1 lineales y un objetivo cuadrático; un negativo no demuestra ausencia de información no lineal ni decide directamente promoción de H8. Se guardan coeficientes, IDs de folds, predicciones OOF, costes CPU y todos los resultados.

No hay gate automático hacia una nueva receta neural: la función de este diagnóstico es distinguir memorizar grupos de una señal que se pueda decodificar linealmente fuera de ellos. Cualquier entrenamiento neural derivado necesita configuración congelada antes de sus updates.
