# Refit final y prueba del mejor sistema

## No cambiar el candidato oficial existente

PHASE_ONLY_S7 ya tiene predictions selladas. Un refit, otro encoder o nueva
calibración produce otro candidato. No reemplazar su ZIP ni presentar la mejora
de uno con la latencia de otro. Tres heads sobre los mismos padres no representan
tres entrenamientos independientes de la percepción.

## Momento del refit

Una vez cerrada la decisión de arquitectura V18 y comparadas las modalidades,
registrar la receta final con datos, inicialización, endpoints y selector antes
de entrenar. Sólo entonces usar más TRAIN, incluidas secuencias previamente V.
Desde ese momento dichas secuencias son entrenamiento del candidato nuevo, no
su validación. Nunca incluir las doce secuencias test ni una caché que contenga
su supervisión. Las repeticiones por clips no añaden escenas independientes.

Para un predictor standalone V18 la receta supervisada completa puede entrenar
sobre TRAIN40 sin construir varios niveles de predicciones OOF. Esto es una
ventaja de simplicidad, no una garantía de precisión.

Para un sistema apilado sigue habiendo dos elecciones legítimas, que no deben
mezclarse:

1. Mantener roles disjuntos de productor y cabeza, declarando qué parte de TRAIN
   vio cada componente. Se usa el dataset en el sistema, no necesariamente todo
   en cada encoder.
2. Cross-fitting de productores y readouts, con exclusiones de grupo de TODOS
   los antecesores, normalizadores, teachers y calibradores. Un output OOF de una
   sola capa no basta si otro padre vio el grupo. Al entrenar productores finales
   sobre más datos, medir y declarar el cambio de distribución de las features.

No tratar el GROUP_CROSSFIT V16 como solución ya validada de ese segundo pipeline:
repitió cabezas de contexto, no todos los productores y generaciones finales.
Tampoco cambiar automáticamente al cross-fitting por razones nominales: debe
conservarse su comparación y su coste. PHASE_ONLY elimina las coordenadas hidden,
pero aún necesita mediciones expertas comparables.

## Réplica integral

Repetir las semillas previstas de productores/encoders y sus cabezas, generando
sus propias cachés. No insertar tres cabezas finales encima de un único encoder
y denominarlo réplica integral. No elegir la seed que salga mejor en transferencia.

## Ranking y cobertura

La meta es una mejora del score que efectivamente define el benchmark bajo
modalidad/contexto admitidos. No existe la exigencia lógica de ganar cada fila
o cada métrica para ganar un agregado, pero una afirmación amplia debe mostrar
regresiones y requisitos de seguridad/coste por separado.

Dev32/FCWD carecen de negativos. Publicar `observed_positive_contribution`, dejar
el componente negativo ausente y no dividir por 0,9 para fingir el oficial.
El mean_MiD local tampoco sustituye al overall. No inferir el score test12 desde
TRAIN40, V o estos datasets.

El envío oficial debe comprobar reglas de input/contexto, permitir formatos
admitidos y guardar score/recibo. No autenticar con tokens en argumentos/logs.
No realizar una serie de submissions adaptadas al score privado. Si el acceso
no permite un score verificable, la afirmación SOTA queda pendiente, no todos
los experimentos de TRAIN.
