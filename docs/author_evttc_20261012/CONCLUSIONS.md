# Resultado y límites de la reconstrucción

La campaña terminó: 43 predicciones y 43 etiquetas admitidas para los doce
endpoints/variantes reportados. No hubo exclusiones por fallo del modelo ni
fases MiD inválidas. Las seis exclusiones del inventario inicial son las dos
observaciones de arranque de cada escenario, necesarias para formar tripletas.

## Qué muestra la comparación local

F_TRUE V13 obtiene RTE pooled **12,010%**, Garl completo **13,674%**: reducción
relativa del 12,17%. MAE es 0,217 s frente a 0,258 s. El MiD crucial es 59,199
frente a 81,641. Todos los targets están en la banda crucial; no hay evidencia
en otras bandas ni un overall oficial de cuatro bandas.

El promedio de RTE dando igual peso a los tres escenarios es **10,237% frente
a 15,399%**. Debe publicarse junto al pooled: el promedio macro da a las seis
consultas CCRm el mismo peso que a las veinte CCRs2-medium.

F_TRUE gana RTE en CCRs2-high (12,048 frente a 13,401) y CCRm-medium (4,404
frente a 21,124). Pierde en CCRs2-medium: **14,260 frente a 11,672**.
Por tanto no hay victoria en todos los escenarios.

El bootstrap exploratorio por tres escenarios, 10.000 réplicas con semilla
20261012, da IC95 de la diferencia macro F_TRUE−Garl full de
**[−16,720; +2,589] puntos porcentuales**, incluyendo cero. Es descriptivo:
solo tres grupos, ya expuestos. No se presenta como confirmación estadística
de superioridad general.

La modalidad importa mucho: R_CTX obtiene RTE pooled 16,403%; E_CTX, 71,932%;
H8 mediana de tres, 47,379%; Garl event-only, 64,522%. La mejora de la fusión
no permite atribuir un resultado SOTA a un modelo solo de eventos. F_ZERO
también es claramente peor que F_TRUE en esta población.

Fuente regenerable: `SCORED_PREDICTIONS.csv`, `RESULT.json`,
`SUMMARY_COMPARISONS.json` y `replay_public.py`.

## Por qué no es una reproducción exacta del paper

Garl reconstruido tiene RTE macro 15,399%, mientras el paper publica 10,60%.
Esa discrepancia es una advertencia de protocolo, no evidencia de que el paper
esté equivocado. No se buscó otro crop, desfase, checkpoint o subconjunto para
reducirla después de ver TTC.

Faltan el inventario oficial de queries, identificación inequívoca de los
archivos y checkpoint original, fase exacta del muestreo y mapping algebraico
completo. La proyección usada aquí ignora traslación, conserva la convención
RGB→event del adaptador anterior y está declarada. Las historias propias
tienen mayor duración que Garl. Se trata de una comparación reproducible
entre sistemas sobre las mismas consultas reconstruidas, con esos límites.

**No se debe comparar nuestro 10,237% directamente con el 10,60% del paper
para declarar SOTA.** La referencia contemporánea local es 15,399%; incluso
frente a ella hay una derrota por escenario y mucha incertidumbre entre grupos.

## Costes observados

Las medianas CPU/FP32 de estas 43 consultas son: Garl completo 1.128 ms desde
entradas crudas, con forward 749 ms; familia V13 completa con seis cabezas
1.538 ms; H8 V12 con tres cabezas 1.924 ms. No se excluyó la primera consulta
como warmup. Los p95 y muestras individuales se conservan en la evidencia.

El temporizador H8 incluye la preparación histórica, que también construye
un tensor Garl no consumido por H8; el de V13 evalúa seis alternativas y no
solo F_TRUE. El host tiene otros trabajos y la GPU no se utilizó. Estas cifras
no permiten comparar con los 12–13 ms GPU del paper ni concluir una latencia
optimizada de despliegue. `TIMING_SUMMARY.json` conserva el perímetro preciso.

## Qué queda después de esta ejecución

Se ha cerrado la evaluación local de la reconstrucción, no la verificación
externa oficial. Para explicar la discrepancia Garl hacen falta correspondencias
geométricas independientes y, si los autores las publican, queries/código de
preprocesado exactos. Dev32 y FCWD conservan sus resultados y no se sustituyen
por este subconjunto más favorable.

Test12 sigue sin score CodaBench acreditado. Los checkpoints finales V13
conservan sus doce hashes; esta campaña consumió cero updates. V14 y su trabajo
en curso no se modificaron.
