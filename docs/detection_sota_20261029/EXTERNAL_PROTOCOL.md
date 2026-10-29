# Protocolo externo: comprobaciones y errata del plan

Comprobación del 29-10-2026. Se conserva sin modificar el plan recibido.

## Competiciones

El [README oficial Garl-TTC fijado en
256661242b8a7f5e56aa3c1c02348b30f6e89de6](https://github.com/NAIL-HNU/Garl-TTC/blob/256661242b8a7f5e56aa3c1c02348b30f6e89de6/README.md)
enlaza **eAP detección: 17291** y **GarlTTC: 17289**. El enlace 16717 del plan
recibido no debe usarse como referencia vigente. El acceso público comprobado no
permitió verificar el leaderboard ni obtener un scorer de detección. Los endpoints
de resultados consultados devolvieron 403 sin sesión; la recuperación web de la
página también produjo contenido inconsistente. La identidad del enlace procede
del repositorio de los autores, no de interpretar ese contenido como resultados.

El mismo repositorio publica un scorer TTC bajo
`garl_ttc_benchmark/codabench/scoring_program/score.py`. Tener ese código permite
reproducir la definición TTC con etiquetas admisibles; no revela las etiquetas
privadas Test12 ni proporciona el scorer de detección. Los resultados TTC
históricos del proyecto conservan su propia procedencia y no se sustituyen por
una evaluación nueva del sistema detector–tracker–PHASE.

## Corpus descrito y medios públicos

La [página del proyecto](https://nail-hnu.github.io/eAP_dataset/) describe 46
secuencias train. El release público fijado
`NAIL-HNU/eAP-dataset@a814a946e60d37f7242358af5f344542177e50c5` tiene 40 secuencias
train, 118.247 frames. La comparación de los IDs del corpus y del catálogo público
identifica estas seis secuencias sin medios en ese release:

`AUIkq1ZJRv`, `CEOFFwY7c8`, `F6XHHaiwJg`, `OwA5qqkewc`, `mXRa2Ux9ok`,
`uXfPJNOXVY`.

Esto delimita una diferencia entre corpus descrito y publicación de medios; no
autoriza contar las seis como diversidad disponible. La campaña usa train40.

## Independencia y TTC conjunto

Los metadatos públicos inspeccionados contienen secuencia, tiempos, calibración,
referencias y condiciones de escena; no proporcionan un ID de adquisición/sesión
que certifique independencia entre todas las secuencias. Se mantiene el split
candidato 24/8/8 y se declara pendiente su independencia por adquisición.

El [release GarlTTC](https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset)
publica etiquetas train40 y entradas test12 sin targets TTC. Un score sobre sus
objetos estructurados puede evaluar un productor TTC admitido; no demuestra la
calidad conjunta con detecciones, identidad y ROIs completamente predichas.
TRAIN40 conserva exposición a las cuarenta secuencias y requiere un gate distinto
para cualquier nueva afirmación de generalización conjunta.
