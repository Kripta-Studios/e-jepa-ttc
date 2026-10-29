# Protocolo externo: comprobaciones y errata del plan

Comprobación del 29-10-2026. Se conserva sin modificar el plan recibido.

## Reconciliación pública v2

La [auditoría reproducible](evidence/external_protocol_v2/README.md) consultó 13
fuentes públicas mediante GET acotados a 2 MB. Conserva el estado HTTP, tamaño y
SHA-256 de cada respuesta, pero no sus cuerpos crudos. No solicitó ni descargó
programas privados, referencias, etiquetas test o recursos autenticados.

La API pública de CodaBench confirma ahora el siguiente contrato:

| Campo | Valor verificado |
|---|---|
| Competición | `17291`, **eAP 3D Detection Benchmark**, publicada |
| Clases | `car`, `bicycle`, `truck`, `pedestrian`, `motorcycle`, `bus`, `tricycle`, `other_vehicle` |
| Umbrales de distancia de centro | 0,5 / 1 / 2 / 4 m |
| Columnas | AP, ATE, ASE, AOE; AP es primaria |
| Fase | `28840`, test, fin `2030-12-31T23:59:59Z` |
| Split público declarado | 40 secuencias train y 12 test |
| Programa de scoring | Privado; solo se publica metadata de tamaño: 9.821 bytes |
| Referencia | Privada; `make_programs_available=false` |

Esto verifica el **contrato**, no los bytes de su implementación métrica. El
endpoint administrativo de resultados devolvió 403 y el endpoint público del
leaderboard `19173` devolvió una lista vacía. Por tanto no hay un líder, score o
ranking oficial verificable desde el acceso público auditado. La AP local sigue
sin poder etiquetarse como score oficial ni como reproducción del scorer privado.

La reconciliación de links queda fechada y fijada por commit:

- `NAIL-HNU/eAP_dataset@2c5f3a74` (27-06-2026) todavía enlaza `16717`; su API
  devuelve actualmente 404.
- `NAIL-HNU/Garl-TTC@25666124` (29-06-2026), commit posterior titulado
  `Update benchmark links`, enlaza detección `17291` y TTC `17289`.

La lista train oficial fijada contiene 46 IDs (SHA-256
`2f92c1c3ed881642190c6d6ddd71f247e6184deb489456e06db2516b81a774fb`).
Las tres revisiones disponibles del catálogo de GitHub Pages contienen 40 assets
y ninguna incluye los seis IDs ausentes. `dataset_info.json` (SHA-256
`209f418b7215a5d5954575d0137240a0d4a5c69db846cd103f810392cf43cd98`)
solo aporta para ellos weather, road type, HDR, densidad categórica, velocidad
categórica y rol train. No aporta URL de medio, tamaño, hash ni labels. Esos seis
IDs no cuentan como datos utilizables, diversidad adicional ni evaluación
independiente.

El script que regenera y valida esta evidencia es
[`scripts/audit_detection_external_protocol.py`](../../scripts/audit_detection_external_protocol.py).
La auditoría inicial se conserva a continuación como registro fechado; donde era
más cauta por falta de acceso estructurado, la reconciliación v2 es la referencia
actual.

## Auditoría inicial: competiciones

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

## Auditoría inicial: corpus descrito y medios públicos

La [página del proyecto](https://nail-hnu.github.io/eAP_dataset/) describe 46
secuencias train. El release público fijado
`NAIL-HNU/eAP-dataset@a814a946e60d37f7242358af5f344542177e50c5` tiene 40 secuencias
train, 118.247 frames. La comparación de los IDs del corpus y del catálogo público
identifica estas seis secuencias sin medios en ese release:

`AUIkq1ZJRv`, `CEOFFwY7c8`, `F6XHHaiwJg`, `OwA5qqkewc`, `mXRa2Ux9ok`,
`uXfPJNOXVY`.

Esto delimita una diferencia entre corpus descrito y publicación de medios; no
autoriza contar las seis como diversidad disponible. La campaña usa train40.

## Auditoría inicial: independencia y TTC conjunto

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
