# Cola G: información vertical antes del pooling global

Hipótesis registrada a partir del diagnóstico P de geometría y la inspección del encoder. No se modifica el encoder: se observa su mapa congelado64×32×32, ya calculado por la ruta original de A5. El pooling global pierde la posición vertical; el token final tampoco garantiza conservar la expansión espacial.

## Extracción y contraste

Extraer solamente P/H, FP32 y TF32off, con el encoder matched A5 original en eval. Pooling promedio a8×1 por canal produce512 valores por observación. Registrar manifests, orden de queries, padre, código, hashes por shard y costes GPU. Los targets no entran en el extractor ni en inferencia. Una prueba compara el mapa directo con `endpoint_dense_features` del productor completo y confirma paridad CPU/GPU dentro de tolerancia registrada1e-5 absoluta/relativa. No hay nuevos updates de encoder, otro encoder ni una CNN adicional en inferencia.

SP0 utiliza los64 promedios globales de esos mismos mapas, seguidos de448 ceros. SP1 conserva los512 valores verticales. Ambos readouts tienen512 coeficientes; los448 nulos de SP0 no añaden información. El contraste cambia información espacial y dimensionalidad efectiva: no identifica exclusivamente una causa.

Ambos ajustan en P el objetivo fijo de altura y cambio temporal de la cola F, normalizado por las respectivas varianzas de targets P, con ridge lambda1. Medias/std de inputs se ajustan sólo P y separadamente para cada brazo. Intercepto P idéntico, clip de corrección[-2,2]. Targets: bbox_height y sus cambios, sin TTC en el ajuste cerrado. Dos ajustes cerrados nuevos, registrados aparte de optimizer updates.

Conservar tokens originales de geometría/transporte, incertidumbre y PAIR133. La corrección altera sólo la altura de la rama analítica y se reprocesan las fórmulas nativas de TTC/incertidumbre/riesgo. El input nuevo se llama `height_features` y debe provenir del encoder original congelado; no se admite sustituirlo por cajas o por la caché de otro productor.

## Entrenamiento y gates

Para cada brazo, reiniciar a cero la última capa del scorer residual C05 y entrenar sólo ese scorer6000updates sobre P, uniformemente, con batch128, FP32, AdamW3e-4→3e-5, warmup100, cosine6000, wd1e-4 y clip1. Pérdida nativa idéntica a E/F. No aprender simultáneamente la calibración ni otros módulos.

Primero SP0/SP1 seed7. Congelar ambos antes de H. SP1 debe pasar todos los gates originales contra SP0 y A5 histórico; si no, cerrar negativo sin otro pooling ni coeficiente. Si pasa, replicar SP0/SP1 seeds13/23. Todos los contrastes deben pasar antes de integrar cabezas. Sólo entonces extraer V sin usarlo para ajustar, reconstruir features y normalizadores H, registrar la integración, entrenar los seis H8 M0/L0 de2500updates y congelarlos antes de evaluar V.

Techo local:36000updates de scorer y15000condicionales de cabezas,40técnicos. Coste de extracción, QA e inferencia dentro de18h GPU globales; actualizaciones dentro de1M físicos globales. Mantener propietarios V13/R1/V16, sin submissions, publicación automática ni datos externos para seleccionar. V continúa siendo desarrollo expuesto.
