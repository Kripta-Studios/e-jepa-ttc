# Análisis prospectivo C0

`analyze.py --family N2` exige los doce endpoints sellados; `--family N3` exige
los seis. Verifica cada checkpoint antes de cualquier forward OLD_DEV. El
supervisor permite un único escritor con `engine.Lease()` y aplica los recursos
registrados mediante `engine.Resources()`; no se lanza análisis junto a otro
trainer pesado.

El protocolo `PROTOCOL_COST_CONTEXT.json` proporciona `launch`, los controles
históricos H8 seed7 y `bootstrap_draws`; por fold, `sources` fija
`parent_dev_sha256` y `wrapped[arm].dev_sha256`. Los hashes de la fuente reducida
codifican su máscara y nunca se sustituyen por la identidad del padre.

Inferencia: fragmentos128 con inputs NPZ deduplicados por contenido, salida NPZ,
recibo y checkpoint hash. Publicación full8192 conserva targets y masas. Para
diagnósticos se leen exclusivamente los expertos actuales compilados y se
contrastan contra H8 publicado. Hull/escapes se etiquetan como referencia
diagnóstica de los tres expertos; no son entradas al modelo reducido ni una
selección científica de experto. Su anchor permitido también queda publicado.
Los heads de costes son inactivos y sus outputs no deciden la estimación.

Las comparaciones están fijadas: FULL_C0−H8, reducidos−FULL_C0,
SET_AGE−FULL_C0 y SET_NOTIME−SET_AGE. Se conservan cada secuencia y fold,
intervalos jerárquicos con los draws históricos y contraste exacto por nueve
secuencias. Bootstrap se recupera por fragmentos256. Nunca se promedian TTC para
crear un ensemble. Todo resultado es exploratorio de OLD_DEV reutilizado.

`execution/analysis/N2` y `N3` contienen `RESULTS.json`, tablas, pérdidas y
recibos. `ANALYSIS_EXPORT_INDEX.json` reúne los miembros numéricos que debe
incluir la entrega. El resumen combinado está en
`METRICS_COSTE_CONTEXTO.csv` del root nocturno. El cribado de precisión usa
límite superior jerárquico <+2MiD, signo ≤+0,005 y crucial ≤+5MiD. La eficiencia
permanece pendiente hasta N4; no se recomienda sustitución de sistema sin una
reducción medida p95≥20% del alcance pertinente.

Al deadline sólo se permite `--partial-inventory <archivo explícito>` con un
inventario fijado antes de evaluar que contiene tres folds completos por brazo.
No se puntúan endpoints parciales2500 ni se cambia la cola por esos scores.

Para regeneración, el bundle necesita `execution/` con pesos compactos, inputs
y predicciones, `CONTENT_MANIFEST.json`, `provenance/src` científico y
`provenance/operational/simplex_t_cost_context/model.py`. Ejecutar
`regenerate.py --root <extracción> --output <recibo>` bajo el supervisor de
recursos compartidos. El regenerador comprueba antes de cada cabeza y fragmento
RAM ≥2 GiB, RSS del árbol Python de entrega ≤4 GiB, disco libre tras reservar
1 GiB ≥10 GB y, en Windows, margen de compromiso ≥1 GiB. No necesita imports del
worktree original. Verifica los miembros e inferencia de cabezas; no promete
reconstrucción desde raw, TRAIN ni nuevas llamadas a expertos.
