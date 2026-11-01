# Historia Git: de la hipótesis JEPA a PHASE-TTC

**Actualización posterior al corte base:** el screen149 terminó durante esta revisión; F0 pasa y el router no. La ampliación de F0 está registrada en curso y la conservación de historias eAP sigue pendiente. Los pasajes que describen el screen como pendiente corresponden al corte inicial. Véase [actualización de cierre](13_ACTUALIZACION_CONCURRENTE.md).

El inventario encuentra 1.516 commits alcanzables en la ventana, de los cuales 309 no son ancestros del HEAD revisado. No se suman como si todos fueran cambios incorporados a la rama actual. Hay 17.146 entradas de rutas modificadas y 12.019 cadenas de ruta distintas; renames y ramas pueden hacer que este número supere los 8.254 archivos del árbol actual.

| Mes de autor | Commits | Registro con cuerpos, padres y rutas |
|---|---:|---|
| Junio de 2026 | 11 | [Junio](git/2026-06.md) |
| Julio de 2026 | 192 | [Julio](git/2026-07.md) |
| Agosto de 2026 | 217 | [Agosto](git/2026-08.md) |
| Septiembre de 2026 | 433 | [Septiembre](git/2026-09.md) |
| Octubre de 2026 | 663 | [Octubre](git/2026-10.md) |

No aparecen commits de abril o mayo en las refs inventariadas. Los metadatos completos están en [commits.json](git/commits.json); las refs y reglas de conteo, en [inventory_summary.json](inventory_summary.json).

## Junio y julio: infraestructura y primeras correcciones

`df4ecaea` inicia el proyecto el 30 de junio. Le siguen el lector EvTTC indexado (`4faea6b4`), baselines triviales (`4081c9e7`), geometría/event-rate (`f25356cb`) y caché voxel/TinyCNN (`81380f2b`). El desarrollo temprano explora predicción latente, máscaras, probes y pocos labels. Los experimentos locales permitieron comprobar entrenamiento e inferencia, pero sus secuencias acabaron reutilizadas durante el desarrollo.

El 13 de julio, `26480260` reclasifica CPLA-high como `reused_test_diagnostic`, añade controles de procedencia y distingue semillas de SSL y fine-tuning. Sus notas no son una sutileza administrativa: invalidan la lectura de las primeras tablas como test ciego y de tres cabezas como tres representaciones independientes. El freeze posterior no borra esa exposición.

El 24 de julio, `1f9e2f99` retira una ruta smoke que escribía JSON `status: success` sin ejecutar la matriz eAP. Se inspeccionó el diff de `scripts/run_all.ps1`: pasa a invocar la matriz también en smoke y añade un test de regresión. Los marcadores anteriores no se deben contar como evidencia. Este defecto histórico corregido no permite deducir que los resultados posteriores sean ficticios; obliga a verificar sus recibos reales.

La [cronología de ingeniería](../engineering_history_20261027/01_SCIENTIFIC_CHRONOLOGY.md) conserva cifras de los primeros smokes con sus límites y referencias Git, incluyendo el cambio desde derivadas centradas a geometría pasada/causal.

## Agosto: objeto, coordenadas, geometría y transporte

La sucesión Object-LHR, Object-Expansion, Object-Event y Causal Scale aborda tres fallos distintos: no identificar el objeto consultado, perder expansión al redimensionar endpoints por separado y aprender foreground cuya derivada no recupera TTC. Las variantes con movimiento de cajas pueden mejorar mediante información privilegiada del objeto; no prueban utilidad de eventos por sí solas.

A4 añade destilación relacional DINOv3; A5 incorpora transporte local. Los mensajes de Git y las recetas documentadas obligan a separar esa supervisión del preentrenamiento sin TTC. El 13 de agosto, `de3349fd` corrige la exposición del parent A4 a grupos reservados. La repetición válida entrena parents dentro de cada fold; los primeros grouped runs quedan como diagnósticos expuestos.

Scientific Recovery V5–V7 no promociona candidato. V7 descubre complementariedad A5/C2F a posteriori; V8 registra la hipótesis de routing antes de contrastarla. En V8, el replay de rutas asigna la mayor diferencia al transporte combinado con residual. D2/D3 no demuestran beneficio JEPA frente a scratch, random-frozen y futuro permutado. Las etiquetas «Causal Scale v5–v8» y «Scientific Recovery V5–V8» designan estudios distintos.

## Septiembre: campo de movimiento, routing y contexto

X0/X0.5 y Stages61–69 comprueban información incremental sobre controles más fuertes que un input destruido. Una diferencia frente a shuffle puede coexistir con ausencia de mejora frente a calibración o pooling. Stage65 conserva bloqueo de integridad; Stage67 termina con resultado negativo válido; Stage68 termina con soporte geométrico insuficiente. No son tres maneras intercambiables de decir «falló».

`3820f76e`, del 6 de septiembre, incorpora el protocolo Stage66–69 y el estimador de normal flow con fallback isotrópico. Su código ya contiene `inverse_ttc_at_anchor`. Esto reduce la novedad de proponer solo un ajuste reducido o la fórmula de transporte.

SIMPLEX-T cambia selección de expertos por corrección residual de fase. Se completan 24 cabezas y 60.000 updates científicos. H16 mejora puntos de tres semillas, pero las dos semillas nuevas no excluyen cero en su intervalo pareado. Los productores compartidos limitan la interpretación de las réplicas. Más observaciones aumentan historia; no equivalen a un estado físico con ley de transición.

## Octubre hasta el 28: RGB, medición conjunta, refit y servicio

V12 aborda entradas, coste y transferencia; V13 añade una ruta RGB y campos semánticos compatibles. V14–V17 exploran pesos de pérdida, sampling, geometría congelada, precisión, bypass y readouts. Varias mejoras del régimen crucial empeoran recesión o MAE. V18 adapta medición y correspondencias; su contraste TRUE/ZERO aporta evidencia de eventos dentro de esa configuración, pero no supera todas las condiciones de promoción. V19 tampoco elimina el compromiso entre bandas.

El refit final sigue la familia V16 con producers OOF por grupos y una cadena de despliegue TRAIN40. El artículo conserva el score oficial de 28 de octubre y diferencia ese resultado de pruebas causales o de runtime. `0219a84f` identifica el corte científico que usa el artículo. Su inventario tenía 1.406 commits; los 1.516 de esta revisión corresponden a un corte posterior y no son una contradicción.

## Del 29 al 31: adaptador, negativos y cadencia

El [dossier 29–31](../research_log_20261029_31/README.md) registra validación por familias, ajuste F1, geometría, calidad y router de arranque. `e482c220` empaqueta el adaptador de tres semillas y 434 salidas. Su precisión local se evalúa sobre 423 targets y no reproduce por completo el PDF del organizador. Debe conservar identidad distinta del modelo del artículo.

Los commits `fdca738d`, `93d58c6f` y `095e1ab9` documentan contexto, soporte y combinación sin promoción. `4926b765` localiza sensibilidad de PAIR a la cadencia. `f1f9c8ed` documenta la intervención canónica y su auditoría eAP parcial. Los tres últimos commits del HEAD (`77113a4e`, `73ea5ec0`, `f8131a03`) registran el screen de 149 consultas, condiciones de cobertura y diagnósticos de signo/rechazo. **Código y preregistro presentes no prueban que el screen ya haya terminado.** No se localizó un resultado versionado completo de ese screen en este corte.

El cierre JEPA del 31 de octubre, recogido en ADR-0045, exige evidencia nueva para reabrir entrenamiento. Esta revisión propone criterios; no revoca ni ejecuta esa campaña.
