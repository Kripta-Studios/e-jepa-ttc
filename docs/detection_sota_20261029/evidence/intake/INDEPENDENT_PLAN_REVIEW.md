# Revisión independiente del plan de continuación

Fecha de cierre: 2026-10-29. Este documento compila los dictámenes de tres revisores distintos del autor del plan. Los informes originales se incluyen completos como anexos, con sus hallazgos y cierres.

## Versión revisada

- Plan: `SOTA_DEVELOPMENT_PLAN.md`.
- SHA-256 final: `d63065f341594613554aecb48219b4bbc967aed2fd5628d3c6bd599cce7013b7`.
- Repositorio de referencia: `24b8f36f6e4aa17c965f9526d8cf486b512a8046`.
- Revisión del plan, del código y de la validez de las conclusiones; no implementación de las fases propuestas.

## Dictámenes

| Revisor | Alcance | Resultado sobre la versión final |
|---|---|---|
| `pilot_code_audit` | Código real, snapshots, interfaces, comandos y primera iteración | Sin hallazgos materiales abiertos. Dos omisiones materiales del borrador corregidas y verificadas. |
| `pilot_science_protocol` | Cifras, población, métricas, splits, causalidad, velocidad y exposición TRAIN40 | Aprobado sin hallazgos materiales abiertos. Tres aclaraciones incorporadas y verificadas. |
| `sota_primary_research` | Fuentes primarias, comparabilidad y estrategia de detector/temporalidad/eventos | Sin hallazgos nuevos; dos precisiones de alcance incorporadas y cerradas. |

## Correcciones que dejó la revisión

1. **Diagnóstico del checkpoint histórico:** el sidecar se implementará como observador externo de outputs. No se modifica el runtime/decoder congelado ni se relajan hashes para cargarlo.
2. **Configuración efectiva y perfilador:** el builder debe aplicar la configuración completa; la restauración de checkpoint/profiler debe recuperar la variante de velocidad, batch y targets correctos. Añadido roundtrip estricto y rechazo de campos ignorados.
3. **Evaluación y roles:** el nuevo GT debe alimentar la selección interna del runner; la admisión de `audit` necesita esquema v2, conservando el histórico.
4. **Oráculos:** la regla que construye las sustituciones se fija, pero el matching del evaluador se vuelve a ejecutar si se reporta AP de las predicciones transformadas.
5. **Velocidad y temporalidad:** flips condicionados a semántica validada o experimento enmascarado; se permite avanzar con el baseline RGB a historia causal aunque una mejora geométrica/ranking falle.
6. **Alcance de SOTA:** una reproducción local solo acredita la comparación entre esos métodos bajo ese protocolo; no acredita por sí sola liderazgo oficial/general.
7. **ImVoxelNet:** poses de ego-movimiento separadas como requisito de fusión BEV temporal, sin bloquear automáticamente un control de un frame.
8. **Preservación:** comparación de tablas en memoria, sin reescribir el histórico. El revisor de código ejecutó el fragmento documentado y obtuvo igualdad exacta del texto.

## Límites del dictamen

El cierre se refiere a la precisión y coherencia del plan. No significa que los gates S0–S6 estén implementados o superados, ni certifica resultados futuros, compatibilidad de nuevos métodos o un scorer oficial. Los revisores no ejecutaron entrenamiento ni nuevas pruebas GPU.

Los bloqueos de scorer, diversidad rara, semántica física de velocidad y generalización TTC siguen expresos. Los documentos históricos y el artículo no se modificaron. La inclusión e integridad del anexo se verifica al preparar la entrega; `SHA256SUMS.txt` permite comprobarla sobre los archivos extraídos.

## Informes originales

- `independent_sota_code_review.md`.
- `independent_sota_science_review.md`.
- `independent_sota_research_review.md`.
