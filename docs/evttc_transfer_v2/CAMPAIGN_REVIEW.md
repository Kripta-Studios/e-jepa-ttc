# Revisión independiente de la campaña

Revisor: agente nuevo `campaign_final_review`, sin historial de implementación,
sin modificaciones ni delegación. Rango revisado: `50b2bb68..c3d51251`, plan de
ejecución, capítulos 09 y 15 del ZIP, código, documentos y artefactos JSON/NPZ.

**Veredicto: GO_CON_ALCANCE_LIMITADO.** No encontró defectos críticos o importantes
en la comparación agrupada principal, el cálculo externo o la reescala solo train.
Reprodujo los tamaños 2.531/24/7, las métricas de las tres semillas, los scores
externos y las nueve condiciones de robustez. Confirmó que la cabeza de cada fold
parte del candidato original y que la familia reservada no interviene en el
ajuste ni en la reescala de intervalos. No equiparó contextos con encoders ni
resultados expuestos con transferencia independiente.

## Dos hallazgos menores y cierre

1. La auditoría de carretera guardaba cobertura por método pero no las tablas
   sobre filas compartidas, aunque el ledger inicial decía que sí. El suplemento
   `REVIEW_SUPPLEMENT.json` recalcula doce comparaciones pareadas desde los mismos
   NPZ sellados. La CLI de análisis también las guarda en ejecuciones futuras.
2. En el piloto STRTTC, la consulta 13 tenía NaN sin causa textual cuando el
   inverse-TTC del endpoint era no positivo o no finito. El suplemento identifica
   esa rama como causa, explícitamente **inferida del flujo del código ejecutado**;
   no inventa un valor de q que no se guardó. La CLI futura registra la causa en
   el momento. Ninguna predicción o métrica previa se cambia.

Ambos comportamientos tienen regresiones que fallaron antes del cambio y pasaron
después: `test_paired_baseline_excludes_exactly_the_comparator_missing_rows` y
`test_invalid_endpoint_estimates_have_explicit_reasons`. No se solicita otra
revisión: se verifica el cambio y se conserva el dictamen original.

## Aspectos que el revisor dejó fuera de su dictamen

- Independencia certificada de adquisiciones, exposición de ancestros/teachers y
  extrínsecas exactas: continúan sin resolver; se limita la afirmación científica.
- Equivalencia oficial de 434 consultas y reproducción oficial de comparadores:
  no se establecen en esta campaña, que presenta diagnósticos locales.
- Tiempo real, backlog y convergencia completa: los tiempos instrumentados y el
  presupuesto fijo no los acreditan; no se anuncian esas propiedades.
- Beneficio de F2/P0/JEPA no ejecutados: la decisión condicional es defendible y
  su beneficio sigue sin medirse.
- Resultado de la batería general: la verifica el ejecutor por separado; el
  revisor no cuenta su éxito anticipado como evidencia.

Las limitaciones aceptadas no se convierten en resultados positivos. La entrega
conserva el modelo exploratorio y el resultado externo negativo.
