# PHASE-TTC: investigación y entrega EvTTC, 29–31 de octubre de 2026

Esta prerelease publica el código, los informes y los paquetes de la campaña
local de transferencia y adaptación. El dossier detallado está en
`docs/research_log_20261029_31/` de la rama `research/evttc-transfer-v2`.

El candidato final `PHASE-TTC-Adapt-20261031` mantiene el grafo TRAIN40 congelado
y utiliza tres routers de 688 parámetros, semillas 7/13/21. El fit usa 2.608
consultas de 24 grabaciones de carretera y Slider-500, excluyendo del ajuste las
grabaciones del benchmark. Se declara la exposición histórica de desarrollo.

- **Para EvTTC:** `PHASE_TTC_Adapt_EvTTC_results_2026-10-31.zip`, con 434 filas en
  diez archivos results.txt, índices/timestamps originales y tiempos por consulta.
- **Modelo y reproducción:** `PHASE_TTC_ADAPT_FINAL_2026-10-31.zip`, con pesos,
  código, inputs numéricos, resultados, manifiesto y evidencias.
- **Historia experimental:** paquetes F1, campaña agrupada, gate robusto y router
  de arranque; también se conservan el envío original y planes/revisión recibidos.

Sobre los 423 targets locales disponibles, aplicando a ambas versiones la regla
de negativos comunicada por EvTTC, RAE pasa de 58,9781 % a 25,8452 % y MAE de
1,057825 a 0,561545 s. No son nuevos scores oficiales: faltan once targets locales
y existen diferencias de alineación con el PDF original en tres secuencias.

La ejecución final mide 1,740290 s por consulta frente a 3,268250 s observados en
el envío original; no es una ablación controlada. La suite final pasó 7.329 tests
y 17 subtests, con 46 omitidos y cero fallos/errores. Las tres semillas reproducen
sus checkpoints byte por byte desde una instalación separada; el ZIP extraído
reproduce la evaluación.

Los pesos all-data de iteraciones anteriores tienen un papel distinto al nuevo
envío. No hay afirmación SOTA, entrenamiento nuevo de encoders/JEPA ni garantía
de seguridad. No se incluyen sensores crudos ni entornos Python. Consulte
`SHA256SUMS.txt` y el inventario de assets para verificar los adjuntos.

Autor único: **Álvaro Schwiedop Souto**, sin afiliación institucional.
