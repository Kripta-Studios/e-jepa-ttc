# Revisión científica de PHASE-TTC y de las propuestas físicas

La dirección propuesta tiene sentido, pero el historial aconseja empezar por una intervención más pequeña: **comprobar y reparar, si mejora la precisión, la relación entre cadencia real y salida de PAIR; estudiar después qué parte de la expansión resulta observable**. Un estado dinámico aprendido merece un piloto condicionado a esos diagnósticos. El modelo generativo del sensor tiene más requisitos y debe ir después. No recomiendo entrenar los tres componentes juntos como primera prueba.

Esta revisión corresponde al árbol `f8131a031f61d9270323fe3961dad88016b582cd`, al 31 de octubre de 2026. El artículo principal es [paper/main.tex](../../paper/main.tex), con versión consolidada [PHASE_TTC_Final.tex](../../paper/PHASE_TTC_Final.tex). Su corte científico es el 28 de octubre; el recibo de compilación es del 29. El antiguo [docs/e_jepa_ttc_paper.tex](../e_jepa_ttc_paper.tex) es histórico y lleva una advertencia de obsolescencia. No se encontró un archivo versionado llamado literalmente `PHASE_TTC_Research_Report`.

Se inventariaron **1.516 commits**, **493 Markdown**, **71 LaTeX**, 1.716 JSON y 136 CSV versionados. El historial empieza el 30 de junio, por lo que no hay seis meses completos de desarrollo registrado. Se revisaron el texto modular del artículo, las familias de experimentos, las correcciones metodológicas y la documentación reciente que afecta a las hipótesis. El inventario no equivale a leer cada línea de cada parche ni a repetir todos los entrenamientos. [Alcance exacto](00_ALCANCE_Y_METODO.md).

## Conclusiones que cambian la decisión

1. **El éxito final no demuestra un beneficio JEPA.** La cadena TRAIN40 tiene nueve componentes entrenados y deriva de V16; incluye supervisión geométrica y de profesores RGB. V8 D2/D3 no superaron sus controles. El piloto de profesor fijo del 31 de octubre tampoco pasó su criterio representacional, aunque predijo mejor el estado futuro del profesor.
2. **La interpretación de V8 en la propuesta es correcta.** Los escalares archivados dan 373,176 MiD para la ruta analítica, 373,596 con residual sin transporte, 163,231 con transporte y residual, y 155,824 al añadir historia. Son intervenciones sobre pesos fijos. Muestran dependencia de rutas del modelo entrenado; no identifican por sí solas qué restricción fue necesaria al entrenarlo.
3. **El problema temporal ya tiene evidencia directa.** El piloto de dos consultas encontró que A5/C2F RGB responde al delta real, mientras PAIR mantiene un TTC parecido al cambiar la cadencia. El screen de 149 consultas terminó durante esta revisión: F0 pasa el criterio registrado, pero el router adaptado falla por regresión en carretera. Esto apoya ampliar F0, no promover toda la cadena. [Actualización concurrente](13_ACTUALIZACION_CONCURRENTE.md).
4. **Stage68 ya intentó un modelo reducido.** El código del commit de entrenamiento tiene fallback de expansión isotrópica y traslación. Además, exige al menos 32 planos normales antes de probar cualquiera de los ajustes; el informe registra una mediana de cinco. La falta de rango de seis parámetros no basta para explicar el rechazo. Hace falta separar escasez de medidas, sesgo del estimador y falta de identificabilidad específica de expansión.
5. **Información de soporte ya se ha probado, sin promoción.** Las campañas F2/Fcap y F2n/Fcapn añadieron relojes/recuentos y reinicios, con controles de capacidad. No cumplieron todas las métricas. Una confianza nueva debe medir ambigüedad geométrica o innovación, no repetir esos diez campos con otro nombre.
6. **La transferencia es el criterio exigente.** En 468 consultas del artículo, PHASE tiene MAE 1,022677 s frente a 1,015377 s de RGB, y cinco errores mayores de diez segundos frente a dos. Todos los intervalos pareados de media, mediana y p95 incluyen cero. Los resultados del adaptador del 31 de octubre pertenecen a otra receta y otra población.

## Dictamen de las ideas

| Idea | Dictamen | Primer contraste útil |
|---|---|---|
| Equivariancia temporal, unidades y coordenadas | Prioridad inmediata | Evaluar ampliación de F0 tras screen favorable; reconstruir historias eAP con genealogía compatible |
| Observabilidad específica de q | Prioridad alta como diagnóstico | Auditoría de rechazos Stage68 y estimabilidad de la traza del campo afín |
| Estado relativo y reloj físico | Piloto condicionado | Estado firmado, tiempos reales y objetivos futuros independientes, con control acelerado |
| Formación de eventos | Prometedora, segunda etapa | Parches calibrados y simulación fotométrica; umbral/iluminación restringidos |
| Silencio sin eventos | Condicionada al dato crudo | Distinguir silencio, dropout, refractario y falta de grabación |
| Correspondencias materiales y visibilidad | Extensión posterior | Trayectorias locales visibles y composición entre tres instantes |
| Reabrir JEPA genérico o escalarlo | No con la evidencia actual | Requiere hipótesis y protocolo nuevos; respetar ADR-0045 |
| Brillo inversamente proporcional a Z² | Descartar como ley general | No corresponde a radiancia de una superficie resuelta |
| Conservación de área o masa de eventos | Descartar | Elimina expansión real o supone conservación que el sensor no cumple |

## Dossier

| Documento | Contenido |
|---|---|
| [00 · Alcance](00_ALCANCE_Y_METODO.md) | Corte, método, verificaciones y límites |
| [01 · Git](01_HISTORIAL_GIT.md) | Cronología interpretada y enlaces a los cinco registros mensuales completos |
| [02 · Artículo](02_AUDITORIA_ARTICULO.md) | Revisión de las secciones, afirmaciones y problemas documentales |
| [03 · Datos y resultados](03_DATOS_RESULTADOS_Y_EXPOSICION.md) | Poblaciones, métricas, semillas y genealogía |
| [04 · Lecciones históricas](04_LECCIONES_EXPERIMENTALES.md) | Qué se intentó y qué conclusión permite cada resultado |
| [05 · Reloj](05_RELOJ_Y_ESTADO_RELATIVO.md) | Matemática, anclaje, aceleración y estado mínimo |
| [06 · Sensor](06_MODELO_DEL_SENSOR.md) | Formación de eventos, fotometría y silencio |
| [07 · Observabilidad](07_OBSERVABILIDAD_Y_FUSION.md) | Auditoría Stage68, Fisher, sesgo y fusión |
| [08 · Simetrías y transporte](08_SIMETRIAS_Y_SUPERFICIES.md) | Tiempo, escala, ROI, correspondencias y visibilidad |
| [09 · JEPA](09_JEPA_Y_RETENCION_FISICA.md) | Cierre reciente, literatura y condiciones de reapertura |
| [10 · Experimentos](10_PLAN_EXPERIMENTAL_ACOTADO.md) | Secuencia, controles, presupuestos propuestos y decisiones |
| [11 · Sistema y documentación](11_SISTEMA_Y_DOCUMENTACION.md) | Runtime, incertidumbre y deuda documental |
| [12 · Bibliografía](12_REFERENCIAS_VERIFICADAS.md) | Fuentes primarias y límites de extrapolación |
| [13 · Actualización concurrente](13_ACTUALIZACION_CONCURRENTE.md) | Screen149 terminado: F0 pasa, router falla; ampliación e historias pendientes |
| [Resultados verificados](RESULTS_VERIFIED.md) | Tablas regeneradas desde predicciones y escalares archivados |
| [Índice documental](DOCUMENT_INDEX.md) | Todos los Markdown y LaTeX inventariados |

La contribución defendible sería demostrar que una representación conserva información sobre aproximación **y que el predictor la usa correctamente bajo intervenciones**, manteniendo precisión, signo, cobertura y coste. La composición nominal de reloj, sensor y confianza no demuestra novedad ni utilidad.

No se modificaron modelos, pérdidas, protocolos históricos ni evidencias. No se ejecutó entrenamiento, inferencia de sensores, publicación, envío o nueva selección de candidatos.
