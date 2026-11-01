# 13 — Referencias, procedencia y alcance de esta investigación

## 1. Qué se hizo

Se consultó GitHub mediante la conexión disponible y se obtuvo un checkout de lectura del commit `30b28b5c14eb9ed60067ed2f1dc262d7b60f2654`. Se leyeron el plan, su anexo, documentación de inferencia y código relevante de lectores, geometría, preparación, productores, fusión y contratos de fase. Se recalcularon agregados a partir de las diez filas publicadas. Se revisaron fuentes primarias de TTC, eventos y JEPA, incluyendo trabajos de septiembre y octubre de 2026.

La consulta del commit confirmó dos archivos añadidos: plan de 318 líneas y anexo JSON de 3.548 líneas. La base de la auditoría histórica declarada por esos archivos es `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`; el commit `30b28b5c…` publica esa evidencia y el plan. Son identidades distintas con funciones distintas.

La inspección del código es estática. No equivale a reejecutar el candidato con sus pesos y datasets. El checkout no se modificó para producir este paquete y no se publicó un nuevo commit.

## 2. Qué no se puede concluir de esta revisión

- No se ha revalidado aquí la identidad de los 365 frames leyendo los HDF5 del ordenador del autor; se ha revisado el recibo versionado que la documenta.
- No se ha demostrado cuál es la causa dominante del error Slider. Geometría, referencia temporal, arranque y fusión siguen siendo hipótesis con pruebas propuestas.
- No se ha entrenado, ejecutado un nuevo modelo ni medido una mejora de exactitud.
- No se ha certificado que las 24 candidatas sean independientes entre sí o de todas las decisiones históricas.
- No se ha obtenido la política privada completa de evaluación ni los targets oficiales alineados de las 434 consultas.
- No se ha certificado la posición actual en vivo de GarlTTC. Se conserva el recibo público del proyecto fechado el 28 de octubre, con 35,80 Overall MiD y la posición entonces registrada.
- La revisión bibliográfica no prueba ausencia de trabajos adicionales. El buscador general integrado falló; se consultaron directamente arXiv, páginas de autores, benchmarks y repositorios oficiales.

Estos límites afectan al alcance de las afirmaciones, no a la utilidad del plan de desarrollo. Las pruebas propuestas especifican qué evidencia falta para resolver cada punto.

## 3. Fuentes internas, fijadas al commit revisado

| ID | Fuente | Uso en el paquete |
|---|---|---|
| I01 | [Plan EvTTC del 30 de octubre](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md) | Orden original, pendientes, cobertura, errores y protocolo inicial. |
| I02 | [Anexo público](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json) | Hashes de frames, inventario, PDF transcrito, diagnósticos y recibos. |
| I03 | [AGENTS.md](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/AGENTS.md) | Contrato histórico y cambio de nombre; no se confunde hipótesis JEPA original con arquitectura final. |
| I04 | [ADR-0043](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/decisions/ADR-0043-evttc-competition-sensor-adapter.md) | Política de sensores, ROIs, arranque, fase, rechazos y límites del adaptador. |
| I05 | [Inferencia PHASE en competición](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/evttc_competition.md) | Comandos existentes, artefactos y frontera del runtime offline. |
| I06 | [Garl congelado sobre las mismas consultas](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/garl_evttc_competition.md) | 424/434, 434/434, 426/434; desacuerdo, inputs y límites de precisión. |
| I07 | [Cabeza `PhaseOnlyFinal`](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/training.py) | Arquitectura, 20 entradas, anclaje RGB y residual. |
| I08 | [Fase firmada y salida](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/simplex_t/phase.py) | Dominio matemático y transformación canónica de 100 ms. |
| I09 | [Adaptador de competición](https://github.com/Kripta-Studios/phase-ttc/tree/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition) | Lectura, geometría, preparación y runtime. |
| I10 | [Manifest de las 32 grabaciones](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/data/manifests/evttc_all32_local.yaml) | IDs y metadatos locales; no se usa como certificación de independencia. |
| I11 | [README del proyecto](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/README.md) | Estado histórico, exposición previa de EvTTC-32 y distinción JEPA/PHASE. |
| I12 | [Recibo GarlTTC del 28 de octubre](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/paper/data/official_benchmark.json) | 35,80145158, 6.762 consultas, pesos de bandas y ranking registrado entonces. |

El archivo de texto adjunto del usuario también se leyó como contexto de la auditoría y de sus aclaraciones. No se atribuye a esta revisión la ejecución descrita en aquella conversación.

## 4. Identidades de las fuentes locales del checkout

SHA-256 calculados sobre los bytes del checkout Git del commit indicado, no sobre una copia reformateada del JSON:

| Fuente | SHA-256 |
|---|---|
| Plan de 318 líneas | `371cfadbe38a189029f4ccd628023a250c7c6846389ab89d9d1509e1ffa86e7c` |
| Anexo JSON | `4342cad998e1dd48b004da7d74c8ed53a18308a9eac447626c54da39c5f97165` |
| `operational/final_refit/training.py` | `cddb650c637e3627bf711ad3571dd9454baecfc997e09cb0ce3675d5cf3a54a4` |
| `src/e_jepa_ttc/simplex_t/phase.py` | `7ee447190ef7b218d0f6c6cf5110a1522c201049749a98b4395eeafefe853cde` |

El anexo identifica el PDF recibido con SHA-256 `2e88892833c5e5ef0d7f1138ff1d0ae5eb9975e882465f6d9130dd66ef8f51be`. Ese es un dato de procedencia del recibo; el PDF no se ha vuelto a comprobar aquí ni se distribuye dentro del ZIP. Un hash vincula bytes, pero no es por sí solo una firma o validación independiente del organizador.

## 5. Registro de fuentes externas

Consulta de esta revisión: 30-10-2026. Cuando una fuente es mutable, fijar commit/checkpoint al implementar su reproducción. El análisis usa la versión indicada del artículo cuando se especifica.

| ID | Referencia primaria | Fecha/versión | Alcance útil |
|---|---|---|---|
| P01 | [EvTTC, competición](https://nail-hnu.github.io/EvTTC/competition/), [formato](https://nail-hnu.github.io/EvTTC/download/data_format/), [calibración](https://nail-hnu.github.io/EvTTC/calibration/) | Web consultada 30-10-2026 | Diez secuencias, entradas, formato y resultados históricos. |
| P02 | [EvTTC, artículo](https://arxiv.org/abs/2412.05053) | 2024/RA-L 2025 | Sensores, dataset y protocolo original. |
| P03 | [Toward Deep Representation Learning… the eAP Dataset](https://arxiv.org/html/2603.16303v1) | Marzo 2026, v1 | Garl, MiD y tablas V–VIII; tabla VI solo tres secuencias EvTTC. |
| P04 | [Garl-TTC, código oficial](https://github.com/NAIL-HNU/Garl-TTC) | Release consultada 30-10-2026 | Modelos, inferencia y contrato de test sin TTC público. |
| P05 | [REACT](https://arxiv.org/html/2609.19204v1) | 16-09-2026, v1 | SSM spiking, tiempo físico, validación y límites de comparabilidad. |
| P06 | [Event-Aided Time-to-Collision Estimation / STRTTC](https://arxiv.org/abs/2407.07324), [código](https://github.com/NAIL-HNU/event_aided_ttc) | ECCV 2024 | Geometría y registro espaciotemporal; FCWD. |
| P07 | [Time-to-Contact Map…](https://openaccess.thecvf.com/content/ICCV2023/html/Nunes_Time-to-Contact_Map_by_Joint_Estimation_of_Up-to-Scale_Inverse_Depth_and_ICCV_2023_paper.html), [ETTCM](https://github.com/neuromorphic-paris/ETTCM) | ICCV 2023 | TTC denso y movimiento/profundidad hasta escala. |
| P08 | [EV-TTC, Bisulco et al.](https://github.com/anthonytec2/EV-TTC) | RA-L 2025 | Multiescala temporal, EV-Slim, T²CEF/M3ED; distinto de EvTTC HNU. |
| P09 | [V-JEPA 2](https://arxiv.org/abs/2506.09985), [código oficial](https://github.com/facebookresearch/vjepa2) | 2025 | Representación visual temporal preentrenada. |
| P10 | [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) | v1 15-03; v3 11-06-2026 | Features densas, supervisión intermedia y temporal. |
| P11 | [MotionJEPA](https://arxiv.org/html/2609.23881v1), [código](https://github.com/mkarmann/motion-jepa) | 20-09-2026, v1 | DISReg y pérdida de características dinámicas. |
| P12 | [LeWorldModel](https://arxiv.org/abs/2603.19312) | Marzo; v3 03-06-2026 | JEPA compacto y regularización de embeddings. |
| P13 | [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) | Junio; v2 28-09-2026 | Predicción de prefijos de acciones; relevancia indirecta. |
| P14 | [Variational JEPA](https://arxiv.org/abs/2601.14354) | 20-01-2026 | Modelado probabilístico latente. |
| P15 | [MDN-VJEPA](https://arxiv.org/abs/2610.13956) | Registro arXiv 2026 consultado al corte | Mezclas probabilísticas; no benchmark TTC. |
| P16 | [The JEPA Predictor](https://arxiv.org/abs/2607.16274v3) | v3 02-10-2026 | Predictor portable y oclusión; utilidad indirecta. |
| P17 | [From Contrast to Consistency](https://arxiv.org/html/2605.25570v2) | 25–26-05-2026 | Flujo continuo y consistencia; revisar bidireccionalidad. |
| P18 | [E-WAVE](https://arxiv.org/abs/2609.34346) | 28-09-2026 | Trayectorias y flujo continuo, no TTC. |
| P19 | [SICAF](https://arxiv.org/abs/2610.29158) | 21-10-2026 | Flujo autosupervisado con SNN y selección temporal. |
| P20 | [CollideNet](https://arxiv.org/abs/2604.16240), [código](https://github.com/DeSinister/CollideNet/) | 17-04-2026 | Forecast de accidente; tarea diferente del TTC geométrico. |

## 6. Qué es evidencia publicada y qué es propuesta propia

| Afirmación | Tipo |
|---|---|
| Nueve componentes, 20 entradas físicas y cabeza de 1.955 parámetros | Lectura y recuento de la arquitectura en código. |
| Proyección que omite traslación y fases canónicas de 100 ms | Contrato explícito del código/ADR. |
| Paralaje cercano podría deteriorar Slider | Inferencia física; causa pendiente de demostración. |
| Fusión con calidad observable puede mejorar transferencia | Hipótesis experimental propuesta. |
| Movimiento constante implica `T(t+dt)=T(t)-dt` | Deducción bajo hipótesis físicas explícitas. |
| MotionJEPA ayuda a preservar dinámica en sus tareas | Resultado de los autores; no prueba de mejora PHASE. |
| J2 mejorará TTC | No establecido; J0/J1/J2 se diseñan para comprobarlo. |
| Objetivos de promoción, presupuesto y gates G0–G5 | Propuesta operativa de este paquete, no regla oficial. |

## 7. Reproducción documental

El [manifiesto del paquete](MANIFIESTO.md) contiene la lista de Markdown, tamaños y hashes. Las tablas numéricas de estado se comprobaron por recálculo. Los enlaces internos y la integridad del ZIP se verificaron antes de la entrega. Los comandos futuros no se presentan como ejecutados: requieren su implementación, datos y entorno según cada documento.

No se incluyen artículos completos, pesos, datos crudos, credenciales ni correspondencia privada. El contenido del ZIP es análisis propio, instrucciones y plantillas, con referencias a sus fuentes.
