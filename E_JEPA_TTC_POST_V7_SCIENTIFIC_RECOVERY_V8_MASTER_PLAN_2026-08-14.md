# E-JEPA-TTC — cierre científico V7 y Master Plan post-V7 / V8

**Fecha de corte:** 2026-08-14 (Europe/Madrid)  
**Repositorio:** `Kripta-Studios/e-jepa-ttc`  
**Rama observada en el bundle local:** `scientific-recovery-v7-balanced-oof`  
**HEAD trackeado del bundle:** `63f48549c920c429210b6d4f9a962d7b477a2f2e` (`Launch preregistered V7 partial-freeze control`)  
**Naturaleza del documento:** handoff científico + especificación de implementación + protocolo de ejecución y decisión.  
**Objetivo:** cerrar V7 sin reinterpretación retrospectiva, consolidar qué familias experimentales ya han sido probadas, evitar repetir experimentos con otro nombre y definir la siguiente investigación mínima capaz de discriminar hipótesis nuevas.

---

## 0. Resumen ejecutivo

La evidencia acumulada del repositorio permite reducir mucho el espacio experimental.

El resultado más importante no es que “V7 haya fallado”, sino que **varias generaciones independientes del proyecto han convergido sobre el mismo patrón**:

1. Los eventos contienen señal útil para TTC.
2. La geometría física del problema —en particular cambio de escala/altura, expansión y magnitudes equivalentes— es una variable útil.
3. El repositorio ha conseguido en distintos momentos aprender mejor geometría, mejor foreground, mejores ratios, mejores correspondencias o mejor estabilidad.
4. Sin embargo, **mejorar o preservar geometría no ha implicado de forma fiable mejorar TTC fuera de distribución**.
5. Los fallos reaparecen especialmente en dinámica temporal, signo, secuencias/regímenes no vistos y calibración del cambio temporal.
6. V7 vuelve a falsar explicaciones sencillas basadas en:
   - más capacidad;
   - más bins temporales fijos;
   - más escala espacial;
   - distillation geométrica convencional;
   - congelar parcialmente el encoder para “proteger” A4.
7. Por tanto, el siguiente ciclo **no debe ser otra optimización de preservación de A4**.
8. El siguiente ciclo debe empezar con una autopsia mecanística de A5 y un control limpio de representación temporal frente a Garl antes de introducir una arquitectura nueva.
9. Solo si esos diagnósticos lo justifican debe ensayarse **soporte temporal causal adaptativo antes/durante la agregación de eventos**, que es distinto de T20 y C2F.
10. La atribución a JEPA debe hacerse **después de encontrar un downstream competitivo**, comparando scratch / JEPA frozen / JEPA partial fine-tuning sobre la misma arquitectura.

### Decisión principal

> **Cerrar Scientific Recovery V7 como negativo bajo sus gates preregistrados y abrir una fase post-V7 centrada en suficiencia física, representación temporal causal y generalización por régimen; la similitud con A4 pasa a ser una métrica diagnóstica, no una condición universal de campeón.**

Esto **no modifica retroactivamente V7**. El gate geométrico V7 se mantiene tal como fue congelado; simplemente no se hereda automáticamente como axioma científico para V8.

---

# 1. Jerarquía de fuentes y límites de interpretación

Este documento se basa en cuatro niveles de evidencia.

## 1.1. Evidencia local más reciente

Bundle recibido:

`E_JEPA_TTC_REVIEW_20260814_123924.zip`

Contiene:

- identidad Git;
- `TRACKED_HEAD.zip`;
- patches del worktree/index;
- artefactos científicos V5/V6/V7 pequeños;
- agregados y auditorías V7 ignorados por Git.

El bundle es especialmente importante porque los artefactos bajo `artifacts/` no están necesariamente versionados en Git y algunos Markdown trackeados quedaron temporalmente por detrás del estado real de ejecución.

## 1.2. Código y documentación trackeados

El `TRACKED_HEAD.zip` contiene, entre otros:

- `src/e_jepa_ttc/`;
- `scripts/`;
- `configs/`;
- `tests/`;
- `docs/object_event_v4_1.md` … `docs/object_event_v4_31.md`;
- documentos Causal Scale;
- Scientific Recovery V5/V6/V7;
- README/STATUS/CODEX handoffs históricos.

## 1.3. Historia Git

El log local demuestra continuidad al menos entre:

- Scientific Recovery V5;
- V6;
- V7.

No debe asumirse que la vista web de ramas de GitHub es un inventario completo de todas las ramas locales/remotas del usuario: la historia local del bundle es la fuente de identidad exacta para este corte.

## 1.4. Literatura externa

La literatura se usa aquí para **motivar mecanismos que merecen ser probados**, no para trasladar cifras como si fueran directamente comparables al MiD local.

Fuentes primarias relevantes:

- **Garl-TTC / eAP (2026):** https://arxiv.org/html/2603.16303
- **ASTW, CVPR 2026:** https://openaccess.thecvf.com/content/CVPR2026/html/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.html
- **TESPEC, ICCV 2025:** https://openaccess.thecvf.com/content/ICCV2025/html/Mohammadi_TESPEC_Temporally-Enhanced_Self-Supervised_Pretraining_for_Event_Cameras_ICCV_2025_paper.html
- **V-JEPA 2.1 (2026):** https://arxiv.org/abs/2603.14482

Las cifras publicadas de Garl/eAP **no se mezclan** con el protocolo local OOF.

---

# 2. Cierre V7: resultado final que debe quedar documentado

El screen V7 inicial ya estaba cerrado sin candidato.

## 2.1. Baselines V7 reevaluados

Bajo predicción puntual a cobertura completa:

| Modelo | MiD |
|---|---:|
| Garl local | **144.353** |
| A5 revaluado | **158.449** |
| V6.1 | 198.889 |
| A8 | 203.243 |

A5 mantiene, por tanto, un gap local frente a Garl de aproximadamente:

\[
158.449 - 144.353 = 14.096\ \text{MiD}.
\]

La comparación local iguala 8.192 tokens, targets, folds OOF, presupuesto y métrica, pero **no iguala toda la representación temporal, preprocessing, topología ni cobertura contractual**. No debe atribuirse el gap exclusivamente a arquitectura.

## 2.2. Screen inicial V7

| Brazo | MiD puntual | Δ vs A5 | IC95% bootstrap Δ | P(Δ<0) | Geometría |
|---|---:|---:|---:|---:|---|
| SOFT | 165.116 | +6.668 | [+3.084,+10.547] | 0.0004 | falla |
| C2F | 158.573 | +0.125 | [-3.025,+3.432] | 0.4460 | falla |
| T20 | 165.260 | +6.812 | [+2.290,+11.514] | 0.0020 | falla |
| CAP-S | 167.025 | +8.576 | [+3.735,+13.630] | 0.0002 | falla |

Interpretación:

- **SOFT:** la distillation no preserva suficientemente A4 y empeora TTC.
- **C2F:** efecto compatible con cero; no justifica una escalada de radios/routers espaciales.
- **T20:** más resolución temporal **fija** empeora; no implica que una ventana adaptativa sea falsa.
- **CAP-S:** la subida controlada de capacidad empeora; no autoriza cap-M por el árbol V7.

## 2.3. SOFT partial-freeze: resultado final ya disponible

El bundle contiene el agregado final que los Markdown operativos anteriores todavía no habían incorporado.

Archivo:

`artifacts/scientific_recovery_v7/results/soft_partial_freeze_seed7_oof.json`

Estado:

`completed_seed7_oof_gate`

Resultado:

- **MiD macro-secuencia puntual:** `167.8263392527`
- **MiD sample-weighted:** `185.6198969110`
- **finite fraction:** `1.0`
- **failure puntual:** `0%`
- **A5 revaluado:** `158.4485793093`
- **Δ candidato − A5:** `+9.3777599434`

Bootstrap emparejado frente a A5:

- IC95%: **[+5.358820, +13.272338]**
- mediana: **+9.427389**
- `P(Δ<0) = 0.0`
- 5.000 remuestras;
- seed bootstrap `20260813`;
- 422 clusters `sequence_id+track_id`;
- 9 secuencias.

Gates:

- `mechanism_positive = false`
- `geometry_positive = false`
- `confirmation_candidate = false`

Auditoría:

`artifacts/scientific_recovery_v7/audit/soft_partial_freeze_geometry.json`

Retención frente a A4:

| Medida | Retención |
|---|---:|
| bbox std-ratio | 29.43% |
| bbox slope | 19.15% |
| physical std-ratio | 29.36% |
| physical slope | 19.06% |

Gate exigido: **60%**.

### Conclusión

El control de congelación parcial:

1. **no mejora A5**;
2. empeora A5 de forma estadísticamente clara bajo el bootstrap preregistrado;
3. **tampoco recupera la geometría A4**;
4. no autoriza seeds 13/23;
5. no autoriza una ablation JEPA bajo el árbol V7;
6. permite cerrar V7 como resultado negativo.

---

# 3. Qué significa realmente “preservar geometría”

Una causa de confusión del árbol experimental ha sido usar la palabra “geometría” para conceptos distintos.

## 3.1. Geometría físicamente suficiente para TTC

Variables como:

- `log(h_t2/h_t1)`;
- expansión;
- divergence/looming;
- `Δt/TTC`;
- una parametrización equivalente y reversible.

Estas variables sí tienen una relación física directa con TTC.

En la formulación Garl:

\[
TTC = \frac{\Delta t}{1-h_{t1}/h_{t2}}.
\]

Aprender una representación de cambio de altura puede simplificar el mapping desde eventos a TTC.

## 3.2. Geometría espacial rica

Por ejemplo:

- altura;
- anchura;
- centroides;
- foreground;
- boundaries;
- correspondencias;
- deformación/flow local.

Puede ser instrumental para obtener la geometría físicamente suficiente, pero no toda ella es obligatoria para predecir TTC.

## 3.3. Retención o semejanza con A4

El gate V7 pregunta si un descendiente conserva una fracción de slopes/std-ratios de los parents A4 fold-locales.

Eso responde una pregunta legítima:

> ¿Puede el descendiente mejorar TTC sin destruir el mecanismo de A4?

Pero no demuestra:

> Todo buen TTC debe parecerse a A4.

La historia del repositorio aporta evidencia contra convertir esa segunda frase en axioma.

---

# 4. Evidencia histórica: mejorar geometría y mejorar TTC no son equivalentes

Este patrón aparece varias veces y bajo mecanismos distintos.

## 4.1. Object Event V4.22 → V4.24

V4.22 consiguió recuperar señal geométrica transferible mediante supervisión object-centric y partial-unfreeze.

Después V4.23/V4.24 intentaron explotar esa geometría con:

- joint geometry + TTC/LHR;
- geometry-only tail;
- TTC-head-only;
- conservative joint;
- geometry-heavy joint.

El resultado fue una mejora aparente de geometría/OOF que no resolvió la generalización del signo/TTC en development.

## 4.2. Scientific Recovery V5/V6

A8 conserva geometría y obtiene aproximadamente `197.691 MiD`.

A5 unfrozen reduce fuertemente las métricas geométricas frente al parent pero obtiene aproximadamente `155.472` en la lectura V6.

A5 es muchísimo mejor en TTC a pesar de parecer menos al mecanismo A4.

Esto no prueba que “destruir geometría sea bueno”; sí demuestra que:

> **retener A4 no es una condición suficiente ni está demostrado que sea necesaria para TTC competitivo.**

## 4.3. V7-SOFT

SOFT añade explícitamente:

- cosine feature distillation;
- log-height;
- log-width;
- centroides.

No mejora TTC.

## 4.4. V7-SOFT partial-freeze

Congelar `encoder.features[0:3]` tampoco:

- preserva el 60%;
- ni mejora TTC.

### Implicación para V8

A4-retention debe mantenerse como **diagnóstico de representación**, pero no debe convertirse automáticamente en gate universal de promoción.

Los gates futuros sí deben proteger:

- causalidad;
- integridad;
- ausencia de leakage;
- invariancias físicas relevantes;
- generalización OOF;
- finitud;
- robustness.

---

# 5. Mapa de experimentos históricos que NO deben ser redescubiertos

Esta sección es deliberadamente extensa. Su función es servir de **registro antirrepetición**.

---

## 5.1. Pre-Scientific-Recovery / arquitectura histórica

Ya existieron:

- EventTubeletTransformer;
- JEPA temporal;
- dense future-token prediction;
- query pooling;
- causal motion/context conditioning;
- tubelet masking;
- integrated-navigation channels;
- low-label studies.

### Lección

“Usar un transformer de eventos + JEPA” por sí solo **no es una nueva hipótesis**.

Una futura contribución JEPA debe diferir en:

- objetivo;
- escala temporal;
- densidad de supervisión;
- niveles de feature;
- protocolo de atribución.

---

## 5.2. FlowMimic / global flow alignment

Se probaron pérdidas/alineamientos relacionados con flow e inverse-TTC y empeoraron respecto al baseline.

### Cerrado

No repetir:

- global optical-flow mimicry;
- global feature-flow alignment;
- añadir otra ponderación de la misma loss.

### Reabrir solo si

aparece un error localizable que requiera **correspondencia local causal** con evidencia nueva y una parametrización diferente de la ya probada posteriormente en V4.19–V4.30.

---

## 5.3. KDA / Object-KDA / AttnRes / bbox-ROI

La historia ya contiene:

- KDA;
- Object-KDA;
- Attention Residuals;
- bbox/ROI experiments;
- variantes de geometría height/area/affine/event-contrast.

### Cerrado como siguiente paso

No volver a proponer “attention residual”, “ROI conditioning” o KDA como si no hubiesen sido probados.

---

## 5.4. Dense Level–Dynamics / NCE / VISReg

Familia ya probada:

- Level;
- temporal residual;
- NCE;
- NCE+VISReg.

El downstream se mantuvo prácticamente colapsado/empatado alrededor de ~201.8 MiD en esos experimentos históricos.

Micro-overfits demostraron además que memorizar TTC no garantizaba una representación útil.

### Lección

No basta con:

> pretrain SSL → fine-tune → esperar transferencia.

Una futura JEPA debe tener una **atribución causal limpia** contra scratch sobre la misma arquitectura.

---

# 6. Historia Object Event V4.1–V4.31

La siguiente tabla resume la pregunta científica de cada generación y qué evita repetir.

| Versión | Pregunta/intervención | Resultado/lectura | Estado para futuro |
|---|---|---|---|
| V4.1 | Event-only learnability gate | Comprueba si existe señal aprendible | Señal existe; no repetir gate básico |
| V4.2 | Full event-only | Pearson validation ~0.56; shuffle destruye señal | Eventos sí contienen TTC |
| V4.3 | Multiseed robustness | mean Pearson ~0.59; ensemble ~0.62; secuencia negativa frágil | Problema de generalización/signo |
| V4.4 | Hand-crafted geometric residual | Geometry-only débil; híbrido mejora poco | No volver a radial moments simples |
| V4.5 | MiD/reciprocity fine-tuning | mejora mínima; seeds adicionales no sostienen | Cierra loss-only tuning |
| V4.6 | Learned foreground height ratio | LHR aprendido débil; GT visible-height excelente | Variable física buena, extracción event-native mala |
| V4.7 | High-res foreground extent | mejora limitada, insuficiente | Más resolución foreground no basta |
| V4.8 | Dense foreground temporal field | mejora dinámica/signo | Dense temporal field útil, no solución final |
| V4.9 | Fixed two-expert fusion | fusionar expertos ayuda | explorado |
| V4.10 | True-seed fusion robustness | confirma parte de señal, persiste track difícil | explorado |
| V4.11 | Sign/magnitude router | insuficiente | No otro router de signo |
| V4.12 | Reversal directional sign probe | recupera muchos negativos, daña otros regímenes | signo aislado no basta |
| V4.13 | Conservative dual-head | threshold/fusion conservadora | explorado |
| V4.14 | True-seed replication | probabilidades no estables | no threshold tuning |
| V4.15 | Shared odd sign×magnitude | calibración/generalización insuficiente | explorado |
| V4.16 | Causal temporal dual head | mejora negativos y worst-seq, no cierra gap | explorado |
| V4.17 | Signed anchor + bounded residual | persiste prior/shift | explorado |
| V4.18 | Physics bottleneck radial/divergence | correlaciones train no transfieren bien | no static handcrafted bottleneck |
| V4.19 | Dense correspondence/divergence | pequeña señal transferible | correspondencia sí tiene señal, limitada |
| V4.20 | Box pseudoflow | OOF mejora, validation cae | no box pseudoflow convencional |
| V4.21 | Oracle target audit | targets geométricos sí son informativos | el target no era el cuello |
| V4.22 | Geometry aux + partial-unfreeze | mejora geometría transferible | demuestra que se puede recuperar geometría |
| V4.23 | Joint geometry + TTC/LHR | aparece bias TTC | conflicto objetivo/mecanismo |
| V4.24 | Cinco schedules | OOF bueno; dev sign/generalization falla | geometry-heavy schedules agotados |
| V4.25 | Anchored geometry TTC readout | falla | no monotonic readout post-hoc |
| V4.26 | Leak-free OOF geometry residual stack | selecciona anchor original | cierra post-hoc stacking |
| V4.27 | Explicit scale-correlation LHR | viable, insuficiente | matcher explícito ya probado |
| V4.28 | Posterior-supervised multiscale correlation | posterior difuso/subescala | ya probado |
| V4.29 | Local affine correspondence | valid-row fuerte, coverage inválida | no repetir mismo matcher |
| V4.30 | Stable multiscale similarity | resultado autoritativo negativo | familia cerrada |
| V4.31 | Box-conditioned causal audit | estabilidad sí; física/oddness/translation fallan | falsifica promoción del matcher |

---

# 7. Familias posteriores ya exploradas

## 7.1. Causal Scale V5–V8 sintético

### Causal Scale V5

- ideal foreground algebra: funciona;
- learned validation: Pearson ~.956;
- translation leakage falla;
- held-out Pearson ~.921.

### V6

- foreground equivariant/separable arregla translation leakage;
- Pearson sigue ~.920;
- pair-ratio mask supervision no soluciona;
- learned height correction no soluciona;
- frozen residual refinement no soluciona.

### V7

- causal transport mejora validation;
- held-out vuelve aproximadamente a `.920`.

### V8

- multigroup selection;
- temporal consensus;
- CVaR;
- llega aproximadamente a `.94621`;
- gate `.95` sigue fallando.

### Lección

El patrón dominante ya no es simplemente “foreground malo”.

Es:

> **generalización temporal/regime-dependent insuficiente incluso cuando foreground, translation invariance y algebra física parecen razonables.**

---

## 7.2. eAP causal-scale: A0/A1/A3 y derivados

Ya se probaron:

- A0;
- Garl matched;
- A1 geometry-only;
- full-resolution raw foreground;
- deep-feature foreground;
- direct pair-ratio supervision;
- SAM train-only distillation.

Resultados históricos aproximados:

| Brazo | MiD macro |
|---|---:|
| A0 | 382.19 |
| Garl matched temprano | 203.63 |
| A1 geometry-only | 346.83 |
| full-res raw | 380.22 |
| deep-feature | 350.30 |
| pair-ratio | 349.86 |
| SAM A3 | 353.64 |

SAM A3 empeoró A1 y no resolvió la dinámica temporal.

### Consecuencia

No volver a máscaras/foreground RGB salvo que una autopsia nueva demuestre específicamente un error de boundary/extent responsable del gap actual.

---

# 8. Registro de caminos cerrados / “blacklist”

Salvo evidencia nueva explícita, las siguientes familias no deben ser la siguiente intervención.

- FlowMimic/global flow alignment.
- KDA/Object-KDA.
- AttnRes.
- bbox-ROI como solución arquitectónica.
- CARLA transfer como ruta prioritaria.
- Level/NCE/NCE+VISReg convencionales.
- sweeps de pesos TTC/reciprocity.
- radial moments/hand-crafted global geometry.
- mask-edge → height-ratio como solución principal.
- full-resolution raw foreground.
- deep-feature foreground convencional.
- SAM distillation/segmentation teacher.
- direct pair-ratio supervision.
- learned height correction convencional.
- sign classifier/router adicional.
- cambiar thresholds approach/recede.
- reversal sign head convencional.
- odd sign head.
- simple causal sign residual.
- static physics bottleneck.
- box pseudoflow.
- geometry partial-unfreeze como objetivo principal.
- geometry-heavy schedules.
- monotonic geometry readout.
- cross-fitted post-hoc geometry stacking.
- scale-correlation LHR equivalente a V4.27.
- posterior-KL scale matcher equivalente a V4.28.
- local-affine matcher equivalente a V4.29.
- stable multiscale similarity equivalente a V4.30.
- radius 3/4/8 como simple extensión de V6/C2F.
- más bins temporales **fijos** como T30/T40/T50.
- capacidad por sí sola como CAP-M/25M.
- A4 final-feature distillation equivalente a SOFT.
- congelar más/menos bloques como sweep de partial-freeze.
- simple previous-pair inverse-TTC transport.
- CVaR/temporal consensus usados como parche arquitectónico.

## 8.1. Regla de reapertura

Una familia cerrada solo se reabre si se cumplen las tres:

1. Un diagnóstico nuevo identifica un error concreto que esa familia podría corregir.
2. La intervención propuesta **no es isomorfa** a una ya probada.
3. Se preregistra qué evidencia falsaría la reapertura antes de ejecutar.

---

# 9. Lectura SOTA 2026 que sí cambia la siguiente hipótesis

## 9.1. Garl-TTC

Garl demuestra la utilidad de una representación LHR/height-ratio:

- la regresión TTC directa event-only empeora frente a LHR;
- la representación física simplifica el mapping.

La lección útil no es:

> “preserva todas las features geométricas”.

Es:

> **usa un bottleneck o variable intermedia task-sufficient.**

## 9.2. ASTW — Adaptive Spatial-Temporal Window

ASTW critica dos formas comunes de discretizar eventos:

- clock windows fijas;
- fixed-event-count windows.

Propone adaptación espacio-temporal dependiente del régimen/event density.

Esto es especialmente relevante porque **T20 no falsifica ASTW**.

T20 responde:

> ¿ayuda meter más bins dentro de la misma lógica temporal fija?

ASTW responde:

> ¿debería cambiar el soporte temporal antes/durante la representación según el movimiento local?

Son hipótesis distintas.

## 9.3. TESPEC

TESPEC refuerza la idea de que copiar SSL de imágenes y mirar intervalos cortos puede desperdiciar el activo principal de event cameras: su historia temporal.

## 9.4. V-JEPA 2.1

Si más adelante se reabre la atribución JEPA, la dirección moderna relevante es:

- dense predictive loss;
- deep/intermediate supervision;
- preservación de estructura espacial y temporal a varios niveles.

Eso es diferente de una simple cosine distillation de la feature final A4.

---

# 10. Nueva hipótesis de trabajo post-V7

La hipótesis central a poner a prueba será:

> **El gap residual A5→Garl está dominado más por cómo se representa y agrega la evidencia temporal causal —y por cómo esa representación se comporta entre regímenes— que por falta de capacidad, más bins fijos o falta de semejanza con A4.**

No se dará por cierta.

Se intentará falsar en el siguiente orden:

1. **V8-A:** autopsia mecanística sin entrenamiento.
2. **V8-B1:** control de kernel temporal Garl-style manteniendo 3 endpoints y ROI común.
3. **V8-B2:** control de dos endpoints, solo si B1 da señal.
4. **V8-C:** soporte temporal causal adaptativo, solo si A/B lo justifican.
5. **V8-D:** atribución JEPA, solo después de seleccionar downstream.

---

# 11. Fase 0 — cerrar V7 correctamente

Antes de crear V8 debe cerrarse la evidencia V7.

## 11.1. Documentación a actualizar

Actualizar como mínimo:

- `docs/SCIENTIFIC_RECOVERY_V7_STATUS.md`
- `CODEX_HANDOFF.md`
- `STATUS.md`
- `README.md` si contiene banner de estado
- cualquier plan V7 que todavía marque partial-freeze como running.

Registrar:

- MiD `167.826339`;
- delta `+9.377760`;
- IC95% `[+5.358820,+13.272338]`;
- `P(Δ<0)=0`;
- retenciones geométricas ~19–29%;
- gates todos false;
- V7 cerrado negativo;
- seeds 13/23 no ejecutadas;
- JEPA V7 no activado;
- tests externos/CodaBench siguen sellados.

## 11.2. No reinterpretar

No modificar:

- gate geométrico de 60%;
- threshold de candidatura;
- bootstrap;
- folds;
- seed;
- definición de A5.

V7 debe morir bajo el contrato que tenía antes del resultado.

## 11.3. Preservar artefactos

Los siguientes outputs deben conservarse junto al commit/handoff:

- aggregate partial-freeze;
- geometry audit;
- folds summaries/predictions;
- protocolo;
- hashes/artifact SHA.

`artifacts/` está ignorado por Git, por lo que el commit de docs **no basta** para reconstruir la evidencia.

## 11.4. Commit sugerido

No usar `git add .`.

Ejemplo:

```powershell
git status --short

git add `
  docs/SCIENTIFIC_RECOVERY_V7_STATUS.md `
  CODEX_HANDOFF.md `
  STATUS.md `
  README.md

git diff --cached

git commit -m "docs: close Scientific Recovery V7 negative"
```

No hacer push/tag/merge automáticamente salvo instrucción explícita.

---

# 12. V8-A — Mechanism Autopsy

## 12.1. Objetivo

Responder antes de gastar GPU:

> ¿Por qué A5 consigue buen TTC mientras pierde las métricas de geometría A4?

Tres hipótesis mutuamente útiles:

### H1 — Reparametrización física

A5 sí aprende una variable físicamente útil, pero las métricas de retención A4 no la capturan.

### H2 — Shortcut/residual dominance

La rama analítica es débil y el residual/head aprende una corrección supervisada que mejora OOF pero puede generalizar peor.

### H3 — Mixture by regime

A5 usa mecanismo físico en algunos regímenes y residual/transport en otros.

V8-A debe poder discriminar entre estas tres.

---

## 12.2. Qué permite el código actual

`CausalScaleTTCOutput` ya expone:

- `ttc_mean_seconds`
- `inverse_ttc_mean`
- `known_mask`
- `log_height_ratio`
- `pair_log_height_ratio`
- `analytic_log_height_ratio`
- `residual_log_height_ratio`
- `pair_ttc_seconds`
- `pair_inverse_ttc`
- `visible_height_normalized`
- `visible_width_normalized`
- `auxiliary_inverse_ttc`
- diagnostics de foreground;
- diagnostics de transport.

La ecuación central actual es:

```text
analytic = corrected_log_height[t+1] - corrected_log_height[t]
residual = antisymmetric_residual(...)
pair_log_ratio = analytic + residual
pair_inverse_ttc = log_ratio_to_inverse_ttc(pair_log_ratio, delta_t)
final inverse TTC = current pair o causal blend con pair previo
```

Esto significa que **no es necesario cambiar el trainer ni volver a entrenar** para hacer la autopsia.

---

## 12.3. Implementación propuesta

Crear:

`scripts/analyze_scientific_recovery_v8_mechanism.py`

No modificar los scripts V5/V6/V7 canónicos salvo reutilizar helpers importables.

Tomar como plantilla:

`scripts/reevaluate_v7_baselines.py`

### Inputs

- protocolo V7 firmado;
- exactos 8.192 tokens;
- checkpoints A5 fold-locales;
- Garl predictions ya firmadas;
- cache actual;
- outer-dev OOF que ya fue abierto bajo V7;
- geometry audit targets ya autorizados dentro del universo OOF.

### Outputs

Crear:

`artifacts/scientific_recovery_v8/diagnostics/mechanism_autopsy_seed7.json`

y, opcionalmente:

`artifacts/scientific_recovery_v8/diagnostics/mechanism_autopsy_rows.parquet`

El JSON debe incluir hashes del parquet/JSONL row-level.

---

## 12.4. Export row-level obligatorio

Por cada token:

```text
sample_token
sequence_id
track_id
fold
target_ttc_s
target_inverse_ttc
delta_t_s

a5_point_ttc_s
a5_final_inverse_ttc
a5_final_log_ratio
a5_analytic_log_ratio
a5_residual_log_ratio
a5_pair_inverse_ttc_current
a5_pair_inverse_ttc_previous
a5_temporal_blend_used

foreground_height_t0/t1/t2
foreground_width_t0/t1/t2
foreground_fraction_t0/t1/t2
sensor_support
guard_margin

transport_flow_magnitude
transport_margin
transport_entropy
transport_cycle_error
transport_fine_weight si existe

garl_point_ttc_s
```

Añadir targets geométricos **solo como diagnóstico OOF**, nunca como inputs del modelo:

```text
bbox_log_height_ratio
physical_log_ratio
movement_quartile
ttc_bucket
```

No exportar información de test/private/CodaBench.

---

# 13. Contrafactuales V8-A

Para cada fila reconstruir, sin reentrenamiento:

## 13.1. Analytic-only

\[
\hat r_a = \text{analytic\_log\_height\_ratio}
\]

\[
\hat{\tau}^{-1}_a = f(\hat r_a,\Delta t).
\]

## 13.2. Residual-only

No interpretar el residual como una altura física absoluta.

Sí medir:

- magnitud;
- correlación con target inverse-TTC/log-ratio;
- signo;
- cuánto cambia la predicción respecto a analytic-only.

## 13.3. Pair final

\[
\hat r = \hat r_a + \hat r_{res}.
\]

## 13.4. Temporal blend

Comparar:

- current pair only;
- final causal blend.

Esto permite saber si la ganancia A5 está en:

1. foreground/analytic;
2. residual;
3. transport;
4. causal blend.

---

# 14. Métricas V8-A

Calcular global y por estrato:

- Pearson;
- Spearman opcional;
- slope robusta;
- std(pred)/std(target);
- sign accuracy;
- balanced sign;
- MiD;
- RTE;
- MAE log-ratio;
- calibration;
- finite rate.

Estratos:

- `0–3 s`;
- `3–6 s`;
- `>6 s`;
- negative/receding;
- 4 cuartiles de movimiento;
- 4 cuartiles de event density;
- 4 cuartiles de guard margin;
- secuencia;
- track;
- folds.

---

# 15. Tests V8-A

Crear, como mínimo:

`tests/unit/test_scientific_recovery_v8_mechanism.py`

### Obligatorios

1. exact 8192 tokens;
2. no duplicates;
3. token hash = protocolo;
4. no forbidden splits;
5. `analytic + residual == pair_log_ratio` dentro de tolerancia;
6. reconstrucción pair inverse-TTC reproduce output del modelo;
7. current-pair counterfactual reproduce el caso `temporal_blend_used=false`;
8. ninguna cifra hardcodeada desde documentos;
9. Garl predictions enlazadas por `sample_token`, no por posición;
10. output determinista;
11. NaN/non-finite aborta;
12. artifact hash reproducible.

---

# 16. Criterio de decisión V8-A

No hay “campeón” en V8-A.

Es un diagnóstico exploratorio.

### Si H1

La rama analytic/final-pair tiene buena suficiencia física y el residual solo calibra.

**Acción:** priorizar representación temporal/entrada.

### Si H2

La mejora A5 está dominada por residual y la rama física sigue pobre.

**Acción:** no añadir más geometry distillation; investigar qué señal consume el residual y cuánto depende del régimen.

### Si H3

El residual ayuda fuertemente solo en determinados cuartiles/buckets.

**Acción:** diseñar adaptación temporal/mixture basada exclusivamente en observables causales.

### Si no discrimina

No entrenar V8-C todavía; ampliar diagnóstico, no arquitectura.

---

# 17. V8-A2 — gradient-interference audit (opcional, explicativo)

Este diagnóstico puede explicar SOFT y partial-freeze, pero **no es una nueva línea de optimización por sí mismo**.

## 17.1. Pregunta

Medir:

\[
\cos(\nabla L_{TTC}, \nabla L_{dense})
\]

y:

\[
\cos(\nabla L_{TTC}, \nabla L_{geometry})
\]

por bloque.

## 17.2. Implementación

Crear:

`scripts/analyze_scientific_recovery_v8_gradient_conflict.py`

Usar:

- fold train-only;
- subset preregistrado;
- checkpoint/initialization definidos;
- cero optimizer steps durante la medición;
- batches y seed fijos.

## 17.3. Interpretación

- coseno negativo sistemático → objetivos compiten;
- ~0 → aux loss consume capacidad pero no alinea;
- positivo → el problema no es simple interferencia.

### Importante

Un resultado negativo **no autoriza PCGrad automáticamente**.

Primero debe demostrarse que preservar la señal auxiliar es útil para TTC.

---

# 18. V8-B — control de representación temporal Garl-style

Esta es la siguiente intervención entrenada con mayor valor causal.

No debe llamarse “Garl parity exacta” porque se mantendrán deliberadamente partes del contrato A5 para aislar variables.

Se divide en dos pasos.

---

# 19. V8-B1 — `TIMEVOL20-3`

## 19.1. Hipótesis

Parte del gap A5→Garl puede proceder del **kernel de representación temporal de eventos**, no de arquitectura downstream.

Garl local utiliza una representación time-volume de 20 planos por intervalo/endpoint bajo su preprocessing.

A5 usa actualmente tres endpoints con otra codificación:

- polaridad separada;
- bins;
- count;
- rate.

T20 cambió la cantidad de bins, pero mantuvo esa familia de voxelización.

Por tanto B1 cambia **la familia de encoding**, no simplemente el número de bins.

## 19.2. Cambio único

Mantener:

- mismos 8.192 tokens;
- mismos folds;
- mismo ROI común V4;
- mismos endpoints `t0/t1/t2`;
- mismo A5;
- mismos optimizer/LR/epochs/batch;
- mismo transport;
- mismo loss;
- mismo seed 7.

Cambiar solo:

```text
endpoint representation:
V4 polarity voxel + count/rate
        ↓
official-Garl-style 20-plane timevolume
```

Shape objetivo:

```text
[B, 3, 20, 128, 128]
```

Nombre recomendado:

`V8-TIMEVOL20-3`

---

# 20. Implementación de B1

## 20.1. No romper V4

No modificar el contrato `EVENT_V4_STEPS=3` para hacerlo ambiguo.

Crear una extensión V8 aislada.

Archivos recomendados:

```text
src/e_jepa_ttc/data/scientific_recovery_v8.py
scripts/build_scientific_recovery_v8_timevolume_cache.py
configs/protocol/scientific_recovery_v8_temporal.json
configs/experiment/scientific_recovery_v8_fold_chain/
scripts/freeze_scientific_recovery_v8_configs.py
scripts/run_scientific_recovery_v8.ps1
scripts/aggregate_scientific_recovery_v8.py
tests/unit/test_scientific_recovery_v8.py
```

Reutilizar:

`official_timevolume_roi_np(...)`

de:

`src/e_jepa_ttc/data/garl_official_preprocessing.py`

## 20.2. ROI

Para B1 usar **exactamente el common square ya congelado para A5**.

No usar todavía el centering/crop oficial completo de Garl porque entonces cambiarían dos cosas:

- kernel temporal;
- spatial preprocessing.

## 20.3. Cache

Nuevo campo/artefacto:

`event_v8_timevolume20_common_roi`

por token:

```text
[3,20,128,128]
```

float16 almacenado si la equivalencia con float32 se valida.

Manifest debe contener:

- protocol hash;
- source cache identity;
- raw source hashes si aplica;
- sample-token hash;
- function/version string;
- planes=20;
- dtype;
- shape;
- ROI contract;
- no-validation-materialization si el protocolo lo exige.

---

# 21. Tests B1

## 21.1. Equivalencia numérica

Sobre un subset real:

```text
cached_float16 -> float32
vs
official_timevolume_roi_np -> float32
```

definir tolerancia antes de full materialization.

## 21.2. Causalidad

Modificar eventos posteriores al endpoint no debe cambiar el tensor de ese endpoint.

## 21.3. ROI

Verificar que B1 usa exactamente el mismo common square que A5.

## 21.4. Identidad de tokens

8192 exactos; cero faltantes/duplicados.

## 21.5. Inputs prohibidos

El tensorizador no puede consumir:

- TTC;
- category;
- sequence ID como feature;
- track ID como feature;
- future bbox;
- test/private metadata.

La bbox autorizada para construir el **mismo oracle common ROI** debe quedar declarada como privilegio compartido del protocolo, no como input del modelo.

## 21.6. Model smoke

`CausalScaleTTCConfig` debe recibir `event_input_channels=20`.

Verificar:

```text
[B,3,20,128,128] -> output finito
```

---

# 22. Ejecución B1

Una vez implementado y congelado:

```powershell
$env:PYTHONPATH = "src;.."

uv run --no-sync python scripts/freeze_scientific_recovery_v8_configs.py `
  --arm timevol20_3

uv run --no-sync python scripts/build_scientific_recovery_v8_timevolume_cache.py `
  --protocol configs/protocol/scientific_recovery_v8_temporal.json `
  --device cuda

uv run --no-sync pytest -q tests/unit/test_scientific_recovery_v8.py

powershell -ExecutionPolicy Bypass `
  -File scripts/run_scientific_recovery_v8.ps1 `
  -Arm timevol20_3 `
  -Device cuda
```

No ejecutar B2/C antes del agregado firmado de B1.

---

# 23. Gate B1

Usar A5 revaluado `158.448579...` como comparador congelado.

### Integridad obligatoria

- 8192 OOF exactos;
- 3 folds;
- 9 secuencias;
- 100% point predictions finitas;
- hashes/firma;
- no forbidden splits.

### Señal de mecanismo

Para justificar B2:

- mejora point MiD ≥ **3 MiD**, y
- `P(Δ<0) ≥ 0.90`.

No es todavía confirmación externa.

### Si B1 es nulo/negativo

Cerrar la hipótesis:

> “el kernel time-volume de Garl por sí solo explica una fracción material del gap”.

No ejecutar B2 por inercia.

---

# 24. V8-B2 — `PAIR20-2`

Solo se ejecuta si B1 es positivo.

## 24.1. Pregunta

¿Parte adicional del efecto proviene de usar solo el par actual, en vez de tres endpoints con posible causal blend previo?

## 24.2. Cambio respecto a B1

Mantener el mismo cache time-volume 20.

Usar:

```text
[t1,t2]
```

en lugar de:

```text
[t0,t1,t2]
```

Shape:

```text
[B,2,20,128,128]
```

El código del modelo ya acepta `steps >= 2`.

Con dos endpoints:

- `pair_count = 1`;
- no existe pair previo;
- `blend_current_inverse_ttc` queda naturalmente reducido al current pair.

## 24.3. Dataset

No debilitar `GarlTTCObjectEventV4Dataset` para que acepte silenciosamente dos steps.

Crear un adapter/collate V8 que valide explícitamente:

```text
arm=timevol20_3 -> T=3
arm=pair20_2    -> T=2
```

## 24.4. Nombre

`V8-PAIR20-2`

No llamarlo “Garl exact parity” todavía porque:

- ROI common-square sigue controlado por A5;
- backbone/topología siguen siendo A5.

---

# 25. Tests B2

Además de los de B1:

1. T=2 forward finito;
2. `temporal_blend_used == false`;
3. output current-pair coincide con pair único;
4. no se referencia t0 en dataset/collate;
5. delta shape correcta `[B,1]`;
6. reversal test de pair;
7. resume/continuous equivalence.

---

# 26. Gate B2

Comparar contra:

- A5;
- B1.

B2 es informativo si:

- mejora A5 ≥3 MiD con `P>=.90`, o
- mejora B1 de forma pareada y consistente.

Si B1 gana pero B2 no:

> el kernel temporal ayuda, pero eliminar la historia previa no.

Si B2 gana:

> el contrato temporal de par actual parece más adecuado que el blend de tres endpoints para este dominio.

---

# 27. Control opcional B3 — spatial preprocessing

**No ejecutar por defecto.**

Solo si B1/B2 indican una mejora material y aún queda gap frente a Garl.

Entonces puede existir un control separado:

- common ROI A5;
- official-Garl-style centering/crop.

Debe ser una ablation nueva porque cambia el privilegio/preprocessing espacial.

No mezclarla con B1/B2.

---

# 28. V8-C — soporte temporal causal adaptativo

Solo se autoriza si V8-A/B muestran dependencia temporal/regime-dependent suficientemente clara.

## 28.1. Qué NO es

No es:

- T30/T40;
- más bins;
- otro r1/r2 router;
- SSM enorme;
- más parámetros.

## 28.2. Qué sí es

Seleccionar o combinar **duraciones efectivas de historia** antes/durante la representación de eventos.

Conceptualmente:

```text
raw causal events
    -> compact temporal primitives
    -> multiple candidate supports / decays
    -> patch-level causal router
    -> endpoint representation
    -> A5 downstream
```

---

# 29. Diseño compacto recomendado para V8-C

Materializar full voxel tensors para muchas ventanas puede explotar disco/VRAM.

En su lugar usar primitives causales compactos por polaridad:

```text
last_event_age_positive
last_event_age_negative
log1p_count_positive
log1p_count_negative
```

Shape por endpoint aproximada:

```text
[4,H,W]
```

A partir de `last_event_age` pueden derivarse en GPU superficies:

\[
S_k(x,y)=\exp(-age(x,y)/\tau_k)
\]

para varios `τ_k` preregistrados.

Ejemplo inicial — solo si cabe dentro del protocolo congelado:

```text
τ ∈ {5, 20, 80, 320} ms
```

Los valores exactos deben congelarse **antes** del screen; no barrer tras ver OOF.

---

# 30. Router temporal V8-C

## 30.1. Inputs permitidos

Solo observables causales, por ejemplo:

- event density;
- local entropy;
- support;
- count/rate;
- quizá un motion proxy derivado del endpoint/prefix.

## 30.2. Inputs prohibidos

- TTC;
- target log-ratio;
- bbox size;
- TTC bucket;
- sequence ID;
- track ID;
- category;
- outer-dev performance;
- future events.

## 30.3. Salida

Weights por patch sobre las escalas temporales:

\[
w_k(x,y),\quad \sum_k w_k=1.
\]

Representation:

\[
F(x,y)=\sum_k w_k(x,y)S_k(x,y).
\]

Puede conservar polaridad/canales separados.

---

# 31. Tests críticos V8-C

Estos tests son más importantes que aumentar la métrica.

## 31.1. Prefix causality

Añadir eventos en el futuro de un endpoint **no cambia** su representación/predicción.

## 31.2. No-lookahead

El builder debe demostrar por timestamp que:

```text
max(event_timestamp_used) <= endpoint_timestamp
```

para cada endpoint.

## 31.3. Horizon correctness

Para cada τ/window, solo puede acceder a su historia causal permitida.

## 31.4. Zero events

Debe producir:

- representación finita;
- baja/zero support;
- point TTC finito/clipped;
- confidence/known coherente.

## 31.5. Temporal reversal

La representación no tiene por qué ser idéntica al invertir, pero la transformación física debe satisfacer el contrato de oddness/reversal que se preregistre.

## 31.6. Router audit

Guardar distribución de weights por:

- event density;
- movement quartile;
- TTC bucket **solo post-hoc**.

El router no recibe esos labels.

## 31.7. Degenerate support

Ventanas sin eventos y eventos todos en un pixel deben ser casos unitarios explícitos.

---

# 32. Gate V8-C

Misma integridad OOF.

Para candidato seed 7:

- mejora vs A5 ≥3 MiD;
- `P(Δ<0) >= 0.90`;
- 100% point finite;
- 0% point failure;
- ningún split sellado;
- causal tests completos.

### Geometría A4

Reportar:

- slope retention;
- std retention;

pero **no gate universal**.

### Física

Reportar obligatoriamente:

- analytic ratio Pearson/slope/std;
- reversal;
- residual contribution;
- causal support.

Si V8-C mejora MiD destruyendo toda señal física, no rechazarlo automáticamente por A4, pero marcarlo como potencial shortcut y exigir robustness adicional antes de test externo.

---

# 33. Confirmación multiseed

Solo el ganador downstream de B/C se confirma.

Seeds:

```text
7, 13, 23
```

Tres folds cada seed.

No seleccionar un brazo distinto por seed.

## 33.1. Estadística

Bootstrap jerárquico:

1. remuestrear secuencias;
2. dentro de secuencias remuestrear tracks;
3. combinar efecto entre seeds.

Reportar:

- delta medio;
- IC95%;
- per-seed delta;
- between-seed std;
- per-sequence;
- buckets;
- movement quartiles.

## 33.2. Contra Garl

Para afirmar únicamente:

> “candidate supera al checkpoint local Garl seed 7”

basta la comparación congelada apropiada.

Para afirmar:

> “la arquitectura supera Garl de forma robusta a seed”

deben entrenarse/evaluarse Garl seeds 13/23 bajo el mismo protocolo.

---

# 34. V8-D — atribución JEPA

No ejecutar hasta que exista un downstream ganador confirmado.

## 34.1. Por qué

A5/V6/V7 causal-scale no constituyen por sí solos evidencia de una mejora causada por JEPA si el `jepa_objective` está inactivo.

El nombre final del modelo no debe atribuir causalidad a JEPA sin ablation.

## 34.2. Tres brazos obligatorios

Misma arquitectura downstream:

### D0 — scratch

Entrenamiento supervisado desde cero.

### D1 — JEPA frozen

- fold-local train-only pretraining;
- encoder congelado;
- linear/small probe TTC.

### D2 — JEPA partial fine-tune

Mismo pretraining D1;
partial fine-tuning congelado/preregistrado.

No permitir una arquitectura diferente para cada brazo.

---

# 35. JEPA objective recomendado

No replicar SOFT.

La pregunta moderna debe acercarse a:

```text
t0,t1 context
    -> predictor
    -> dense future tokens t2
target encoder EMA
```

Con:

- dense token prediction;
- intermediate/deep supervision;
- collapse diagnostics;
- no TTC/bbox/mask/category labels durante SSL;
- solo train del fold;
- target encoder sin gradiente.

## 35.1. Deep supervision

Inspirado por V-JEPA 2.1, investigar supervisión en más de una profundidad del encoder **como ablation preregistrada**, no como sweep post-hoc.

---

# 36. Gates JEPA

JEPA solo merece atribución si aporta una ventaja respecto a scratch en al menos una dimensión previamente congelada:

- full-label TTC;
- low-label TTC;
- robustness;
- sample efficiency.

Idealmente debe mejorar varias.

Si no mejora:

> describir el modelo como `causal event TTC model`, no como resultado positivo de E-JEPA.

---

# 37. Low-label attribution

Dado que JEPA puede ayudar más por eficiencia de representación que por full-data asymptote, congelar antes:

```text
1%
5%
10%
25%
100%
```

o un subconjunto más pequeño si compute lo exige.

No crear los porcentajes después de ver resultados.

Mismo fold grouping y número de updates efectivo.

---

# 38. Protocolo de integridad común V8

Cada run seleccionable debe demostrar:

1. exact token identity;
2. grouped folds por secuencia/track según protocolo;
3. no train/outer-dev leakage;
4. no private/test/CodaBench;
5. no future event access;
6. deterministic config hash;
7. checkpoint hash;
8. prediction artifact hash;
9. full finite output;
10. resume equivalence;
11. sampler order reproducible;
12. no metric-coded constants en agregador.

---

# 39. Revisión específica de leakage/lookahead

Antes de ejecutar B/C/D, añadir tests que busquen de forma explícita:

## Datos

- timestamp endpoint vs event max timestamp;
- ROI derivado únicamente de información autorizada;
- ninguna columna TTC entra en tensorizador;
- no sequence/category leakage accidental;
- no target normalization calculada con outer-dev.

## Modelo

- router no recibe bucket/target;
- transport no recibe bbox;
- no state carry entre sequences;
- reset entre tracks/sequences;
- no batch-stat contamination si existe BatchNorm.

## Selección

- early stopping solo en fold train/dev permitido;
- no reutilizar el mismo OOF como confirmación independiente;
- seeds adicionales solo después de frozen winner.

---

# 40. Sanity/degenerate cases comunes

Testear explícitamente:

### Caso A — cero eventos

Debe ser finito y con soporte bajo.

### Caso B — ratio físico cero

`log_height_ratio = 0`.

TTC es singular/very-large físicamente.

La implementación debe:

- clip finitamente;
- no crear NaN;
- marcar `known_mask` según contrato.

### Caso C — soporte cero

Mismo requisito de finitud.

### Caso D — temporal reversal

Debe respetar la transformación preregistrada.

### Caso E — duplicated timestamps

No división por cero no controlada.

### Caso F — one-pixel events

No foreground/entropy NaN.

### Caso G — extreme event count

No overflow de count/log1p/timevolume.

---

# 41. Observabilidad durante entrenamiento

No seleccionar por estos diagnósticos salvo que el protocolo lo permita, pero registrarlos:

- analytic vs residual norm;
- fraction residual-dominant;
- foreground height std;
- pair log-ratio std;
- inverse-TTC std;
- event support;
- transport flow/margin/entropy/cycle;
- router temporal entropy (V8-C);
- per-τ weights (V8-C);
- gradient norm por módulo.

Esto facilita detectar otra vez un modelo que “funciona” solo por un shortcut.

---

# 42. Artefactos V8 recomendados

```text
artifacts/scientific_recovery_v8/
  protocol/
    protocol.json
    frozen_manifest.json
  diagnostics/
    mechanism_autopsy_seed7.json
    mechanism_autopsy_rows.parquet
    gradient_conflict.json
  cache/
    timevol20_common_roi/
    adaptive_temporal_primitives/
  runs/
    ...
  results/
    timevol20_3_seed7_oof.json
    pair20_2_seed7_oof.json
    adaptive_support_seed7_oof.json
  audit/
    temporal_causality.json
    physical_mechanism.json
    representation.json
```

Cada JSON final debe incluir:

`artifact_sha256`.

---

# 43. Scripts/archivos que hay que crear

## Obligatorios para V8-A

- `scripts/analyze_scientific_recovery_v8_mechanism.py`
- `tests/unit/test_scientific_recovery_v8_mechanism.py`

## V8-B

- `src/e_jepa_ttc/data/scientific_recovery_v8.py`
- `scripts/build_scientific_recovery_v8_timevolume_cache.py`
- `scripts/freeze_scientific_recovery_v8_configs.py`
- `scripts/run_scientific_recovery_v8.ps1`
- `scripts/aggregate_scientific_recovery_v8.py`
- `configs/protocol/scientific_recovery_v8_temporal.json`
- `configs/experiment/scientific_recovery_v8_fold_chain/*`
- `tests/unit/test_scientific_recovery_v8.py`

## V8-C, condicional

- `src/e_jepa_ttc/data/adaptive_event_support.py`
- `src/e_jepa_ttc/models/adaptive_temporal_support.py`
- cache builder/primitives;
- tests de causalidad.

## V8-D, condicional

- configs JEPA;
- freeze script;
- runner attribution;
- aggregator multiseed/low-label.

---

# 44. Qué código existente conviene reutilizar

No duplicar lógica si ya existe de forma auditada.

Reutilizar como plantilla:

- `scripts/reevaluate_v7_baselines.py`
- `scripts/aggregate_v7_fold_results.py`
- `scripts/audit_v7_fold_geometry.py`
- `src/e_jepa_ttc/training/causal_scale_eap.py`
- `src/e_jepa_ttc/models/causal_scale_ttc.py`
- `src/e_jepa_ttc/data/garl_official_preprocessing.py`
- `src/e_jepa_ttc/data/garlttc_lhr_cache.py`

Pero:

> **no modificar silenciosamente scripts V7 de modo que dejen de regenerar los resultados históricos.**

Si se refactoriza código compartido, añadir regression tests que demuestren bit/equality o tolerancia definida sobre artefactos V7.

---

# 45. Secuencia exacta de trabajo

## Paso 0 — cierre V7

- actualizar docs;
- firmar/preservar artifacts;
- commit de cierre.

## Paso 1 — V8-A

- implementar analyzer;
- tests;
- ejecutar;
- congelar informe;
- decidir H1/H2/H3.

## Paso 2 — congelar protocolo B

Solo entonces:

- crear protocol V8;
- configs;
- manifest;
- source hashes.

## Paso 3 — cache B1

- build;
- audit numérico;
- disk/hash verification.

## Paso 4 — B1 seed 7 OOF

- tres folds;
- aggregate;
- bootstrap;
- decision.

## Paso 5 — B2 solo si B1 pasa

Mismo proceso.

## Paso 6 — C solo si A/B justifican adaptive support

No abrir C automáticamente.

## Paso 7 — multiseed winner

Seeds 13/23.

## Paso 8 — JEPA attribution

Scratch/frozen/partial.

## Paso 9 — sealed evaluation

Solo después de:

- winner congelado;
- commit limpio;
- configs/hash/checkpoints firmados;
- autorización explícita.

---

# 46. Comandos de verificación antes de cualquier run largo

Desde raíz:

```powershell
$env:PYTHONPATH = "src;.."

git status --short --branch
git rev-parse HEAD

uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q
```

Si el repo global tiene deuda lint histórica ya conocida, no “arreglarla toda” dentro de V8.

Ejecutar además lint focalizado de archivos V8.

---

# 47. Smoke gate antes de OOF

Cada brazo nuevo debe pasar:

1. 32 filas reales;
2. 1 epoch o número mínimo de batches;
3. forward/backward;
4. checkpoint;
5. resume;
6. 100% point finite;
7. no OOM;
8. no forbidden source access;
9. deterministic token order.

Un smoke **no cuenta como evidencia científica**.

---

# 48. Overfit sanity

Para cada arquitectura entrenable nueva:

- subset pequeño train-only;
- demostrar que la loss relevante puede bajar;
- comprobar que output tiene varianza no nula;
- comprobar gradients en módulos esperados.

Un overfit que falla = bug/optimización antes de OOF.

Un overfit que pasa = infraestructura, no prueba de generalización.

---

# 49. Stop/go tree

```text
V7 partial-freeze final
        |
        +-- FAIL (ya observado)
              |
              v
          cerrar V7
              |
              v
          V8-A autopsy
              |
      +-------+--------+
      |                |
   temporal         no temporal
   evidence            |
      |                v
      |          investigar causa antes
      |          de entrenar B/C
      v
 V8-B1 TIMEVOL20-3
      |
 +----+----+
 |         |
FAIL      PASS
 |         |
stop B   V8-B2 PAIR20-2
           |
        +--+--+
        |     |
       eval  elegir mejor B
              |
              v
       ¿queda evidencia de
       soporte temporal adaptativo?
              |
          +---+---+
          |       |
         no      sí
          |       |
       confirm   V8-C
          |       |
          +---+---+
              |
        winner multiseed
              |
              v
        V8-D JEPA attribution
              |
              v
       sealed benchmark only
       after explicit authorization
```

---

# 50. Qué NO hacer durante V8

- No abrir public validation/private test/CodaBench para elegir B/C.
- No cambiar gates tras ver seed 7.
- No crear B2 si B1 es claramente negativo.
- No crear C “porque ASTW es SOTA” sin evidencia local.
- No hacer sweep de τ con el mismo OOF.
- No hacer CAP-M.
- No aumentar radius.
- No aumentar fixed bins.
- No reintroducir SAM.
- No hacer PCGrad automáticamente.
- No llamar al downstream “E-JEPA” sin attribution.
- No mezclar cifras del paper Garl con el MiD local.
- No usar `git add .`.
- No borrar resultados negativos.

---

# 51. Claims permitidos tras cerrar V7

Se puede decir:

- Garl es el mejor comparador local bajo el protocolo emparejado actual.
- A5 es el mejor TTC propio actual del árbol limpio, pero pierde gran parte de las métricas geométricas A4.
- V7-SOFT, T20 y CAP-S empeoran A5.
- V7-C2F es compatible con efecto nulo.
- V7 partial-freeze empeora A5 y no restaura el gate geométrico.
- V7 se cierra sin candidato.
- La evidencia histórica no demuestra que retención A4 sea una condición necesaria para TTC competitivo.

No se puede decir:

- SOTA;
- “superamos Garl”;
- JEPA causa la mejora;
- adaptive support funcionará;
- A5 generalizará mejor al test;
- Garl gana solo por su representation kernel;
- “la geometría no importa”.

La frase científicamente correcta es más estrecha:

> **la similitud con la geometría A4 no está demostrada como requisito necesario para TTC; la geometría físicamente suficiente sigue siendo relevante y debe auditarse directamente.**

---

# 52. Qué resultado falsaría la nueva hipótesis temporal

Es importante definir también cómo puede morir V8.

La hipótesis temporal debe considerarse debilitada/negativa si:

1. V8-A no muestra dependencia de error/mecanismo con event density/motion/regime;
2. B1 `TIMEVOL20-3` es nulo o negativo;
3. B2 tampoco aporta señal si B1 lo autorizó;
4. un adaptive-support C causal correctamente implementado no mejora A5;
5. los gains solo existen en un fold/sequence y bootstrap no los sostiene.

En ese caso no se debe seguir escalando ASTW-like modules.

La siguiente investigación debería volver a la autopsia de error y buscar otro eje.

---

# 53. Hipótesis de trabajo final

La mejor hipótesis actual, **no todavía demostrada**, es:

\[
\boxed{
\text{el cuello de botella residual es la estimación y agregación temporal
task-sufficient bajo cambios de régimen, más que la capacidad o la
similitud geométrica con A4}
}
\]

La formulación evita dos errores:

1. No dice que la geometría física no importe.
2. No presupone que adaptive temporal support vaya a funcionar.

La próxima fase está diseñada precisamente para falsarla con el menor coste posible.

---

# 54. Checklist de implementación

## Cierre V7

- [ ] Actualizar `SCIENTIFIC_RECOVERY_V7_STATUS.md`.
- [ ] Actualizar `CODEX_HANDOFF.md`.
- [ ] Actualizar `STATUS.md`.
- [ ] Actualizar banner README si procede.
- [ ] Preservar agregado partial-freeze.
- [ ] Preservar geometry audit.
- [ ] Registrar SHA.
- [ ] Commit específico de cierre.
- [ ] No ejecutar seeds 13/23.

## V8-A

- [ ] Crear analyzer row-level.
- [ ] Exportar analytic/residual/final.
- [ ] Exportar transport diagnostics.
- [ ] Join Garl por token.
- [ ] Estratificar por TTC/motion/event density.
- [ ] Tests identity/equations.
- [ ] Ejecutar sin entrenamiento.
- [ ] Firmar artifact.
- [ ] Dictamen H1/H2/H3.

## V8-B1

- [ ] Implementar timevolume20 common-ROI.
- [ ] Manifest.
- [ ] Equivalencia float32/float16.
- [ ] Prefix causality.
- [ ] Config 20 channels.
- [ ] Smoke GPU.
- [ ] 3 folds seed 7.
- [ ] Aggregate/bootstrap.
- [ ] Stop/go B2.

## V8-B2 condicional

- [ ] Adapter T=2.
- [ ] Sin t0.
- [ ] Sin previous-pair blend.
- [ ] Tests.
- [ ] 3 folds.
- [ ] Aggregate.

## V8-C condicional

- [ ] Causal age/count primitives.
- [ ] τ values preregistrados.
- [ ] Router causal.
- [ ] No labels en router.
- [ ] Future-perturbation test.
- [ ] Degenerate tests.
- [ ] Smoke.
- [ ] OOF.

## Confirmación

- [ ] Frozen winner.
- [ ] seeds 13/23.
- [ ] Hierarchical bootstrap.
- [ ] No reselección por seed.

## JEPA

- [ ] scratch.
- [ ] dense JEPA frozen.
- [ ] dense JEPA partial.
- [ ] low-label si preregistrado.
- [ ] collapse diagnostics.
- [ ] attribution report.

---

# 55. Definition of Done post-V7

La fase post-V7 se considera científicamente completa cuando se cumple una de dos condiciones.

## Resultado positivo

Existe un candidato que:

- pasa integridad;
- mejora A5 con efecto preregistrado;
- se sostiene multiseed;
- no usa lookahead/leakage;
- tiene mecanismos auditados;
- queda congelado antes de cualquier evaluación sellada.

## Resultado negativo

A/B/C se ejecutan según stop/go y ninguna hipótesis temporal produce un candidato.

En ese caso se publica/cierra:

> **la representación temporal evaluada —incluyendo Garl-style timevolume y soporte adaptativo causal bajo este presupuesto— no explica ni cierra el gap A5→Garl.**

Ese resultado también es valioso porque cierra otra familia grande sin contaminar test.

---

# 56. Referencias internas prioritarias

Antes de implementar, leer en este orden:

1. `CODEX_HANDOFF.md`
2. `docs/SCIENTIFIC_RECOVERY_V7_STATUS.md`
3. `docs/SCIENTIFIC_RECOVERY_V6_STATUS.md`
4. `docs/SCIENTIFIC_RECOVERY_V5_STATUS.md`
5. `docs/object_event_v4_21.md`
6. `docs/object_event_v4_22.md`
7. `docs/object_event_v4_23.md`
8. `docs/object_event_v4_24.md`
9. `docs/object_event_v4_26.md`
10. `docs/object_event_v4_27.md`
11. `docs/object_event_v4_29.md`
12. `docs/object_event_v4_30.md`
13. `docs/object_event_v4_31.md`
14. documentos `causal_scale_v5.md` … `causal_scale_v8.md`
15. `src/e_jepa_ttc/models/causal_scale_ttc.py`
16. `src/e_jepa_ttc/training/causal_scale_eap.py`
17. `src/e_jepa_ttc/data/event_v4_geometry.py`
18. `src/e_jepa_ttc/data/object_event_v4.py`
19. `src/e_jepa_ttc/data/garl_official_preprocessing.py`
20. `src/e_jepa_ttc/data/garlttc_lhr_cache.py`
21. `scripts/reevaluate_v7_baselines.py`
22. `scripts/aggregate_v7_fold_results.py`
23. `scripts/audit_v7_fold_geometry.py`

---

# 57. Instrucción final para quien continúe el repo

No empezar implementando una arquitectura.

Primero:

1. cerrar V7;
2. reconstruir exactamente la autopsia A5;
3. identificar qué parte de A5 produce el gain;
4. probar un único control temporal limpio;
5. dejar que el resultado decida el siguiente brazo.

El objetivo post-V7 no es “conseguir una mejora como sea”.

Es:

> **descubrir qué mecanismo explica el gap sin volver a gastar compute en familias que el propio repositorio ya falsó.**
