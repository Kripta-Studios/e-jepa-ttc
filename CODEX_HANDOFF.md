# CODEX HANDOFF: cierre V6 y Scientific Recovery V7/V8

Fecha de corte: 2026-08-14 (Europe/Madrid).

Este archivo es el contrato canónico de continuación. El acta V6 permanece en
[`docs/SCIENTIFIC_RECOVERY_V6_STATUS.md`](docs/SCIENTIFIC_RECOVERY_V6_STATUS.md) y no
se reinterpreta con resultados posteriores.

## 1. Estado Git y límites de trabajo

- Base V6 inmutable:
  `scientific-recovery-v6-oof-diagnostics@28c1efb50622255719f239622ba07858ce704535`.
- Rama V7 creada exactamente desde esa base: `scientific-recovery-v7-balanced-oof`.
- Los worktrees A6 y CUDA robustness siguen intactos.
- La entrega se publica solo en `scientific-recovery-v7-balanced-oof`. No se
  avanzó `main` ni se crearon tags.
- Public validation, private test, EvTTC test y CodaBench no se abrieron para
  selección.
- `artifacts/` contiene salidas ignoradas por Git. Un commit de código no basta para
  recuperarlas: cada resultado aceptado debe conservar su archivo, SHA-256 y firma
  `artifact_sha256`.
- La entrega usa staging explícito; no se usó `git add .`.

La propuesta anterior se archivó, sin reescribir su cuerpo, en
[`docs/archive/E_JEPA_TTC_SCIENTIFIC_RECOVERY_V6_POSTMORTEM_AND_V7_V8_MASTER_PLAN_2026-08-13.md`](docs/archive/E_JEPA_TTC_SCIENTIFIC_RECOVERY_V6_POSTMORTEM_AND_V7_V8_MASTER_PLAN_2026-08-13.md).

## 2. Dictamen científico

La formulación permitida es:

> E-JEPA y Garl pueden compararse por rendimiento bajo una evaluación emparejada,
> pero la diferencia no puede atribuirse exclusivamente a la arquitectura.

La paridad local cubre las mismas 8.192 muestras, targets, particiones OOF,
presupuesto, métrica y privilegio de ROI oracle. No iguala preprocessing,
representación temporal, capacidad, topología ni contrato de cobertura.

Correcciones que no deben volver a perderse:

- El estado `epoch 8 / MiD 185.266` quedó obsoleto. V6.1 F0 terminó en epoch 18
  con `181.351`; el agregado final fue `194.122`, failure `6.689%`, y falló el
  gate.
- Garl usa 40 planos totales: dos intervalos de 20. El modelo causal-scale local
  usa tres pasos de 12 canales, 36 planos totales.
- El Garl local es event-only y declara `with_decoder:false`. No recibió
  supervisión SAM. El foreground del artículo es una hipótesis externa, no una
  causa demostrada del resultado local.
- Garl calcula `Δt / (1-h₀/h₁)`. Su failure de 0% es empírico. Si `h₀=h₁`,
  el denominador es cero; el test local conserva esa singularidad.
- A5 y V6 usan height-ratio, geometría y transporte, pero sus configuraciones
  declaran `jepa_objective:false`. Son brazos causal-scale/transport del árbol
  E-JEPA, no evidencia de un objetivo JEPA activo.
- A3 ya probó distillation SAM train-only y empeoró A1 en `+7.539 MiD`, IC95%
  `[1.553, 10.638]`. V7 no repite foreground/SAM como primera intervención.
- Los tres folds V6 son particiones OOF, no semillas. El bootstrap de 422 clusters
  mide incertidumbre de evaluación condicionada al entrenamiento seed 7; no mide
  variabilidad de optimización.
- Existe un repositorio oficial con benchmarks CodaBench de eAP y GarlTTC. No
  existe un leaderboard unificado entre datasets y protocolos.

## 3. Resultado V6 congelado

| Modelo | MiD | Failure | Interpretación |
|---|---:|---:|---|
| Garl local | 144.353 | 0% | Mejor comparador bajo el protocolo local emparejado |
| A5 causal | 155.472 | 4.761% | Mejor TTC propio; no preserva la geometría |
| V6.1 r2 | 194.122 | 6.689% | Mejora media frente a A8 sin evidencia suficiente |
| A8 r1 | 197.691 | 7.019% | Geometría preservada; TTC restringido |
| A6 | 211.509 | 7.849% | Inferior a A5, A8 y V6.1 |

Estadística emparejada firmada:

- `A5−Garl = +11.119`, IC95% `[4.271, 17.527]`.
- `V6.1−A8 = −3.570`, IC95% `[−8.187, 1.004]` al redondear la lectura
  bootstrap solicitada. El acta V6 conserva `[−8.190, 0.999]` del agregado
  firmado original; la decisión no cambia.
- `A5−V6.1 = −38.650`, IC95% `[−46.050, −31.246]` en la lectura solicitada.
  El agregado V6 registra `[−46.052, −31.250]` por precisión de serialización.

Los buckets `0–3 s` y `3–6 s` explican casi todo el gap A5–Garl. R2 perjudica
los cuartiles bajos de movimiento y mejora los altos; eso apoya adaptación
condicional, no r2 fijo. A5 reduce las slopes aproximadamente de `.163→.027`
frente a bbox y `.269→.041` frente a física.

Fuentes locales:

- `artifacts/scientific_recovery_v6/results/aggregate.json`, artifact
  `ed6da1c77a211870406810d4d6b446d450845b021f211f45d0485b257945e77a`.
- `artifacts/scientific_recovery_v6/diagnostics/a8_oof_failure_modes.json`.
- `artifacts/scientific_recovery_v6/diagnostics/oof_garl_gap.json`, artifact
  `f0ebc082d06571c645c42542d53a39324c22723f346b17dcac2e49d9ae646b9c`.
- `scripts/aggregate_v6_fold_results.py`, `scripts/analyze_v6_oof_garl_gap.py` y
  `scripts/audit_v5_fold_geometry.py` regeneran las lecturas; no hay cifras
  incrustadas en el código de agregación.

## 4. Nueva lectura V7 de los baselines

V7 separa predicción puntual y abstención. El archivo histórico
`prediction_ttc_s` conserva `NaN` cuando `known_mask=false`.
`point_prediction_ttc_s` toma `ttc_mean_seconds` antes del gate y debe ser finito.
La reevaluación está firmada en
`artifacts/scientific_recovery_v7/baselines/manifest.json`, artifact
`de5ff61811f7eb7579797c7131a74b279481b2ac9ed005726885e4d6434378c9`.

| Modelo | MiD puntual, cobertura completa | MiD selectivo histórico | Cobertura selectiva | Failure puntual |
|---|---:|---:|---:|---:|
| Garl | 144.353 | 144.353 | 100.000% | 0% |
| A5 | 158.449 | 155.374 | 95.227% | 0% |
| V6.1 | 198.889 | 194.156 | 93.335% | 0% |
| A8 | 203.243 | 197.620 | 93.005% | 0% |

Esta tabla no reemplaza V6. Usa el mismo universo de tokens y checkpoints, pero
reconstruye la salida puntual bajo el contrato V7. En A5, la diferencia media
entre predicciones selectivas antiguas y reevaluadas es `0.045 s`; siete máscaras
cambian. Cerca de la singularidad del cociente, diferencias numéricas pequeñas
producen cambios grandes de TTC. Los gates V7 usan el A5 revaluado, no cambian el
postmortem V6.

## 5. Contrato V7.0 implementado

El protocolo firmado es
[`configs/protocol/scientific_recovery_v7_balanced_oof.json`](configs/protocol/scientific_recovery_v7_balanced_oof.json),
artifact `7267421c288f6a5e68e779e344dabebb62c9d04df5e100186a8013dfe4a93cf9`.
Congela 8.192 tokens, tres folds, seed 7, 18 epochs, hashes, bootstrap y splits
prohibidos.

El export de `causal_scale_eap.py` produce:

- `prediction_ttc_s`: salida selectiva histórica;
- `point_prediction_ttc_s`: punto finito anterior a `known_mask`;
- `auxiliary_prediction_ttc_s`: diagnóstico, nunca métrica principal;
- `known_mask`, `guard_margin`, log-varianza, varianza, fold, seed y hashes;
- `guard_margin=min(|log_ratio|/0.002, support/0.0001)`.

El agregado informa MiD puntual a cobertura completa, failure Garl, MiD
selectivo, cobertura, curvas riesgo–cobertura en
`[100, 99, 97.5, 95, 90, 80, 70, 50]%`, secuencias, tracks y cuartiles de
movimiento. La salida puntual finita no se cuenta como cobertura selectiva.

## 6. Matriz V7.1

Todos los brazos mantienen muestras, targets, folds, optimizer, LR, 18 epochs,
batch, DINO fold-local, loss TTC y r1 de A5 salvo el cambio declarado.

| Brazo | Cambio único | Pregunta |
|---|---|---|
| V7-SOFT | Distillation desde A4 fold-local congelado | ¿A5 conserva geometría sin inmovilizar el encoder? |
| V7-C2F | Transporte fine-r1/coarse-r2 con router causal | ¿La escala debe depender del régimen? |
| V7-T20 | Diez bins por polaridad, 22 canales por paso | ¿Falta resolución temporal? |
| V7-CAP-S | hidden 96, geometry 192, depth 3, r1 | ¿Una subida controlada a 1.107M ayuda? |

### V7-SOFT

El estudiante A5 parte de cero y permanece entrenable. El A4 del fold se carga
en `eval`, con gradientes desactivados y fuera del optimizer. Solo vio train del
fold y no se usa en outer-dev ni inferencia. Las dos pérdidas tienen peso `1.0`:
distancia coseno de features densas finales y Smooth L1 de log-altura,
log-anchura y centroides. Si falla geometría, solo se permite un control posterior
que congele `encoder.features[0:3]`; no se barren capas ni pesos.

### V7-C2F

`CausalScaleTTCConfig` acepta `transport_mode: legacy | adaptive_pyramid` sin
romper configs antiguas. Calcula r1 en `32×32` y r2 tras average-pooling en
`16×16`; cada escala genera los mismos nueve descriptores físicos. Un router
sigmoide parte con 90% de peso r1. Sus seis entradas son event count, event rate,
flow magnitude, margin, entropy y cycle error actuales. TTC, secuencia, track,
bbox y bucket no entran al router.

### V7-T20

`bins_per_polarity` conserva `5` como default. T20 genera
`[3,22,128,128]`: 10 bins positivos, 10 negativos, count y rate por paso. El
caché train-only tiene 8.192 filas en 32 shards float16, ocupa 17,52 GiB y se
convierte a float32 al cargar. No materializó validation. Manifest artifact:
`dea7974896825d8c633adfd0e96ff3a39f43ad332fc3ce7478e48949e6ec1b6f` en
`artifacts/cache/garl_object_event_common_roi_train8192_t20_v1/manifest.json`.
La materialización aborta si hay menos de 25 GiB libres. Es una ablation de
resolución temporal, no paridad exacta con Garl: mantiene polaridad y tres pasos;
Garl usa dos intervalos sin polaridad. El aumento del primer `Conv2d` queda
separado del control general de capacidad.

### V7-CAP-S

El modelo r1 tiene 1.106.786 parámetros y deriva del patrón cap-S preregistrado.
No reutiliza los runs cap-S/cap-M históricos con r4. Cap-M, alrededor de 2,25M,
solo se autoriza si cap-S mejora al menos 5 MiD y obtiene
`P(Δ<0)≥0.90`. Garl-small queda fuera: el head oficial fija 2.048 features y
reducir ResNet cambia la topología.

## 7. Archivos y ejecución

Implementación principal:

- `scripts/freeze_scientific_recovery_v7_configs.py` congela protocolo y doce
  configs fold/arm.
- `configs/experiment/scientific_recovery_v7_fold_chain/` contiene las doce
  configs seed 7 y su `frozen_manifest.json` firmado.
- `scripts/reevaluate_v7_baselines.py` reconstruye y firma baselines V7.
- `scripts/run_scientific_recovery_v7.ps1` valida firmas y hashes antes de CUDA,
  reanuda runs parciales, se detiene ante corrupción y ejecuta auditoría/agregado.
- `scripts/run_scientific_recovery_v7_soft_partial_freeze.ps1` ejecuta como
  máximo dos folds del control a la vez, deja el tercero en cola y genera su
  auditoría y agregado al completar los tres.
- `scripts/aggregate_v7_fold_results.py` valida el OOF exacto, calcula bootstrap,
  estratos y gates, y firma el resultado.
- `scripts/audit_v7_fold_geometry.py` calcula retención frente a A4 fold-local.
- Resultados ignorados por Git:
  `artifacts/scientific_recovery_v7/{baselines,protocol,results,audit,diagnostics}`.

Comandos desde la raíz:

```powershell
$env:PYTHONPATH = "src;.."
uv run --no-sync python scripts/freeze_scientific_recovery_v7_configs.py
uv run --no-sync python scripts/reevaluate_v7_baselines.py --device cuda
powershell -ExecutionPolicy Bypass -File scripts/run_scientific_recovery_v7.ps1 `
  -Device cuda -SkipBaselines
uv run --no-sync python scripts/freeze_scientific_recovery_v7_configs.py `
  --soft-partial-freeze-control
powershell -ExecutionPolicy Bypass `
  -File scripts/run_scientific_recovery_v7_soft_partial_freeze.ps1 `
  -Device cuda -MaximumParallel 2 -PollSeconds 900
```

El runner no avanza al siguiente brazo si un run falla. Un summary existente con
firma inválida también detiene la cadena. Solo añade `--resume` cuando existe
`state/last.pt`; un directorio nuevo comienza desde cero. Freeze/resume conserva
optimizer, scheduler, RNG y hashes mediante el state del trainer existente.

## 8. Gates seed 7

Integridad exige, a la vez:

1. 8.192 predicciones OOF exactas, sin duplicados ni ausencias;
2. tres folds y nueve secuencias con métricas finitas;
3. 100% de puntos finitos;
4. ningún split prohibido abierto;
5. configs, checkpoints, predicciones y agregado firmados.

`mechanism_positive` exige mejora puntual frente al A5 revaluado de al menos
5 MiD, `P(Δ<0)≥0.90` y pérdida de cobertura selectiva no mayor de 1 pp.

`geometry_positive` exige signo positivo y retención mínima del 60% del A4
fold-local en slope y std-ratio, tanto frente a bbox como frente a física.

Un candidato a confirmación debe pasar integridad y geometría, mejorar A5 al
menos 3 MiD y alcanzar `P(Δ<0)≥0.90`.

Si ningún brazo individual pasa:

1. si SOFT preserva geometría y otro brazo es `mechanism_positive`, se combinan
   solo esos dos cambios y se repiten tres folds seed 7;
2. si SOFT falla geometría, se ejecuta el único control de congelación parcial;
3. si no aparece un candidato Pareto, V7 cierra negativo.

No se añaden foreground, SSM, radios mayores ni una curva de 25M parámetros.
Entre varios candidatos: menor MiD completo, menor MiD `0–3 s`, menor failure,
menor latencia y parámetros. Solo el ganador se repite con seeds 13 y 23.

### 8.1 Resultado del screen inicial

Los doce runs terminaron el 2026-08-14. Cada brazo produjo 8.192 puntos OOF
únicos y finitos, tres folds y nueve secuencias sin abrir splits prohibidos.

| Brazo | MiD puntual | Delta vs A5 | IC95% bootstrap del delta | P(delta<0) | Geometría |
|---|---:|---:|---:|---:|---:|
| SOFT | 165.116 | +6.668 | [+3.084,+10.547] | 0.0004 | falla |
| C2F | 158.573 | +0.125 | [-3.025,+3.432] | 0.4460 | falla |
| T20 | 165.260 | +6.812 | [+2.290,+11.514] | 0.0020 | falla |
| CAP-S | 167.025 | +8.576 | [+3.735,+13.630] | 0.0002 | falla |

Todos tienen `mechanism_positive=false`, `geometry_positive=false` y
`confirmation_candidate=false`. C2F es compatible con efecto nulo; los otros tres
empeoran A5. T20 no autoriza ASTW y CAP-S no autoriza cap-M.

SOFT retiene solo 20.6%/19.8% de las slopes bbox/física y 28.6% de ambos std
ratios. Esto activa el único control previsto: el mismo SOFT con
`encoder.features[0:3]` congelado. Las tres configs y su causa están firmadas en
`soft_partial_freeze_manifest.json`, artefacto
`c41fabd0c4e12220d531bc39748aea40de078a316e14c9eca5b599bd93ecf174`.
No se permite barrer capas ni pesos. El acta detallada está en
`docs/SCIENTIFIC_RECOVERY_V7_STATUS.md`.

### 8.2 Control partial-freeze y cierre V7

El control terminó los tres folds. Obtiene `167.826 MiD` puntual, delta `+9.378`
frente a A5, mediana bootstrap `+9.427`, IC95% `[+5.359,+13.272]` y
`P(delta<0)=0`. Su MiD selectivo es `164.744`, con `94.885%` de cobertura.
Retiene aproximadamente 19% de las slopes y 29% de los std-ratios. Integridad
pasa; mecanismo, geometría y candidatura fallan.

V7 queda cerrado como resultado negativo. No hay ganador, seeds 13/23 ni ablation
JEPA. La geometría densa permanece como requisito para atribuir el mecanismo, no
como condición demostrada para maximizar MiD.

### 8.3 Hipótesis posterior: A5/C2F-MoE

Un diagnóstico no preregistrado encuentra complementariedad A5/C2F: oracle
`133.074 MiD`; router logístico leave-one-fold-out con ocho señales causales
`153.519 MiD`, delta mediana `−4.919`, IC95% `[−7.033,−2.910]` y
`P(delta<0)=1.0`. Mejora los tres folds, pero se diseñó tras observar V7. No es
un candidato confirmado ni puede reetiquetarse como resultado V7.

Una continuación TTC-first debe abrir un protocolo nuevo, mantener fold,
secuencia, track, bbox, TTC y bucket fuera del router, usar stacking anidado y
confirmar con seeds/datos no usados para diseñarlo. La geometría se reportará como
métrica secundaria y seguirá bloqueando claims mecanísticos, no el ranking TTC.

## 9. Atribución JEPA

El ganador causal-scale no se llama E-JEPA final hasta comparar la misma
arquitectura bajo:

1. supervisado desde cero;
2. pretraining JEPA denso sin TTC, bbox, máscaras ni categorías, seguido de
   encoder frozen/linear probe;
3. el mismo pretraining seguido de partial fine-tuning.

El pretraining usa solo train del fold, predice tokens de `t2` desde `t0,t1`,
emplea target encoder EMA y aborta ante colapso. Debe reutilizar
`dense_level_dynamics_jepa.py` y el trainer existente. Si JEPA no mejora TTC,
low-label o robustez, se publica el resultado negativo y el modelo se describe
como `causal-scale event model`.

## 10. V8: gate para superar al Garl local

El ganador confirmado debe cumplir:

- seeds `7, 13, 23`, cada una con tres folds OOF;
- mismos 8.192 tokens, targets, folds y métrica que Garl;
- media `candidate−Garl < 0`;
- IC95% jerárquico completo bajo cero;
- delta puntual negativo en cada seed;
- 100% de puntos finitos y failure puntual 0%;
- retención geométrica mínima del 60%;
- ningún acceso a public validation, private test, EvTTC test o CodaBench
  durante selección.

La inferencia final remuestrea secuencias y, dentro de ellas, tracks; usa 10.000
réplicas y seed congelada; promedia el efecto entre seeds dentro de cada réplica.
También informa un IC compatible por `sequence_id+track_id`, desviación entre
seeds y resultados por secuencia/bucket. Garl seed 7 sigue siendo el comparador
congelado. Para atribuir la diferencia a arquitectura deben entrenarse Garl seeds
13/23; superar un solo checkpoint no basta para ese claim.

Solo después de congelar commit, configs, checkpoints y predicciones se puede
preparar una evaluación CodaBench, con autorización explícita. Ese resultado forma
un track externo separado y nunca se mezcla con el MiD local.

## 11. Evidencia externa y alcance

Hay tres niveles distintos:

1. **Comparador local.** Garl `144.353`, exact-sample y relevante para este repo.
2. **Resultados publicados.** El artículo Garl/eAP informa `79.7→66.2 MiD` al
   pasar de regresión directa event-only a LHR, `45.0 MiD` para RGB+event completo
   y `10.60% RTE / 12.67 ms` en EvTTC sin fine-tuning. No son comparables de forma
   directa con el MiD local. Fuente: [artículo Garl/eAP](https://arxiv.org/html/2603.16303).
3. **Benchmark externo.** El [repositorio oficial Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC)
   enlaza CodaBench de eAP y GarlTTC y mantiene labels de test privados.

Las siguientes fuentes justifican mecanismos, no resultados TTC transferibles:

- [V-JEPA 2.1](https://arxiv.org/html/2603.14482): supervisión predictiva densa y
  profunda para features densas.
- [Event-Aided TTC](https://arxiv.org/abs/2407.07324): refinamiento geométrico
  coarse-to-fine.
- [TMA](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_TMA_Temporal_Motion_Aggregation_for_Event-based_Optical_Flow_ICCV_2023_paper.html):
  agregación temporal para optical flow.
- [ASTW](https://openaccess.thecvf.com/content/CVPR2026/html/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.html):
  ventanas adaptativas en velocidades heterogéneas.
- [TESPEC](https://openaccess.thecvf.com/content/ICCV2025/html/Mohammadi_TESPEC_Temporally-Enhanced_Self-Supervised_Pretraining_for_Event_Cameras_ICCV_2025_paper.html):
  pretraining recurrente de historia larga.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) y
  [CQR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html):
  riesgo–cobertura e intervalos.
- [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html):
  solo si se prueba interferencia sistemática de gradientes.

Foreground/boundary requiere un nuevo diagnóstico de error en bordes. CQR exige
un calibration split separado y no mejora el punto. Recurrencia larga exige fallo
residual en baja densidad o aceleración. ASTW completo exige que T20/C2F confirme
dependencia temporal.

## 12. Verificación y casos límite

Pruebas V7 implementadas:

- default `[3,12,128,128]` y T20 `[3,22,128,128]` finito contra referencia
  float32;
- configs antiguas sin campos V7;
- teacher SOFT congelado, `eval` y sin parámetros entrenables;
- router C2F con seis entradas permitidas e inicio 90% r1;
- separación punto/abstención con todas las filas unknown;
- rechazo de puntos no finitos en riesgo–cobertura;
- singularidad Garl para `h₀=h₁`;
- bootstrap con una sola secuencia falla con mensaje explícito;
- hash de tokens del agregador idéntico al protocolo congelado.

El smoke real de 32 filas y una epoch pasó en la RTX 5070 Ti para SOFT, C2F,
T20 y CAP-S, con 32 puntos finitos por brazo. Su manifiesto tuvo identidad
`560a3453a31b785957d1523364ac93e33662679cfc3e424ae940fe30957417f2`.
La limpieza final retiró ese smoke regenerable. No contiene una métrica
científica ni cambia las configs OOF.

La suite funcional completa previa al cierre pasa, con siete skips. Los archivos
Python V7 modificados pasan Ruff. El comando global
`uv run --no-sync ruff check src scripts tests` encuentra 1.353 incidencias
históricas fuera del cambio V7; no se ocultaron ni se reescribieron de forma
masiva en esta rama.

Comandos de verificación:

```powershell
$env:PYTHONPATH = "src;.."
uv run --no-sync pytest
uv run --no-sync ruff check src scripts tests
uv run --no-sync ruff check `
  src/e_jepa_ttc/data/evttc_object_cache.py `
  src/e_jepa_ttc/data/event_v4_geometry.py `
  src/e_jepa_ttc/data/object_event_v4.py `
  src/e_jepa_ttc/evaluation/selective_ttc.py `
  src/e_jepa_ttc/models/causal_scale_ttc.py `
  src/e_jepa_ttc/training/causal_scale_eap.py `
  scripts/freeze_scientific_recovery_v7_configs.py `
  scripts/reevaluate_v7_baselines.py `
  scripts/aggregate_v7_fold_results.py `
  scripts/audit_v7_fold_geometry.py `
  tests/unit/test_scientific_recovery_v7.py
```

El cierre no añadió nuevos overfits ni una nueva equivalencia resume/continuo.
Estos controles vuelven a ser obligatorios si se abre otro protocolo. Una ventana
sin eventos debe dar un punto finito con baja confianza; `log_height_ratio=0` o
`sensor_support=0` debe dar punto finite/clipped, `known_mask=false` y salida
selectiva `NaN`.

## 13. Claims permitidos y prohibidos

Permitidos ahora:

- Garl es el mejor comparador local seed 7 bajo el protocolo emparejado.
- A5 es el mejor brazo TTC propio de V6 y pierde geometría.
- V6.1 mejora la media de A8 sin evidencia suficiente y falla su gate.
- El screen V7 inicial fue negativo: C2F es compatible con efecto nulo y SOFT,
  T20 y CAP-S empeoran A5 bajo OOF train-only.
- La congelación parcial SOFT es un control preregistrado, no un candidato ni
  evidencia a favor de JEPA.
- V7 completo, incluido partial-freeze, es negativo bajo sus gates congelados.
- El A5/C2F-MoE es una hipótesis post hoc con señal OOF, no confirmación.

Prohibidos ahora:

- `SOTA`, `superamos Garl`, `JEPA causa la mejora` o `Garl garantiza siempre una
  salida`;
- comparar las cifras publicadas de eAP/EvTTC como si fueran el MiD local;
- llamar a T20 paridad exacta con Garl;
- usar el mismo OOF exploratorio como confirmación independiente;
- abrir test o CodaBench antes de congelar un candidato y recibir autorización.

## 14. Estado de ejecución

- [x] Rama V7 creada desde el commit V6 exigido.
- [x] Protocolo, doce configs y caché T20 congelados y firmados.
- [x] Baselines V7 reevaluados sobre 8.192 tokens.
- [x] Contratos SOFT, C2F, T20, CAP-S, export, agregado y auditoría implementados.
- [x] Suite Pytest completa y lint focalizado ejecutados.
- [x] Smokes GPU de los cuatro brazos.
- [x] Doce runs OOF seed 7.
- [x] Cuatro agregados y auditorías seed 7 firmados; todos fallan los gates.
- [x] Control SOFT partial-freeze congelado antes de entrenar.
- [x] Tres folds del control SOFT partial-freeze y agregado firmado.
- [x] V7 cerrado negativo; no existe ganador.
- [x] Seeds 13/23 no ejecutadas por ausencia de ganador.
- [x] Ablation JEPA no ejecutada por ausencia de ganador.

Los resultados del screen constan en agregados firmados bajo
`artifacts/scientific_recovery_v7/results/`; ese directorio está ignorado por Git y
debe preservarse con el commit. El paper Markdown/TeX sigue desactualizado. No se
hicieron tag, avance de `main` ni acceso a test/CodaBench durante selección.

## 15. Retención local tras el cierre

La limpieza del 14 de agosto redujo `artifacts/` de 115,87 GiB a 446 MiB. Conservó:

- `artifacts/scientific_recovery_v5/`, `v6/` y `v7/` con agregados, predicciones y
  auditorías;
- los runs V5–V7 completados con `summary.json`, predicciones y `model_best.pt`
  firmado;
- manifiestos, configuraciones de preprocessing y metadatos de caché.

Se borraron tensores de caché, estados `best/last` usados para reanudar, smokes,
logs de supervisión, runs abortados, screens anteriores y paquetes temporales. Los
datasets fuente permanecen intactos. Los artefactos borrados se pueden regenerar;
Git no los recupera porque estaban ignorados.

El diario fold-local final queda en
`artifacts/scientific_recovery_v7/diagnostics/FINAL_TRAINING_RESULTS.md`. Para
transferir el proyecto hay que entregar el commit y el paquete de evidencia local;
el repositorio Git por sí solo no incluye `artifacts/`.
