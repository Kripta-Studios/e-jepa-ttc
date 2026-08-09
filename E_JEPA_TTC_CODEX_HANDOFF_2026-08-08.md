# E-JEPA-TTC — Codex Research Handoff
## Estado al 8 de agosto de 2026 — rama `scientific-recovery-v3-hardening`

> **Propósito de este documento**
>
> Este archivo es un handoff técnico para un agente Codex que debe continuar el desarrollo del repositorio:
>
> `https://github.com/Kripta-Studios/e-jepa-ttc/tree/scientific-recovery-v3-hardening`
>
> El objetivo de investigación es construir un estimador TTC por objeto basado en cámaras de eventos, con aprendizaje JEPA/representaciones espacio-temporales y geometría explícita, que pueda compararse limpiamente con Garl-TTC bajo el protocolo eAP y, solo si supera las métricas oficiales bajo el mismo protocolo, sostener un claim SOTA.
>
> **No asumir que el repositorio público contiene todos los experimentos descritos aquí.** La rama pública verificada el 8 de agosto de 2026 llega hasta v4.25 (`f26b4ee`). Las secciones históricas v4.26–v4.30 posteriores no son instrucciones pendientes: v4.30 terminó negativamente y v4.31 reemplaza su siguiente paso bajo el contrato descrito abajo.

> **Actualización fix-first, 9 de agosto de 2026 — sustituye el estado pendiente
> v4.28–v4.30 de este handoff.** v4.29/v4.30 ya no son candidatos pendientes:
> v4.30 terminó con SHA
> `9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689` y
> `completed_oof_gate_failed`. Sus controles post-hoc (no preregistrados) sugieren
> correspondencia local rota (`swap` correlación positiva, cero flips), pero no son
> evidencia de un nuevo brazo. v4.31 es solamente una auditoría
> TTC-label-free **box-conditioned**: parquet Garl proyectado, HDF5 eAP relativo,
> ROI común v4.30 t0/t1/t2, cache sanitizada, split train-only y stage2 independiente
> 3×2048. El diagnóstico train-only real de 512 filas terminó el 9 de agosto: la
> estabilidad pasó, pero fallaron equivariancia analítica, pendiente, signo, oddness,
> leakage de traslación y cobertura swap. Stage 2 no se suministró y el manifest
> registra worktree sucio, por lo que el resultado es `not_issued_diagnostic`, no
> seleccionable ni autoritativo. No hay resultado SOTA ni promoción; full queda cerrado.
> Consultar `docs/object_event_v4_31.md`, no las secciones históricas v4.28 de abajo,
> para comandos, logs, roots configurables, batch ≤8, signos y artefactos.

---

# 0. Instrucciones obligatorias para el agente

Antes de modificar nada:

1. Leer completamente:
   - `AGENTS.md`
   - `PLAN.md`
   - `STATUS.md`
   - `README.md`
   - documentación `docs/object_event_v4*.md` que exista en el checkout.
2. Inspeccionar el estado real local:
   ```powershell
   git status --short
   git branch --show-current
   git log -15 --oneline
   git rev-parse HEAD
   git diff --check
   ```
3. **No hacer `git reset --hard`, `git clean`, `git restore .` ni borrar artefactos locales sin necesidad.**
4. No asumir que `origin/scientific-recovery-v3-hardening` coincide con el workspace.
5. No abrir eAP test oficial ni EvTTC para seleccionar arquitectura/hiperparámetros.
6. No hacer split aleatorio por ventana: los folds deben ser **agrupados por secuencia**.
7. No usar boxes, visible heights ni TTC como features de entrada del modelo event-only. Se permiten como **targets de entrenamiento** cuando el experimento lo declare.
8. No inventar resultados ni llamar SOTA a resultados de train/OOF/development.
9. Mantener resultados negativos y fallos importantes.
10. Si aparece NaN, cobertura incompleta, mismatch del loader o un test crítico falla, **detener y diagnosticar**, no continuar silenciosamente.
11. Antes de cualquier claim, congelar config/checkpoints/preprocessing/protocolo y reproducir el benchmark comparable.

Las reglas anteriores son coherentes con `AGENTS.md` y `PLAN.md`: no seleccionar con test, no mezclar secuencias entre splits, no afirmar SOTA sin comparación reproducida y preservar provenance.

---

# 1. Qué significa realmente “batir Garl-TTC”

Hay que separar tres objetivos.

## Track A — eAP test oficial

El plan del repositorio registra para Garl-TTC RGB+eventos sobre el test eAP:

- `MiDc/MiDs/MiDl/MiDn = 53.1 / 37.6 / 40.6 / 31.3`
- weighted MiD publicado ≈ `45.02`
- `RTEc/RTEs/RTEl/RTEn = 16.6 / 20.0 / 34.1 / 28.2`
- `FR = 0` en los cuatro buckets
- `6762` muestras

El GT privado del test no está en el release público. El claim oficial requiere submission/evaluador válido.

**Importante:** Garl-TTC completo es RGB+eventos. Un modelo event-only que lo supere sería un resultado muy fuerte, pero no es una comparación de modalidad idéntica. No ocultar esa diferencia.

## Track B — eAP → EvTTC zero-shot, Tabla VI

El plan registra para Garl-TTC:

- CCRs2-medium: RTE `8.31 %`
- CCRs2-high: RTE `10.56 %`
- CCRm-medium: RTE `12.93 %`
- media: `10.60 %`

Para batirlo de forma limpia:
- entrenar solo con eAP/Garl train público;
- cero fine-tuning/calibración/selección con EvTTC;
- congelar modelo antes;
- usar protocolo/ROI comparable;
- obtener RTE medio `< 10.60 %`.

## Track C — grouped CV / development

Sirve para desarrollar arquitectura y medir robustez, **no prueba SOTA**.

---

# 2. Datos y contrato local

Raíces históricas del equipo:

```text
EAP_ROOT=E:\eAP_dataset
GARLTTC_DATA_ROOT=E:\GarlTTC_dataset
GARLTTC_RELEASE_ROOT=E:\Garl-TTC
```

No hardcodearlas dentro del paquete.

Datos conocidos:
- eAP contiene medios y metadatos de cámara/eventos.
- `GarlTTC_dataset` aporta anotaciones TTC y referencias a medios eAP.
- Se verificó localmente una correspondencia completa para el subconjunto usado:
  - 88,744 entradas
  - 88,744 anotaciones
  - 0 duplicados
  - 0 TTC ausentes
  - 40 secuencias eAP train enlazadas
- TTC es **por objeto/track**, no “un TTC por escena”.
- Hay objetos con TTC negativo (alejamiento), y esta clase minoritaria ha sido uno de los principales failure modes.

El cache Object Event usado por v4.27/v4.28 es event-only y utiliza tres estados temporales `t0/t1/t2`.

---

# 3. Evolución de la última semana

## 3.1 2 de agosto — Level-Dynamics JEPA y hardening del piloto

En la rama pública aparecen, entre otros:

- `8e60e8e` — Implement dense level-dynamics JEPA core
- `7d89444` — Harden matched Level-Dynamics pilot execution
- `2780184` — Allow canonical GarlTTC dataset root
- `1b7279b` — Freeze matched eAP Level-Dynamics subsets
- `41f6e45` — Restore CUDA RNG state safely on resume
- commits de límites de workers/RAM y refreeze del subset
- `9a5cc5c` — Seed matched JEPA initialization deterministically
- `6e4ad4b` — Bind matched subset to deterministic trainer

Se construyó un subset eAP/Garl determinista y balanceado para investigar si JEPA aprendía señal transferible.

Arms principales:
- `level`
- `temporal_residual`
- `nce`
- `nce_visreg`

Todos se entrenaron ~1000 updates.

Resultados de pérdidas:
- `level`: ~1.22 → ~0
- `temporal_residual`: ~2.17 → ~0.91
- `nce`: ~2.55 → ~1.33
- `nce_visreg`: ~2.58 → ~1.35

La caída de loss **no bastó** para demostrar TTC transferible.

### Diagnóstico de colapso/readout

Audit de 2048/pequeños subsets mostró que:
- scratch y `level` producían `pred_std` extremadamente pequeña;
- la representación contenía señal débil pero el readout colapsaba hacia casi constante;
- `query pooling` podía sobreajustar un subset y alcanzar Pearson alto (~0.8+), mientras `mean pooling` quedaba mucho peor.

Conclusión:
> existe información temporal utilizable, pero el readout/objetivo inicial no la estaba transformando en TTC robusto.

---

## 3.2 5–6 de agosto — Object Event TTC v4 temprano y baseline v4.10

La línea Object Event v4 se diseñó para eliminar atajos falsados por v3:

```text
eventos t0/t1/t2
  -> ROI temporal común
  -> canales event activos
  -> encoder event-only
  -> tokens/local temporal geometry
  -> TTC firmado
```

Contrato:
- sin RGB en el modelo principal;
- sin boxes/motion embedding como shortcut forward;
- evento debe tener dependencia observable;
- TTC firmado, incluyendo negativos.

Commits públicos relevantes:
- `e16e6e7` — Record Object Event TTC v4 failed event-learning screen
- `1d11b3e` — Added v4.2
- `488b433` — Added v4.3
- `b84dcfc` — Checkpoint Object Event TTC v4.10 true-seed pipeline

### Baseline v4.10 que se mantiene como ancla de development-validation

Métricas conocidas:

```text
count                         2048
negative_count                 335
positive_count                1713

Pearson                     0.676004
expansion MAE               0.014938
prediction std              0.019490
target std                  0.025213

positive accuracy           0.891419
negative accuracy           0.638806
balanced sign accuracy      0.765112

minimum-sequence Pearson    0.557129
minimum-sequence neg acc    0.107143

weighted MiD               186.308
weighted RTE               433.007 %
```

El baseline no es SOTA, pero es el benchmark interno estable que las nuevas arquitecturas deben superar antes de gastar evaluations más sensibles.

---

# 4. 7 de agosto — v4.15 a v4.25: aislar la física que sí transfiere

## v4.15 / v4.16 / v4.17 — signo y temporalidad

Se comprobó:
- existe señal direccional;
- se puede reparar parcialmente el signo;
- heads temporales/ajustes directos generalizan mal entre secuencias;
- algunos métodos corrigen prior pero el residual sigue sequence-specific.

Esto llevó a abandonar la idea de que bastaba un head TTC más flexible.

---

## v4.18 — geometría foreground agregada

La geometría agregada simple resultó prácticamente inútil.

Conclusión:
> no basta con resumir foreground/extent; hay que conservar correspondencias espaciales/temporales.

---

## v4.19 — correspondencia densa

Fue uno de los primeros indicios claros de transferencia:

```text
validation divergence Pearson ≈ 0.250
```

Mostró que la correspondencia densa event-native contenía física útil.

---

## v4.20 — decoder de pseudoflow basado en boxes

Intentó aprender un pseudoflow affine derivado de boxes.

Resultado:
- train/OOF podía mejorar;
- validation empeoró respecto a v4.19;
- señal muy variable por secuencia.

Ejemplo histórico:
- `DGqicHUGWb` ~0.025
- `pBqGOb2vYq` ~0.231
- `qoohcdtLDH` ~0.443

Pregunta que surgió:
> ¿el target box-pseudoflow es físicamente válido o estamos enseñando ruido de truncamiento/lateralidad?

---

## v4.21 — auditoría oracle del target geométrico

No entrenó el modelo. Midió directamente si geometría derivada de boxes se relacionaba con TTC.

Hallazgos principales:

```text
oracle box divergence validation Pearson   ≈ 0.670
oracle box height-ratio validation Pearson ≈ 0.760
```

Conclusión muy importante:
> el target de escala/altura contiene mucha información TTC. El problema no es que la física no exista; el modelo todavía no sabe extraerla bien desde eventos.

Esto justificó mover supervisión geométrica dentro del encoder.

Commits públicos:
- `2194770` — Audit object-centric box pseudoflow target v4.21
- `5354c4f` — Add object-centric geometry supervision v4.21-v4.22

---

## v4.22 — partial unfreeze del geometry encoder

Se desbloquearon solo las últimas partes del encoder con regularización de drift.

Resultados:

```text
v4.19 frozen divergence       ≈ 0.250
v4.20 decoder/frozen          ≈ 0.183
v4.22 partial-unfreeze        ≈ 0.309
v4.22 vertical scale          ≈ 0.299
```

Por secuencia, vertical scale mantuvo Pearson positivo:
- ~0.272
- ~0.342
- ~0.256

Drift del encoder:
- ~0.5 %

Conclusión:
> mover supervisión geométrica al encoder funciona y no requiere destruir la representación.

Pero seguía un gap grande frente al oracle height ratio (~0.760).

---

## v4.23 — joint geometry + TTC

Commit público:
- `d558bea` — Integrate joint geometry TTC fine-tuning v4.23

Mejoró geometría y algunos errores:

```text
divergence         ~0.309 -> ~0.339
vertical scale     ~0.299 -> ~0.331
expansion MAE      ~0.01494 -> ~0.01416
weighted MiD       ~186.31 -> ~175.11
weighted RTE       ~433 % -> ~232 %
min-seq Pearson    ~0.557 -> ~0.593
```

Pero apareció un sesgo de signo:
- positive accuracy subió ~0.964
- negative accuracy cayó ~0.445

Train llegaba casi perfecto mientras validation fallaba.

Conclusión:
> la geometría mejora, pero el fine-tuning TTC sobreajusta el prior/signo de las 9 secuencias train.

---

## v4.24 — orchestrator de schedules train-only

Commit público:
- `e0ae800` — Add train-only geometry TTC orchestrator v4.24

Se dejó de ejecutar un probe minúsculo cada vez.

Un solo orchestrator hizo 36 entrenamientos:
- 5 arms
- grouped folds por secuencia
- stage 1 + confirmación multiseed
- champion full-train

Arms:
1. `v423_control`
2. `geometry_only_regularized`
3. `motion_only_control`
4. `joint_conservative`
5. `joint_geometry_heavy`

Ganó:
```text
geometry_only_regularized
```

pero casi empatado con `joint_geometry_heavy`.

OOF train era muy alto, pero validation seguía peor que v4.10:
```text
Pearson             ~0.635
negative accuracy   ~0.513
worst seq neg acc   ~0.071
```

Mientras la geometría siguió mejorando:
```text
divergence       ~0.358
vertical scale   ~0.340
```

Conclusión:
> tampoco se puede culpar únicamente al motion head. La geometría transfiere mejor que el TTC final.

---

## v4.25 — anchored geometry-conditioned readout

Commit público:
- `f26b4ee` — Add train-only geometry TTC orchestrator v4.25

Intentó:

```text
prediction =
  a * baseline
+ b * divergence
+ c * vertical_scale
```

con coeficientes no negativos, bias cero y anchor hacia `(1,0,0)`.

Resultado:
```text
selected_readout = baseline_control
coefficients = [1.0]
validation Pearson = 0.676004
```

Es decir, no cambió el baseline.

### Problema metodológico descubierto después

La geometría usada para el meta-readout era OOF, pero la predicción baseline de train procedía de un predictor evaluado sobre su propio train y alcanzaba ~0.978 Pearson.

Comparación injusta:

```text
baseline train in-sample    ~0.978 Pearson
geometry OOF                held-out real
```

Por definición, la meta-regresión tendía a ignorar la geometría.

Además la recomendación final tenía un bug lógico:
si se elegía `baseline_control`, sus métricas eran idénticas al baseline y las comparaciones `>=` lo marcaban erróneamente como “supported”.

**No interpretar v4.25 como prueba de que geometría no sirve.**

---

# 5. Trabajo local posterior a la rama pública

## v4.26 — leak-free OOF residual stack

Objetivo:
corregir el defecto metodológico de v4.25.

Se reconstruyeron en los mismos grouped folds:
- TTC anchor OOF
- divergence OOF
- vertical scale OOF

Y se probó:

```text
prediction =
anchor
+ c_div * residual_div
+ c_vertical * residual_vertical
```

con anchor fijado a 1 y coeficientes geométricos no negativos.

### Resultado

La selección OOF siguió prefiriendo:
```text
anchor_control
```

Mejor residual:
```text
residual_vert_r1e4
```

pero la mejora de objective fue ~`0.000042`, sin mejora real de sign accuracy.

OOF:
```text
Pearson                ~0.958
negative accuracy      ~0.988
balanced sign          ~0.994
```

Full-train → validation:
```text
Pearson                ~0.635
negative accuracy      ~0.513
balanced sign          ~0.727
```

Por secuencia:
```text
DGqicHUGWb    Pearson ~0.560   neg acc ~0.071
pBqGOb2vYq    Pearson ~0.665   neg acc ~0.508
qoohcdtLDH    Pearson ~0.693   neg acc ~0.587
```

Se identificó un failure mode especialmente claro:

```text
track DGqicHUGWb_000778
28/28 samples con target negativo
baseline/anchor interpreta sistemáticamente mal el movimiento
```

Conclusión:
> no seguir con readouts lineales post-hoc. El problema está en cómo la representación aprende la variable física TTC/scale.

---

# 6. v4.27 — Scale-Correlation LHR event-native

Objetivo:
dejar de pedir a un CNN/head que “invente” `log_eta` y estimarlo por correspondencia física de escala.

Concepto:

```text
eventos t1                  eventos t2
   -> geometry features        -> geometry features
          \                      /
           correlation en 45 escalas
                    |
                soft-argmax
                    |
          log(h_prev / h_curr)
                    |
            g = 1 - exp(log_eta)
                    |
                   TTC
```

La inferencia sigue siendo event-only.
Visible heights/boxes son training-only supervision.

## v4.27 primer intento — fallo numérico

El primer run produjo NaNs desde epoch 1.

Se localizó una singularidad de backward:

```python
sqrt(warped_weight * curr_weight)
```

`grid_sample(... padding_mode="zeros")` produce ceros exactos y `sqrt(0)` tiene gradiente infinito.

No fue un resultado científico.

## v4.27.1 — hotfix

Se sustituyó el weighting por una forma sin `sqrt(0)` singular y se añadieron guards:
- `isfinite` en inputs/outputs/losses;
- grad norm finito;
- `error_if_nonfinite=True`;
- comprobación de parámetros después de optimizer.

Después del hotfix el entrenamiento fue estable.

---

# 7. Resultado científico v4.27

Se ejecutaron:
```text
3 seeds × 3 grouped folds × 8 epochs
```

Tiempo:
```text
3408.56 s ≈ 56.8 min
```

Todos los folds mostraron descensos estables de:
- total loss
- LHR loss
- correlation loss
- sign loss

Ejemplo seed 7/fold 1:

```text
epoch 1:
loss 1.5100
lhr  0.0239
corr 0.7237
sign 0.6466

epoch 8:
loss 0.4956
lhr  0.0111
corr 0.1362
sign 0.2851
```

No hay colapso numérico.

## OOF final

```text
count                              2048
negative                           569
positive                          1479

Pearson                        0.602651
expansion MAE                  0.019120
prediction std                 0.019506
target std                     0.032042

positive accuracy              0.872211
negative accuracy              0.652021
balanced sign                  0.762116

minimum sequence neg accuracy  0.506667
minimum sequence Pearson       0.405639

log_eta Pearson                0.592408
log_eta MAE                    0.019094
min sequence log_eta Pearson   0.419539
```

Track metrics:

```text
track_count                     422
eligible_track_count             93
negative_track_count              43
track_macro_pearson           0.545858
minimum_track_pearson        -0.562723
negative_track_macro_acc      0.674419
minimum_negative_track_acc    0.25
```

Gates v4.27:

```text
Pearson                       >= 0.65      FAIL (0.6027)
negative accuracy             >= 0.65      PASS (0.6520)
balanced sign                 >= 0.78      near, FAIL (0.7621)
log_eta Pearson               >= 0.60      near, FAIL (0.5924)
negative-track macro acc      >= 0.70      near, FAIL (0.6744)
```

Resultado:
```text
status = completed_oof_gate_failed
```

**Development-validation no se abrió.**
eAP official test no se abrió.
EvTTC no se abrió.

Esto debe conservarse.

---

# 8. Interpretación de v4.27

v4.27 no fue un fracaso binario.

Hallazgos:

1. El matcher LHR event-native **sí aprende señal física**.
2. Negative accuracy ya supera el gate de 0.65.
3. `log_eta Pearson ~0.592` está cerca de 0.60.
4. El problema importante está en:
   - Pearson global insuficiente;
   - heterogeneidad por secuencia/track;
   - magnitud comprimida.

## Compresión de magnitud

OOF:

```text
prediction std = 0.0195
target std     = 0.0320
ratio          ≈ 0.61
```

El modelo reproduce solo ~61 % de la dispersión física.

En expansiones grandes, tiende a predecir valores demasiado cerca de cero.

Interpretación:
> suele identificar razonablemente el signo, pero no cuánto looming/receding existe.

## Posterior demasiado difuso

La distribución de correlación sobre escalas mantiene entropía normalizada cercana a `0.95`, donde `1.0` sería casi uniforme.

El soft-argmax sobre una distribución muy plana se encoge hacia el centro:

```text
log_eta -> 0
expansion -> 0
```

Esto explica parte del MiD/RTE pobre.

## Limitación del matcher 1-D

Usar principalmente escala vertical no modela bien:
- movimiento lateral;
- cambio de pose;
- rotación aparente;
- truncamiento;
- desplazamiento del centro;
- edges de eventos distintos entre frames.

Por eso no se debe:
- bajar gates post-hoc;
- abrir development manualmente;
- entrenar simplemente 30 épocas más;
- hacer otro sweep trivial de ridge/readout.

---

# 9. Siguiente experimento: v4.28

## Nombre

**Object Event v4.28 — posterior-supervised multiscale event correlation**

## Hipótesis

V4.28 prueba dos explicaciones de v4.27 en un solo orchestrator.

### Arm A — `profile_posterior`

Mantiene el matcher vertical de v4.27, pero cambia la supervisión.

En lugar de supervisar solo el valor esperado `log_eta`, crea un target gaussiano sobre el grid físico de escalas:

```text
q(scale_i) ∝ exp(
  -(log(scale_i) - log_eta_GT)^2 / (2*sigma^2)
)
```

y optimiza KL:

```text
KL(q || p_model)
```

Objetivo:
> obligar al correlation volume a localizar la escala verdadera y evitar un posterior casi uniforme cuyo soft-argmax colapse hacia cero.

Configuración principal:
```yaml
matcher: profile
correlation_dim: 48
log_scale_min: -0.22
log_scale_max: 0.22
scale_bins: 45
correlation_temperature: 0.040
rotation_degrees: [0.0]
pyramid_factors: [1]
batch_size: 8
```

### Arm B — `spatial_rotation_posterior`

Conserva features espaciales 2-D.

Busca:
- escala;
- pequeñas rotaciones nuisance;
- varios niveles espaciales.

Configuración:

```yaml
matcher: spatial_rotation
correlation_dim: 32
log_scale_min: -0.22
log_scale_max: 0.22
scale_bins: 37
correlation_temperature: 0.045
rotation_degrees: [-6, -3, 0, 3, 6]
pyramid_factors: [1, 2]
batch_size: 6
```

Usa los tres estados temporales y la estructura temporal ya presente en v4.8.

La rotación se marginaliza; TTC sigue dependiendo de la escala.

## Loss v4.28

```yaml
lhr_weight: 4.0
expansion_weight: 1.0
correlation_weight: 1.0
sign_weight: 1.0
posterior_weight: 0.75
entropy_weight: 0.0
smooth_l1_beta: 0.004
sign_temperature: 0.015
posterior_sigma: 0.015
max_abs_expansion: 0.25
```

Nota:
- `entropy_weight` se pone a 0 porque la KL al posterior físico ya controla la distribución.
- no introducir simultáneamente muchas penalizaciones difíciles de interpretar.

## Training

```yaml
fold_count: 3
seeds: [7, 13, 23]
epochs: 10
final_epochs: 12
geometry_tail_tensors: 8
projection_learning_rate: 1e-4
geometry_learning_rate: 1e-5
weight_decay: 1e-4
geometry_anchor_weight: 0.01
max_grad_norm: 1.0
```

OOF total:
```text
2 arms × 3 seeds × 3 folds = 18 trainings
```

Solo el campeón OOF puede abrir development-validation.

---

# 10. Gates v4.28

El campeón debe superar gates absolutos:

```text
OOF Pearson                         >= 0.635
OOF negative accuracy               >= 0.650
OOF balanced sign                   >= 0.775
OOF log_eta Pearson                 >= 0.615
OOF negative-track macro accuracy   >= 0.690
OOF minimum sequence Pearson        >= 0.430
prediction_std / target_std         >= 0.70
```

Y mejoras sobre v4.27:

```text
Pearson gain             >= +0.025
log_eta Pearson gain     >= +0.015
negative-track gain      >= +0.010
```

Esto impide declarar victoria por un cambio trivial.

Si el gate falla:
- no abrir development-validation;
- no bajar gates;
- no ejecutar EvTTC;
- no ejecutar eAP test.

Si pasa:
- elegir campeón solo con OOF;
- entrenar full-train seeds 7/13/23;
- evaluar development-validation una sola vez;
- comparar contra v4.10.

Decision final contra v4.10:

```text
Pearson gain                 >= +0.005
negative accuracy gain       >= +0.020
balanced sign gain           >= +0.010
log_eta Pearson              >= 0.45
negative-track macro gain    >= +0.020
```

---

# 11. Patch v4.28 pendiente

Archivo esperado en la raíz del repo:

```text
e_jepa_ttc_object_event_v4_28_multiscale_posterior.patch
```

SHA-256:

```text
858814b4bd513646ff21380c8e75374c75ff51a09e6451cc0b27ff97a364eac9
```

El patch añade exactamente:

```text
configs/experiment/e_jepa_garl_object_event_multiscale_posterior_v4_28.yaml
docs/object_event_v4_28.md
scripts/analyze_object_event_v4_28_multiscale_posterior.py
scripts/preflight_object_event_v4_28.py
scripts/run_object_event_v4_28_multiscale_posterior.ps1
src/e_jepa_ttc/models/object_event_v4_28.py
src/e_jepa_ttc/training/object_event_v4_28.py
tests/unit/test_object_event_v4_28.py
```

No contiene `.pyc` ni `__pycache__`.

---

# 12. Comandos exactos para continuar

## 12.1 Verificar estado

```powershell
git status --short
git branch --show-current
git log -15 --oneline
git rev-parse HEAD
git diff --check
```

**No asumir HEAD.**

La rama pública verificada al redactar este handoff llega a:
```text
f26b4ee Add train-only geometry TTC orchestrator v4.25
```

El workspace local puede estar por delante con v4.26/v4.27.

---

## 12.2 Si v4.27/v4.27.1 aún no está commiteado

Revisar primero el diff. Si corresponde exactamente al trabajo descrito y los tests pasan, guardar el hito:

```powershell
git add -- `
  configs/experiment/e_jepa_garl_object_event_scale_correlation_lhr_v4_27.yaml `
  docs/object_event_v4_27.md `
  scripts/analyze_object_event_v4_27_scale_correlation_lhr.py `
  scripts/preflight_object_event_v4_27.py `
  scripts/run_object_event_v4_27_scale_correlation_lhr.ps1 `
  src/e_jepa_ttc/models/object_event_v4_27.py `
  src/e_jepa_ttc/training/object_event_v4_27.py `
  tests/unit/test_object_event_v4_27.py

git diff --cached --check
git diff --cached --stat
git status --short
```

Si todo es correcto:

```powershell
git commit -m "Record scale-correlation LHR OOF result v4.27"
git push origin scientific-recovery-v3-hardening
git log -1 --oneline
```

**No incluir**:
```text
*.zip
*.patch
__pycache__
*.pyc
```

Si ya está commiteado, no repetir.

---

## 12.3 Verificar SHA de v4.28

```powershell
$Patch = "e_jepa_ttc_object_event_v4_28_multiscale_posterior.patch"
$ExpectedSHA = "858814b4bd513646ff21380c8e75374c75ff51a09e6451cc0b27ff97a364eac9"

$ActualSHA = (Get-FileHash $Patch -Algorithm SHA256).Hash.ToLower()

"Esperado: $ExpectedSHA"
"Obtenido: $ActualSHA"

if ($ActualSHA -ne $ExpectedSHA) {
    throw "El SHA-256 de v4.28 no coincide"
}
```

---

## 12.4 Aplicar

```powershell
git apply --check $Patch
```

Solo si pasa:

```powershell
git apply $Patch

git diff --check
git status --short
```

No hacer commit todavía.

---

## 12.5 Compilar

```powershell
uv run --no-sync python -m py_compile `
  src\e_jepa_ttc\models\object_event_v4_28.py `
  src\e_jepa_ttc\training\object_event_v4_28.py `
  scripts\preflight_object_event_v4_28.py `
  scripts\analyze_object_event_v4_28_multiscale_posterior.py `
  tests\unit\test_object_event_v4_28.py
```

---

## 12.6 Tests unitarios

```powershell
uv run --no-sync pytest -q `
  tests\unit\test_object_event_v4_28.py
```

Esperado:
```text
......                                                                   [100%]
6 passed
```

Si no pasan, arreglar antes de ejecutar GPU.

---

## 12.7 Validar PowerShell

```powershell
$Tokens = $null
$ParseErrors = $null

[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path "scripts\run_object_event_v4_28_multiscale_posterior.ps1"),
    [ref]$Tokens,
    [ref]$ParseErrors
) | Out-Null

if ($ParseErrors.Count -gt 0) {
    $ParseErrors | Format-List
    throw "El wrapper PowerShell v4.28 contiene errores"
}

"PowerShell v4.28: OK"
```

---

## 12.8 Ejecutar

Primera ejecución:

```powershell
& .\scripts\run_object_event_v4_28_multiscale_posterior.ps1 `
  -Device "cuda"

$V428ExitCode = $LASTEXITCODE
"Exit code v4.28: $V428ExitCode"
```

No usar `-Force` si el output todavía no existe.

Si existe un output parcial/fallido que se quiere reemplazar, solo entonces:

```powershell
& .\scripts\run_object_event_v4_28_multiscale_posterior.ps1 `
  -Device "cuda" `
  -Force
```

Output:
```text
artifacts\debug\object_event_v4_28_multiscale_posterior
```

---

# 13. Qué inspeccionar tras v4.28

```powershell
$Out = "artifacts\debug\object_event_v4_28_multiscale_posterior"
$S = Get-Content "$Out\summary.json" -Raw | ConvertFrom-Json

$S.status
$S.champion
$S.oof_gate_passed
$S.oof_gate_checks
$S.decision
```

Ranking:

```powershell
Import-Csv "$Out\oof_arm_ranking.csv" |
    Format-Table -AutoSize
```

Comparación esencial:

```text
profile_posterior
vs
spatial_rotation_posterior
```

Preguntas científicas a responder:
1. ¿La KL al posterior físico reduce la entropía y aumenta `prediction_std_ratio`?
2. ¿A mejora v4.27 sin necesidad de spatial 2-D?
3. ¿B mejora específicamente worst-sequence/negative-track?
4. ¿La mejora viene de Pearson global o realmente de magnitud/calibración?
5. ¿Hay trade-off de signo positivo/negativo?
6. ¿Alguna seed domina artificialmente?
7. ¿Los mismos tracks siguen invertidos?

Si OOF pasa, inspeccionar también:

```powershell
$S.v410_validation_metrics
$S.validation_metrics
$S.validation_track_metrics
$S.decision.comparisons
```

Y:

```powershell
Import-Csv "$Out\validation_per_sequence.csv" |
    Format-Table -AutoSize
```

```powershell
Import-Csv "$Out\validation_per_track.csv" |
    Where-Object { [int]$_.negative_count -ge 4 } |
    Sort-Object `
      @{Expression={[double]$_.negative_accuracy}},
      @{Expression={[double]$_.pearson}} |
    Format-Table -AutoSize
```

---

# 14. Comprimir resultados

```powershell
$Out = "artifacts\debug\object_event_v4_28_multiscale_posterior"

Compress-Archive `
  -Path "$Out\*" `
  -DestinationPath "object_event_v4_28_multiscale_posterior_results.zip" `
  -Force
```

Conservar el ZIP fuera de Git.

---

# 15. Qué hacer según el resultado

## Caso A — `profile_posterior` gana claramente y pasa OOF

Interpretación:
> la hipótesis LHR 1-D era esencialmente correcta; el cuello de botella era la identificación del posterior de escala.

Acción:
- no añadir más arquitectura todavía;
- confirmar multiseed full-train;
- evaluar development una vez;
- si supera v4.10 de forma convincente, congelar esta familia y escalar entrenamiento;
- no abrir test oficial todavía hasta freeze formal.

## Caso B — `spatial_rotation_posterior` gana claramente y pasa OOF

Interpretación:
> la representación TTC necesita estructura espacial 2-D y manejar movimiento nuisance además de escala.

Acción:
- bloquear este matcher;
- estudiar después solo cambios que preserven su principio físico;
- medir coste/VRAM/latencia;
- entrenar multiseed largo antes del test.

## Caso C — ambos mejoran pero no pasan gates

No bajar gates.

Determinar:
- si falla principalmente magnitud;
- si falla worst sequence/track;
- si el posterior sigue plano;
- si el matcher spatial sí separa rotation/scale pero el readout TTC falla.

Solo entonces diseñar v4.29.

## Caso D — ambos fallan de forma clara

No hacer v4.28.1 con temperatura/bins/weights arbitrarios.

La siguiente familia propuesta es **event-native affine / normal-flow field**, estimando algo como:

```text
u(x,y) = a0 + a1*x + a2*y
v(x,y) = b0 + b1*x + b2*y
```

y separando:
- translation;
- rotation/shear;
- expansion/divergence.

TTC se deriva de la componente de expansión/divergencia.

Motivación:
> si scale matching 1-D y 2-D no transfieren suficientemente, el fenómeno debe representarse como un campo de movimiento local y no como un único escalar de escala.

---

# 16. Cosas que NO hay que volver a probar sin nueva evidencia

Evitar repetir:
- simple global/mean pooling como solución;
- más MLPs TTC opacos;
- router de signo post-hoc;
- ridge/readout lineal de divergence + vertical;
- boxes como input forward;
- pseudoflow de boxes como única supervisión;
- tuning contra development-validation;
- bajar gates después de ver resultados;
- seleccionar por EvTTC;
- abrir test oficial antes de freeze.

Resultados negativos son parte del conocimiento acumulado.

---

# 17. Hipótesis científica actual

La hipótesis de trabajo ya no es:

> “un Transformer debería estimar TTC”.

Es más concreta:

> **Los eventos contienen correspondencias espacio-temporales suficientes para recuperar la expansión/escala física de objetos; una representación JEPA/densa event-only que preserve esa geometría y la convierta en LHR/TTC mediante una restricción física debe generalizar mejor que un readout TTC directo.**

Evidencia a favor:
- oracle height ratio ~0.76 Pearson;
- dense correspondence transfiere mejor que geometry aggregation;
- partial-unfreeze mejora divergence/vertical;
- v4.27 recupera `log_eta Pearson ~0.592` OOF;
- negative accuracy v4.27 alcanza ~0.652 OOF.

Evidencia en contra / problemas pendientes:
- gap grande oracle→learned scale;
- posterior v4.27 demasiado difuso;
- magnitud comprimida;
- worst-sequence y worst-track aún insuficientes;
- no existe todavía resultado oficial comparable que supere Garl-TTC.

---

# 18. Relación con JEPA

No perder el objetivo original.

El proyecto no debe degenerar en “otro Garl-TTC”.

La ruta final pretende que:
- JEPA/event encoder aprenda tokens densos espacio-temporales;
- la geometría LHR/flow sea una variable física supervisable/diagnosticable;
- la inferencia principal siga usando eventos;
- se mida si el pretraining JEPA aporta transferencia frente a la misma arquitectura scratch;
- se preserve el control scratch / pretrained / partial FT.

Un matcher geométrico fuerte sin demostrar beneficio JEPA puede ser un excelente baseline event-only, pero no demostraría todavía la contribución JEPA.

Por eso, una vez estabilizada la arquitectura geométrica, habrá que reabrir la ablation:
```text
same architecture scratch
vs
JEPA pretrained frozen/partial FT
```
bajo protocolo idéntico.

---

# 19. Cuándo abrir eAP test / EvTTC

No ahora.

Orden recomendado:

1. pasar OOF robusto;
2. development-validation una sola vez por familia preregistrada;
3. fijar arquitectura;
4. entrenamiento largo multiseed;
5. reproducibilidad;
6. congelar:
   - config;
   - preprocessing;
   - checkpoints;
   - ensemble rule;
   - hashes;
7. solo entonces:
   - eAP official submission / CodaBench;
   - EvTTC zero-shot protocol.
8. claim SOTA solo si realmente supera Garl bajo el protocolo correspondiente.

---

# 20. Fuentes públicas que el agente debe consultar

Repositorio:
- https://github.com/Kripta-Studios/e-jepa-ttc/tree/scientific-recovery-v3-hardening

Historial:
- https://github.com/Kripta-Studios/e-jepa-ttc/commits/scientific-recovery-v3-hardening

Documentos:
- `AGENTS.md`
- `PLAN.md`
- `STATUS.md`
- `README.md`

Garl/eAP:
- https://arxiv.org/abs/2603.16303
- https://github.com/NAIL-HNU/Garl-TTC
- https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset

El release Garl local es referencia/oracle de preprocessing, métricas y arquitectura; no copiar código sin respetar licencia/provenance.

---

# 21. Objetivo inmediato que debe asumir Codex

**No empezar una arquitectura nueva antes de ejecutar/analizar v4.28.**

Tarea inmediata:

1. verificar workspace y commit;
2. aplicar `e_jepa_ttc_object_event_v4_28_multiscale_posterior.patch`;
3. ejecutar compilación/tests/preflight;
4. ejecutar el orchestrator v4.28;
5. inspeccionar el ranking OOF;
6. no abrir development si el gate falla;
7. analizar por secuencia y track;
8. conservar resultados;
9. proponer la siguiente modificación solo a partir de la evidencia producida.

El criterio no es “hacer que pase el gate”.
El criterio es identificar una representación física event-native que generalice de secuencias train a secuencias no vistas y finalmente pueda batir Garl-TTC de forma reproducible.

---

# 22. Resumen en una frase

**La última semana pasó de demostrar que los readouts TTC directos/lineales sobreajustan, a aislar que la geometría de escala/height-ratio sí existe en eventos; v4.27 ya recupera esa física parcialmente OOF pero comprime magnitud, y v4.28 es el experimento decisivo para distinguir entre “posterior de escala mal supervisado” y “necesitamos correspondencia espacial 2-D con nuisance de rotación”.**
