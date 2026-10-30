# Progreso

## 2026-08-14: Scientific Recovery V7

- completados 15 runs: SOFT, C2F, T20, CAP-S y partial-freeze, tres folds seed 7;
- integridad OOF: 8.192 tokens únicos, nueve secuencias y 100% de puntos finitos;
- MiD puntual: A5 158.449; SOFT 165.116; C2F 158.573; T20 165.260;
  CAP-S 167.025; partial-freeze 167.826;
- todos los brazos fallan mecanismo, geometría y candidatura; V7 cierra negativo;
- partial-freeze congela 9.664 parámetros, pero solo retiene ~19% de slopes y
  ~29% de std-ratios;
- diagnóstico post hoc A5/C2F-MoE: 153.519 MiD, delta bootstrap mediana
  `−4.919`, IC95% `[−7.033,−2.910]`; requiere protocolo nuevo;
- no se usaron public validation, private test, EvTTC test ni CodaBench.

Detalles y hashes: [estado canónico V7](SCIENTIFIC_RECOVERY_V7_STATUS.md).

## 2026-08-13 — Scientific Recovery V5

- reparados provenance, claim readiness y stale-output handling P0;
- congelados tres folds train-only antes de A8;
- invalidados para promoción los A6 parent-exposed sin borrar su evidencia;
- ejecutados secuencialmente 3×A4, 3×A6, 3×A8.0 y 3×Garl;
- probada geometry exacta y prefix causality de A6/A8;
- agregado outer-dev exacto: A8.0 197,69, A6 211,51, Garl 144,35 MiD;
- A8.0 mejora A6, pero falla el gate 175; A8.1 no se promueve;
- public validation no usada y private/test cerrado.

Detalles y hashes: [estado canónico V5](SCIENTIFIC_RECOVERY_V5_STATUS.md).

## 2026-08-10 — A0, Garl matched y A1 exact-2048

- terminado A0 seed 7 como resultado negativo;
- construido subset exacto Garl de 2.048 tokens;
- evaluada referencia event-only release y auditada exposición de secuencias;
- generado comparador firmado con bootstrap por secuencia y cola de outliers;
- preregistrado A1 geometry-only como siguiente hipótesis aislada;
- descompuesto A0: la bbox sí contiene señal física, pero el foreground no aprende
  su extensión temporal; residual y consenso no la recuperan;
- implementado runner Garl matched desde cero y completado smoke batch 32 sin OOM;
- materializado preprocessing oficial matched exacto 2.048/2.048 en 322,08 s,
  identidad firmada `92af2810…f564b52`;
- implementado trainer sobre cache con selección validation-only, checkpoints
  atómicos y resume fail-closed ante cambios de protocolo.
- terminado Garl matched seed 7: best 11/16, MiD macro `203.6342`, global
  `203.0982`, cero failures, 274,98 s y 1.317,6 MiB VRAM;
- actualizado comparador firmado con tabla matched, win rate pareado y bootstrap
  por secuencia: identidad `e6344713…c254de`;
- A0 pierde en las tres secuencias y por `+180.7031` MiD macro; Garl matched
  conserva un fallo visible en receding/negative al predecir solo TTC positivo.
- preregistrado e implementado A1 geometry-only sin cambiar parámetros ni forward;
  tests prueban ausencia de rasterización weak-box y bbox como input;
- añadidos diagnósticos por época de geometría absoluta/diferencial globales y
  macro-secuencia;
- ejecutado A1 seed 7 en GPU desde el preregistro publicado: best 18/18, MiD macro
  `346.8295`, failure `9.9609%`, Pearson log-ratio `.110821`, 631,88 s;
- A1 mejora A0 en las tres secuencias, pero pierde contra Garl matched por
  `143.1953` MiD macro, IC95% por secuencia `[115.1042,166.6705]`;
- localizado el remanente: altura absoluta `.4708`, anchura/centros `<=.079`,
  `delta log h`/bbox `.0591`, `delta log h`/física `.1048`; geometry-only no basta;
- generado comparador A1/Garl exact-token firmado, identidad
  `471fa106…7ba485d`. No se promueve ni escala A1.
- añadida auditoría GPU por endpoint y actividad event-only: anchura objetivo
  variable pero predicción `r=.048/.105`; actividad cruda casi uniforme y sin
  delta geométrico; artifact firmado `737a3663…f0aa635d`;
- fijada siguiente hipótesis mínima: decoder full-resolution 2-D contra el `amax`
  axial separable, sin introducir teachers, JEPA o pair-ratio.
- preregistrado A1-FR con igualdad exacta de data/training/loss y un solo cambio
  de decoder; config `7ceb1149…772c42e`, 340.870 parámetros, pendiente del run GPU.
- detectado selector que aceptaba macro con secuencia NaN durante el primer intento
  A1-FR; run preservado/inválido (`fd5bde50…c9be8f19b`), sin usar sus métricas;
- selector corregido y probado para exigir cobertura finita de todas las secuencias;
  repetición A1-FR obligatoriamente desde cero.
- completada repetición A1-FR válida: best 11/16, macro `380.2202`, failure
  `28.7598%`, ratio `-.0181`; peor que A1 y no promovida;
- firmado comparador A1-FR/Garl (`b0251860…53c3c35`); la hipótesis fullres raw
  queda rechazada y el siguiente control propuesto usa features `resize_conv`.
- preregistrado A1-DF (`dddfb393…0b284`) con un único cambio a `resize_conv`,
  features profundas, misma loss A1 y sin teachers/pair-ratio.
- ejecutado A1-DF seed 7: macro `350.3020`, failure `21.0938%`, ratio `.1865`;
  señal temporal mejor pero sin mejora principal, no promovido;
- firmados comparador `003c3867…e0a1d0c` y descomposición candidate-aware
  `5a9c4293…75141da`; siguiente hipótesis propuesta: pair-ratio directo único.
- implementado pair-ratio geometry-only sin máscaras y warm-up seguro;
  preregistrado A1-DF-R peso `5.0`, experimento `b3f9eb9e…de43f67`.
- ejecutado A1-DF-R: macro `349.8628`, failure `19.8242%`; mejora marginal y
  concentrada en qooh, no promovido ni escalado;
- firmados comparador `05601545…2dd6205` y descomposición
  `0bc741a5…49900e7`; siguiente rama representación/pretraining event-native;
- auditadas 88.744 filas train: 177.488 referencias de máscara/64.629 únicas,
  cero ficheros; 64.629 RGB únicos presentes en 135 TAR;
- auditados seis snapshots HF sin red ni carga de pesos: SAM ViT-L y DINOv3
  ConvNeXt-Tiny quedan utilizables; artefacto `6e910ec2…f1e246`;
- siguiente intervención: smoke SAM bbox-prompt train-only en CUDA, con protocolo
  separado `event-only inference with RGB distillation`.
- preregistrado el smoke en `e4969f1` y ejecutado sin pesos remotos: pasa todos los
  gates (`.4207 s`, `1691.39 MiB`, máscara finita `.0663`), identidad
  `be097e6c…2af5e9`; sigue sin ser evidencia TTC.
- preregistrado/ejecutado audit SAM train-only 36 pares/72 endpoints: todos los
  gates pasan; área temporal `.6471`, signo `.8286`, bbox–mask IoU `.5761`, una
  degenerada; artefacto portable `0922d540…73dd44`.
- materializadas 4.096 máscaras SAM para los 2.048 train exactos: `.9492` endpoint,
  `.7822` par, 32 shards, `1784.63 s`, `1691.89 MiB`; manifest
  `aaa60090…0426b0`. Validation permanece sin teacher.
- implementado/preregistrado A3 sobre A1 sin cambiar arquitectura: SAM BCE `1.0` +
  Dice `.5` solo en 3.204 endpoints aceptados; join exacto por token/secuencia/crop,
  fallback geometry-only y validation sin campos teacher. Config SHA
  `83e8c716…9b7754`.
- A3 seed 7 cerrado negativo: best 8/13, macro `353.6351`, failure `10.8887%`,
  ratio `.1053`; pierde frente a A1 en las tres secuencias. Bootstrap A3−A1
  `+7.5388`, IC95% `[1.5525,10.6383]`; artefactos `ffa968a8…26ee63`,
  `5e24d901…1bfcf0`, `fd637354…1c46d9`. No sweep/escalado/seeds.

## 2026-08-10 — Transporte temporal Causal Scale v7

- implementado transporte físico del inverse TTC anterior al timestamp actual;
- blend `.75/.25` sin parámetros aprendidos y fallback seguro ante denominador inválido;
- corregidos controles para separar oddness T=2 y translation de la salida T=3;
- selección validation penaliza explícitamente checkpoints que fallan gates;
- validation 401/502: Pearson `.96126`, slope `.92744`, sign `.99115`, IoU `.89268`,
  TTC `.24345`, coverage `.79941`, translation `.00351`; todos los gates pasan;
- artefacto firmado: `artifacts/metrics/causal_scale_v7_diagnostic_comparison_v1.json`,
  identidad `eb3497fafad8a4d23284b263303628be8ad025fd61bac57ad5f54580d142ee82`;
- test 603 y todos los datos reales permanecen cerrados.

La suite completa, el commit/push y el único test clean-tree 603 se completaron.

El test se ejecutó después desde `0bc781f` y falló solo Pearson: `.9201432 < .95`.
Translation `.0033841`, IoU `.8896096`, TTC `.2457614`, slope `.9278544`, sign
`.9911308` y coverage `.7827051` pasaron. Seed 603 queda consumida; v7 cerrado.

Se preregistró V8 sin abrir datos: tres grupos train (701–703), tres validation
(801–803) y tres tests sellados (901–903). La selección pondera por igual score macro
y peor grupo manteniendo el presupuesto total V7. Pendiente: verificación, commit y
diagnóstico validation-only.

El diagnóstico base seleccionó epoch 16: Pearson macro `.80631` y por grupo
`.64269/.89168/.88458`. IoU `.89016`, TTC `.28206` y translation `.00469` pasaron.
Los mayores errores proceden de endpoints aislados con foreground inestable; se
implementó consenso temporal simétrico `.10/.15` para entrenar dos brazos controlados.

## 2026-08-10 — Adaptación causal-scale a eAP/Garl public validation

- auditado `E:\Garl-TTC` release oficial y sus checkpoints event/visual/full;
- auditados `E:\eAP_dataset` (~691,5 GiB) y `E:\GarlTTC_dataset` públicos;
- congelado cache existente 2.048 train + 2.048 validation, 9/3 secuencias disjuntas;
- confirmada cobertura de los cuatro buckets en cada secuencia validation;
- implementado weak-box target explícito, excluyendo t0 proxy y sin cajas como input;
- implementado trainer BF16, CVaR, early stopping, guard 6 h y resume atómico por época;
- añadido runner fail-closed con hashes y config congelada;
- benchmark 128+128: 5,289 s, 395,6 MiB VRAM; no es resultado eAP;
- checks focalizados Ruff/Pyright/Pytest verdes;
- resume end-to-end exacto aprobado; cambios de contrato se rechazan fail-closed;
- builder subset Garl exacto implementado y probado con manifest firmado;
- este estado histórico quedó supersedido: A0, Garl matched y A1 están completos;
  consultar el bloque superior de 2026-08-10.

CVaR 10% con `w=.15` seleccionó epoch 32: Pearson macro `.94621`, grupos
`.94812/.94567/.94486`, TTC `.29547`. Es el mejor V8 pero no pasa; test sintético no
se abrió. Se inicia diseño eAP train/validation limitado a menos de seis horas.

## 2026-08-10 — Foreground equivariante Causal Scale v6

- preregistrados grupos nuevos 401/502/603; no se reutilizó test v5 seed 303;
- implementadas ramas full-resolution y separable row/column sin strides;
- la rama separable mejora IoU `.1824 -> .8932` y translation `.0279 -> .00462`;
- validation seleccionada: Pearson `.9204`, slope `.9488`, sign `.9912`, TTC
  simétrico `.2663`, cobertura `.7994`; falla solo Pearson `.95`;
- pair-ratio mask loss (`.8717`), learned height correction (`.8586`) y refinamiento
  residual congelado (`.9225`) no mejoran de forma material y quedan rechazados;
- test seed 603, eAP, EvTTC, RGB y CodaBench no se abrieron;
- comparación firmada: `artifacts/metrics/causal_scale_v6_diagnostic_comparison_v1.json`,
  identidad `e00a64a90aee5c302ad486763ed147a2af590a7d3575191395e0f0d374d6191f`.

Pendiente: v7 multigrupo/multiescala; no ampliar el decoder ni abrir real-data.

## 2026-08-09 — Gate held-out Causal Scale v5

- publicado el protocolo exacto en `d9d20af` y ejecutado desde worktree detached limpio;
- abierto test sintético seed 303 exactamente una vez (`test_evaluation_count=1`);
- resultado `completed_gate_failed`: Pearson `.9213532 < .95` y translation p95
  `.0274930 > .02`;
- pasaron slope `.9691788`, sign `.9941860`, IoU `.8724040`, TTC simétrico
  `.2592012`, cobertura `.7761628`, oddness, known y controles de vacío;
- artefacto completo firmado:
  `artifacts/metrics/causal_scale_v5_synthetic_learning_gate_v1.json`, identidad
  `ce42fe957c4944a72bf38b5b134df7dfd0809ccc1c87b6cff6a749662093ea29`;
- seed 303 queda consumida; no se ajustarán umbrales ni se repetirá v5 como evidencia;
- eAP, EvTTC, RGB, Garl-TTC y CodaBench siguieron cerrados.

Pendiente: v6 con nuevos grupos sintéticos predeclarados y operador de geometría
explícitamente translation-equivariant; no abrir real-data antes de un nuevo pass.

## 2026-08-09 — Aprendizaje sintético Causal Scale v5

- implementado dataset causal event-only con seeds disjuntas 101/202/303;
- implementados trainer, selección post-warm-up, logs completos por época,
  calibración de varianza solo en validation y runner full fail-closed;
- ejecutados nueve diagnósticos train/validation; el test seed 303 no se construyó;
- la secuencia de mejoras llevó Pearson `.2678 -> .5941 -> .9064 -> .9329 -> .9560`;
- el candidato deconv+translation+cosine alcanza slope `.9686`, signo `.9957`, IoU
  `.8640`, MAE ratio `.01903`, error TTC simétrico `.2639` y cobertura `.7974`;
- solo falla translation leakage: `.02399` frente al gate congelado `.02`;
- Huber adicional (`.8724`) y resize-conv (`.9430`, translation `.0320`) quedan
  preservados como resultados negativos;
- comparación firmada y regenerable:
  `artifacts/metrics/causal_scale_v5_diagnostic_comparison_v1.json`, identidad
  `27053c853b93b1ff14ec32f4db79e4e216c6a5c3929f385998b549c1dee2fe80`;
- no se abrió eAP, EvTTC, RGB, CodaBench ni ninguna etiqueta TTC real.

Ese test clean-tree se ejecutó después y falló; véase la sección superior. No existe
autorización SOTA ni real-data.

## 2026-08-09 — Causal Scale v5

- preservado el resultado negativo v4.31 y descartado añadir otro readout al matcher;
- implementado un core común de foreground/altura/log-ratio para eventos y futuro RGB;
- eliminados bbox, categoría e ID de secuencia de la API del candidato v5;
- añadido residual antisimétrico acotado, incertidumbre en ratio y riesgo derivado;
- añadido loss con máscara training-only, NLL física, consistencia y auxiliar aislado;
- añadidos config, ADR, runner fail-closed y nueve tests unitarios;
- Pyright 0, Ruff focalizado limpio y Pytest completo aprobado con 7 skips;
- publicados `cae7d1f`, `c58d07f` y `7945e99` en la rama activa;
- ejecutado gate clean-tree ideal-foreground: Pearson `1.0`, slope `.9999995232`,
  sign `1.0`, oddness `0/0`, translation `0`, rotation-square `.00171029`,
  zero-unknown `1.0`, 336.398 parámetros;
- no se abrió eAP, EvTTC, RGB ni ninguna etiqueta TTC real.

Pendiente inmediato: implementar el generador/runner de aprendizaje event-only
sintético y exigir los mismos gates sobre máscaras predichas held-out. Este resultado
no autoriza real-data, comparación Garl ni claim SOTA.

## 2026-08-02

- auditado el repositorio completo y preservados cambios ajenos;
- corregidos contratos Garl, calibración, timestamps, sampling y métricas firmadas;
- verificada paridad raw/resized y de salida de modelo;
- corregido el pooling BF16 high-resolution;
- implementados padding/máscaras, atención por ventanas, merge 2x2 y mixer temporal;
- rechazado KDA por regresión;
- implementado trainer event-only raw/on-demand sin cache global;
- añadido gradient accumulation, orden determinista por época y perfil full con Git
  limpio;
- añadido runner `screen/full`, seeds 7/13/23, freeze, predict/score EvTTC y
  validación local de submission;
- completado smoke real 16/16; MiD macro `1868,3186`, no promocionable;
- comprobado shard cache 256 y descartado 4.096 por presión de RAM;
- eliminados launchers/aliases rotos, protocolos/manifests duplicados y código
  engañoso;
- eliminados CARLA local, `artifacts/runs` y `artifacts/features`; ~101 GiB
  adicionales recuperados en esta pasada;
- preservados resúmenes compactos de resultados negativos;
- corregidos tests de evidencia para que un clon limpio omita artefactos locales
  opcionales;
- Ruff, Pyright y Pytest completo pasan sin los árboles generados;
- implementada auditoría falsable de shortcut semántico, cinco brazos x tres seeds
  y control frame-varying, sin dataset real;
- demostrado que varianza/VISReg pueden estar sanos estadísticamente y conservar
  el shortcut; R²-lite no supera el gate y queda rechazado;
- residual temporal pasa el shortcut lento, pero falla el control frame-varying:
  se conserva solo como canal `z_delta` candidato junto a `z_level`;
- publicados `7ec2b90`, `cbdf54c` y `7d33989` en la branch activa.

Pendiente prioritario:

1. JEPA high-resolution compatible con salida de nivel y residual;
2. `level` vs `level+temporal_residual` en screen pareado y probes reales;
3. RGB-E causal como ablación separada;
4. inputs EvTTC Tabla VI label-free;
5. full 7/13/23 solo tras mejorar claramente el screen;
6. evaluación oficial eAP/CodaBench.

No implementar R²/HSIC/CMI ni INTACT antes de ese gate: el primero fue rechazado
y el segundo requiere acciones expertas ausentes en este problema.

No existe claim SOTA.


## 2026-10-09 — Correcciones directas RGB-PORT

Por petición del usuario, se corrigen los módulos originales y se conserva la
geometría BF16. Véase `docs/decisions/ADR-0002-rgb-port-audit-corrections.md`.
Se corrigen el router RGB, la reducción efectiva T2/T3, caché event, intervalos
TTC, historias Dev32, costes desde HDF5/VRAM e informe de latencias.
`artifacts/rgb_port_20261008/AUDIT_CODE_MIGRATION.json` conserva la transición
exacta desde las congelaciones históricas y los checkpoints event existentes.
La pausa anterior permanece; esta tarea no consume updates ni reinicia los fits.
Evidencia reproducible y comprobaciones negativas: `audit_fixes_20261009/`.

Validación: suite `test_rgb_port*` excepto el test técnico opt-in que realiza
updates; Ruff y Pyright sobre los módulos afectados; paridad de datos reales,
compatibilidad de cola y restauración aislada de estados completos.
Los benchmarks finales de GPU conjunta y de todos los endpoints siguen pendientes
de su ejecución; no se extrapolan tiempos CPU a throughput científico.

Cierre de esta corrección: 133 pruebas de la suite y 23 comprobaciones finales
centradas en productores/regresiones (134 casos únicos). Ruff y Pyright pasan.
Restauración exacta de ambos checkpoints y paridad CPU de pérdida/componentes/
gradientes event contra el código archivado: PASS. Resultado:
`artifacts/rgb_port_20261008/audit_fixes_20261009/RESULT.json`.

## 2026-10-12: cierre y publicación V14 RANGE-TRANSFER

Se ejecutaron seis cabezas M0/M1/M2 x L0/L1, seed7, 2500 updates por fit,
con padres matched finales y caché H admitida. Los seis endpoints quedaron
congelados antes de evaluar las 16432 consultas V. Ninguna receta cumple
los gates: NO_CANDIDATE. No se ejecutan réplicas ni transferencia nueva.
Contabilidad: 15000 científicos + 20 técnicos, cero recuperación.

Los [informes detallados](range_transfer_results_20261012/README.md) contienen
metodología, atribución H/V12, tablas por banda/grupo/subrango, bootstrap,
fallos, coste con scope, pruebas y comandos de reproducción. El generador
scripts/build_range_transfer_reports.py verifica y reconstruye la entrega
sin entrenamiento. Las fuentes científicas ejecutadas conservan sus hashes.

La petición del usuario del 12 de octubre autoriza publicar esta rama V14,
tras la prohibición de push del encargo anterior. No autoriza merge ni ampliar
experimentos. V13 y sus admisiones no se modifican.

Validación local: 41 tests y 17 subtests aprobados; Ruff, formato y Pyright
sobre la integración y el generador. La copia histórica del handoff se excluye
del formateo para conservar los hashes del paquete recibido.

## 2026-10-28 — Cierre del consejo de runtime TRAIN40

Por petición del usuario, se termina el diagnóstico en curso y se prepara la
pausa de la investigación tras publicar código, recibos y documentación. El
[handoff detallado](train40_runtime_council_20261028/HANDOFF.md) sustituye al
handoff de la ronda anterior para continuar este trabajo. No se reentrenó ni se
alteraron los pesos, normalizadores o predicciones originales.

El mejor replay de un objeto completa 250/250 consultas con paridad, cero
descartes y cero errores durante cinco segundos ofertados. La respuesta desde
el deadline previo a adquisición hasta todos los TTC tiene p95 de 29,143155 ms
y máximo de 35,3326 ms. En 200/250 deadlines todavía había una actualización
previa pendiente. No admite 50 Hz sostenidos, todos los updates bajo 20 ms ni
ausencia de acumulación; dos objetos sostenidos y veinte quedan pendientes.

El perfil efectivo conserva PIL ROI-first, E reuse OFF, R reuse ON y FP32;
adopta transporte exacto reverse-summaries y lanzamiento temprano de eventos.
El runner paced usa RAW CPU precargado, primación sintética nativa y caché de
features RGB con identidad inmutable SHA-256 V1. La frontera de memoria CPU
está declarada: no es adquisición de un sensor físico. Crop cache, hash V2 y
subpaquetes de 5 ms siguen fuera del perfil por defecto.

El último diagnóstico de 5 ms completó tres casos, con gates de estados,
representaciones y TTC correctos y 626 archivos preservados; no mide latencia.
Se conserva también su fallo inicial de presupuesto RAW, la pérdida de cobertura
del ensayo con crop cache y los fallos numéricos de transporte compilado. Las
tablas, tests, limitaciones y fuentes exactas están enlazados en el handoff.

## 2026-10-28 — EvTTC competition inference completed

Implemented the ten-sequence competition adapter and completed local inference
with the preserved nine-component TRAIN40 candidate. All 22 downloaded input
files were present; no additional calibration downloads or training were needed.
The complete run is `output/evttc_competition_20261028_v2`, with 434/434 finite
predictions in ten `results.txt` files and `results.zip`. Original CSV indices
and timestamp text are preserved. The ZIP SHA-256 is
`2c79847516993924229799696b05d84185e1dc32d90cbc1ac6016532e9552733`.

The first attempt remains intact: after 367 successful predictions it stopped
on an unrepresentable event-expert TTC in Slider-750. Adapter v2 uses fixed
causal history anchors with real startup prefixes, records whole-observation
phase-domain rejection independently of confidence, and preserves explicit
unavailable handling. Seven of 1,626 event observations were rejected in the
complete run; all queries retained enough valid history. There were 426 fused
outputs and eight native event fallbacks. Two negative TTC outputs were retained
without clipping. See [ADR-0043](decisions/ADR-0043-evttc-competition-sensor-adapter.md)
for the adaptation, prior failure, calibration interpretation, and limitations.

The independent audit passed on its first execution. It verified all 434 raw
query identities, the exact ten-file archive, the 22 raw inputs, 23 executed
source files, 13 model-package files, and three files from the preserved parent
attempt. The maximum complete sensor span was 641.313 ms. The run did not read
TTC labels or published result plots. Earlier manual inspection of the public
curves was not used for offsets, fitting, or output selection.

Validation: 61 focused tests passed with one conditional CUDA test skipped in
the combined suite; that sealed CUDA parity test passed separately. Ruff,
format checks, and Pyright passed. Prediction and measured-cost figures are
regenerated from the complete ledger, including every signed value and the
startup/support diagnostics. Median query cost was 2.68688 s and p95 was
9.07308 s for this offline history-recomputing adapter; these are not optimized
streaming latency measurements.

[Reproduction and artifact guide](evttc_competition.md) documents the run,
verification, and package formats. Official EvTTC accuracy and leaderboard
placement remain unmeasured; local coverage is not a benchmark-defined failure
rate. No application email or leaderboard submission was performed.

## 2026-10-28 — Public Garl checkpoints compared on EvTTC

Completed inference with the released full RGB-event, event-only LHR, and
visual-only LHR checkpoints on the same 434 competition query identities as
the preserved PHASE run. The three checkpoints match Hugging Face
`NAIL-HNU/GarlTTC-model` revision `b676fcdaf26c04bcf896cdb2b208c9c424e8462a`;
the native graph is pinned to upstream commit
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`. No training, target access, or
test-based checkpoint selection occurred. Execution used commit `1d8de7d9`.

The complete population is accounted for in
`output/garl_evttc_competition_20261028`: 1,302 terminal records. Event-only
has 434 finite outputs, full has 424, and visual-only has 426. Eight road
startup queries lack two causal RGB frames; the full model also lacks its
earlier event endpoint at Slider-750 indices 1 and 2. These are explicit nulls,
not filled with another checkpoint. No numerical model failure occurred.
Finite signed outputs, including all negative values and extremes, are intact.
The aggregate status is correctly `PARTIAL`; event-only has a complete archive.

The adapter preserves native preprocessing and FP32 height-ratio conversion,
with measured sensor spacing substituted only in the TTC conversion. Full
event windows end independently at their RGB endpoints and are bounded by
`min(actual spacing, 100 ms)`. A label-free clock audit found camera jitter up
to 24 microseconds beyond 100 ms; the bounded support and resulting gap are
explicitly recorded. Native graph `dT=0.1` remains unchanged.

Both independent audits passed on all 1,302 rows, exact query/ZIP identities,
causality and FP32 conversions. All 22 raw files, three checkpoints/configs,
executed sources and preserved PHASE artifacts retained their hashes.
The run freeze SHA-256 is
`3168520154bcde371e66986804f559241319fb4af1279e82c4acc226e8eed4db`.
The event-only results ZIP SHA-256 is
`fcca74928f56d958b6f7264c8ba2691a2313d3e77edc8efa30e87a407e9ace33`.

The regenerated comparison includes all per-query values, a 30-row per-sequence
table, full-range and signed-log plots, runtime telemetry, and artifact hashes.
Median absolute disagreement with PHASE is 0.261 s for full, 0.681 s for event,
and 0.236 s for visual. These are disagreement statistics, not errors. Exact
ground-truth correspondence is unverified, so local accuracy and SOTA are
unmeasured. Fixed variant order and shared cache/warm state prevent a controlled
latency ranking; PHASE also uses a different history and offline adapter.

The primary eAP/Garl paper reports 10.60% average RTE on three EvTTC sequences,
not all ten competition sequences. Literature evidence, commands, temporal
adaptations and checkpoint-lineage limitations are documented in
[the reproduction guide](garl_evttc_competition.md).

Validation: 41 focused tests passed and one conditional CUDA test was skipped
in the combined suite. Native CUDA parity for all three checkpoint models passed
separately. Ruff, formatting and integrated Pyright passed. No external
submission or publication was performed.

## 2026-10-28 — Paper revision with official and EvTTC competition evidence

Reorganized the manuscript around the deployed estimator, training/evaluation,
official GarlTTC results, ten-sequence EvTTC competition inference, labeled
development transfer, controlled design analysis, runtime and discussion.
The full historical experiments and method families remain in appendices.
The prior manuscript and the user-provided structural reference remain intact.

The revised source is `output/pdf/phase_ttc_evttc_paper`. Its public competition
projection contains 1,736 model-query rows with all signed finite outputs and
explicit nulls, together with regenerated coverage, disagreement, per-sequence
and observational-runtime tables. Published Garl EvTTC RTE values form a separate
table. The paper assigns no PHASE-TTC RTE to the unscored competition population.
The official benchmark section includes all four bands and an additive weighted
gain decomposition with its 0.001-point display-rounding residual.

Delivery: `output/pdf/PHASE_TTC_Research_Report.pdf` (51 pages) and
`output/pdf/PHASE_TTC_LaTeX_Source.zip` (120 source/data/figure files plus the
archive manifest). Scientific review matched every public competition row to
the preserved inference ledgers. Visual review corrected float placement and
page pacing. The final build has no undefined references or overfull boxes.
The archive was extracted and rebuilt independently; extracted PDF text and
pagination match the delivered PDF exactly. The public build verifies 13
derived-data hashes and regenerates all added numerical tables and charts.

PDF SHA-256:
`04b3f126ab4fc86e9ffd5b75d545564369a0c9d17a052efc8c4c9add3387650f`.
Source ZIP SHA-256:
`1fdd298557927a234d12f080df22a6b88af1add0c88e3c378c6da90f9f80acb5`.
The delivery receipt is `output/pdf/source_package_receipt.json`.

## 2026-10-29 — Final article refinement and reproducible source delivery

Refined the manuscript using the supplied Improved TeX/source reference while
preserving the 28 October scientific evidence snapshot and all 13 public
ledger hashes. The final source is `output/pdf/phase_ttc_final_paper`; the prior
51-page source and user references remain intact. The article now has 56 pages,
24 numbered figures, and 27 tables, with a separate technical supplement.

The method exposition uses modality-specific architecture, fusion-interface,
out-of-fold training, and causal-history diagrams. Paired ablation figures,
complete error-exceedance curves, all-sequence transfer comparisons, and
unscored competition traces preserve their distinct populations and estimands.
Scientific and visual reviewers checked the complete manuscript. Final fixes
resolved sentence-interrupting floats, subplot-label collisions, numeric-column
spacing, and an isolated table page. Public text and package metadata contain
no aesthetic-reference or internal authoring-process commentary.

R01 adds an exploratory analysis of the existing 468 eligible labeled transfer
queries from 32 distinct recordings. An independent multiplicity-weighted
implementation reproduced all 30,000 paired bootstrap values within
8.88e-16 seconds. PHASE-minus-RGB mean, median and p95 difference intervals
all include zero. The paper retains the stronger RGB averages and the worse
PHASE extreme tail (five versus two errors above ten seconds). No fitting,
model selection, new inference, or benchmark submission was performed.

R02 documents the native finite phase-output map and the exact L40 paced-run
GPU, PyTorch CUDA build, precision, worker/queue configuration, initialization
boundaries, and parity tolerance. Source hashes and executed Git objects bind
the metadata to the original run. CPU model, driver version, and a full
component parameter/peak-memory inventory remain unreported. R02 SHA-256 is
`c97dbdc25d7d28089d31ff60de0d0fa24b8e8a82744e78fc832e784e39c4447c`.

Delivery:

- `output/pdf/PHASE_TTC_Final.pdf`
- `output/pdf/PHASE_TTC_Final.tex`
- `output/pdf/PHASE_TTC_Final_Complete_Source.zip`

The ZIP contains 197 manuscript, data, figure and generator files plus its
SHA-256 manifest. A fresh extraction passed every member hash, regenerated all
assets, compiled both source forms without LaTeX warnings, and reproduced the
delivered PDF's extracted text and pagination exactly. Consolidated-source
regeneration is byte-identical. All 56 delivered pages passed visual review.
The delivery receipt is `output/pdf/final_delivery_receipt.json`.

PDF SHA-256:
`b07b078c03bee3393924bd6c451fb0961699343573c7b5f28c9ddbaa5a7985e5`.
Source ZIP SHA-256:
`563b948d6a9d64a41896739fc1739afc8ecdad1968d3b5ac8074f1037fbef098`.

## 2026-10-29 — Scientific prose and caption revision

Revised the full manuscript in `output/pdf/phase_ttc_editorial_paper` after
paragraph-level editorial and scientific review. Captions now describe the
displayed content directly; repeated qualifications are consolidated in the
methods and discussion. The prose uses simpler terminology, fewer formulaic
conclusions, and rounded metrics where precision does not affect interpretation.
Figure and table generators carry the revised wording into future rebuilds.
The preceding edition remains in `output/pdf/phase_ttc_final_paper`.

All 21 distributed data files, all citations and cross-reference labels, the
paired bootstrap values, and the numerical analyses remain unchanged. Scientific
review retained the dated leaderboard scope, unequal temporal support, mixed
RGB/fusion error distribution, inconclusive paired intervals, historical JEPA
controls, and the distinction between labeled transfer and unscored competition
queries. No training, inference, model selection, or submission was performed.

The revised article has 52 pages, 24 numbered figures, and 27 tables. Every page
passed visual inspection after correcting a sentence-interrupting float and
the appendix table placement. Both modular and consolidated builds have zero
LaTeX warnings, overfull boxes, missing glyphs, or undefined references. A fresh
extraction of the 197-file source package regenerated every asset, reproduced
the consolidated TeX byte for byte, and matched PDF text and pagination exactly.
The source manifest adds one archive member. Public text scans found no local
machine paths or internal authoring commentary.

Validation commands:

```text
python output/pdf/phase_ttc_editorial_paper/scripts/build_final.py
python tmp/pdfs/editorial_revision/package_final.py
python tmp/pdfs/editorial_source_roundtrip_final/PHASE_TTC_Final_Source/scripts/build_final.py
python tmp/pdfs/editorial_revision/finalize_delivery.py
```

Verified copies replace the previous `PHASE_TTC_Final.pdf`,
`PHASE_TTC_Final.tex`, and `PHASE_TTC_Final_Complete_Source.zip` in Downloads.
The final receipt is `output/pdf/final_delivery_receipt.json`.

PDF SHA-256:
`70aa8a9757777286cbeb4d8b934cde5c5bd67e84ab4467e445c84c57c0c79fed`.
Source ZIP SHA-256:
`1d3137c0522fca53bcb3aa409bd93bc3298a6a61835a1055f1c578e3c0a683d2`.

## 2026-10-29 — Publication and main-branch integration

The publication source now lives in `paper/`, alongside its compiled PDF,
consolidated TeX, generators, numerical inputs, and complete source archive.
The cover names Álvaro Schwiedop Souto as the sole author, includes the supplied
logo unchanged, and links the requested email, LinkedIn profile, and repository.
There is no organization affiliation or additional author credit.

The article retains 52 pages, 24 numbered figures, and 27 tables. All 21 data
files remain unchanged. Both TeX builds pass without warnings; the source
package contains 200 manifested files and the manifest itself. A staged-byte
check confirms that Git preserves every packaged SHA-256, including original
line endings. The PDF SHA-256 is
`bbec72cf1a52591416edcb780bd5a3772412bf96c1fbf8c60800b31ee63f590f`.

Clean-clone validation now covers the full test suite, including top-level
test modules previously omitted by CI. Unit tests use deterministic synthetic
inputs through production operators instead of depending on private local
artifacts. Optional external historical evidence remains explicitly labeled.
Resource-boundary rejection, numerical tolerances, immutable source hashes,
and production admission rules are retained. ADR-0044 records the byte-preserved
orchestration sources. Offline archive verification no longer imports a missing
session-only transport.

The paper workflow verifies the committed archive against its manifest before
regenerating figures and compiling both TeX forms. The main and release workflows
run lint, formatting, type checks, the complete tests, wheel building, and both
command entry points. The smoke workflow also runs on changes to `main`.

Local validation completed before publication: Ruff lint and formatting,
Pyright, wheel build, both CLI help commands, manuscript compilation, package
verification, and the independent focused scientific review (108 passed, two
optional external-data skips). Full-suite and GitHub Actions results are recorded
on the integration pull request; merging requires those checks to pass.

No new training, model selection, benchmark submission, or scientific-result
change is part of this integration.

## 2026-10-29 — Detection continuation S0–S6

The explicitly authorized continuation uses isolated branch
`dev/eap-s0-evaluation-20261029`, based on pilot commit `24b8f36f`.
The received plan and audit package remain byte-preserved. A detector-independent
local evaluation protocol, ignored-object handling, error diagnostics, constant
and permutation controls, and a label-free external observer are implemented and
independently reviewed. The public evidence bundle regenerates without private
`output/` files:

```text
python scripts/build_detection_sota_evidence.py --from-bundle --check
```

S1 added a diverse D4096 manifest, full training-state continuation, strict frozen
inference and periodic validation. A real four-to-eight-update continuation and
independent inference gave exact prediction parity. R0 subsequently stopped after
725 completed updates because a PNG in the pinned public TAR is intrinsically
truncated; the durable checkpoint is update 512. All failure evidence is retained.
A complete media decode/hash preflight is underway before a revised fixed-cohort
restart. This is not a completed RGB baseline or a new AP claim.

Preparatory S2–S4 modules now cover train-only quality ranking, causal exposure
history and exact chunked event accumulation. The multiobject PHASE service uses
one packet ingestion and bounded object chunks while preserving native phase
quantiles. These engineering checks do not demonstrate new TTC generalization.

Detailed commands, scoped verification and remaining scientific gates are in
[the execution ledger](detection_sota_20261029/EXECUTION.md). No cloud training,
benchmark submission or external publication was performed in this continuation.

## 2026-10-30 — Authorized A40 cloud campaign and evaluation optimization

The subsequent user-authorized campaign runs on the user's RunPod A40. Exact
input residency removed repeated TAR indexing and reached approximately 256
updates/minute with warm training inputs. R0 seed 7 remains in progress; the
release model and six-replica confirmation are not complete.

An evaluation-only CPU prefetch candidate measured 1.5004x throughput on a fixed
128-frame selection panel with exact evaluation payload parity apart from timing
fields. It preserves single-frame FP32 inference and the complete scorer.
The admitted handoff retains the active run and applies the revision to pending
replicas after the existing R0 completes. Details, validation scope and rollout
receipts are in [the A40 evaluation report](detection_sota_20261029/RUNPOD_EVALUATION_OPTIMIZATION.md).
