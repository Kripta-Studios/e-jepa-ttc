# Progreso

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
