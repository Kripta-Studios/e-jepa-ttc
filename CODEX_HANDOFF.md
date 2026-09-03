# CODEX_HANDOFF: Scientific Recovery V8

**Fecha de corte:** 2026-08-14, Europe/Madrid

**Repositorio:** `Kripta-Studios/e-jepa-ttc`

**Rama de partida:** `scientific-recovery-v7-balanced-oof`

**Commit de partida:** `f9331b29596c4107430af5a8c78935bd127ccf94`

**Rama nueva recomendada:** `scientific-recovery-v8-temporal-mechanisms`

**Función de este archivo:** contrato único de implementación, ejecución, auditoría y decisión para V8.

Este documento reemplaza por completo el handoff anterior. Contiene el estado verificable del proyecto, el inventario de familias ya probadas, la evaluación del plan post-V7 existente y el trabajo que debe implementarse y ejecutarse. Un agente nuevo debe poder continuar desde aquí sin reconstruir la historia experimental.

Este protocolo sustituye el gate provisional post-V7 de retención geométrica del 60% documentado en `scientific-recovery-v7-balanced-oof@f9331b29596c4107430af5a8c78935bd127ccf94`. El gate histórico V7 sigue intacto. V8 separa dos claims: el gate de candidato TTC ordena por MiD macro-secuencia y no exige A4; el gate mecanístico registra A4, correlación física, slope y reversal para interpretar el resultado.

## 1. Decisión científica

V7 queda cerrado como negativo bajo sus gates preregistrados. Ningún brazo supera a A5 y ninguno cumple el mecanismo geométrico exigido por V7.

V8 investigará tres preguntas, en este orden:

1. ¿Qué mecanismo usa A5 para mejorar TTC y en qué regímenes falla?
2. ¿Cuánto del margen entre A5 y Garl procede de la representación temporal de entrada?
3. ¿Aporta JEPA información temporal útil cuando se compara con scratch, encoder aleatorio y futuro barajado bajo el mismo downstream?

La retención de geometría A4 pasa a ser una métrica diagnóstica. No será un requisito universal para elegir un modelo TTC. Este cambio no reinterpreta V7: sus gates permanecen intactos en los resultados históricos.

V8 no empieza con otra arquitectura grande. Primero ejecuta una autopsia sin entrenamiento, un router prospectivo con validación anidada y dos controles temporales fijos. Solo se abre una representación temporal adaptativa si esos resultados aportan evidencia de dependencia por régimen.

## 2. Estado que no debe alterarse

### 2.1. Git y entrega

- `main` no se modificó durante V7.
- El commit local y remoto de V7 coinciden en `f9331b29596c4107430af5a8c78935bd127ccf94`.
- Se conservaron `main`, A4, A6 y V7.
- Los worktrees A6 y CUDA deben permanecer intactos.
- La historia accesible contiene 376 commits, 374 son ancestros de V7 y 295 son exclusivos de la línea V7 respecto a `main`.
- Las ramas históricas eliminadas eran ancestros completos de V7. Su eliminación no borró el historial experimental.

No reescribir historia, no hacer `git reset --hard` y no mover checkpoints de A4, A5, A6, A8, Garl o V7.

### 2.2. Paquete de evidencia V7

Ruta:

```text
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/E_JEPA_TTC_V7_EVIDENCE_f9331b2.zip
```

- Tamaño: 22,19 MiB.
- SHA-256: `187bdaeda0b5347db437db96e8e4a6acc4c0dcad837c21185ea16b3a93ec65ac`.
- Contenido: 241 archivos de documentación, configuración y evidencia.
- Exclusión deliberada: checkpoints pesados.
- La auditoría local validó 34 checkpoints contra sus SHA-256.

No reconstruir cifras desde Markdown si existe un JSON o CSV firmado. Los artefactos son la fuente autoritativa.

### 2.3. Datos sellados

Durante selección V8 queda prohibido abrir o usar:

- public validation;
- private test;
- EvTTC test;
- CodaBench;
- cualquier split que no figure en el protocolo V8 congelado.

La validación pública solo se podrá abrir después de congelar un único candidato final, completar semillas, auditorías, robustez y exportación, y recibir autorización explícita del usuario. El resultado público será una confirmación única, no otra etapa de ajuste.

## 3. Resultado final V7

| Brazo | MiD puntual | Diferencia frente a A5 | Estado |
|---|---:|---:|---|
| Garl local | **144.353** | -14.096 | referencia local |
| A5 revaluado | **158.449** | 0.000 | baseline V8 |
| C2F | 158.573 | +0.125 | negativo |
| SOFT | 165.116 | +6.668 | negativo |
| T20 | 165.260 | +6.812 | negativo |
| CAP-S | 167.025 | +8.576 | negativo |
| SOFT partial-freeze | 167.826 | +9.378 | negativo |

El router A5/C2F construido después de ver los resultados obtiene `153.519 MiD`, una mejora de `-4.929` frente a A5. Sus tres folds mejoran y el bootstrap emparejado tiene mediana `-4.919`, IC95% `[-7.033, -2.910]` y `P(Δ<0)=1`. Es una hipótesis útil, pero no es un resultado confirmatorio: el diseño se eligió post hoc. V8 debe reconstruirlo prospectivamente con predicciones internas OOF y evaluación exterior intacta.

## 4. Qué se probó ya

Esta sección funciona como lista de exclusión. No abrir una variante si su diferencia consiste solo en cambiar pesos, nombres o tamaños dentro de una familia cerrada.

### 4.1. JEPA y objetivos latentes históricos

`main` ya incluía EventTubeletTransformer, JEPA temporal, predicción densa de tokens futuros, condicionamiento causal, query pooling, tubelet masking y experimentos low-label.

Dense Level Dynamics comparó `level`, `temporal_residual`, `nce` y `nce_visreg`. Todos quedaron alrededor de `201.8 MiD` y produjeron downstream casi indistinguible. NCE apenas movió el encoder; VISReg cambió geometría sin resolver TTC; temporal residual cambió la representación sin mejorar el resultado. Un backbone aleatorio congelado con pool y head podía memorizar el micro-overfit, por lo que memorizar TTC no demuestra que el preentrenamiento haya aprendido una dinámica útil.

No repetir Level, NCE o VISReg convencionales ni atribuir una mejora a JEPA sin controles de scratch, encoder aleatorio y futuro barajado.

### 4.2. Flow, KDA y atención geométrica

FlowMimic empeoró BASE con alignment, inverse-TTC y la combinación. KDA, Object-KDA, Attention Residuals y bbox-ROI tampoco superaron los gates.

No reabrir:

- optical-flow imitation global;
- inverse-TTC auxiliary global;
- KDA u Object-KDA;
- Attention Residuals;
- bbox-ROI como intervención principal;
- CARLA pretraining, que ya produjo transferencia negativa.

### 4.3. Señal event-only y fallo temporal

V4.1 a V4.3 demostraron que los eventos contienen señal TTC. En held-out se observaron Pearson aproximados de `0.56` a `0.62`; barajar eventos llevó la correlación cerca de cero. El fallo persistente apareció en dirección temporal, signo y secuencias no vistas. Una secuencia negativa llegó a `1/28` aciertos.

La pregunta ya no es si existen eventos informativos. La pregunta es cómo elegir y codificar el soporte temporal para que la dinámica se transfiera entre regímenes.

### 4.4. Geometría manual, foreground y height ratio

V4.4 probó extent radial, actividad, anisotropía, movimiento de centroide y ridge train-only. V4.5 probó weighted eAP log-eta, balance de signo, recíproco temporal y regularización de reciprocidad. La mejora MiD fue mínima y se cerró el ajuste de loss como eje principal.

V4.6 a V4.10 probaron foreground aprendido, mayor resolución, height ratio, dense temporal log-eta y fusiones. El ground truth visible-height ratio era físicamente fuerte, pero su extracción desde eventos fallaba fuera de train. La formulación densa ayudó, pero no eliminó el fallo de tracks negativos.

La etapa eAP posterior añadió full-resolution raw, deep-feature foreground, supervisión directa de ratio y distillation SAM. Todos empeoraron los baselines relevantes.

No repetir máscaras, edges de máscara, SAM, full-resolution foreground, otro ridge radial o un nuevo barrido de pesos geométricos.

### 4.5. Sign heads y routers de signo

V4.11 a V4.17 cubrieron router signo/magnitud, temporal reversal, thresholds conservadores, ensemble multiseed, odd heads, dual heads causales y signed anchor con residual acotado. Hubo mejoras locales, pero la probabilidad y el signo no generalizaron entre seeds o secuencias.

No crear otro clasificador approach/recede, threshold, odd head o factorization de signo.

### 4.6. Bottlenecks físicos y correspondencia

V4.18 probó un bottleneck físico estático. Las correlaciones útiles en train desaparecieron en validation. V4.19 obtuvo una señal modesta con correspondencia local. V4.20 probó pseudoflow con boxes y V4.21 confirmó que el target bbox era bueno: la representación no recuperaba esa geometría de forma fiable.

V4.27 a V4.31 cubrieron scale correlation vertical, posterior KL, scale más rotación, correspondencia afín local, estabilización multiescala y auditoría física. V4.29 fue prometedor en filas válidas, pero falló cobertura. V4.30 colapsó al estabilizarlo. V4.31 falló zoom analítico, slope, signo, oddness, translation leakage y swap coverage.

No reabrir un matcher de escala o afinidad que consuma la misma tensorización fija.

### 4.7. Preservación geométrica

V4.22 consiguió mejorar geometría con partial unfreeze. V4.23 a V4.25 demostraron que esa mejora podía convivir con peor TTC fuera de distribución. V4.26 corrigió el stacking con cross-fitting completo y volvió a seleccionar el anchor sin modificar.

Scientific Recovery repitió el patrón:

- V5: A4 `291.088`, A6 `211.509`, A8 `197.691`, Garl `144.353`.
- V6: A5 alcanzó `155.472` en la evaluación original mientras perdía fidelidad geométrica.
- V7: SOFT y partial-freeze intentaron proteger A4 y empeoraron TTC.

No usar “retención A4 >= 60%” como condición de campeón V8. Sí registrar la métrica para comprender el mecanismo.

### 4.8. Escala espacial, bins y capacidad

- Radius 2 produjo una mejora pequeña e incierta y empeoró low-motion.
- C2F fue indistinguible de A5 en promedio.
- T20 empeoró A5 con más bins fijos.
- CAP-S empeoró A5 con más parámetros.

No ejecutar r3/r4/r8, T30/T40, CAP-M o una pirámide espacial equivalente sin una observación nueva que la justifique.

### 4.9. Lista de exclusión V8

Quedan cerradas estas familias:

- FlowMimic y alignment global.
- KDA, Object-KDA y AttnRes.
- bbox-ROI como solución arquitectónica.
- CARLA pretraining.
- Level, NCE y NCE+VISReg convencionales.
- loss-only tuning.
- geometría estática por momentos globales.
- foreground a mask-edge a height.
- SAM y teachers RGB de segmentación.
- sign routers y direction heads.
- pseudoflow con boxes.
- geometry-heavy schedules y post-hoc geometry stacking.
- scale-correlation y local affine matcher actuales.
- radios fijos mayores.
- más bins fijos por sí solos.
- capacity scaling por sí solo.
- A4 feature distillation convencional.
- previous-pair causal transport simple.
- CVaR o temporal consensus como sustituto de una intervención representacional.

## 5. Evaluación del master plan post-V7 existente

Archivo revisado:

`E_JEPA_TTC_POST_V7_SCIENTIFIC_RECOVERY_V8_MASTER_PLAN_2026-08-14.md`

### 5.1. Acuerdo

El diagnóstico central es correcto:

- V7 debe cerrarse sin candidato.
- La preservación A4 deja de ser un axioma.
- La autopsia de A5 debe preceder a otra ronda de GPU.
- La representación temporal merece un control aislado.
- La ventana temporal adaptativa es distinta de aumentar bins fijos.
- JEPA necesita atribución separada del rendimiento downstream.
- La selección debe seguir OOF agrupado y mantener test sellado.

### 5.2. Correcciones obligatorias

No se ejecutará ese documento literalmente. V8 incorpora estas correcciones:

1. **Identidad Git actual.** El punto de partida es `f9331b2`, no el HEAD anterior `63f4854` descrito en el borrador.
2. **Router prospectivo.** El efecto A5/C2F post hoc es suficientemente grande para merecer un brazo formal con validación anidada.
3. **Control EV-TTC fijo.** Antes de aprender ventanas adaptativas se probará un banco causal fijo de seis estados exponenciales. Separa “memoria temporal útil” de “adaptación aprendida”.
4. **Paridad precisa.** `TIMEVOL20-3` no reproduce Garl completo. Solo cambia la tensorización temporal dentro de A5. Debe describirse como ablación de frontend, no como réplica exacta.
5. **Autopsia por replay.** Los CSV OOF actuales no contienen todos los tensores internos. V8-A debe cargar checkpoints y repetir inferencia.
6. **JEPA obligatorio.** La atribución se ejecutará aunque ningún frontend nuevo supere A5. En ese caso A5 será el downstream común.
7. **Validación pública sellada.** No forma parte de la selección ni se abre automáticamente al final del screen.
8. **Calidad del repositorio.** Antes de entrenar se corrigen cinco tests rotos por fixtures históricos eliminados y se impide añadir deuda Ruff nueva.

Con esas correcciones, este handoff sustituye al master plan como especificación operativa.

## 6. Evidencia externa y límite de las comparaciones

No existe un único SOTA 2026 comparable con el MiD OOF local. Los trabajos usan datasets, targets, modalidades y protocolos diferentes. V8 citará resultados externos como contexto y no como línea directa en una tabla local.

### 6.1. Garl-TTC y eAP

Fuentes:

- Paper: <https://arxiv.org/abs/2603.16303>
- Código oficial: <https://github.com/NAIL-HNU/Garl-TTC>

El paper publica en eAP test `79.7 MiD` para event-only directo, `66.2` para event-only LHR y `45.0` para RGB más eventos. Esas cifras no son comparables con `144.353` local porque cambian split, disponibilidad de modalidades y protocolo de evaluación. El release oficial deja test privado detrás de CodaBench.

El repositorio local ya fija el release de referencia en el commit `256661242b8a7f5e56aa3c1c02348b30f6e89de6` y reproduce su time-volume en `src/e_jepa_ttc/data/garl_official_preprocessing.py`.

### 6.2. EV-TTC

Fuentes:

- Código oficial: <https://github.com/anthonytec2/EV-TTC>
- DOI: <https://doi.org/10.1109/LRA.2025.3565150>

EV-TTC utiliza memoria temporal causal con seis constantes exponenciales `[0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035]`, actualización interna de `0.2 ms` y salida periódica. V8 toma estas constantes como control fijo de representación. No trasladará su resultado publicado ni afirmará que `EXP6-3` reproduce toda su red.

### 6.3. ASTW

Fuente primaria:

<https://openaccess.thecvf.com/content/CVPR2026/html/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.html>

ASTW adapta el soporte temporal a nivel de patch según densidad de eventos y máxima entropía. Se evaluó en detección y tracking, no en TTC. V8 usa la idea de adaptación causal y local como motivación. El brazo adaptativo de este plan no se llamará reproducción ASTW.

### 6.4. V-JEPA 2.1 y TESPEC

Fuentes:

- V-JEPA 2.1: <https://arxiv.org/abs/2603.14482>
- TESPEC: <https://arxiv.org/abs/2508.00913>

V-JEPA 2.1 refuerza pérdida predictiva densa, supervisión profunda y contribución de tokens visibles y enmascarados. TESPEC estudia preentrenamiento autosupervisado temporal específico de eventos. El código local ya contiene los componentes relevantes en `src/e_jepa_ttc/training/jepa.py` y `src/e_jepa_ttc/models/dense_level_dynamics_jepa.py`. V8 debe reutilizarlos y añadir controles de atribución, no copiar una arquitectura de vídeo a escala incompatible.

### 6.5. Fuentes y herramientas consultadas

- Context7, documentación estable de scikit-learn: confirmó el uso de `Pipeline(StandardScaler(), LogisticRegression(...))` para ajustar el scaler dentro de cada fold y evitar fuga.
- Hugging Face Hub CLI: confirmó la ficha y metadatos de V-JEPA 2.1. TESPEC y Event3R no estaban indexados en Papers al consultar, por lo que se mantienen sus fuentes primarias.
- DeepWiki: Garl-TTC y EV-TTC no estaban indexados. No se usa una respuesta de DeepWiki como evidencia.

Sentry, W&B y OpenAI Developer Docs no se consultan para diseñar V8: no contienen la verdad experimental de este repositorio ni son necesarios para la implementación propuesta. W&B seguirá siendo opcional; los JSON y CSV locales firmados son la fuente obligatoria.

## 7. Protocolo V8

### 7.1. Unidad experimental

- 8.192 tokens fijos.
- 9 secuencias.
- 3 outer folds agrupados por secuencia.
- `sequence_id` y `track_id` son unidades de dependencia estadística.
- Seed de screen: `7`.
- Seeds de `multiseed_replication`: `13` y `23` para un único ganador fijado en seed 7. Miden estabilidad de optimización, no aportan datos nuevos.
- Targets, filas, pesos MiD y folds deben coincidir exactamente con V7.

El protocolo se materializará en:

`configs/protocol/scientific_recovery_v8_temporal.json`

Debe incluir hashes de:

- commit base;
- dataset manifest;
- split;
- lista ordenada de token IDs;
- targets;
- pesos MiD;
- definición de folds;
- checkpoints parent;
- código Garl oficial fijado;
- configuraciones congeladas.

### 7.2. Métrica primaria

La métrica primaria sigue siendo MiD macro-secuencia a cobertura completa, calculada por el mismo código usado en V7. También se reportarán:

- MiD sample-weighted;
- MAE;
- error relativo;
- finite fraction;
- failure rate;
- cobertura;
- métricas por secuencia, track y bucket TTC;
- negative accuracy y balanced sign cuando correspondan.

No cambiar clipping, fallback o política de NaN después de observar resultados.

### 7.3. Gate de screen frente a A5

Un brazo pasa a `multiseed_replication` si cumple todos los puntos:

1. `MiD(candidate) - MiD(A5) <= -3.0`.
2. `P(Δ<0) >= 0.90` en bootstrap jerárquico emparejado.
3. `finite_fraction = 1.0`.
4. `failure_rate = 0.0`.
5. Identidad exacta de filas, folds, targets y pesos.
6. Caída de cobertura no superior a `1` punto porcentual.
7. Todas las pruebas de causalidad e integridad pasan.

Un resultado mejor que A5 pero peor que `-3 MiD` se registra como señal débil. No habilita automáticamente otra arquitectura.

### 7.4. Gate de multiseed replication

El único ganador elegido en seed 7 se repite sin cambios en seeds 13 y 23. La replicación exige:

- diferencia negativa frente a A5 en cada seed;
- diferencia media `<= -3.0 MiD`;
- IC95% jerárquico de la diferencia completamente por debajo de cero;
- misma configuración salvo seed;
- ningún fallo de integridad o cobertura.

No hay reselección, rescue ni cambios de hiperparámetros entre seeds. Si D2 o D3 da una señal JEPA causal positiva en seed 7, repetir D0, D1, el mejor de D2/D3 y D4 en seeds 13 y 23. Para afirmar que el ganador supera a la arquitectura Garl local se deben entrenar también los controles Garl de seeds 13 y 23 y cumplir el mismo criterio emparejado. Sin esos controles solo se permite decir que supera A5.

La confirmación externa es distinta: una única apertura posterior de validación pública sellada, sin tuning ni selección y con autorización explícita del usuario.

### 7.5. Bootstrap

Usar bootstrap jerárquico emparejado:

1. Remuestrear las 9 secuencias con reemplazo.
2. Dentro de cada secuencia remuestreada, remuestrear tracks con reemplazo.
3. Mantener juntas las filas emparejadas de candidato y baseline.
4. Recalcular MiD completo por réplica.
5. Usar al menos 5.000 réplicas y una seed congelada.

No hacer bootstrap por ventanas independientes.

## 8. Cambios de infraestructura antes de entrenar

### 8.1. Rama y ledger

Crear la rama desde el commit V7 exacto:

```powershell
git switch scientific-recovery-v7-balanced-oof
git pull --ff-only
git switch -c scientific-recovery-v8-temporal-mechanisms
```

Crear:

- `docs/EXPERIMENT_LEDGER.md`
- `configs/protocol/scientific_recovery_v8_temporal.json`
- `scripts/freeze_scientific_recovery_v8_configs.py`

El ledger tendrá una fila por brazo con estado `planned`, `frozen`, `running`, `completed`, `failed_integrity`, `failed_gate` o `multiseed_replicated`. Cada cambio de estado debe apuntar al JSON firmado que lo demuestra.

### 8.2. Cinco tests históricos rotos

El suite completo recoge 1.174 tests. Los cinco fallos actuales no son regresiones de modelo: tres pertenecen a `test_freeze_scientific_recovery_v5_a8_configs`, uno a `test_freeze_scientific_recovery_v5_fold_parents` y uno a `test_freeze_scientific_recovery_v5_garl_grouped`. Buscan YAML pequeños que se borraron de `artifacts/scientific_recovery_master_v3` durante la limpieza.

Solución:

1. Copiar las configuraciones mínimas autoritativas desde el paquete de evidencia o Git history a `tests/fixtures/scientific_recovery_v5/`.
2. Actualizar los cinco tests para usar fixtures versionados.
3. No restaurar artefactos generados dentro de `artifacts/`.
4. Verificar que los hashes esperados no cambian.

### 8.3. Calidad sin reformat masivo

El repositorio tiene deuda histórica de Ruff y formato. No hacer un cambio mecánico de cientos de archivos en V8.

Implementar `scripts/check_quality_baseline.py` y `configs/quality/ruff_baseline_v8.json`:

- guardar la lista normalizada de violaciones históricas;
- fallar si aparece una violación nueva;
- exigir `ruff check` y `ruff format --check` limpios en archivos V8 nuevos o modificados;
- ejecutar Pyright con `uvx --from pyright==1.1.411 pyright`;
- mantener `pytest` completo verde.

### 8.4. Estructura de artefactos

Usar:

```text
artifacts/scientific_recovery_v8/
├── protocol/
├── cache/
├── diagnostics/
├── runs/
├── results/
├── audit/
└── package/
```

`artifacts/` permanece ignorado por Git. Cada JSON de control y resultado debe llevar `artifact_sha256`; cada checkpoint debe tener un manifiesto con SHA-256. No borrar runs negativos hasta empaquetar la evidencia final.

## 9. Contratos de código nuevos

### 9.1. Representación temporal de endpoints

Crear `src/e_jepa_ttc/data/scientific_recovery_v8.py` con:

```python
@dataclass(frozen=True)
class TemporalRepresentationOutput:
    tensor: torch.Tensor
    endpoint_us: int
    support_start_us: int
    support_end_us: int
    event_count: int
    finite: bool
    source: str
    diagnostics: dict[str, float]


class TemporalEndpointRepresentation(Protocol):
    def encode(
        self,
        events: EventBatch,
        endpoint_us: int,
        roi: torch.Tensor,
    ) -> TemporalRepresentationOutput: ...
```

Invariantes:

- `support_end_us <= endpoint_us`;
- ningún evento futuro puede afectar al tensor;
- ROI, resolución y normalización quedan registradas;
- ventanas vacías devuelven un tensor finito y `event_count=0`;
- la salida es determinista para la misma entrada y configuración.

### 9.2. Batch V8 aislado

`src/e_jepa_ttc/data/object_event_v4.py` fija `EVENT_V4_STEPS=3`. No modificar ese contrato histórico.

Definir en el módulo V8:

```python
@dataclass(frozen=True)
class ScientificRecoveryV8Batch:
    representations: torch.Tensor  # [B, steps, channels, H, W]
    endpoint_us: torch.Tensor       # [B, steps]
    token_id: list[str]
    sequence_id: list[str]
    track_id: list[str]
    target_ttc: torch.Tensor
    sample_weight: torch.Tensor
    metadata: dict[str, Any]
```

`steps` podrá ser 2 o 3 únicamente en el loader V8. Los loaders y tests V4 deben conservar su comportamiento.

### 9.3. Esquema OOF V8

Cada CSV de predicciones debe contener, como mínimo:

```text
token_id
sequence_id
track_id
outer_fold
seed
target_ttc
sample_weight
prediction_ttc
prediction_log_variance
finite
failure_reason
event_count
event_rate
support_ms
model_name
config_sha256
checkpoint_sha256
```

Los diagnósticos de autopsia añadirán columnas del mecanismo descritas en la fase A.

### 9.4. Esquema de agregado

Cada JSON agregado debe incluir:

```text
schema_version
status
git_commit
protocol_sha256
config_sha256
seed
folds
row_identity_sha256
target_sha256
prediction_sha256
checkpoint_sha256
metrics
per_sequence
per_bucket
bootstrap
integrity_checks
gate_decision
artifact_sha256
```

## 10. Fase A: autopsia mecanística

**Coste:** inferencia, sin entrenamiento.

**Objetivo:** decidir si A5 aprende una variable física útil, un atajo supervisado o una combinación dependiente del régimen.

### 10.1. Archivos

Crear:

- `src/e_jepa_ttc/evaluation/scientific_recovery_v8.py`
- `scripts/replay_scientific_recovery_v8_mechanisms.py`
- `scripts/aggregate_scientific_recovery_v8_autopsy.py`
- `tests/unit/test_scientific_recovery_v8_autopsy.py`

Reutilizar patrones de:

- `scripts/analyze_v5_a8_oof_failure_modes.py`
- `scripts/reevaluate_v7_baselines.py`
- `scripts/aggregate_v7_fold_results.py`
- `scripts/audit_v7_fold_geometry.py`

### 10.2. Modelos y filas

Reproducir inferencia de A5, C2F y Garl sobre las 8.192 filas exactas. No basta con leer los CSV V7 porque no contienen todos los tensores internos.

Para A5 y C2F exportar desde `CausalScaleTTCOutput`:

- predicción TTC final;
- log-varianza e incertidumbre;
- known mask, support y margin;
- `pair_log_height_ratio`;
- `analytic_log_height_ratio`;
- `residual_log_height_ratio`;
- pair TTC e inverse TTC;
- contribución de pair actual y anterior;
- peso o salida de blend;
- foreground y masa efectiva;
- geometry tokens y pair tokens;
- transport raw y transport tokens;
- endpoint feature norms;
- event count, rate, occupancy y entropía;
- cycle consistency cuando sea definible.

### 10.3. Contrafactuales

Ejecutar sobre el mismo checkpoint, sin reentrenar:

1. residual fijado a cero;
2. transport fijado a cero;
3. solo pair actual;
4. solo pair anterior;
5. blend neutralizado;
6. eventos fijados a cero;
7. inversión causal del orden temporal;
8. perturbación del prefijo futuro, que no debe cambiar ninguna salida anterior;
9. permutación espacial controlada;
10. dropout de eventos por intensidad, solo para medir sensibilidad.

Cada contrafactual debe conservar row identity y producir su propio hash de predicciones.

Registrar además esta tabla factorial de replay para A5. Cada fila usa el mismo checkpoint y las mismas filas OOF.

| Analítico | Residual | Transport | Blend de pair anterior |
|---|---|---|---|
| activo | cero | cero | cero |
| activo | activo | cero | cero |
| activo | cero | activo | cero |
| activo | activo | activo | cero |
| activo | activo | activo | activo |

Reportar cada delta por bucket TTC, secuencia, densidad de eventos, movimiento y signo.

### 10.4. Cortes de análisis

Calcular error y contribuciones por:

- TTC `0-3`, `3-6`, `>6`;
- targets negativos o receding;
- cuartil de event density;
- cuartil de motion magnitude;
- entropía de ocupación;
- secuencia;
- track;
- categoría, solo para analizar y nunca como feature del modelo;
- guard margin;
- confianza o log-varianza.

### 10.5. Clasificación del mecanismo

- **H1, física no capturada por la auditoría A4:** las salidas analíticas o residuales conservan correlación out-of-fold con dinámica y explican la mejora de A5 sin depender de identidad de secuencia.
- **H2, shortcut supervisado:** la mejora desaparece bajo cambios inocuos, se concentra por secuencia o el residual domina sin relación estable con dinámica.
- **H3, mezcla por régimen:** A5 y C2F ganan en regiones separadas y esas regiones se predicen con variables causales disponibles en inferencia.

Guardar la decisión en `artifacts/scientific_recovery_v8/diagnostics/mechanism_autopsy.json`. La decisión debe derivarse de reglas codificadas, no de una frase escrita después de ver gráficos.

### 10.6. Diagnóstico opcional SOFT

Se puede medir por batch:

```text
cos(grad L_TTC, grad L_geometry)
```

Esto sirve para explicar V4.23, V4.24 y V7-SOFT. No abre PCGrad ni otro brazo de preservación geométrica.

## 11. Fase R: router A5/C2F prospectivo

**Coste:** entrenamiento anidado.

**Objetivo:** comprobar si el resultado post hoc `153.519` refleja complementariedad transferible.

### 11.1. Archivos

Crear:

- `src/e_jepa_ttc/models/causal_expert_router.py`
- `src/e_jepa_ttc/evaluation/nested_router.py`
- `scripts/run_scientific_recovery_v8_nested_router.py`
- `scripts/aggregate_scientific_recovery_v8_router.py`
- `configs/experiment/scientific_recovery_v8_fold_chain/router_fold{0,1,2}_seed7.yaml`
- `tests/unit/test_scientific_recovery_v8_router.py`
- `tests/integration/test_scientific_recovery_v8_router_smoke.py`

### 11.2. Cross-fitting anidado

Para cada outer fold:

1. Mantener intactas sus 3 secuencias dev.
2. Tomar las 6 secuencias outer-train.
3. Formar 3 inner folds agrupados. Cada inner dev tendrá dos secuencias, una de cada uno de los dos grupos exteriores restantes, emparejadas en orden lexicográfico congelado.
4. Entrenar A5 y C2F en las 4 secuencias inner-train.
5. Predecir las 2 secuencias inner-dev.
6. Concatenar las tres predicciones inner OOF. Ninguna fila usada para ajustar el router puede ser in-sample para su experto.
7. Ajustar el router solo con esas predicciones inner OOF.
8. Entrenar A5 y C2F finales en las 6 secuencias outer-train.
9. Evaluar ambos una vez en outer-dev.
10. Aplicar el router congelado a outer-dev.

### 11.3. Label y features

Label por fila:

```text
loss_i = |log(1 - 0.1 / TTC_i) - log(1 - 0.1 / pred_i)| * 10^4
y_router = 1 si loss_C2F_i < loss_A5_i, en otro caso 0
```

`loss_i` es la contribución cruda MiD oficial. El peso base por fila es el coeficiente del bucket MiD dividido por las filas de ese bucket dentro de la secuencia y por nueve secuencias. El peso efectivo de `LogisticRegression.fit` es `peso_base * abs(loss_C2F_i - loss_A5_i)`. Los empates pueden tener peso cero. Firmar el hash ordenado, suma, mínimo, máximo y masa por clase de esos pesos.

Features permitidas, en orden fijo:

```text
shared_event_count_log1p
shared_event_rate_log1p
a5_flow
a5_margin
a5_log_variance
c2f_flow
c2f_margin
c2f_log_variance
```

Prohibido incluir TTC target, predicción TTC cruda, bbox, bucket TTC, fold, sequence ID, track ID, categoría o cualquier dato futuro.

### 11.4. Modelo exacto

Usar un único `sklearn.pipeline.Pipeline`:

```python
Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "router",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight=None,
                solver="liblinear",
                max_iter=1000,
                random_state=seed,
            ),
        ),
    ]
)
```

El scaler se ajusta dentro del inner OOF de cada outer fold. Pasar los pesos solo a `router__sample_weight`; `class_weight` queda en `None`. Threshold fijo `0.5`. Routing duro. Sin tuning de C, threshold o features.

### 11.5. Auditorías

- comprobar que los grupos inner train/dev no se solapan;
- comprobar que outer dev nunca se usa durante fit;
- firmar scaler, coeficientes, intercept y orden de features;
- comparar con A5, C2F, promedio 50/50 y oracle por fila;
- reportar frecuencia de elección por secuencia y régimen;
- ejecutar future-prefix invariance.

El router compite bajo el gate general. Si falla, la familia de meta-routing A5/C2F se cierra.

## 12. Fase B: controles de representación temporal

**Objetivo:** cambiar la información temporal que llega a A5 sin tocar el resto del contrato.

### 12.1. B0, golden tests de preprocesado

Antes de entrenar, construir `scripts/build_scientific_recovery_v8_cache.py` y pruebas contra `official_timevolume_roi_np` de `src/e_jepa_ttc/data/garl_official_preprocessing.py`.

Comparar:

- fixtures sintéticos con timestamps en bordes;
- polaridad alterna;
- pixels repetidos;
- ventanas vacías;
- 64 tokens reales tomados solo de outer-train.

Tolerancias:

- float32: `atol=1e-6`, `rtol=1e-5`;
- roundtrip float16: `atol=5e-4`, `rtol=1e-3`.

Verificar hash del código Garl fijado, causalidad, ROI, orden de canales, duración, normalización y determinismo.

### 12.2. B1, TIMEVOL20-3

Config:

`configs/model/e_jepa_causal_scale_event_v8_timevol20_3.yaml`

Contrato:

- mismos tres endpoints `t0`, `t1`, `t2` que A5;
- misma ROI 128x128;
- mismo downstream, topología, optimizador, epochs, sampler, loss, folds y seed que A5;
- modificación estructural obligatoria: el input stem cambia de 12 a 20 canales para aceptar cada endpoint `official_timevolume_roi_np` de 100 ms;
- tensor `[B, 3, 20, 128, 128]`;
- sin canales extra de count o rate dentro del tensor.

El congelador construye ambos modelos canónicos y registra A5: 424.274 parámetros totales y 9.600 en el stem; B1: 430.674 y 16.000; delta: 6.400 parámetros, aproximadamente 1,5%. También firma FLOPs y MACs de un forward eval `[1,3,C,128,128]`. Si B1 pasa el gate por un margen cercano a 3 MiD, ejecutar un control de capacidad del stem antes de atribuir la ganancia al frontend. Ese control no se abre por defecto.

Este brazo es una ablación de tensorización temporal. No es paridad exacta con Garl porque mantiene tres endpoints, spatial operator, head y entrenamiento A5.

Ejecutar seed 7 en los tres folds y aplicar el gate general.

### 12.3. B2, EXP6-3

Config:

`configs/model/e_jepa_causal_scale_event_v8_exp6_3.yaml`

Implementar `CausalExponentialStateRepresentation` en `src/e_jepa_ttc/data/scientific_recovery_v8.py` con seis estados firmados:

```text
alpha = [0.1, 0.05, 0.025, 0.0125, 0.0075, 0.0035]
internal_dt_ms = 0.2
```

Requisitos:

- actualizar estado por secuencia y polaridad firmada;
- reset obligatorio al cambiar `sequence_id` o detectar rollback;
- snapshot causal en `t0`, `t1`, `t2`;
- tensor `[B, 3, 6, 128, 128]`;
- mismo downstream y entrenamiento A5, con la adaptación obligatoria del input stem a seis canales;
- warm-up de estado definido y registrado;
- fixture lento de referencia para verificar la implementación vectorizada.

Congelar `EV-TTC@59c498b71ae526bc2d7e570c82a078306a996b93`, `ev_ttc/include/ev_ttc/ev_processor.h` SHA-256 `439384787969f36f72bdc72e3f6a058c33847f7f8a70454a44313ffc0e9d511e` y `ev_ttc/include/ev_ttc/config.h` SHA-256 `d30bfe8b292cb8505b1e1841bb76ebbeb2e1f34b3dce13c85b383252d4a44fe7`. La ecuación de referencia añade en bin `j` masa espacial firmada `polarity * alpha * (1-alpha)^(-j)` y en el snapshot de 7 ms multiplica el estado completo por `(1-alpha)^35`. V8 usa polaridad normalizada `{-1,+1}`, origen estable `t_start_us`, bins fijos de 0,2 ms y snapshot de frontera antes de insertar el evento de la frontera. Reset por secuencia, rollback o cambio de resolución; warm-up desde estado cero; `track_id` no participa. La paridad cubre esa ecuación raster en coordenadas enteras, no ROS, lente, bilinear placement, crop o downsampling del release oficial.

B2 se ejecuta aunque B1 falle. Permite distinguir un time-volume puntual de una memoria causal multiescala.

### 12.4. B3, PAIR20-2 condicional

Config:

`configs/model/e_jepa_causal_scale_event_v8_pair20_2.yaml`

Abrir solo si B1 supera el gate de screen.

- usar dos endpoints y 20 canales por endpoint;
- batch V8 con `steps=2`;
- no modificar `EVENT_V4_STEPS`;
- mantener ROI y arquitectura espacial A5;
- documentar qué transport terms dejan de estar definidos.

B3 responde si parte de la mejora de B1 depende de tres endpoints o de la representación time-volume. No introducir a la vez crop o centering de Garl.

### 12.5. Qué no cambia en B

Queda congelado:

- filas y folds;
- labels y pesos;
- sampling;
- ROI y resolución;
- augmentations;
- backbone, head y transport operator;
- budget de entrenamiento;
- selección de checkpoint;
- fallback de inferencia.

## 13. Fase C: soporte temporal adaptativo

Esta fase está cerrada por defecto. Se abre si ocurre una de estas condiciones:

- la autopsia clasifica H3 y demuestra regímenes separables con features causales;
- B2 mejora de forma heterogénea y estable por densidad o movimiento, aunque su media no pase el gate;
- el router R pasa el gate TTC y un análisis preregistrado muestra dependencia estable de features temporales o de densidad causales en cada outer fold y secuencia, junto con causal invariance.

Cada ruta tiene un plan firmado antes de resultados en `configs/protocol/scientific_recovery_v8_c1_analysis_plans/`: `autopsy_h3`, `exp6_regime` o `router_regime`. El protocolo y el manifest fijan su ruta, SHA-256 de bytes, firma y contrato del agregado fuente seed 7. Los agregados R y EXP6 deben completar el esquema congelado: commit, hashes de config por fold, tres folds completados, hashes de predicción y checkpoint, 8.192 filas, métricas, secuencia, bucket, bootstrap e integrity checks. El agregado debe tener `status=completed`, contrato cerrado, hashes exactos de filas, targets, pesos y folds, y cobertura exacta de los tres outer folds. EXP6 y R además deben informar componentes numéricos finitos del gate TTC: delta MiD frente a A5, probabilidad bootstrap, fracción finita, failure rate y caída de cobertura. El runner recalcula los cinco criterios y rechaza un booleano `passed` que no coincida. H3 exige un agregado de autopsia que referencia, con tipo, hash de bytes y firma, un replay factorial con las cinco combinaciones congeladas y un diagnóstico completado por bucket TTC, secuencia, densidad, movimiento y signo. La regla congelada da H3 solo si hay complementariedad, predictabilidad causal por régimen, estabilidad en los tres outer folds y secuencias, e invariancia ante cambios inocuos. Si no, da H1 solo si la física analítica o residual está soportada y no hay concentración por secuencia ni residual ajeno a dinámica. En los demás casos da H2. La decisión recalculada del diagnóstico debe coincidir con el agregado fuente. La apertura usa cinco artefactos separados: el plan congelado, `scientific_recovery_v8_c1_opening_decision_v1`, un agregado fuente tipado de seed 7, `scientific_recovery_v8_regime_evidence_v1` y `scientific_recovery_v8_causal_invariance_v1`. Las referencias de evidencia viven bajo `artifacts/scientific_recovery_v8/`, incluyen SHA-256 del archivo y vinculan el SHA del protocolo, su hash de bytes y el `sample_contract` completo con `fold_definitions`. La cobertura debe contener exactamente las claves `"0"`, `"1"` y `"2"` y las nueve secuencias dev congeladas por fold. El runner rechaza autorreferencia, agregados no permitidos, planes creados durante el run, rutas absolutas, hashes declarativos sin archivo, incertidumbre aislada y dependencia aleatoria.

### 13.1. Brazo único C1, GATED-EXP6-3

Config:

`configs/model/e_jepa_causal_scale_event_v8_gated_exp6_3.yaml`

Entrada base: los seis estados de B2. B2 es su control exacto.

Para una rejilla de patches 4x4 sobre 128x128, crear ocho features causales por patch:

1. media absoluta de cada uno de los seis estados;
2. `log1p(event_count)` en los últimos 250 ms;
3. entropía binaria de ocupación en los últimos 250 ms.

Router:

```text
Conv1x1(8,16) -> GELU -> Conv1x1(16,6) -> softmax
```

Los seis pesos se interpolan a 128x128 y multiplican los seis estados. Se conservan seis mapas de salida; no se suman a un solo canal.

### 13.2. Restricciones

- sin metadata de secuencia, categoría, bbox o TTC;
- sin eventos posteriores al endpoint;
- sin sweep de patch size, hidden dim, soporte o temperatura;
- sin nuevos losses geométricos;
- mismo A5 aguas abajo;
- registrar distribución y entropía de pesos por patch y régimen.

### 13.3. Tests

- prefix invariance;
- future perturbation invariance;
- ventana vacía;
- hot pixel;
- timestamp rollback;
- temporal reversal sensitivity;
- determinismo CPU/GPU dentro de tolerancia;
- gradientes finitos;
- equivalencia con B2 cuando los logits del gate son uniformes.

El nombre del brazo debe dejar claro que es “ASTW-inspired”. No afirmar reproducción ASTW.

## 14. Fase D: atribución JEPA obligatoria

**Objetivo:** responder la pregunta original del proyecto con controles causales adecuados.

Usar el mejor downstream admisible de R, B o C. Si ninguno supera A5, usar A5. La arquitectura, representación, head, presupuesto etiquetado y splits deben ser iguales entre brazos.

### 14.1. Archivos

Crear:

- `configs/experiment/scientific_recovery_v8_jepa/d0_scratch.yaml`
- `configs/experiment/scientific_recovery_v8_jepa/d1_random_frozen.yaml`
- `configs/experiment/scientific_recovery_v8_jepa/d2_jepa_frozen.yaml`
- `configs/experiment/scientific_recovery_v8_jepa/d3_jepa_partial_ft.yaml`
- `configs/experiment/scientific_recovery_v8_jepa/d4_shuffled_future.yaml`
- `scripts/pretrain_scientific_recovery_v8_jepa.py`
- `scripts/run_scientific_recovery_v8_jepa_attribution.py`
- `scripts/aggregate_scientific_recovery_v8_jepa.py`
- `tests/integration/test_scientific_recovery_v8_jepa_smoke.py`

Reutilizar:

- `src/e_jepa_ttc/models/dense_level_dynamics_jepa.py`;
- `src/e_jepa_ttc/training/jepa.py`;
- checkpointing y hashing existentes.

### 14.2. Preentrenamiento

Por outer fold, preentrenar solo con secuencias outer-train. Prohibido leer TTC, bbox, mask, categoría o cualquier label supervisado.

Contrato inicial:

- contexto causal en `t0` y `t1`;
- target encoder EMA sobre tokens densos de `t2`;
- predictor de tokens futuros por horizonte;
- pérdida predictiva densa final e intermedia;
- contribución de tokens visibles y target;
- target encoder con stop-gradient;
- EMA programada y registrada;
- diagnóstico de std por dimensión, effective rank, covariance y cosine inter-sample;
- abortar ante colapso según la política existente;
- igualar updates, batch y seed entre JEPA correcto y shuffled-future.

No introducir NCE, VISReg o geometry loss en este ciclo.

### 14.3. Brazos

- **D0 scratch:** encoder y head supervisados desde inicialización aleatoria.
- **D1 random frozen:** encoder aleatorio congelado, mismo head y presupuesto.
- **D2 JEPA frozen:** encoder preentrenado congelado, mismo head.
- **D3 JEPA partial FT:** descongelar solo las dos últimas etapas del encoder con LR menor congelado en config.
- **D4 shuffled future:** mismo pretraining que D2, pero targets futuros se asignan mediante un derangement determinista dentro de outer-train. `pi(i) != i`, `track_pi(i) != track_i` y el proceso falla cerrado si no puede construir el emparejamiento cross-track. `track_id` sirve solo para construir ese emparejamiento. Después congelar y entrenar el mismo head.

D4 conserva inicialización, optimizer, número de updates, batches, masking, EMA, predictor, augmentations y compute de D2. D4 se ejecuta en seed 7 aunque D2 no gane. Sin D1 y D4 no se permite una afirmación causal sobre JEPA.

### 14.4. Low-label

Construir subconjuntos anidados de `1%`, `5%`, `10%`, `25%` y `100%` por secuencia y track. Los IDs del 1% deben estar contenidos en 5%, y así sucesivamente. Congelar sus hashes antes de entrenar.

Endpoint principal JEPA:

- área bajo la curva MiD frente a fracción de labels para `1%` a `25%`;
- IC95% emparejado completamente por debajo de scratch;
- D2 o D3 debe superar también a D1 y D4;
- en `100%`, la regresión frente a scratch no puede superar `3 MiD`.

Reportar 100% por separado. No dejar que un resultado full-label oculte una curva low-label negativa.

### 14.5. Resultado permitido

Una mejora downstream sin ventaja frente a D4 se atribuye a inicialización o regularización, no a predicción futura. Una mejora frente a scratch pero no frente a encoder aleatorio se considera no concluyente. Una mejora en un solo fold se registra como señal débil.

## 15. Fase E: multiseed replication y entrega

### 15.1. Seeds

Elegir un único candidato por el árbol congelado. Ejecutar seeds 13 y 23 sin cambiar hiperparámetros ni candidato. Esta fase mide estabilidad de optimización y no es confirmación externa. Si el candidato pretende superar Garl, ejecutar Garl 13 y 23 bajo el mismo protocolo.

### 15.2. Robustez

Aplicar sin reentrenar:

```text
event_dropout: 0.1, 0.3, 0.5, 0.7
timestamp_jitter_us: 50, 200, 1000
background_event_rate: 0.01, 0.05, 0.10
hot_pixel_fraction: 0.001, 0.005
dead_pixel_fraction: 0.01, 0.05
polarity_drop: positive, negative
temporal_window_scale: 0.5, 0.75, 1.25, 1.5
spatial_crop_fraction: 0.9, 0.75
```

Registrar MiD, degradación absoluta, degradación relativa, failure rate y cambio de incertidumbre. Una cabeza probabilística debe aumentar incertidumbre bajo corrupción; mantener MAE con confianza errónea no cuenta como robustez suficiente.

### 15.3. Calibración y riesgo

Calcular:

- NLL;
- coverage y anchura de intervalos 50%, 80% y 95%;
- error por cuantil de incertidumbre;
- ECE, Brier, AUROC, AUPRC, FNR y lead time para thresholds TTC congelados.

El calibrador se ajusta solo en outer-train o inner OOF, nunca en outer-dev.

### 15.4. Eficiencia

Medir por separado:

- indexación y lectura;
- tensorización temporal;
- inferencia de red;
- pipeline completo.

Reportar CPU y GPU, batch 1 y batch de evaluación, warm-up, 1.000 iteraciones o todas las disponibles, mediana, p90 y p99, eventos/s, ventanas/s, peak VRAM, RAM, parámetros y tamaño de checkpoint.

Host de referencia actual: RTX 5070 Ti Laptop con aproximadamente 11,94 GiB de VRAM. El script debe registrar el hardware real y no asumir ese nombre.

### 15.5. ONNX

Crear:

- `scripts/export_scientific_recovery_v8_onnx.py`
- `model.onnx`
- `model_metadata.json`
- `normalization.json`
- `example_input.npz`
- `example_output.json`

Verificar batch 1 en CPU con ONNX Runtime y tolerancia congelada frente a PyTorch. Si la representación requiere estado causal, exportar la red dense y documentar el state adapter fuera del grafo, o exportar estado de entrada y salida de forma explícita. No ocultar esta diferencia.

### 15.6. Informe y paquete

Crear:

- `docs/SCIENTIFIC_RECOVERY_V8_STATUS.md`
- `docs/SCIENTIFIC_RECOVERY_V8_REPORT.md`
- `scripts/build_scientific_recovery_v8_report.py`
- `scripts/package_scientific_recovery_v8_evidence.py`

Las tablas se regeneran desde JSON y CSV firmados. El paquete final debe incluir configs, manifests, hashes, logs compactos, predicciones, agregados, auditorías, figuras y documentación. Excluir datasets y checkpoints pesados; incluir un manifest de checkpoints con rutas y SHA-256.

## 16. Árbol de ejecución

El orden es obligatorio.

```text
P0 integridad y tests
|
+-- A autopsia sin entrenamiento
|
+-- R router prospectivo
|
+-- B1 TIMEVOL20-3
|
+-- B2 EXP6-3
|    |
|    +-- B3 PAIR20-2, solo si B1 pasa
|    |
|    +-- C1 GATED-EXP6-3, solo si se activa el gate mecanístico
|
+-- seleccionar mejor downstream admisible, o A5
     |
     +-- D atribución JEPA obligatoria
          |
          +-- E multiseed replication de un único ganador, robustez, exportación y paquete
```

Reglas:

- R, B1 y B2 son independientes y pueden ejecutarse después de congelar todos sus configs.
- C1 no se implementa hasta resolver el gate de apertura.
- No ejecutar semillas 13/23 para varios brazos.
- No crear una variante de rescate después de ver outer-dev.
- Un fallo de integridad invalida el run, aunque la métrica sea buena.

## 17. Runner y comandos que deben existir

### 17.1. Congelación

```powershell
uv run --no-sync python scripts/freeze_scientific_recovery_v8_configs.py `
  --protocol configs/protocol/scientific_recovery_v8_temporal.json
```

Salida:

`configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json`

### 17.2. Preflight

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python scripts/check_quality_baseline.py
uvx --from pyright==1.1.411 pyright
uv run --no-sync pytest -q
uv run --no-sync python scripts/smoke_scientific_recovery_v8.py --device cpu
```

### 17.3. Autopsia

```powershell
uv run --no-sync python scripts/replay_scientific_recovery_v8_mechanisms.py `
  --protocol configs/protocol/scientific_recovery_v8_temporal.json `
  --models a5 c2f garl `
  --device cuda
```

### 17.4. Screen completo

Crear `scripts/run_scientific_recovery_v8.ps1` con parámetros:

```powershell
./scripts/run_scientific_recovery_v8.ps1 `
  -Device cuda `
  -Stage screen
```

Stages admitidos:

```text
preflight
autopsy
router
temporal
adaptive
jepa
multiseed_replication
robustness
export
package
screen
all
```

El runner debe:

- validar firmas y hashes antes de cada stage;
- reanudar desde `state/last.pt`;
- saltar un run solo si `summary.json` existe y su firma es válida;
- detenerse ante corrupción, mismatch o NaN;
- escribir un estado por stage;
- no abrir validación pública ni CodaBench.

### 17.5. Agregado

```powershell
uv run --no-sync python scripts/aggregate_scientific_recovery_v8.py `
  --protocol configs/protocol/scientific_recovery_v8_temporal.json `
  --results-root artifacts/scientific_recovery_v8/results
```

### 17.6. Multiseed replication

```powershell
./scripts/run_scientific_recovery_v8.ps1 `
  -Device cuda `
  -Stage multiseed_replication `
  -Candidate <frozen_candidate_id>
```

El runner debe rechazar un candidato que no figure como `multiseed_replication_candidate=true` en el agregado firmado de seed 7. La confirmación externa permanece fuera del runner y requiere autorización del usuario para abrir la validación pública sellada.

## 18. Naming, resume y fallo

Nombre de run:

```text
scientific_recovery_v8_<arm>_outer<fold>_seed<seed>
```

Para inner folds del router:

```text
scientific_recovery_v8_router_<expert>_outer<fold>_inner<fold>_seed<seed>
```

Cada run debe guardar:

```text
config.yaml
environment.json
state/last.pt
checkpoints/model_best.pt
checkpoint_manifest.json
train_metrics.csv
dev_predictions.csv
summary.json
stdout.log
stderr.log
```

Política de fallo:

- `summary.json` firmado y completo: no reentrenar.
- `last.pt` válido sin summary: reanudar.
- checkpoint corrupto: marcar `failed_integrity`; no sobrescribirlo.
- OOM: registrar el fallo; solo se permite reducir batch con gradient accumulation si el effective batch permanece idéntico y la regla estaba congelada.
- NaN o cobertura incompleta: invalidar el run; no imputar salvo fallback ya preregistrado.
- interrupción manual: conservar estado y log.

## 19. Tests mínimos V8

### 19.1. Unitarios

- golden Garl time-volume;
- EXP6 contra implementación lenta;
- causalidad por prefijo;
- reset por secuencia y rollback;
- ventanas vacías;
- shapes para steps 2 y 3;
- preservación del contrato V4;
- row identity y hashes;
- bootstrap jerárquico;
- nested fold disjointness;
- router sin features prohibidas;
- scaler ajustado solo en train;
- determinismo del router;
- schema de autopsia;
- counterfactual replay;
- collapse diagnostics JEPA;
- shuffled-future con compute equivalente;
- subconjuntos low-label anidados;
- firma de artefactos.

### 19.2. Integración

- synthetic fixture a cache V8 a training a checkpoint a aggregate;
- resume de A5 con un frontend V8;
- router inner OOF a outer inference;
- JEPA pretrain a frozen probe;
- ONNX parity;
- runner que reanuda y no duplica runs completos.

### 19.3. Regresión

- A5 revaluado vuelve a `158.449` dentro de tolerancia congelada;
- Garl local vuelve a `144.353` dentro de tolerancia;
- hash de las 8.192 filas no cambia;
- outputs del loader V4 no cambian;
- ninguna prueba accede a public validation o test.

## 20. Instrumentación

Los artefactos locales firmados son obligatorios. W&B puede reflejar curvas si ya existe una conexión configurada, pero debe funcionar en modo offline y nunca ser la única fuente. Sentry no forma parte del entrenamiento científico; no introducirlo salvo que exista un requisito separado de servicio en producción.

Cada run registrará:

```text
experiment_id
run_name
git_commit
config_hash
seed
dataset_manifest_hash
split_version
row_identity_sha256
host
python_version
torch_version
cuda_version
gpu_name
start_time
end_time
status
checkpoint_path
checkpoint_sha256
metrics_path
artifact_sha256
```

## 21. Presupuesto y disciplina de búsqueda

V8 limita deliberadamente el número de decisiones adaptativas:

- 1 autopsia;
- 1 router prospectivo;
- 2 frontends temporales fijos obligatorios;
- 1 control pair condicional;
- 1 frontend adaptativo condicional;
- 5 brazos de atribución JEPA;
- 1 candidato de multiseed replication.

No hacer barridos cartesianos. Si un valor no está fijado por A5, Garl, EV-TTC o el protocolo anterior, debe justificarse antes de congelar y permanecer fijo durante el screen.

Estimación relativa de coste:

- P0 y A: CPU/GPU de inferencia, sin entrenamiento nuevo.
- R: 18 entrenamientos internos más 6 expertos outer si no se pueden reutilizar de forma íntegra.
- B1 y B2: 6 entrenamientos de fold.
- B3 o C1: 3 entrenamientos adicionales cada uno, solo si se abre.
- D: pretraining por fold más heads low-label; reutilizar encoder por fracción.
- E: dos seeds adicionales para un candidato y, si se reclama comparación arquitectónica, dos seeds Garl.

Antes de lanzar R, calcular el coste real con un fold smoke y registrarlo en el ledger. Si el coste excede el presupuesto disponible, no degradar el protocolo de forma silenciosa: congelar una revisión explícita.

## 22. Criterios de parada

Cerrar V8 como negativo si:

- R, B1 y B2 fallan el gate;
- C1 no se abre o falla;
- JEPA no supera scratch y shuffled-future en low-label;
- no queda un candidato reproducible para multiseed replication.

Un cierre negativo sigue siendo una entrega completa si conserva predicciones, hashes, auditorías y explica qué hipótesis quedó falsada.

Detener inmediatamente un stage si:

- detecta fuga entre folds;
- cambia el hash de filas o targets;
- usa eventos futuros;
- abre un split sellado;
- falla una firma;
- produce predicciones no finitas sin fallback preregistrado.

## 23. Definición de terminado V8

V8 termina cuando se cumple todo lo siguiente:

- [ ] rama creada desde `f9331b2`;
- [ ] protocolo y configs congelados y firmados;
- [ ] cinco tests históricos reparados con fixtures versionados;
- [ ] suite completa verde;
- [ ] no hay deuda Ruff nueva en archivos tocados;
- [ ] autopsia A5/C2F/Garl completa;
- [ ] router prospectivo completo;
- [ ] TIMEVOL20-3 completo;
- [ ] EXP6-3 completo;
- [ ] decisiones B3 y C1 documentadas por gates;
- [ ] atribución JEPA D0 a D4 completa;
- [ ] low-label completo;
- [ ] un candidato replicado en seeds 7/13/23 o cierre negativo formal;
- [ ] robustez, calibración y eficiencia ejecutadas para el candidato final;
- [ ] exportación ONNX verificada si existe candidato;
- [ ] todos los resultados proceden de CSV/JSON firmados;
- [ ] informe regenerable;
- [ ] paquete de evidencia con SHA-256;
- [ ] public validation, test y CodaBench siguen sellados salvo autorización explícita posterior.

## 24. Primeras acciones del siguiente agente

Ejecutar en este orden:

1. Verificar `git status`, branch y commit.
2. Crear `scientific-recovery-v8-temporal-mechanisms` desde `f9331b2`.
3. Leer este archivo, `docs/SCIENTIFIC_RECOVERY_V7_STATUS.md`, V6, V5 y el protocolo V7.
4. Crear el ledger V8.
5. Reparar los cinco tests mediante fixtures versionados.
6. Crear el protocolo V8 y congelar hashes de filas, folds, targets y parents.
7. Implementar contratos V8 y golden tests B0.
8. Implementar y ejecutar la autopsia A.
9. Congelar antes de entrenar todos los configs de R, B1 y B2.
10. Ejecutar el screen según el árbol de la sección 16.

No empezar por C1, no cambiar A5, no restaurar 115 GiB de artefactos y no usar test para decidir.

## 25. Archivos de referencia que se deben leer

En orden:

1. `CODEX_HANDOFF.md`
2. `docs/SCIENTIFIC_RECOVERY_V7_STATUS.md`
3. `docs/SCIENTIFIC_RECOVERY_V6_STATUS.md`
4. `docs/SCIENTIFIC_RECOVERY_V5_STATUS.md`
5. `STATUS.md`
6. `configs/protocol/scientific_recovery_v7_balanced_oof.json`
7. `src/e_jepa_ttc/models/causal_scale_ttc.py`
8. `src/e_jepa_ttc/data/object_event_v4.py`
9. `src/e_jepa_ttc/data/garl_official_preprocessing.py`
10. `src/e_jepa_ttc/models/dense_level_dynamics_jepa.py`
11. `src/e_jepa_ttc/training/jepa.py`
12. `scripts/analyze_v5_a8_oof_failure_modes.py`
13. `scripts/reevaluate_v7_baselines.py`
14. `scripts/aggregate_v7_fold_results.py`
15. `scripts/run_scientific_recovery_v7.ps1`

## 26. Regla final de interpretación

El patrón estable del repositorio es este: los eventos contienen señal, la escala aparente contiene señal y el foreground puede aprenderse; lo que falla es estimar y calibrar el cambio temporal correcto en secuencias y regímenes no vistos.

V8 debe aislar esa afirmación. Si un frontend temporal mejora A5, el resultado apoya la importancia de la representación temporal. Si el router prospectivo mejora, el resultado apoya una mezcla de mecanismos por régimen. Si JEPA supera scratch, encoder aleatorio y futuro barajado, el resultado apoya predicción latente futura. Ninguna de esas conclusiones se puede sustituir por una sola cifra agregada.

## V8 completion/hardening patch — one-command training DAG

The V8 implementation is completed/hardened by the post-`191632b` patch.  Its
canonical local training entrypoint is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_scientific_recovery_v8_all_trainings.ps1 `
  -Device cuda `
  -MaxParallel 2 `
  -EapRoot "E:\eAP_dataset" `
  -GarlTtcRoot "E:\GarlTTC_dataset"
```

The orchestrator executes scientific stages serially while allowing at most two
training processes concurrently *inside* a stage.  It freezes/verifies V8 before
training, runs the no-training mechanism autopsy, prospective nested router,
fixed temporal controls, gate-authorized B3/C1 controls, mandatory JEPA
attribution, and gate-authorized multiseed replication.  A downstream stage is
never opened by an unsigned or stale artifact.

Top-level logs are written under
`artifacts/scientific_recovery_v8/master_logs/<stage>/`, including separate
`stdout.log`, `stderr.log`, and `command.txt`.  Individual training jobs retain
per-run command/stdout/stderr logs and resumable state through the V8 job
substrate.  `artifacts/scientific_recovery_v8/master_state.json` records the last
master-stage transition.

This local training DAG never opens public validation, private test, EvTTC test,
or CodaBench.  Seeds 13/23 are optimization-stability replication on the same
OOF universe, not external confirmation.

The completion patch also makes these contracts explicit:

- B1/B2/B3/C1 trainers emit the exact schema consumed by the canonical aggregate.
- B3 uses the last two TIMEVOL20 endpoints and is opened only by a signed B1 pass.
- C1 starts exactly from uniform EXP6 channel weights and is opened only by signed
  preregistered mechanism/regime evidence.
- prospective router experts preserve exact macro-sequence MiD weights and outer
  dev is not used for checkpoint selection.
- the autopsy computes fold/sequence stability instead of hard-coding it and uses
  clean factorial replay contrasts.
- JEPA low-label IDs are frozen before D0--D4; shuffled-future remains a
  deterministic cross-track derangement; PAIR20 keeps a two-endpoint downstream
  while JEPA pretraining still uses `t0,t1 -> t2`.
- a router winner remains the primary TTC winner, but JEPA attribution is made on
  its A5 constituent encoder because a meta-router is not one transferable encoder.

## V9 E-Clock X0 implementation pointer

The approved, implementation-only X0 status is recorded in
`docs/SCIENTIFIC_RECOVERY_V9_ECLOCK_X0_STATUS.md`. No X0 scientific OOF result exists
at this handoff. X0-DYN-W remains non-executable, and all height-bypass claims retain
the upstream box-conditioned ROI provenance stated there.
