# PLAN.md — Recuperación científica y ruta ejecutable hacia E-JEPA-TTC frente a Garl-TTC

Estado del documento: especificación de implementación.

Fecha de la auditoría de partida: 2026-08-01.

Última actualización de ejecución: 2026-08-02.

Repositorio de trabajo: `E-JEPA-TTC`.

Objetivo: construir y validar un modelo de TTC por objeto basado en pretraining
self-supervised JEPA, tokens densos y Tubelet Transformers que pueda compararse de
forma limpia con Garl-TTC y, solo si supera sus métricas bajo el mismo protocolo,
reclamar un nuevo estado del arte.

Este documento sustituye cualquier plan anterior como orden de ejecución. No
sustituye `AGENTS.md`: todas sus reglas de integridad científica, reproducibilidad,
tipado, tests y documentación siguen siendo obligatorias.

## Estado ejecutado al 2026-08-02

La infraestructura crítica está saneada y publicada en el commit `7ec2b90`:

- Ruff, formato, Pyright, la suite completa de Pytest y los smokes CLI/ONNX/
  streaming terminan con código cero;
- el error BF16 del pooling high-resolution está corregido y cubierto por una
  regresión;
- existe un trainer event-only que lee eventos eAP/Garl bajo demanda y no crea la
  caché densa global;
- una caché de 256 muestras fue verificada y consumida; el intento de 4.096
  muestras se detuvo al aproximarse a 11 GiB de RAM y se conserva como evidencia
  negativa;
- la caché completa se estima en aproximadamente 455 GiB y queda prohibida en el
  camino activo;
- el smoke real high-resolution de 16/16 muestras completó, pero su MiD de
  validación macro por secuencia fue `1868,3186`; está marcado
  `claim_eligible=false` y no es evidencia de calidad ni SOTA;
- el pretraining JEPA high-resolution compatible con tokens densos todavía no está
  implementado; el alias antiguo falla de forma explícita en vez de entrenar una
  arquitectura incompatible;
- el auditor de shortcut semántico `7d33989` reprodujo un latente con rango sano
  pero dominado por una nuisance lenta: VISReg no lo corrigió, R²-lite falló el
  gate TTC y el residuo temporal fue la mejor solución solo cuando el shortcut era
  constante; no hay autorización para incorporar R² al candidato final;
- la rama RGB-E continúa bloqueada hasta disponer de un trainer multimodal causal
  y comparable; el camino operativo actual es event-only;
- siguen pendientes las seis secuencias eAP ausentes, tres semillas, calibración y
  robustez reales, EvTTC Tabla VI, export del checkpoint final, demo real y
  CodaBench.

Phase 0 se revalidó sobre `1d9e23a` antes de implementar Level–Dynamics:

- Ruff check/format, Pyright (`0` errores/warnings), Pytest completo y
  `git diff --check` terminaron con código cero;
- el dry-run full produjo exactamente seeds `7/13/23` y freeze sin entrenar;
- el auditor semántico regeneró sus tres JSON y conservó el veredicto: residual
  condicional, VISReg solo insuficiente y R²-lite rechazado;
- el smoke GPU 16/16 actual terminó en `17,99 s` con MiD validation macro
  `2117,5968`; continúa como evidencia de integración negativa, no calidad;
- la arquitectura de cinco partes quedó predeclarada en
  `docs/dense_level_dynamics_jepa_spec_v1.md`; tras incorporar siete blockers de
  un primer review, un segundo Sol fresco emitió `proceed`.

Este commitment no cambia ningún checkbox experimental ni autoriza pilotos antes de
que pasen los tests mecanísticos y el manifest label-free firmado.

Este estado valida contratos y ejecutabilidad con coste acotado. No valida la
hipótesis científica ni autoriza un claim SOTA.

---

## 0. Contrato para el agente implementador

El agente que ejecute este plan debe obedecer estas reglas sin interpretaciones
creativas:

1. Implementar las fases en orden. No iniciar una fase cuyo gate anterior esté rojo.
2. No ejecutar entrenamientos largos hasta que datos, métricas, causalidad y paridad
   de preprocessing estén cubiertos por tests.
3. No usar EvTTC para elegir arquitectura, hiperparámetros, época o checkpoint del
   experimento denominado `zero-shot`.
4. No usar TTC, `frame_ttc`, profundidad, `box3d_Fcam`, altura 3D, categoría ni
   máscaras de evaluación como entrada del encoder JEPA.
5. Las cajas 2D pueden ser entrada solo en los protocolos que lo declaren de forma
   explícita. Una ejecución condicionada por cajas GT debe llamarse
   `oracle_bbox_roi`, nunca `raw_stream` ni `bbox_free`.
6. No escribir “SOTA” basándose en validation local, en una réplica aproximada, en
   una sola semilla, en EvTTC-32 agrupado o en cifras copiadas del artículo.
7. Toda cifra debe proceder de un JSON/CSV generado por código y debe incluir hashes
   de configuración, datos, split, checkpoint y commit.
8. Las rutas de `E:\...` son entradas locales de esta máquina. Deben recibirse por
   CLI o configuración local ignorada por Git; no deben quedar hardcodeadas en el
   paquete Python ni en configuraciones portables.
9. No modificar `E:\Garl-TTC`, `E:\eAP_dataset` ni `E:\GarlTTC_dataset`. Son fuentes
   de referencia de solo lectura. Los derivados se escriben en `artifacts/` o en un
   directorio de caché suministrado explícitamente.
10. No silenciar excepciones por muestra sin contarlas, clasificarlas y comprobar un
    umbral de descarte. Un split vacío o sesgado debe abortar.
11. No continuar automáticamente tras un test fallido, NaN, colapso JEPA, falta de
    cobertura de secuencias o discrepancia con el loader oficial.
12. Cada checkbox se marca solo después de generar el artefacto de evidencia indicado.

La prioridad real es:

1. corregir la infraestructura rota;
2. reproducir el baseline oficial;
3. construir datos idénticos y sin fuga;
4. demostrar que JEPA aprende algo transferible;
5. demostrar que los tokens densos y el Tubelet Transformer mejoran al mismo modelo
   desde cero;
6. ejecutar zero-shot EvTTC congelado;
7. solicitar evaluación oficial eAP/CodaBench;
8. formular el claim.

---

## 1. Fuentes de verdad y entorno local

### 1.1 Raíces locales verificadas

El código debe aceptar los siguientes argumentos o variables de entorno:

```text
EAP_ROOT=E:\eAP_dataset
GARLTTC_DATA_ROOT=E:\GarlTTC_dataset
GARLTTC_RELEASE_ROOT=E:\Garl-TTC
EVTTC_MANIFESTS=<lista de manifests suministrada por el usuario>
```

Significado:

- `E:\eAP_dataset`: medios eAP. Contiene `data/train.parquet` con 118.247
  filas y, entre otros campos, `K_event` y `T_event_ego` por muestra.
- `E:\GarlTTC_dataset`: anotaciones públicas Garl-TTC. Contiene:
  - `data/train.parquet`: 88.744 filas y columnas de identidad, timestamps,
    rutas RGB/eventos, ventanas, cajas 2D y `mask_paths`;
  - `annotations/train.parquet`: 88.744 filas con `ttc`, `frame_ttc`,
    `box3d_h` y `box3d_Fcam`;
  - `data/test_inputs.parquet`: entradas públicas del test;
  - `splits/train.txt` y `splits/test.txt`;
  - no contiene TTC GT privado del test.
- `E:\Garl-TTC`: release oficial de código y checkpoints. Commit auditado:
  `256661242b8a7f5e56aa3c1c02348b30f6e89de6`.

Checkpoints oficiales auditados:

```text
E96A613A4FB877A1969D57AB562CADBA89961FB202F5F2F2F0658F333A0D443E  paper_ours_full.pth
02D900D0DDF81086F4176B63663F952A7C753EEB033B13ACC5030A892A88CF70  paper_visual_only_lhr.pth
FCAF9BE47E2DAFC6F73C6C3EBD102595AE06119DCAE78AEA698A42627B2B4FEF  paper_event_only_lhr.pth
```

Crear `configs/local/paths.example.yaml` con placeholders, añadir
`configs/local/*.yaml` a `.gitignore` y permitir un archivo local real como:

```yaml
eap_root: 'E:\eAP_dataset'
garlttc_data_root: 'E:\GarlTTC_dataset'
garlttc_release_root: 'E:\Garl-TTC'
evttc_manifests: []
```

Los manifests de resultados sí pueden registrar las rutas resueltas para auditoría,
pero también deben incluir los hashes de los archivos; una ruta no es identidad de
dataset.

### 1.2 Referencias científicas y uso permitido

Referencias primarias:

- Garl-TTC/eAP: https://arxiv.org/abs/2603.16303
- release oficial: https://github.com/NAIL-HNU/Garl-TTC
- JEPADepth: https://arxiv.org/abs/2607.26600
- Patch Policy: https://arxiv.org/abs/2607.18236
- INTACT: https://arxiv.org/abs/2607.26056
- Kimi K3: https://arxiv.org/abs/2607.24653

Decisiones de diseño:

- De JEPADepth se adopta la separación online/target por EMA, predicción de regiones
  latentes bajo máscaras estructuradas y eliminación de predictor/target en
  inferencia. Su resultado también obliga a comparar inicialización fuerte frente a
  JEPA desde cero; una pérdida decreciente no basta.
- De Patch Policy se adopta el principio de conservar patches densos y usar atención
  block-causal: atención bidireccional dentro del mismo instante y causal entre
  instantes.
- INTACT estudia transición intención-acción. No es una justificación directa para
  TTC sin acciones. Sus ideas solo pueden entrar en una ablation separada si se
  define una acción física observable disponible tanto en eAP como en inferencia.
- Kimi K3 es principalmente un trabajo de modelos de lenguaje, por lo que sus
  resultados no validan directamente una arquitectura para eventos. Sí aporta dos
  mecanismos que deben evaluarse de forma separada: (a) Kimi Delta Attention (KDA)
  como memoria recurrente de coste lineal en la longitud temporal y estado de tamaño
  fijo, y (b) el camino visual MoonViT-V2, que factoriza atención espacial/temporal,
  aplica pooling temporal progresivo y reduce tokens con una transformación 2×2 tipo
  pixel-shuffle/space-to-depth. El segundo mecanismo es el más directamente aplicable
  al problema de conservar resolución espacial.
- KDA se aplicará únicamente sobre el eje temporal de patches alineados
  `[B,T,P,D]`; nunca se aplanará la rejilla 2D en una secuencia raster causal. La
  atención espacial seguirá siendo bidireccional y local/windowed. El patrón Kimi de
  tres capas KDA y una Gated MLA será una ablation, no un default: una MLA global
  reduce KV cache, pero mantiene el término cuadrático de atención token-a-token.
- `src/e_jepa_ttc/models/temporal_kda.py` ya contiene una referencia PyTorch y
  `src/e_jepa_ttc/models/hybrid_spatiotemporal_mixer.py` ya contiene arms KDA. No se
  reescriben: se auditan y extienden. El KDA de objeto K1 ya observado como negativo
  se conserva como resultado negativo; la nueva pregunta es distinta: si KDA permite
  duplicar resolución o historia temporal bajo el mismo presupuesto de memoria.
- No mezclar todas las ideas en un único modelo. Cada incorporación requiere un
  control idéntico salvo por el componente evaluado.

Mapa del código oficial que el implementador debe tratar como oracle:

| Archivo de `E:\Garl-TTC` | Comportamiento que se debe reproducir o envolver |
|---|---|
| `configs/garl_ttc_eventdecoder.yaml` | Modalidad RGBE, 128×128, delta 0.1 s, `fy`, ResNet50, late fusion, branch checkpoints y decoder. |
| `garl_ttc/datasets/event_representation.py` | Construcción exacta de los 20 planos de eventos por endpoint. |
| `garl_ttc/datasets/ttc_dataset.py` | Selección temporal, square ROI compartida, RGB/event crop, visible height y masks. |
| `garl_ttc/models/ttc_network.py` | Arquitectura, two-height LHR, carga de branches y pérdidas/pesos oficiales. |
| `garl_ttc/engine/metrics.py` | Conversión ratio→TTC, MiD, RTE, FR, buckets firmados y weighted MiD. |
| `garl_ttc/engine/trainer.py` | Scheduler, checkpoints y diferencias que habrá que suplir con validation explícita. |
| `tools/infer.py` | Inferencia pública y schema de submission CodaBench. |

No copiar estos archivos dentro del repo sin conservar licencia/provenance. Preferir
un adapter o una reimplementación pequeña validada numéricamente.

---

## 2. Qué significa “batir Garl-TTC”

Hay tres tracks distintos. Sus datos y claims no se pueden mezclar.

### 2.1 Track A — benchmark oficial Garl-TTC sobre eAP test12

El artículo reporta para Garl-TTC RGB+eventos:

```text
MiDc/MiDs/MiDl/MiDn = 53.1 / 37.6 / 40.6 / 31.3
FRc/FRs/FRl/FRn     = 0.0 / 0.0 / 0.0 / 0.0
RTEc/RTEs/RTEl/RTEn = 16.6 / 20.0 / 34.1 / 28.2
num_samples          = 6762
```

Los pesos oficiales de MiD son `0.5/0.3/0.1/0.1`; la combinación de las cifras
publicadas es 45.02. La fuente pública no incluye TTC GT del test. Por tanto:

- la inferencia local solo produce `submission.json`;
- la métrica oficial procede de CodaBench o de etiquetas privadas legítimas;
- el candidato SOTA principal debe ser RGB+eventos, porque el Garl completo también
  lo es;
- un modelo event-only puede ser un resultado científico, pero no una comparación
  de modalidad equivalente al Garl completo.

Gate para claim Track A:

- score oficial mejor que Garl-TTC en el evaluador vigente;
- 6.762 predicciones, tokens únicos y cobertura 100 %;
- FR no peor, salvo que la mejora del score y el trade-off se reporten explícitamente;
- arquitectura, datos y checkpoint congelados antes de la submission;
- no más de tres submissions del estudio, declaradas de antemano;
- evidencia guardada en `artifacts/official/garlttc_codabench/<submission_id>/`.

### 2.2 Track B — zero-shot eAP → EvTTC, protocolo de la Tabla VI

La Tabla VI de Garl-TTC reporta:

```text
CCRs2-medium: RTE 8.31 %, runtime 13 ms
CCRs2-high:   RTE 10.56 %, runtime 13 ms
CCRm-medium:  RTE 12.93 %, runtime 12 ms
Average:      RTE 10.60 %, runtime 13 ms
```

Crear un protocolo separado de los splits generales de EvTTC. Debe mapear de forma
explícita los nombres del artículo a assets locales. El mapping candidato que debe
verificarse contra los metadatos originales es:

```text
CCRs2-medium -> CCRs-2-medium-100%
CCRs2-high   -> CCRs-2-high-100%
CCRm-medium  -> CCRm-medium-100%
```

No asumir el mapping por similitud de nombre. Crear
`data/protocols/garl_evttc_table_vi_v1.yaml` con ID local, ruta del manifest, hash,
número de labels, timestamp inicial/final y regla de agregación. Un test debe fallar
si aparecen varias secuencias compatibles o falta una.

Gate para claim Track B:

- modelo entrenado únicamente con eAP/Garl train público;
- cero fine-tuning, calibración, early stopping o selección con EvTTC;
- mismo protocolo de caja/ROI que Garl-TTC;
- RTE medio < 10.60 % y resultado por las tres secuencias;
- bootstrap por secuencia/track o bloques temporales, nunca por ventanas i.i.d.;
- runtime medido de forma emparejada en el mismo host para nuestro modelo y el
  checkpoint oficial. Las cifras del paper se citan, no se comparan como benchmark
  de hardware local.

### 2.3 Track C — EvTTC-32 grouped CV

El grouped CV de 32 secuencias existente es un análisis secundario útil para
robustez y arquitectura. Si se entrena o selecciona con sus labels, el resultado es
`EvTTC supervised/grouped-CV`, no zero-shot y no demuestra superar la Tabla VI.

Los resultados históricos A0/A1 se preservan como evidencia negativa/diagnóstica:

- A0 global: score 0.58452;
- A1 dense: score 0.59312, 1.47 % peor en selección aunque su MAE fue 0.41 % mejor.

No eliminar estos resultados. Indican que “dense” no mejora automáticamente; el
nuevo diseño debe demostrar que los patches llegan al readout TTC y que la atención
los utiliza.

### 2.4 Escalera de claims

Los reports solo pueden usar el claim más alto cuyo gate esté verde:

1. `implementation_smoke_passed`;
2. `official_preprocessing_parity`;
3. `official_checkpoint_reproduced_locally`;
4. `eap_validation_improvement`;
5. `frozen_zero_shot_evttc_improvement`;
6. `official_codabench_improvement`;
7. `new_sota_under_garlttc_protocol`.

---

## 3. Estado actual auditado y bloqueos que se corrigen primero

Branch auditada: `scientific-recovery-v3-hardening`.

Commit base observado: `3edf11c7cfdfe5a101a7a8441966bc41d078cfaf`.

El worktree contiene cambios del usuario. No resetear ni sobrescribir cambios
ajenos. Antes de implementar, guardar `git status --short`, `git diff --binary` y el
hash base dentro del run.

### B0 — el run v3 está roto por un contrato de artefactos incompatible

Evidencia:

- run: `artifacts/runs/eap_lhr_v3_hardening_20260801_115850`;
- `src/e_jepa_ttc/training/eap_jepa.py` escribe `metrics.json` en torno a la línea
  1013;
- `scripts/repair_eap_geo2_provenance.py` exige `summary.json` en torno a la línea
  78;
- el stage `repair_geo2_provenance` termina con `FileNotFoundError`.

Corrección:

- definir un único contrato de salida en
  `schemas/jepa_pretrain_run_v4.schema.json`;
- `metrics.json` será el artefacto canónico del pretraining;
- el reparador debe aceptar explícitamente versiones antiguas y migrarlas a v4, no
  adivinar nombres;
- añadir `--input-artifact` y `--dry-run` a
  `scripts/repair_eap_geo2_provenance.py`;
- renombrar la operación a migración, no reparación silenciosa;
- registrar `source_sha256`, `migration_version`, cambios de campos y hash de salida;
- añadir integración que entrene un mini-run real y llame a `migrate()` sobre su
  salida, no solo un unit test de un diccionario fabricado.

Gate B0:

- `tests/integration/test_eap_geo2_artifact_migration.py` verde;
- reanudación del run fallido en el stage posterior sin repetir el piloto;
- el run reanudado conserva hashes del checkpoint y de `metrics.json`.

### B1 — la caché Garl-TTC pide una columna que no existe

Evidencia:

- `src/e_jepa_ttc/data/garlttc_lhr_cache.py::_visible_heights()` exige `K_event`;
- `E:\GarlTTC_dataset\data\train.parquet` no contiene `K_event`;
- `K_event` sí existe en `E:\eAP_dataset\data\train.parquet`;
- el loader oficial usa `fy: 1694.1323524131867` en
  `E:\Garl-TTC\configs\garl_ttc_eventdecoder.yaml`.

Corrección:

- crear `src/e_jepa_ttc/data/garlttc_calibration.py`;
- implementar `CalibrationResolver` con dos modos explícitos:
  - `official_constant_fy`: usa exactamente 1694.1323524131867 para paridad;
  - `per_sample_eap_intrinsics`: join por identidad estable con
    `E:\eAP_dataset\data\train.parquet`, valida matriz 3x3 y usa `K_event[1,1]`;
- nunca buscar `K_event` en una fila Garl sin join;
- el modo oficial es default del baseline de paridad;
- el modo por muestra es ablation física y no puede compararse como réplica exacta
  sin declararlo;
- guardar `calibration_source`, `fy`, claves del join y hash del parquet eAP por
  muestra/caché.

Gate B1:

- cero descartes por `K_event` ausente;
- igualdad del visible height con el loader oficial en al menos 100 fixtures
  aleatorios, `rtol <= 1e-5`, `atol <= 1e-4` en modo oficial;
- cobertura de join por muestra 100 % o aborto en modo per-sample.

### B2 — el límite de 4.096 muestras sesga el train a tres secuencias

Evidencia sobre `data/splits/eap_pilot12_v1.json`:

```text
merged rows: 26.206
train:       21.471
validation:   4.735
```

La ordenación global actual y el `continue` al alcanzar el cap dejan en train:

```text
2cyv0Oedzg: 1531
5ilM1PX2vz: 2127
6h5yRW2LGc:  438
otras seis secuencias train: 0
```

Corrección en `src/e_jepa_ttc/data/garlttc_lhr_cache.py`:

- separar selección de filas y materialización;
- implementar `select_balanced_cache_rows()`;
- agrupar primero por split y secuencia;
- dentro de secuencia, estratificar por `public_track_id`, categoría, bucket TTC
  firmado y `sampling_group` observable;
- aplicar round-robin determinista con semilla congelada;
- ningún cap global puede aplicarse antes de asegurar mínimo por secuencia;
- registrar candidatos, seleccionados y descartados por secuencia/track/bucket;
- si `cap < sequence_count`, abortar con mensaje accionable;
- no balancear usando EvTTC ni métricas downstream.

Gate B2:

- las 9 secuencias train y las 3 validation aparecen en cualquier caché de al menos
  12 muestras;
- diferencia entre cuotas por secuencia explicada por falta de candidatos, no por
  orden lexicográfico;
- el mismo seed produce los mismos tokens y hash;
- otro seed cambia selección dentro de estratos sin cambiar cobertura.

### B3 — las labels Garl contienen TTC negativo y la selección actual exige positivo

Evidencia:

```text
train:      9.686 TTC negativos de 21.471
validation: 1.455 TTC negativos de 4.735
cap train:  2.069 TTC negativos de 4.096
```

`src/e_jepa_ttc/evaluation/object_ttc.py` rechaza ground truth no positivo,
mientras `src/e_jepa_ttc/training/eap_lhr_jepa_ttc.py::_evaluate()` llama siempre a
esa selección.

Corrección:

- crear `src/e_jepa_ttc/evaluation/garl_ttc_protocol.py` como implementación única
  de MiD, RTE, FR y buckets firmados oficiales;
- buckets exactos: crucial `(0,3]`, small `(3,6]`, large `(6,10]`, negative
  `[-10,0)`;
- failure: NaN/Inf o `abs(pred_ttc) < 0.1`, idéntico al evaluador oficial;
- preservar ratio `eta = 1 - delta_t / ttc` y documentar dominios inválidos para log;
- checkpoint selection debe usar solo validation eAP, admitir TTC firmado y combinar:
  `paper_MiD_overall`, FR y macro por secuencia;
- no llamar `grouped_ttc_selection_components()` para Garl firmado;
- añadir una API con protocolo explícito, por ejemplo
  `select_checkpoint(metrics, protocol='garl_signed_v1')`.

Gate B3:

- predicción perfecta positiva y negativa da MiD/RTE/FR cero;
- evaluación con los 4 buckets coincide con
  `E:\Garl-TTC\garl_ttc\engine\metrics.py` sobre fixtures;
- entrenar/evaluar una batch con TTC negativo no lanza excepción ni NaN de loss.

### B4 — la segunda tolerancia temporal se calcula pero no se valida

`select_temporal_indices()` calcula `second_error` y no lo usa. Corregir para exigir:

```text
abs(t_second - anchor) <= endpoint_tolerance
abs(t_first - (anchor - target_delta)) <= endpoint_tolerance
t_first < t_second
```

La auditoría del pilot12 observó dos frames por fila, ambos errores de endpoint 0 us
y equivalencia actual entre `frame_ttc[second]` y `ttc`. Esto no autoriza un fallback
silencioso.

Corrección:

- `_official_ttc_at_endpoint()` debe requerir `frame_ttc[second_index]`;
- `ttc` de fila puede usarse solo tras un `assert_allclose` documentado y como
  compatibilidad de versión explícita;
- registrar `ttc_label_index`, `ttc_label_timestamp_us` y `ttc_label_source`;
- añadir property tests con anchors desplazados que demuestren que ambos lados de la
  tolerancia se comprueban.

### B5 — JEPA está desactivado en todas las filas públicas del pilot12

Todas las 26.206 filas auditadas contienen exactamente dos endpoints. El cache marca
`jepa_valid=False` si no existe un tercer `t0`. Después, `_losses()` solo aplica JEPA
cuando `jepa_valid=True`. Resultado: la rama JEPA del entrenamiento LHR recibe loss
cero aunque `t1 -> t2` es un par causal válido.

Corrección:

- separar dos flags:
  - `jepa_pair_valid`: existe `t1 < t2` y target futuro;
  - `precontext_motion_valid`: existe `t0 < t1` para movimiento previo opcional;
- la pérdida JEPA usa `jepa_pair_valid`;
- el predictor usa tokens de t1, tiempo/horizonte y, si existe, movimiento t0→t1;
- si se requiere t0, construirlo mediante un índice por
  `(sequence_id, public_track_id, timestamp_us)` que enlace filas anteriores, sin
  consultar TTC;
- nunca invalidar t1→t2 solo porque falta t0;
- manifest debe reportar ambas tasas; el gate exige JEPA pair valid > 99 % en los
  datos oficiales auditados.

### B6 — el “zero-shot” actual no tiene el contrato de entrada Garl

Los manifests usados por el comando actual declaran:

```text
include_context_events=true
include_garl_pair=false
include_rgb=false
context shape aproximada=[3,10,90,160]
```

`_model_inputs()` exige `garl_event_roi` y `garl_delta_t_s`; el full encoder está
hardcodeado a 21 canales. El adaptador no crea esos tensores. La evaluación falla
antes de producir una comparación válida.

Corrección:

- crear `src/e_jepa_ttc/data/garl_input_contract.py` con dataclasses tipadas;
- crear `src/e_jepa_ttc/data/evttc_garl_adapter.py` que materialice exactamente la
  representación del modelo seleccionado;
- no intentar reinterpretar automáticamente un cache de 10 canales como uno de 21;
- cada manifest debe incluir `input_schema_version`, shapes, channel names,
  normalization, endpoint timestamps y protocolo de caja;
- `scripts/evaluate_eap_lhr_zero_shot.py` valida el schema antes de cargar GPU;
- un cache incompatible debe fallar con una lista exacta de campos/shapes faltantes.

### B7 — pooling global prematuro invalida la hipótesis de tokens densos

En `src/e_jepa_ttc/models/eap_lhr_jepa_ttc.py::_full_tokens()` se ejecuta
`forward_tokens(...).mean(1)`. Los patches full-frame desaparecen antes de la fusión
TTC. Los tokens densos ROI solo alimentan la cabeza foreground; el TTC utiliza
global tokens.

Corrección:

- conservar `[B,T,P,D]` desde los encoders hasta el mixer block-causal;
- usar `src/e_jepa_ttc/models/block_causal_transformer.py` para mezcla temporal;
- usar attention pooling con query de objeto para TTC, no media simple;
- permitir un global token solo como baseline A0/diagnóstico;
- añadir gradient tests que comprueben que la loss TTC alcanza patches no uniformes;
- añadir ablation `global_pool` vs `dense_block_causal` con mismo backbone y número de
  parámetros comparable.

### B8 — la fórmula LHR no tiene política física suficiente

El modelo actual usa `dt/(1-ratio)` con floor 1e-3 y residual ±0.25 s. Puede producir
valores enormes, discontinuos o negativos sin diagnóstico.

Corrección:

- predecir `log_height_t1`, `log_height_t2` o directamente `log_ratio` y recuperar
  `ratio = exp(log_ratio)`;
- conservar TTC negativo: `ratio > 1` representa alejamiento;
- definir zona casi estacionaria `abs(1-ratio) < epsilon` como TTC censurado/infinito,
  no como valor arbitrario silencioso;
- aplicar rango oficial de evaluación `[-10,10]` solo donde el protocolo lo exija;
- devolver `valid_prediction`, `failure_reason`, `raw_ratio`, `raw_ttc` y
  `clipped_ttc`;
- comparar tres heads: LHR puro, LHR+residual acotado y direct signed-log TTC;
- no permitir que el residual oculte un LHR fallido sin reportar su contribución.

### B9 — foreground supervision no está conectada

La loss exige `garl_foreground_mask`, pero la caché no lo escribe. El loader parquet
oficial también rellena `masks=[None]` aunque el parquet contiene `mask_paths`.

Corrección:

- primero reproducir Garl con decoder desactivado y declarar la diferencia;
- luego implementar lectura/generación de máscaras en un módulo separado;
- resolver `mask_paths` de forma trazable o generar teacher SAM con checkpoint/hash;
- guardar máscaras binarias comprimidas, no logits gigantes;
- validar alineación ROI/máscara después de square crop y resize 256;
- reproducir el mínimo de soporte foreground del código oficial;
- la máscara es supervisión, nunca input del modelo en inferencia;
- no sustituir una bbox rectangular por máscara y llamarla foreground GT.

### B10 — la réplica local no sustituye al baseline oficial

`src/e_jepa_ttc/models/garl_ttc_replica.py` y
`src/e_jepa_ttc/training/garl_ttc.py` son útiles como aproximación interna, pero la
comparación primaria debe ejecutar el release oficial y sus checkpoints.

Corrección:

- crear un wrapper de solo lectura que invoque el código de `E:\Garl-TTC` o replique
  sus tensores con tests de paridad;
- registrar commit, config y hashes de los tres checkpoints;
- descargar/verificar los checkpoints de ablation que se usen; si faltan, el runner
  debe fallar, no sustituir pesos;
- generar una submission con `paper_ours_full.pth` antes de entrenar el candidato.

### B11 — el repositorio no está verde

Estado observado el 2026-08-01, resuelto en `7ec2b90`:

- Ruff: 36 errores;
- `git diff --check`: whitespace en el launcher v3 eliminado;
- `tests/integration/test_no_fabricated_evidence.py`: falla con 15 falsos positivos
  sobre nombres legítimos `manifest.json`/`summary.json`;
- unit tests del run pasaron, pero la suite completa no es verde;
- no hay type checker instalado/ejecutado en CI.

Corrección:

- arreglar los 36 errores Ruff sin desactivar reglas globalmente;
- reparar whitespace;
- reemplazar el test heurístico de evidencia por validación semántica:
  - prohibir funciones que creen éxitos/metrics inventados;
  - permitir lectura/escritura legítima de manifests mediante helpers auditados;
  - inspeccionar `artifact_type`, provenance y hashes, no palabras de filename;
- añadir Pyright al extra `dev` y a `.github/workflows/ci.yaml`;
- CI debe ejecutar unit, integration, regression, Ruff, format y Pyright.

Gate B11:

```powershell
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest tests/unit -q
uv run --no-sync pytest tests/integration -q
uv run --no-sync pytest tests/regression -q
git diff --check
```

Todo debe terminar con código 0.

### B12 — el launcher modifica código y corrompe rutas Unicode en el handoff

El launcher v3 intentaba aplicar un patch durante el propio run y existían dos
copias divergentes. Ambos launchers obsoletos fueron eliminados. El traceback
guardado en `FAILURE.json`
representa `Álvaro` como `├ülvaro`, por lo que el artefacto de diagnóstico no es
fiable para rutas Unicode.

Corrección:

- un runner científico no debe ejecutar `git apply` ni mutar fuente;
- mover la aplicación/verificación del patch a un comando de setup separado;
- el runner registra commit y dirty diff al inicio;
- los runs confirmatorios exigen commit limpio; los smoke pueden aceptar dirty tree
  solo con `--allow-dirty-smoke` y hash del diff;
- el wrapper raíz delega al script canónico y no duplica lógica;
- configurar stdout/stderr/JSON como UTF-8 y añadir un test con una ruta temporal que
  contenga `Á`, espacios y caracteres no ASCII;
- el resume lee un stage manifest con hash, no infiere éxito por la existencia de un
  directorio;
- no repetir un stage completado cuyo artifact valida contra schema y hashes.

Gate B12: un fallo intencional bajo una ruta Unicode produce `FAILURE.json` parseable
con la ruta original exacta y el resume continúa desde el siguiente stage válido.

### B13 — el release declara 46 secuencias train, pero los parquets locales cubren 40

La auditoría encontró:

```text
E:\Garl-TTC\configs\splits\train.txt: 46 IDs
E:\eAP_dataset\data\train.parquet: 40 sequence_id únicos
E:\GarlTTC_dataset\data\train.parquet: 40 sequence_id únicos
```

Faltan en ambos parquets locales:

```text
AUIkq1ZJRv
CEOFFwY7c8
F6XHHaiwJg
OwA5qqkewc
mXRa2Ux9ok
uXfPJNOXVY
```

Corrección:

- añadir este chequeo al audit del release;
- registrar revisión/hash del snapshot Hugging Face y `dataset_info.json`;
- comprobar si las seis secuencias existen en el snapshot oficial actual antes de
  descargar nada;
- si están disponibles, preparar un comando explícito de adquisición/verificación
  y pedir autorización antes de una descarga material;
- si no están publicadas, denominar los entrenamientos locales
  `public_train40_retraining`, nunca “reproducción exacta del paper train46”;
- no inventar anotaciones, no reconstruir TTC para suplir Garl y no contar el
  checkpoint oficial como entrenamiento local;
- mantener el checkpoint oficial como oracle de inferencia y comparar el nuevo
  modelo bajo los datos públicos realmente accesibles;
- todos los reports deben incluir `train_sequence_coverage=40/46` hasta resolverlo.

Gate B13: `dataset_coverage.json` explica 40/46 o 46/46 y bloquea claims de
retraining exacto cuando no hay cobertura completa.

### B14 — la rama full-frame pierde detalle espacial y no escala con atención global

La configuración actual de `garlttc_lhr_cache.py` reduce el frame eAP de 1280×720 a
160×90. Con el `patch_size=16` por defecto de `EventTubeletTransformerEncoder`, la
rejilla espacial es solo 10×5 patches; además se descartan silenciosamente los diez
pixels inferiores porque 90 no es divisible por 16. Para cinco slices temporales son
250 tokens y cada patch representa aproximadamente 128×128 pixels del sensor.

La auditoría local preliminar de las 88.744 filas Garl train encontró bbox mediana de
176×143 pixels originales: la caja mediana ocupa solo 1,38×1,12 patches de la rama
full actual; 43,80 % de las cajas tienen alguna dimensión menor que un patch y
81,55 % tienen área menor que cuatro patches. Estas cifras deben regenerarse mediante
script y guardarse como JSON/CSV; no se copiarán como resultado final sin hash del
manifest.

Subir resolución sin cambiar el mixer tampoco es viable. Con `T=5`:

| Entrada/patch | Rejilla `P` | Tokens `N=T·P` | `N²` por capa global | frente a actual |
|---|---:|---:|---:|---:|
| 160×90, p16, sin padding | 10×5=50 | 250 | 62.500 | 1× |
| 256×144, p16 | 16×9=144 | 720 | 518.400 | 8,29× |
| 320×192, p16 | 20×12=240 | 1.200 | 1.440.000 | 23,04× |
| 320×192, p8 | 40×24=960 | 4.800 | 23.040.000 | 368,64× |

Corrección:

- mantener 128×128 en la ruta de paridad oficial ROI, pero usar p8 para conservar
  una rejilla 16×16 hasta el readout TTC;
- crear una ruta full-frame 256×144/p16 como escalón barato y una ruta
  320×192/p8 como arm accuracy-first;
- nunca recortar bordes para hacer divisible la entrada: padding explícito y
  `valid_patch_mask` obligatorio;
- para 320×192/p8, usar atención espacial por ventanas, mezcla temporal factorized,
  reducción 2×2 space-to-depth después de 1–2 bloques locales y atención global solo
  sobre tokens ya reducidos o un número fijo de queries;
- no ejecutar un entrenamiento largo de atención global sobre los 4.800 tokens: ese
  arm es solo un control teórico/OOM protegido;
- ejecutar el análisis de tamaño de objeto por buckets y exigir que la ruta nueva no
  empeore el bucket de objetos pequeños.

Gate B14: ninguna fila/pixel válido se pierde por divisibilidad, el profiler reproduce
los conteos teóricos y la ruta 320×192/p8 entra en 12 GB con batch/config comparable
sin degradar más de 1 % la métrica eAP validation frente al control 320×192/p16.

---

## 4. Estructura de archivos que se debe implementar

No duplicar funciones ya correctas. Extraer contratos canónicos y adaptar los
módulos existentes.

### 4.1 Configuración y protocolos

Crear:

```text
configs/local/paths.example.yaml
configs/protocol/garlttc_official_v1.yaml
configs/data/garlttc_public_train_v4.yaml
configs/model/garl_official_reference.yaml
configs/model/e_jepa_tubelet_lhr_small.yaml
configs/model/e_jepa_tubelet_lhr_rgbe.yaml
configs/model/e_jepa_highres_factorized_kda.yaml
configs/train/eap_jepa_pretrain_v4.yaml
configs/train/garl_supervised_v4.yaml
configs/experiment/garl_reproduction_v1.yaml
configs/experiment/e_jepa_garl_sota_v1.yaml
configs/experiment/highres_token_scaling_v1.yaml
data/protocols/garl_evttc_table_vi_v1.yaml
schemas/garlttc_cache_v4.schema.json
schemas/garlttc_metrics_v1.schema.json
schemas/jepa_pretrain_run_v4.schema.json
schemas/zero_shot_evttc_v1.schema.json
schemas/garlttc_submission_v1.schema.json
schemas/patch_resolution_audit_v1.schema.json
schemas/token_scaling_benchmark_v1.schema.json
```

`configs/protocol/garlttc_official_v1.yaml` debe congelar:

```yaml
protocol_id: garlttc_official_v1
seeds: [7, 13, 23]
official_release_commit: 256661242b8a7f5e56aa3c1c02348b30f6e89de6
input_size: [128, 128]
endpoint_delta_s: 0.1
event_planes_per_endpoint: 20
ttc_range: [-10.0, 10.0]
fy: 1694.1323524131867
selection_metric: paper_mid_overall_with_fr_guard
test_labels_available: false
zero_shot_selection_uses_evttc: false
codabench_submission_budget: 3
```

Se congelan `[7,13,23]` porque son las semillas solicitadas en el run v3. Los
resultados históricos con `[7,13,21]` se conservan, pero no se mezclan estadísticamente
con esta serie.

### 4.2 Datos

Crear o consolidar:

```text
src/e_jepa_ttc/data/garl_input_contract.py
src/e_jepa_ttc/data/garlttc_calibration.py
src/e_jepa_ttc/data/garlttc_preprocessing.py
src/e_jepa_ttc/data/garlttc_sampling.py
src/e_jepa_ttc/data/evttc_garl_adapter.py
```

Modificar:

```text
src/e_jepa_ttc/data/garlttc_eap.py
src/e_jepa_ttc/data/garlttc_lhr_cache.py
src/e_jepa_ttc/data/evttc_object_cache.py
src/e_jepa_ttc/data/official_protocol.py
scripts/build_eap_lhr_cache.py
scripts/audit_garlttc_lhr_v2.py
scripts/audit_official_garlttc_release.py
```

Responsabilidades:

- `garl_input_contract.py`: dataclasses y validación de shapes/roles.
- `garlttc_calibration.py`: resolución de intrínsecas y provenance.
- `garlttc_preprocessing.py`: square crop, RGB pair, time-volume oficial, resize,
  masks y visible height.
- `garlttc_sampling.py`: selección balanceada determinista antes de leer medios.
- `evttc_garl_adapter.py`: materialización exacta para P0/P1/P2 de EvTTC.
- `garlttc_lhr_cache.py`: orquestar lo anterior y emitir cache v4; no mantener copias
  privadas de las mismas fórmulas.
- `official_protocol.py`: mantener protocolo general EvTTC y añadir una clase
  separada para Tabla VI; corregir nomenclatura que confunde tablas del artículo
  EvTTC con la Tabla VI de Garl.

### 4.3 Modelos

Reutilizar:

```text
src/e_jepa_ttc/models/token_transformer.py
src/e_jepa_ttc/models/dense_patch_ttc.py
src/e_jepa_ttc/models/block_causal_transformer.py
src/e_jepa_ttc/models/temporal_kda.py
src/e_jepa_ttc/models/hybrid_spatiotemporal_mixer.py
src/e_jepa_ttc/training/jepa.py
```

Crear:

```text
src/e_jepa_ttc/models/e_jepa_tubelet_lhr.py
src/e_jepa_ttc/models/garl_reference_adapter.py
src/e_jepa_ttc/models/ttc_readout.py
src/e_jepa_ttc/models/window_spatial_attention.py
src/e_jepa_ttc/models/highres_token_pyramid.py
src/e_jepa_ttc/losses/garl_ttc.py
src/e_jepa_ttc/losses/jepa_dense.py
```

Modificar/deprecar:

```text
src/e_jepa_ttc/models/eap_lhr_jepa_ttc.py
src/e_jepa_ttc/training/eap_lhr_jepa_ttc.py
```

El módulo actual no se borra hasta que una regresión pueda cargar sus checkpoints.
Marcarlo `legacy_v3` en manifests y ofrecer migración explícita. El candidato v4
debe vivir en un nombre nuevo para impedir cargar pesos incompatibles por accidente.

### 4.4 Evaluación y scripts

Crear:

```text
src/e_jepa_ttc/evaluation/garl_ttc_protocol.py
src/e_jepa_ttc/evaluation/garl_evttc_zero_shot.py
src/e_jepa_ttc/evaluation/scientific_gates.py
src/e_jepa_ttc/evaluation/token_complexity.py
scripts/pretrain_eap_tubelet_jepa.py
scripts/train_e_jepa_tubelet_lhr.py
scripts/evaluate_garl_evttc_table_vi.py
scripts/build_garlttc_submission.py
scripts/benchmark_token_scaling.py
scripts/analyze_patch_resolution.py
```

Modificar:

```text
scripts/evaluate_eap_lhr_zero_shot.py
scripts/aggregate_eap_lhr_zero_shot.py
scripts/compare_eap_lhr_zero_shot.py
scripts/repair_eap_geo2_provenance.py
Makefile
src/e_jepa_ttc/cli.py
src/e_jepa_ttc/training/eap_jepa.py
```

No reintroducir launchers PowerShell que muten la fuente. Los comandos Python
canónicos deben ser resumibles y detectar artefactos por schema/hash, no solo por
existencia de filename.

Infraestructura que también se modifica:

```text
pyproject.toml
uv.lock
.gitignore
.github/workflows/ci.yaml
tests/integration/test_no_fabricated_evidence.py
```

---

## 5. Contrato de datos canónico v4

### 5.1 Dataclass de entrada

Implementar en `garl_input_contract.py`:

```python
@dataclass(frozen=True)
class GarlTTCModelInput:
    event_roi_endpoints: torch.Tensor   # [B, T=2, C=20, 128, 128]
    endpoint_timestamps_us: torch.Tensor  # [B, 2]
    delta_t_s: torch.Tensor             # [B]
    boxes_xyxy: torch.Tensor | None     # [B, 2, 4], solo protocolos bbox
    full_event_context: torch.Tensor | None  # [B, T, C, H, W]
    rgb_endpoints: torch.Tensor | None  # [B, 2, 3, 128, 128]
    input_valid: torch.Tensor           # [B]
    protocol_id: str
```

Supervisión separada:

```python
@dataclass(frozen=True)
class GarlTTCSupervision:
    ttc_s: torch.Tensor                 # [B], firmado
    visible_heights_px: torch.Tensor    # [B, 2]
    foreground_mask: torch.Tensor | None
    geometry_target: torch.Tensor | None
    category_index: torch.Tensor | None
```

Metadatos no tensoriales:

```python
@dataclass(frozen=True)
class GarlTTCSampleMeta:
    sequence_id: tuple[str, ...]
    public_track_id: tuple[str, ...]
    sample_token: tuple[str, ...]
    endpoint_label_index: torch.Tensor
    bbox_source: tuple[str, ...]
    calibration_source: tuple[str, ...]
```

El `forward()` solo acepta `GarlTTCModelInput`. La training loop mantiene
`GarlTTCSupervision` fuera del diccionario de kwargs del modelo. Esto sustituye el
guard basado únicamente en nombres de keys.

### 5.2 Semántica temporal

Para cada muestra:

```text
t1 = endpointo anterior
t2 = anchor actual
delta = (t2 - t1) / 1e6, objetivo 0.1 s
TTC GT = frame_ttc[index(t2)]
contexto JEPA = solo datos con timestamp <= t1
target JEPA = representación EMA de t2 o regiones futuras
```

Si se construye multihorizonte:

```text
t0 -> target t1  (100 ms)
t0 -> target t2  (200 ms)
t0 -> target t3  (300 ms, si existe)
```

La unión de filas por track debe:

- ordenar por timestamp;
- deduplicar `sample_token` y endpointos;
- no cruzar gaps > tolerancia;
- no cruzar secuencias;
- no consultar `frame_ttc` para decidir qué ventanas existen;
- generar máscaras de horizonte cuando falta futuro.

### 5.3 Representación de eventos oficial

Para el baseline de paridad, implementar exactamente
`E:\Garl-TTC\garl_ttc\datasets\event_representation.py`:

- 20 planos por endpoint;
- ventana de 0.1 s;
- por bin espacio-temporal, usar la diferencia entre el evento más reciente y el
  evento previo según la implementación oficial;
- decaimiento exponencial;
- mismo `event_pixel_diff=5`;
- square ROI compartida determinada por el máximo edge de los dos endpoints;
- resize 128×128.

No llamar “oficial” al voxel grid lineal de 10 bins existente. Mantener ambas
representaciones con nombres inequívocos:

```text
garl_timevolume20_v1
voxel_grid10_separate_polarity_v1
base_compatible_fullframe21_v1
```

Crear fixture `.npz` pequeño a partir de eventos sintéticos conocidos y comparar
salida oficial frente a la implementación local.

### 5.4 RGB y ROI

El modelo completo usa dos crops RGB de 3 canales y dos endpointos de eventos. El
crop debe usar la misma square ROI para ambos tiempos, derivada del máximo edge, como
el loader oficial. Registrar:

```text
raw_boxes_t1_t2
square_box
clipped_square_box
resize_scale
roi_out_of_frame_fraction
```

No usar crops diferentes por endpoint en el modo de paridad. Puede evaluarse como
ablation, no como reproducción.

### 5.5 Cajas, máscaras y shortcut audit

Definir protocolos:

- `P0_oracle_bbox_roi`: cajas GT/proporcionadas por benchmark. Comparable con Garl.
- `P1_predicted_bbox_roi`: detector/tracker congelado, sin cajas GT en inferencia.
- `P2_raw_fullframe`: no caja, detector/query interna. Investigación futura.

En P0, `observable_motion` derivado de las cajas de t1 y t2 es una entrada legítima
solo si se declara. Es un shortcut fuerte. Ejecutar siempre:

- baseline `bbox_only_lhr` sin eventos/RGB;
- candidato sin `observable_motion` explícito;
- candidato con cajas solo para crop y no como vector;
- candidato con vector de movimiento.

Si el vector bbox-only explica la mejora, no atribuirla a JEPA.

### 5.6 Manifest de caché v4

Debe contener como mínimo:

```text
artifact_type=garlttc_object_cache_v4
schema_version
created_at
code_commit
dirty_diff_sha256
eap_data_parquet_sha256
garl_data_parquet_sha256
garl_annotations_parquet_sha256
split_sha256
protocol_sha256
selection_seed
candidate_count_by_split_sequence_track_bucket
selected_count_by_split_sequence_track_bucket
discard_count_by_reason
discard_fraction
input_schema
channel_names
normalization
calibration_mode
bbox_protocol
jepa_pair_valid_fraction
precontext_motion_valid_fraction
ttc_positive_negative_counts
ttc_bucket_counts
shard_paths_and_sha256
```

Umbrales:

- discard total > 1 %: abortar salvo allowlist versionada y justificada;
- una secuencia esperada con cero muestras: abortar;
- una clase TTC oficial sin soporte cuando existía en candidatos: abortar;
- hash de shard distinto al manifest: abortar al cargar.

---

## 6. Reproducción obligatoria del baseline oficial Garl-TTC

### Fase G0 — auditoría automática del release

Implementar `scripts/audit_official_garlttc_release.py` para comprobar:

- root existe y es repo Git limpio;
- commit esperado;
- config oficial parseable;
- checkpoints presentes y hashes exactos;
- imports del release funcionan en su entorno;
- test inputs y assets referenciados existen;
- espacio de salida suficiente;
- GPU/CPU y versiones registradas.

Artefacto: `artifacts/audits/garl_release_v1/audit.json`.

Gate: `status=pass`, sin warnings de pesos sustituidos.

### Fase G1 — paridad de preprocessing

Seleccionar de forma determinista 100 muestras train que cubran:

- las 12 secuencias pilot;
- TTC positivo/negativo;
- cajas pequeñas/grandes/parcialmente fuera;
- baja/alta tasa de eventos.

Para cada muestra, guardar solo hashes y estadísticas, más un máximo de 10 fixtures
compactos redistribuibles si la licencia lo permite. Comparar:

```text
square ROI
RGB crop normalizado
time-volume20 t1/t2
visible heights t1/t2
delta_t
target TTC
```

Artefacto: `artifacts/parity/garl_preprocessing_v1/parity.json`.

Gate: 100/100 muestras dentro de tolerancia. Cualquier diferencia debe resolverse o
documentarse como modo no-oficial.

### Fase G2 — paridad de modelo

Con `paper_ours_full.pth`:

- ejecutar una batch con el release oficial;
- ejecutar el adapter local sobre los mismos tensores;
- comparar heights, ratio y TTC;
- verificar `eval()`, normalization, dtype y orden de canales;
- comprobar que se cargan los branch checkpoints del modelo completo cuando se
  entrena desde cero, no durante inferencia del full checkpoint.

Artefacto: `artifacts/parity/garl_model_v1/parity.json`.

Gate: output TTC dentro de tolerancia numérica acordada. Si el wrapper ejecuta
directamente el modelo oficial, el gate es identidad de inputs y outputs.

### Fase G3 — submission oficial de referencia

Ejecutar el equivalente a:

```powershell
uv run python E:\Garl-TTC\tools\infer.py `
  --config E:\Garl-TTC\configs\garl_ttc_eventdecoder.yaml `
  --checkpoint E:\Garl-TTC\checkpoints\paper_ours_full.pth `
  --data-root E:\eAP_dataset `
  --garlttc-annotation-root E:\GarlTTC_dataset `
  --output-json artifacts\official\garl_reference\submission.json
```

Validar schema, 6.762 tokens esperados si esa es la cobertura del test release,
finitud, duplicados y rango. No enviar automáticamente a CodaBench sin autorización
del usuario.

### Fase G4 — reproducción train/validation pública

El trainer oficial no selecciona best validation de forma suficiente para el nuevo
estudio. Antes de cambiarlo:

- reproducir un smoke con su config;
- crear un split train/validation por secuencias dentro de las 40 train públicas;
- congelar el split antes de resultados;
- seleccionar checkpoint solo con validation;
- mantener un arm que reproduce hiperparámetros oficiales: ResNet50 por rama,
  late fusion, 128×128, batch 128 si memoria permite, 50 epochs, Adam 1e-3,
  milestones 10/20/30/40, gamma 0.5, decoder y branch pretraining.

Si la GPU no permite batch 128, usar gradient accumulation y demostrar paridad de
batch efectivo; registrar la diferencia.

---

## 7. Arquitectura E-JEPA Tubelet LHR v4

### 7.1 Hipótesis

La hipótesis no es “un transformer es mejor”. Es:

> El pretraining JEPA causal y multihorizonte sobre eventos eAP aprende tokens
> espacio-temporales densos que preservan expansión, bordes del objeto y movimiento;
> una fusión block-causal puede usar esos tokens para estimar el height ratio/TTC con
> mejor generalización que Garl-TTC y que la misma arquitectura entrenada desde cero.

Esta hipótesis requiere cuatro controles independientes:

1. misma arquitectura desde cero;
2. inicialización JEPA label-free;
3. global pooling frente a dense block-causal;
4. ResNet50 Garl oficial frente a Tubelet Transformer.

### 7.2 Input y tokenizer

El endpoint de eventos oficial tiene 20 planos. No concatenar endpoints en 40
canales antes de definir la semántica temporal. Reestructurar a:

```text
[B, endpoint=2, plane=20, H=128, W=128]
```

Dos opciones de tokenizer, como ablation:

- `endpoint_2d_patch`: patch embed 2D por endpoint, pesos compartidos;
- `tubelet_3d`: Conv3D sobre endpoint/planos con kernel temporal pequeño.

No mezclar la dimensión de bins internos de la time-volume con la dimensión de
endpoint sin documentar el orden. `EventTubeletTransformerEncoder` debe aceptar
metadata de layout o un tensor ya reordenado; añadir asserts de shape y channel
names.

Config inicial recomendada para consumo:

```yaml
embedding_dim: 256
depth: 6
heads: 4
patch_size: 8
tubelet_size_endpoint: 1
mlp_ratio: 4
drop_path: 0.1
positional_encoding: factorized_3d_rope
input_endpoints: 2
event_planes_per_endpoint: 20
```

No aumentar tamaño hasta superar al small bajo el mismo presupuesto.

### 7.3 Dense tokens y block-causal mixer

El encoder devuelve:

```python
DenseTemporalFeatures(
    tokens=[B,T,P,D],
    spatial_shape=(Hp,Wp),
    global_diagnostic=[B,T,D],
    intermediate_tokens=dict[int, Tensor],
)
```

Reglas de atención:

- un token de t1 puede ver todos los patches de t1;
- un token de t1 no puede ver t2;
- un token de t2 puede ver t1 y t2;
- no usar máscara triangular a nivel patch que impida interacción dentro del mismo
  frame;
- los tokens de caja/tiempo se asignan al endpoint correspondiente;
- el query TTC final se coloca después de t2 y puede ver ambos endpoints.

Reutilizar `block_causal_attention_mask()` y ampliar tests. No implementar una
segunda máscara independiente.

### 7.4 Pirámide high-resolution inspirada por Kimi K3

#### Qué se adopta y qué no

La solución al crecimiento cuadrático no es sustituir toda la atención por KDA. Se
adopta la separación usada por el camino visual de Kimi K3/MoonViT-V2:

1. interacción espacial y temporal factorized;
2. procesamiento local a resolución alta;
3. reducción progresiva de tokens 2×2 preservando los cuatro subpatches en canales;
4. memoria temporal lineal solo cuando la historia es suficientemente larga;
5. refresco global ocasional, pero después de reducir tokens.

No adoptar estas dos simplificaciones incorrectas:

- aplanar `(y,x)` en orden raster y pasar `T·P` tokens por KDA causal: introduce un
  orden espacial arbitrario, pierde bidireccionalidad dentro del frame y hace que el
  resultado dependa de si se recorre izquierda-derecha o derecha-izquierda;
- asumir que Gated MLA elimina `N²`: comprime el KV cache, pero su softmax global
  sigue comparando tokens entre sí. Una sola capa global sobre 4.800 tokens puede
  dominar todo el coste aunque se intercale después de tres KDA.

#### Topología candidata

Implementar `HighResolutionTokenPyramid` con esta ruta, sin sustituir la ruta oficial
hasta superar sus gates:

```text
full event volume 320×192
  -> patch embed p8                         [B,T,24,40,D]
  -> local spatial attention 8×8, shifted [B,T,24,40,D]
  -> temporal mixer por patch              [B,T,P=960,D]
  -> local spatial attention 8×8, shifted [B,T,24,40,D]
  -> 2×2 space-to-depth + projection       [B,T,12,20,D2]
  -> local spatial + temporal mixer        [B,T,P=240,D2]
  -> M=8..16 object/global queries por cross-attention
  -> como máximo un refresh block-causal sobre patches reducidos + queries
  -> query TTC/LHR; no global mean de patches
```

Con ventanas 8×8, la parte de atención espacial high-resolution requiere
`T·15·64² = 307.200` pares para 320×192/p8/T=5, frente a 23.040.000 pares de la
atención global, antes de contar proyecciones. La atención temporal ordinaria por
patch a T=5 añade solo `P·T² = 24.000` pares; por ello KDA no es la fuente principal
de ahorro con la historia actual. El ahorro principal debe venir de ventanas y
reducción espacial.

#### Contrato de los nuevos módulos

`window_spatial_attention.py`:

- entrada/salida `[B,T,Hp,Wp,D]` y `valid_patch_mask`;
- atención bidireccional independiente por frame;
- ventanas 8×8 configurables, alternando regular/shifted para cruzar fronteras;
- relative position bias 2D o RoPE 2D consistente con las coordenadas originales;
- padding siempre enmascarado y sin contribuir a softmax, pooling o pérdidas;
- nunca consulta el TTC, bbox futuro, categoría ni mask GT.

`highres_token_pyramid.py`:

- `SpaceToDepthPatchMerge` reordena 2×2 patches de `D` a un vector `4D` sin pérdida
  antes de la proyección; no usa average pooling;
- concatena o aplica explícitamente los cuatro bits de validez, pone a cero entradas
  padded y emite el nuevo mask;
- conserva coordenadas/escala para que foreground, JEPA targets y readout puedan
  volver a la rejilla inicial;
- la proyección `4D -> D2` es la única compresión aprendida y debe compararse con
  `D2=2D`, `D2=4D` y average-pool como control negativo;
- ofrece taps dense antes y después del merge; la pérdida JEPA localizada se puede
  aplicar antes del merge y el readout TTC después del merge.

`HybridSpatiotemporalMixer`:

- extender, no duplicar, los modos existentes `block_causal`, `object_kda` y
  `aligned_patch_kda`;
- añadir un modo `highres_factorized_kda` que siempre ejecuta spatial local antes de
  KDA temporal;
- aplicar `KimiDeltaAttention` sobre `[B,T,P,D]`, manteniendo un estado independiente
  por patch/head; `P` es batch lógico, no longitud causal;
- mantener el estado recurrente FP32 bajo AMP/bfloat16;
- permitir chunked recurrence para streaming y reset por secuencia/timestamp rollback;
- el refresco global equivalente al patrón 3:1 de Kimi se ejecuta solo tras el merge
  o sobre `M` queries fijas. Comparar ratios `sin refresh`, `3:1` y `1:1`;
- no añadir FlashKDA/custom CUDA en esta fase. La referencia PyTorch debe demostrar
  corrección primero; cualquier kernel externo posterior exige licencia, commit
  fijado, paridad numérica y fallback.

KDA se activa por defecto solo si `temporal_steps >= 8` o en streaming con historia
larga. Para los dos endpoints oficiales (`T=2`) y para cinco bins (`T=5`), usar
atención temporal factorized normal salvo que la ablation mida una ventaja real.
`endpoint`, `event_bin` e `history_window` son ejes semánticos distintos y no se
concatenan para inflar artificialmente T.

La alineación espacial tampoco se da por supuesta. En ROI normalizada, `P` representa
posición relativa al objeto; en full-frame representa coordenada de sensor. Comparar:

1. patch fijo + spatial local antes de KDA;
2. alineación aprendida mediante offsets/cross-attention derivados solo de eventos;
3. object crop condicionado por bbox, únicamente en SSL-Object-Conditioned/P0.

No usar flow, depth, caja futura ni TTC para alinear el arm SSL-Pure. Un test con un
objeto sintético desplazado debe revelar si la memoria de patch fijo pierde la pista.

#### Resoluciones a soportar

- `R0`: 160×90/p16 legacy, solo regresión; documentar fila inferior perdida.
- `R1`: 256×144/p16, 16×9 patches; primer full-frame dense económico.
- `R2`: 320×192/p16, 20×12 patches; control de resolución por resize.
- `R3`: 160×96/p8, 20×12 patches; separa patch pequeño de más pixels de entrada.
- `R4`: 320×192/p8, 40×24 patches; arm high-resolution con ventanas/pirámide.
- ROI oficial: 128×128/p8, 16×16; ablation p4 solo con ventanas y merge temprano.

Todo resize de eventos debe efectuarse desde coordenadas/eventos fuente o una
representación de suficiente resolución; está prohibido hacer upsample de la cache
160×90 y llamarlo R2/R4. Los caches incluyen resolución fuente, método de binning,
antialiasing si aplica, padding, mask y hashes.

#### Gate de promoción KDA/high-resolution

Promocionar `K3_HIGHRES_FACTORIZED_KDA` solo si, frente a
`R4 + window + merge + temporal attention` con idéntico resto:

- permite al menos 2× más historia temporal o reduce peak VRAM ≥20 %;
- mejora throughput/latencia o conserva precisión con coste claramente menor;
- `paper_MiD_overall` y RTE eAP validation no empeoran más de 1 % relativo;
- no empeora el bucket de bbox pequeña ni foreground boundary metrics;
- la ventaja persiste en al menos dos longitudes T y no solo en un batch sintético.

El arm `object_kda` K1 ya negativo no se vuelve a etiquetar como éxito. El nuevo ID,
config hash y pregunta experimental deben ser distintos. Si KDA no supera el gate,
se conserva `window + merge + temporal attention`; las ideas de reducción espacial
siguen siendo válidas independientemente del resultado de KDA.

### 7.5 Rama RGB

El candidato comparable principal debe tener RGB+eventos. Implementar:

- baseline exacto ResNet50 late-fusion del release;
- variante Tubelet event + encoder RGB fuerte;
- inicialización RGB con el checkpoint visual oficial o backbone preentrenado
  documentado;
- proyección de patches RGB y eventos al mismo `D`;
- late fusion como primer control;
- cross-modal dense fusion solo después de que late fusion sea reproducible.

Un CNN RGB de tres capas como el actual no es una comparación suficiente con el
Garl completo. Puede quedar como smoke, no como candidato SOTA.

### 7.6 Predictor JEPA

Online path:

```text
eventos t<=t1
 -> online tubelet encoder
 -> tokens de contexto visibles
 -> predictor condicionado por horizon embedding y posiciones objetivo
 -> tokens predichos de t2/t3
```

Target path:

```text
eventos futuros completos
 -> target tubelet encoder EMA
 -> stop-gradient target patches
```

Requisitos:

- target inicializado como copia exacta del online;
- `requires_grad=False` y siempre `eval()`;
- EMA cosine schedule 0.99 → 0.9999;
- predictor transformer pequeño, no solo MLP global;
- embeddings de horizonte Fourier + learned;
- máscaras objetivo espaciales en bloques, entre 30 % y 60 %;
- target positions conocidas, contenido target oculto al predictor;
- pérdida en patches objetivo, no únicamente media global;
- predictor y target encoder excluidos de `inference_state_dict()`;
- downstream puede cargar solo el online encoder con reporte de keys exacto.

### 7.7 Pretraining realmente self-supervised

Separar dos regímenes en provenance:

#### SSL-Pure

- usa stream de eventos, timestamps y límites de secuencia;
- no usa TTC, cajas, categorías, masks, depth ni 3D;
- sampling temporal independiente de labels;
- es el régimen principal para claim self-supervised.

#### SSL-Object-Conditioned

- puede usar cajas 2D para crop o muestreo;
- no usa TTC/depth/3D;
- debe llamarse `annotation-conditioned self-supervision`, no unlabeled puro;
- es ablation útil para compararse con el protocolo objeto de Garl.

El flag actual `uses_labels_for_window_sampling=true` debe descomponerse en:

```text
uses_ttc_for_sampling
uses_boxes_for_sampling
uses_category_for_sampling
uses_depth_for_sampling
uses_future_labels_for_sampling
```

Gate anti-leakage: todos falsos para SSL-Pure.

### 7.8 Readout TTC/LHR

Implementar en `ttc_readout.py`:

```python
@dataclass
class TTCReadoutOutput:
    log_height_t1: Tensor
    log_height_t2: Tensor
    log_ratio: Tensor
    ratio: Tensor
    raw_ttc_s: Tensor
    ttc_s: Tensor
    valid_prediction: Tensor
    failure_code: Tensor
    residual_s: Tensor | None
    uncertainty: Tensor | None
```

Heads obligatorios:

- `lhr_two_height`: dos heights positivos;
- `lhr_direct_ratio`: ratio positivo;
- `direct_signed_log_ttc`: control directo;
- `lhr_bounded_residual`: solo si LHR puro ya funciona.

No escoger head por EvTTC. Selección en validation eAP firmada.

### 7.9 Foreground decoder

El decoder se usa solo en training y debe poder descartarse. Implementarlo después
del JEPA básico. Comparar:

- sin decoder;
- mask decoder Garl con teacher masks;
- patch-objectness self-supervised derivado de actividad de eventos, marcado como
  proxy y no como GT.

La mejora debe medirse tanto en eAP validation como en transferencia congelada. Una
mejora de IoU sin mejora TTC no promociona el componente.

### 7.10 Incertidumbre

No bloquear el primer candidato. Tras estabilizar TTC:

- predecir distribución de `log_ratio` o signed-log TTC;
- propagar a intervalos TTC evitando singularidad en ratio≈1;
- calibrar solo en eAP validation;
- evaluar coverage 50/80/95 %, NLL y error-vs-uncertainty;
- en EvTTC zero-shot no recalibrar;
- medir si incertidumbre aumenta bajo event dropout/jitter.

---

## 8. Pérdidas y selección de checkpoint

### 8.1 Pretraining JEPA

Pérdida inicial:

```text
L_ssl = L_patch_cosine
      + 0.25 * L_context_token
      + lambda_var * L_variance
      + lambda_cov * L_covariance
```

Requisitos:

- normalización L2 para cosine;
- variance/covariance sobre muestras, nunca mezclando posiciones como si fueran
  ejemplos independientes;
- registrar effective rank online/pred/target;
- registrar std por dimensión y porcentaje colapsado;
- abortar si >80 % de dimensiones tienen std < 1e-3 durante 3 evaluaciones;
- añadir gate de effective rank relativo, no solo std.

No activar geometría, foreground, categoría y JEPA a la vez en el primer run. Orden:

1. JEPA puro;
2. JEPA + anti-collapse;
3. JEPA + foreground;
4. JEPA annotation-conditioned/geometry como ablation.

#### 8.1.1 Gate de capacidad semántica y residuos temporales

El guard `std < 1e-3` detecta colapso estadístico, no garantiza que el latente
represente dinámica. El smoke eAP versionado confirma esta distinción: con dimensión
192, validation registra fracción colapsada `0,03125`, pero rangos efectivos
context/predictor/target `2,255/1,095/5,105`. Esto es una advertencia de rango; no
demuestra por sí sola qué semántica ocupa el embedding.

El artefacto `artifacts/metrics/jepa_semantic_capacity_audit_v1.json` ejecutó cinco
brazos x tres semillas sobre una dinámica TTC sintética con shortcut de 12 bits. Los
probes se entrenaron después de congelar el encoder; TTC y los bits nunca entraron en
la loss de representación. Con shortcut fijo por secuencia:

| brazo | R² dinámica | MAE log-TTC | acc. shortcut | duplicación |
|---|---:|---:|---:|---:|
| varianza actual | 0,15 | 0,39 | 0,84 | 1,93 |
| VISReg | 0,20 | 0,38 | 0,92 | 1,63 |
| residuo temporal | **0,72** | **0,29** | **0,65** | 1,06 |
| R² rate+dependencia | 0,29 | 0,36 | 0,88 | 1,92 |
| residuo+R² | 0,48 | 0,34 | 0,68 | **0,68** |

R²-lite queda rechazado: no alcanza la reducción predeclarada del 15 % en MAE
log-TTC. VISReg mejora estadísticas globales pero aumenta la decodificación del
shortcut. El residuo temporal pasa todos sus gates en la nuisance lenta.

El control pixel-matched actualiza el shortcut cada frame. En ese régimen, la
varianza actual obtiene R² `0,74` y MAE `0,19`, mientras el residuo obtiene R²
`-0,05` y MAE `0,40`. El residuo no reemplaza al objetivo de nivel. La arquitectura
mínima a probar es:

```text
tokens densos causales
  -> z_level: escala/contenido necesario
  -> z_delta: expansión/movimiento temporal

L = L_predict_level + lambda_delta * L_predict_delta
  + anti-collapse separado sobre nivel y delta
```

Gate real obligatorio, sobre las mismas 256–2.048 filas raw y seeds:

1. `level` frente a `level+temporal_residual` con igual encoder/predictor/compute;
2. probes congelados de expansión, event rate, ID de secuencia y TTC;
3. el residual debe mejorar expansión/TTC sin aumentar el probe de ID de secuencia;
4. no promocionar si solo aumenta rango o reduce la loss SSL;
5. no implementar CMI, total correlation, HSIC de entrenamiento ni optimización
   dual antes de este gate. Batch 4 no estima esas cantidades con fiabilidad.

INTACT requiere acciones expertas y no aplica al protocolo TTC perceptivo actual.
Podría reabrirse solo si un dataset futuro aporta steering/freno/aceleración y el
objetivo cambia explícitamente de estimación a control.

### 8.2 Supervised Garl

Reproducir primero la loss oficial:

```text
visible_height SmoothL1, peso 1
MiD, peso 1, activo después de epoch 5
foreground focal t1/t2, peso oficial cuando haya masks
```

El código oficial define pesos altos para mask focal y usa el LHR como mecanismo
principal. Implementar una config `official_exact` antes de modificar pérdidas.

Luego evaluar:

```text
L = lambda_height * L_height
  + lambda_mid * L_mid
  + lambda_ttc * L_signed_log_ttc
  + lambda_fg * L_foreground
  + lambda_unc * L_uncertainty
```

No usar una loss directa TTC en el arm “official exact” si el release no la usa.

### 8.3 Checkpoint SSL

El piloto Geo2 observado eligió epoch 3 por total validation:

```text
val total:               0.14991 -> 0.10669
val JEPA:                0.04208 -> 0.00618
val geometry:            0.14294 -> 0.14991 (empeora)
val patch IoU:           0.12302 -> 0.14452
val context effective rank: 14.45 -> 4.97
```

Esto demuestra que minimizar suma total puede seleccionar una representación menos
diversa y peor geométricamente. Corregir selección:

- SSL best primario por validation JEPA solo entre checkpoints que pasan health
  gates;
- guardar también best geometry y best transfer-probe, sin mirar EvTTC;
- ejecutar linear probe eAP validation congelado sobre todos los checkpoints
  candidatos;
- elegir regla antes del entrenamiento y registrarla;
- no seleccionar por un probe construido con el test.

### 8.4 Checkpoint downstream

Usar validation eAP firmado:

1. FR guard;
2. `paper_MiD_overall`;
3. macro por secuencia;
4. RTE macro como desempate;
5. nunca training loss o EvTTC.

Guardar `best.pt`, `last.pt` y top-3 con score. `best.pt` debe registrar todos los
componentes que justificaron la selección.

---

## 9. Fases de implementación y gates

### Fase 0 — saneamiento y contratos

Tareas:

- [ ] preservar diff/status del usuario;
- [ ] corregir B0 y migración de artefactos;
- [ ] corregir Ruff/whitespace/test de evidencia;
- [ ] añadir Pyright;
- [ ] crear schemas v4;
- [ ] congelar protocolo y semillas;
- [ ] hacer resumible el runner por stages.

Gate: CI local completa verde. No GPU larga.

### Fase 1 — oracle oficial reproducido

Tareas:

- [ ] auditar release/checkpoints;
- [ ] implementar fixtures de preprocessing;
- [ ] comparar 100 muestras;
- [ ] reproducir outputs del checkpoint oficial;
- [ ] generar submission reference sin enviarla;
- [ ] generar adapter exacto para las tres secuencias Tabla VI;
- [ ] ejecutar el checkpoint oficial en EvTTC y comprobar proximidad a 10.60 %.

Si el checkpoint oficial no reproduce la tabla dentro de una tolerancia predefinida,
detener. El problema es adapter/protocolo, no arquitectura candidata.

Gate recomendado para Tabla VI local:

- diferencia absoluta media <= 0.5 puntos RTE;
- ninguna secuencia difiere > 1 punto sin explicación trazable;
- mismo count de muestras que el protocolo congelado.

### Fase 2 — cache Garl v4

Tareas:

- [ ] resolver calibración;
- [ ] validar ambos endpoints;
- [ ] eliminar fallback TTC silencioso;
- [ ] sampler balanceado;
- [ ] distinguir JEPA pair y precontext;
- [ ] soportar TTC firmado;
- [ ] añadir RGB opcional;
- [ ] añadir masks opcionales;
- [ ] emitir manifest/hashes/discard report;
- [ ] smoke 12 muestras, piloto 4.096 y full train/val.

Gate: cache audit v4 verde, cobertura total, discard <=1 %.

### Fase 3 — baseline local exacto

Tareas:

- [ ] entrenar event-only oficial;
- [ ] entrenar visual-only oficial;
- [ ] entrenar RGBE late fusion oficial;
- [ ] usar branch pretraining oficial donde corresponde;
- [ ] ejecutar seeds `[7,13,23]` solo tras smoke;
- [ ] reportar métricas validation firmadas y recursos.

Gate: comportamiento coherente con checkpoints/paper; cualquier desviación se
documenta antes de comparar el nuevo modelo.

### Fase 4 — JEPA smoke sintético y eAP pequeño

Tareas:

- [ ] overfit de 32–128 muestras sintéticas;
- [ ] predictor reduce loss;
- [ ] EMA se actualiza y no recibe gradiente;
- [ ] máscara estructurada realmente oculta target;
- [ ] causal perturbation test;
- [ ] health metrics sin colapso;
- [ ] checkpoint resume bitwise o dentro de tolerancia.

Gate: todos los tests y figura `embedding_health.png`.

### Fase 5 — pretraining eAP SSL-Pure

Curriculum:

1. 100 ms single horizon, masks suaves;
2. 100/200/300 ms si el índice por track lo permite;
3. masks 30/45/60 %;
4. event dropout, jitter y background;
5. full train40.

No usar TTC ni cajas en el arm Pure. Screen seed 7 y promocionar una sola config a
tres seeds.

Gate:

- loss predictive mejora frente a predictor congelado/aleatorio;
- no colapso;
- linear probe eAP mejora sobre random init;
- throughput y VRAM dentro del presupuesto.

### Fase 6 — Tubelet LHR supervised

Inicializaciones:

- scratch;
- JEPA-Pure;
- JEPA-Object-Conditioned;
- opcional encoder oficial event branch.

Fine-tuning:

- frozen linear/readout;
- partial last 2 blocks;
- full con LR backbone 0.1×.

Gate principal JEPA: misma arquitectura, mismo seed, mismo split y scheduler;
JEPA debe mejorar validation eAP de forma consistente. Si solo acelera convergencia,
reportarlo como tal.

### Fase 7 — dense block-causal

Arms mínimos:

- global mean pool;
- attention pool sin temporal mixer;
- dense block-causal;
- dense triangular causal incorrecto solo como test negativo, no como candidato.

Gate: dense block-causal mejora el score predefinido sin empeorar runtime >25 % ni
usar más información.

### Fase 7B — escalado high-resolution factorized y KDA temporal

Esta fase empieza solo cuando B14, el baseline 256×144/p16 y la ruta dense de Fase 7
son verdes. No usar EvTTC para elegir resolución o mixer.

Orden obligatorio para que las causas sean identificables:

1. generar `patch_resolution_audit.json` desde los parquets/caches reales;
2. hacer que todos los tokenizers emitan padding mask y no descarten bordes;
3. medir R1/R2 con el mixer existente;
4. implementar R4 con window attention, sin KDA ni merge;
5. añadir el merge 2×2 manteniendo atención temporal estándar;
6. añadir KDA únicamente en el eje temporal;
7. añadir, por último, un refresh global tras el merge o queries fijas;
8. repetir el candidato promovido con seeds `[7,13,23]`.

Cada paso registra:

```text
input_resolution, source_resolution, patch_size, temporal_axis_semantics
tokens_before_merge, tokens_after_merge, valid/padded_token_count
attention_pair_estimate_by_stage, parameters, FLOPs
peak_VRAM_train, peak_VRAM_infer, samples_per_second, p50/p95
MiD/RTE/FR overall, small-object bucket, JEPA patch loss, foreground boundary F1
```

Comandos que se deben implementar:

```powershell
uv run python scripts/analyze_patch_resolution.py `
  --data-root "E:\eAP_dataset" `
  --garl-root "E:\GarlTTC_dataset" `
  --output artifacts/audits/patch_resolution_v1

uv run python scripts/benchmark_token_scaling.py `
  --config configs/experiment/highres_token_scaling_v1.yaml `
  --device cuda --memory-budget-gb 12
```

El benchmark debe predecir memoria antes de reservar el tensor de atención y marcar
`theoretical_oom_guard=true` para R4 global. No provocar OOM deliberadamente. Un arm
solo pasa a entrenamiento largo si el smoke forward/backward, la causalidad, el mask
y el profiler son verdes.

Gate de fase:

- R4 window+merge cabe en el presupuesto y supera R2 en small-object/overall o muestra
  un tradeoff accuracy-cost útil;
- KDA se compara contra atención temporal estándar a igualdad exacta de R4/window/
  merge y solo se conserva si pasa el gate de §7.4;
- los resultados K1/K2 previos aparecen en la tabla como evidencia histórica, sin
  mezclarse con el nuevo protocolo;
- una regresión automatizada demuestra que ningún output depende del orden raster de
  patches.

### Fase 8 — RGBE y foreground

Orden:

1. event-only;
2. RGB-only;
3. late-fusion RGBE;
4. dense cross-modal;
5. foreground decoder.

Cada paso se compara con su control. El candidato SOTA Track A sale de RGBE salvo
que se declare una comparación event-only distinta.

### Fase 9 — congelación

Antes de EvTTC final:

- [ ] congelar config y hash;
- [ ] congelar checkpoint por seed;
- [ ] congelar ensemble rule si existe;
- [ ] congelar preprocessing;
- [ ] congelar protocol mapping Tabla VI;
- [ ] escribir `FROZEN.json` firmado por hashes;
- [ ] bloquear código que cargue EvTTC labels durante predict;
- [ ] registrar que EvTTC no participó en selección.

### Fase 10 — zero-shot Tabla VI

Pipeline en dos pasos:

1. `predict`: lee eventos/RGB/cajas permitidas, no TTC; escribe predictions.
2. `score`: proceso separado une predictions con TTC y calcula RTE.

Ejecutar:

- checkpoint oficial Garl;
- nuestro scratch control;
- nuestro JEPA candidato;
- bbox-only control.

No cambiar nada después de ver resultados y seguir llamándolo mismo experimento
zero-shot. Una corrección de bug crea `protocol_version+1` y obliga a reevaluar todos
los baselines.

### Fase 11 — CodaBench

- [ ] generar submission candidata;
- [ ] validar offline;
- [ ] comparar schema/token coverage con reference;
- [ ] obtener autorización del usuario;
- [ ] enviar una de las submissions presupuestadas;
- [ ] archivar recibo/score/fecha;
- [ ] no ajustar con feedback del leaderboard salvo declarar otra ronda.

### Fase 12 — robustez, export y report

- [ ] robustness suite;
- [ ] latencia PyTorch/ONNX;
- [ ] RAM/VRAM/parámetros/FLOPs;
- [ ] incertidumbre/calibración;
- [ ] low-label;
- [ ] tablas regenerables;
- [ ] model/dataset cards;
- [ ] limitaciones y claims correctos.

---

## 10. Tests obligatorios

### 10.1 Unit tests de datos

Crear:

```text
tests/unit/test_garl_input_contract_v4.py
tests/unit/test_garl_calibration_v4.py
tests/unit/test_garl_preprocessing_parity_v4.py
tests/unit/test_garl_sampling_v4.py
tests/unit/test_garl_signed_metrics_v4.py
tests/unit/test_garl_temporal_pairing_v4.py
tests/unit/test_evttc_garl_adapter_v4.py
tests/unit/test_garl_cache_manifest_v4.py
```

Casos mínimos:

- longitudes/shape/dtype incorrectos fallan;
- channel order incorrecto falla;
- join de calibración único y 100 %;
- duplicado en join falla;
- modo official fy exacto;
- ambos errores de endpoint se validan;
- target t2 correcto;
- no fallback a row TTC sin equivalencia;
- TTC negativo preservado;
- time-volume contra referencia lenta/oficial;
- ROI square/crop fuera de límites;
- sampler cubre todas las secuencias;
- cap demasiado bajo falla;
- sampler determinista;
- discard report exacto;
- manifest detecta shard alterado;
- `jepa_pair_valid=True` con dos endpoints;
- `precontext_motion_valid=False` no anula JEPA.

### 10.2 Unit tests de modelo

Crear:

```text
tests/unit/test_e_jepa_tubelet_shapes.py
tests/unit/test_e_jepa_target_ema.py
tests/unit/test_e_jepa_structured_mask.py
tests/unit/test_block_causal_dense_ttc.py
tests/unit/test_window_spatial_attention.py
tests/unit/test_highres_token_pyramid.py
tests/unit/test_temporal_kda_axis.py
tests/unit/test_kda_chunk_recurrent_equivalence.py
tests/unit/test_token_complexity.py
tests/unit/test_ttc_readout_signed.py
tests/unit/test_ttc_gradient_routing.py
tests/unit/test_inference_state_dict_v4.py
tests/unit/test_foreground_decoder_v4.py
```

Casos:

- `[B,T,P,D]` se conserva hasta mixer;
- patches del mismo frame se ven entre sí;
- t1 no ve t2;
- t2 sí ve t1;
- window attention es bidireccional dentro del frame y shifted windows transmiten
  información a través de una frontera tras dos bloques;
- padding/border tokens nunca reciben ni aportan masa de atención;
- 90 pixels de alto con p16 produce padding explícito, no truncado a 80;
- space-to-depth 2×2 se puede invertir exactamente antes de su proyección;
- un patrón checkerboard/high-frequency se conserva antes de proyección y se pierde
  en el control average-pool esperado;
- los conteos de tokens antes/después del merge son exactos para R0–R4;
- KDA recibe `[B,T,P,D]` y falla ante layouts ambiguos sin metadata;
- permutar P, ejecutar KDA e invertir la permutación produce el mismo resultado:
  prueba de que P no se ha convertido en eje causal raster;
- perturbar un tiempo futuro no modifica outputs KDA pasados;
- forward completo, por chunks y paso recurrente coinciden en FP32 y dentro de
  tolerancia explícita en BF16;
- el estado KDA permanece FP32 bajo autocast y se resetea entre secuencias;
- retention permanece en `(exp(-5), 1)` con la config Kimi y gradientes finitos;
- el profiler calcula 250/720/1.200/4.800 tokens para los casos nominales y activa
  el OOM guard antes de materializar atención global R4;
- perturbar futuro no cambia embeddings de contexto;
- target encoder sin grad;
- online sí recibe grad;
- EMA exacta en un modelo pequeño;
- predictor conoce posiciones pero no valores target;
- loss TTC llega al patch encoder;
- intercambiar patches espacialmente cambia predicción dense;
- el global baseline puede ser invariante como control;
- ratio <1 da TTC positivo;
- ratio >1 da TTC negativo;
- ratio≈1 produce failure/censura explícita;
- predictor y target no aparecen en pesos de inferencia;
- export no incluye ramas training-only.

### 10.3 Tests anti-leakage

Crear `tests/scientific/test_garl_no_leakage_v4.py` o ubicar bajo integration si no
existe la carpeta.

Tests:

1. `forward()` no acepta `ttc_s`, `frame_ttc`, depth, 3D, category ni masks.
2. Cambiar TTC manteniendo inputs no cambia predicción antes de calcular loss.
3. Cambiar future target t2 no cambia online context t1.
4. El sampler SSL-Pure da mismos IDs aunque se permuten/cambien TTC labels.
5. Split train/val por secuencia disjunto.
6. Ningún track aparece en ambos splits.
7. Normalización/pos-weight se calcula solo en train.
8. Checkpoint selection no recibe test/EvTTC metrics.
9. `predict` zero-shot funciona si el TTC CSV se mueve temporalmente fuera del root.
10. `score` no modifica predictions ni config congelada.
11. Oracle boxes quedan declaradas en manifest.
12. Un run P0 no puede emitir `bbox_free=true`.

### 10.4 Paridad oficial

Crear:

```text
tests/integration/test_garl_release_audit.py
tests/integration/test_garl_preprocessing_100_sample_parity.py
tests/integration/test_garl_checkpoint_output_parity.py
tests/integration/test_garl_submission_reference.py
tests/regression/test_garl_fixture_hashes.py
```

Los tests que necesitan `E:\...` deben tener marker `external_data` y saltarse en CI
con mensaje claro cuando no existan roots. En esta máquina deben ejecutarse antes de
los runs.

### 10.5 Integración de entrenamiento

Crear:

```text
tests/integration/test_eap_geo2_artifact_migration.py
tests/integration/test_garl_cache_v4_smoke.py
tests/integration/test_jepa_tubelet_overfit.py
tests/integration/test_jepa_resume_determinism.py
tests/integration/test_garl_signed_train_eval.py
tests/integration/test_evttc_zero_shot_two_process.py
tests/integration/test_e_jepa_onnx_parity.py
tests/integration/test_highres_memory_scaling.py
tests/integration/test_highres_shifted_object_tracking.py
```

Gates:

- smoke CPU termina en minutos;
- CUDA smoke cuando disponible;
- loss finita;
- checkpoint reload reproduce salida;
- resume no repite steps ni cambia scheduler;
- output artifacts validan contra JSON schema;
- ONNX/PyTorch dentro de tolerancia.
- R2 y R4 completan forward/backward sin pérdida de border patches;
- R4 window+merge respeta el presupuesto configurado y el arm global se omite por
  guard, no por capturar un OOM;
- una secuencia sintética con objeto que cruza patches distingue patch-fijo de la
  variante de alineación aprendida sin usar labels futuras.

### 10.6 Collapse/health tests

Crear fixtures degenerados:

- encoder constante;
- batch de una sola muestra repetida;
- predictor que copia media;
- target aleatorio.

El monitor debe detectar los tres primeros y no confundir batch pequeño legítimo con
colapso sin suficiente soporte. Registrar effective rank con método estable cuando
`B < D`.

### 10.7 Robustness tests

Perturbaciones:

```yaml
event_dropout: [0.1, 0.3, 0.5, 0.7]
timestamp_jitter_us: [50, 200, 1000]
background_event_rate: [0.01, 0.05, 0.1]
hot_pixel_fraction: [0.001, 0.005]
dead_pixel_fraction: [0.01, 0.05]
polarity_drop: [positive, negative]
temporal_window_scale: [0.5, 0.75, 1.25, 1.5]
bbox_jitter_fraction: [0.01, 0.05, 0.10]
bbox_scale_fraction: [0.9, 1.1, 1.25]
rgb_blackout: [true]
event_blackout: [true]
```

Cada perturbación debe ser determinista por seed, no alterar TTC salvo time scaling
físico declarado y reportar degradación absoluta/relativa más incertidumbre.

---

## 11. Análisis experimental secuencial

No ejecutar un cartesiano. Usar promoción por gates.

### A0 — saneamiento de datos

Outputs:

- distribución TTC firmada por split/secuencia/track;
- tasa de eventos;
- tamaño/visibilidad ROI;
- cobertura RGB/masks/calibración;
- delta temporal;
- discard reasons;
- duplicados y overlap.

Salida: auditoría JSON del selector raw y, cuando se construya un shard de
diagnóstico, el resumen emitido por `scripts/build_eap_lhr_cache.py`.

### A1 — oracle y baselines

Comparar:

```text
mean/median train
bbox-only analytic LHR
bbox-motion MLP
official Garl event-only checkpoint
official Garl RGB-only checkpoint
official Garl full checkpoint
local official-exact training
```

Este análisis cuantifica cuánto aporta la caja y evita atribuir shortcut a JEPA.

### A2 — representación de eventos

Misma arquitectura pequeña:

```text
official time-volume20
voxel grid 10 bins
event count
time surface simple
multi-timescale
```

La comparación primaria debe incluir official time-volume20.

### A3 — arquitectura temporal

```text
ResNet50 late fusion
2D shared endpoint encoder + global pool
Tubelet Transformer + global pool
Tubelet Transformer + attention pool
Tubelet Transformer + dense block-causal
```

Igualar razonablemente parámetros y reportar también compute.

### A3B — resolución, pirámide de tokens y KDA de Kimi K3

Ejecutar en este orden, con seed 7 para screen y exactamente el mismo tokenizer/head
cuando una comparación pretende aislar un componente:

| Arm | Resolución/patch | Spatial | Reducción | Temporal | Global refresh | Propósito |
|---|---|---|---|---|---|---|
| S0 | R1 256×144/p16 | global/referencia | no | block-causal | n/a | baseline barato |
| S1 | R2 320×192/p16 | global/referencia | no | block-causal | n/a | efecto de resolución |
| S2 | R4 320×192/p8 | global | no | block-causal | global | control teórico/OOM, sin long run |
| S3 | R4 320×192/p8 | window 8 | no | atención temporal | no | efecto de ventanas |
| S4 | R4 320×192/p8 | window 8 | 2×2 | atención temporal | tras merge/queries | control recomendado |
| S5 | R4 320×192/p8 | window 8 | 2×2 | KDA aligned-patch | no | efecto KDA puro |
| S6 | igual S5 | window 8 | 2×2 | KDA | 3:1 antes del merge | test negativo/smoke |
| S7 | igual S5 | window 8 | 2×2 | KDA | 3:1 después del merge/queries | análogo Kimi promovible |

Añadir controles R3 160×96/p8 y R2 con window+merge para separar “más pixels de
entrada”, “patch más pequeño” y “mixer distinto”. No presentar S4/S7 contra S0 como
evidencia de KDA: esa comparación cambia tres componentes.

Para S4/S5/S7 barrer `T=[2,5,8,16,32]` en benchmark sintético y solo las longitudes
disponibles sin fuga en datos reales. Se espera que KDA tenga poca o ninguna ventaja
en T=2/5; esa predicción se registra antes del experimento. Reportar el cruce donde
memoria/latencia KDA resulta favorable, incluyendo el coste secuencial de la
recurrence PyTorch.

Análisis por bucket obligatorio:

```text
bbox min-dimension en patches: <1, [1,2), [2,4), >=4
bbox area en patches: <1, [1,4), [4,16), >=16
event density, TTC firmado, speed, truncation y scenario family
```

La promoción se decide en eAP validation por MiD/RTE/FR, small-object delta y coste.
EvTTC se ejecuta solo tras freeze. Preservar K1 Object-KDA/K2 como runs históricos y
no agregarlos con S5/S7 porque resolución, layout temporal y protocolo son distintos.

### A4 — valor de JEPA

```text
same Tubelet architecture, random init
same architecture, reconstruction/masked-input control
same architecture, JEPA-Pure
same architecture, JEPA-Object-Conditioned
JEPA no anti-collapse
JEPA variance/covariance
```

No comparar backbones distintos para probar JEPA.

### A5 — objetivo JEPA

```text
global future embedding
dense all-token future
dense masked regions
single horizon
multi-horizon
MLP predictor
Transformer predictor
```

Promocionar solo si mejora el linear probe eAP y downstream validation.

### A6 — head TTC

```text
direct TTC SmoothL1
direct signed-log TTC
LHR two heights official
direct positive ratio
LHR + bounded residual
```

Reportar singularidades/failures, no solo media.

### A7 — modalidad y foreground

```text
event-only
RGB-only
RGBE late fusion
RGBE dense fusion
RGBE + foreground decoder
```

### A8 — caja y despliegue

```text
P0 oracle crop + motion vector
P0 oracle crop, no motion vector
P1 predicted boxes
P2 bbox-free, si está implementado
```

No mezclar métricas entre protocolos.

### A9 — low-label

Fine-tuning con 1/5/10/25/100 % de labels eAP, selección por secuencia/track y mismos
IDs para scratch/JEPA. Es el análisis principal para demostrar utilidad SSL.

### A10 — zero-shot congelado

Solo después de freeze. Tres secuencias exactas Tabla VI, más EvTTC-32 como análisis
de cobertura secundaria sin retocar el modelo.

### A11 — robustez e incertidumbre

Comparar baseline oficial, scratch y JEPA al mismo nivel de corrupción. Reportar
curvas y área de degradación, no seleccionar por una sola intensidad.

### A12 — eficiencia

Medir por separado:

```text
event IO
representation
ROI crop
encoder event
encoder RGB
fusion/readout
total end-to-end
window spatial stages
temporal mixer/KDA
space-to-depth merge
global refresh/query cross-attention
```

Protocolo:

- batch 1;
- 50 warmups, 500 iteraciones;
- CUDA synchronize;
- mediana, p90, p95;
- mismo host/dtype;
- PyTorch eager, compile opcional y ONNX separados;
- peak VRAM/RAM, parámetros, archivo ONNX.

---

## 12. Diseño estadístico y reglas de promoción

### 12.1 Screens

- seed 7;
- máximo de samples/epochs predefinido;
- validation eAP únicamente;
- sin EvTTC;
- un componente se promociona si mejora el score primario y no viola health/runtime.

### 12.2 Confirmación

- seeds `[7,13,23]`;
- misma configuración;
- media, desviación y resultados individuales;
- bootstrap jerárquico por secuencia y track, 10.000 réplicas;
- reportar CI 95 % del delta emparejado frente al control;
- no bootstrap por ventanas como independientes.

### 12.3 Gate de mejora

Promoción recomendada:

- mejora relative ≥2 % en `paper_MiD_overall` validation o CI favorable;
- FR no empeora materialmente;
- mejora consistente en al menos 2/3 seeds;
- ninguna familia crítica colapsa;
- runtime no aumenta >25 % salvo mejora clara y objetivo accuracy-first declarado.

Una mejora menor puede conservarse como “prometedora”, no como candidato final.

### 12.4 Multiple comparisons

Registrar todas las arms antes de ejecutarlas. No ocultar runs negativos. En el
report, separar:

- hipótesis confirmatorias pre-registradas;
- exploración;
- correcciones de bugs;
- reruns por fallo infraestructural.

---

## 13. Comandos que deben existir al finalizar

### 13.1 Saneamiento

```powershell
uv sync --extra dev --extra export
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

### 13.2 Auditoría oficial

```powershell
uv run python scripts/audit_official_garlttc_release.py `
  --release-root 'E:\Garl-TTC' `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-dir artifacts/audits/garl_release_v1
```

### 13.3 Shard de caché opcional y acotado

```powershell
uv run python scripts/build_eap_lhr_cache.py `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --split data/splits/eap_pilot12_v1.json `
  --output-dir artifacts/cache/garlttc_lhr_probe `
  --max-samples-per-split 256
```

No aumentar este shard por defecto. El pipeline activo no depende de él; el full
cache estimado en aproximadamente 455 GiB no cabe en el host.

### 13.4 Pretraining high-resolution (gate cerrado)

```powershell
uv run python scripts/pretrain_eap_tubelet_jepa.py `
  --eap-root 'E:\eAP_dataset' `
  --config configs/train/eap_jepa_pretrain_v4.yaml `
  --seed 7 `
  --output-dir artifacts/runs/eap_jepa_v4_seed7
```

Este comando es actualmente un guard explícito y termina con error accionable.
Solo se habilitará cuando el target/context encoder produzca los mismos tokens
densos y resolución que el trainer downstream. No sustituirlo por el pretrainer
pooled legacy.

### 13.5 Fine-tuning

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile screen --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --split data/splits/eap_pilot12_v1.json `
  --output-root artifacts/runs/e_jepa_garl_event_screen_v1
```

La configuración `event_screen` es un screen no promocionable. El perfil `full`
usa todas las filas, seeds 7/13/23 y exige commit limpio. RGB-E no está
implementado por este comando.

### 13.6 Zero-shot Tabla VI

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages evttc-predict evttc-score `
  --output-root artifacts/runs/e_jepa_garl_event_full_v1 `
  --evttc-config configs/local/evttc_table_vi_inference.yaml `
  --evttc-predictions artifacts/official/evttc_table_vi/predictions.json `
  --evttc-targets configs/local/evttc_table_vi_targets.json `
  --evttc-metrics artifacts/official/evttc_table_vi/metrics.json
```

Este comando permanece bloqueado hasta crear y verificar el config/manifest
label-free real. Predict y score son procesos distintos dentro del runner.

### 13.7 Submission

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages submission-validate `
  --submission artifacts/official/candidate_01/submission.json `
  --sample-submission configs/local/sample_submission.json `
  --submission-validation artifacts/official/candidate_01/validation.json
```

La etapa solo valida un archivo ya generado y nunca contacta el benchmark. La
generación real de eAP test continúa bloqueada por los assets/secuencias ausentes.

---

## 14. Artefactos y provenance

Cada run debe contener:

```text
config.resolved.yaml
protocol.snapshot.yaml
run_manifest.json
environment.json
git_status.txt
git_diff.patch
dataset_manifest.snapshot.json
split.snapshot.json
history.jsonl
metrics.json
best.pt
last.pt
checkpoint_index.json
predictions.parquet
embedding_health.png
latency.json
FAILURE.json, solo si falla
```

`run_manifest.json` debe registrar lo exigido por `AGENTS.md` más:

```text
claim_track
bbox_protocol
modality
ssl_regime
uses_ttc_in_pretraining
uses_boxes_in_pretraining
uses_boxes_at_inference
uses_rgb_at_inference
official_release_commit
official_checkpoint_hashes
cache_manifest_hash
protocol_hash
selection_rule
selected_epoch
evttc_seen_before_freeze
codabench_submission_index
```

No escribir `no_privileged_model_inputs=true` como booleano general. Registrar cada
fuente por separado. Una caja oracle es privilegiada para despliegue aunque sea
legítima en el protocolo P0.

### 14.1 Estados de run

```text
created
validated
running
completed
failed
invalidated
```

Un run `failed` no se convierte en `completed` mediante edición de JSON. Solo un
resume real puede producir un nuevo state transition con timestamps y hashes.

### 14.2 Invalidation

Invalidar automáticamente resultados si cambia:

- join de datos;
- semántica temporal;
- label index;
- representación/channel order;
- split;
- protocolo de caja;
- métrica;
- checkpoint selection;
- código entre predict y score sin version bump.

Conservar artefactos invalidados y el motivo.

---

## 15. Resultados y tablas que debe generar el análisis

### Tabla 1 — paridad oficial

```text
official release checkpoint
local adapter
preprocessing max error
output max error
sample count
status
```

### Tabla 2 — baselines eAP validation

```text
mean/median
bbox-only
Garl official event
Garl official RGB
Garl official RGBE
local retrained Garl
```

Métricas: MiD por bucket, FR por bucket, weighted MiD, RTE, macro secuencia.

### Tabla 3 — JEPA

```text
scratch
JEPA-Pure frozen
JEPA-Pure partial FT
JEPA-Pure full FT
JEPA-Object-Conditioned
```

### Tabla 4 — arquitectura

```text
global pool
attention pool
dense block-causal
Tubelet vs ResNet
```

### Tabla 4B — resolución y escalado de tokens

```text
S0 R1/p16 baseline
S1 R2/p16 resolution
S3 R4/p8 window
S4 R4/p8 window + 2x2 merge + temporal attention
S5 R4/p8 window + 2x2 merge + temporal KDA
S7 S5 + reduced global refresh 3:1
```

Columnas obligatorias: grid/tokens por stage, T, MiD/RTE/FR, small-object delta,
peak VRAM, throughput, p50/p95 y motivo de promoción/rechazo. S2 global R4 aparece
como estimación/guard y no como OOM ejecutado.

### Tabla 5 — modalidad/foreground

```text
event
RGB
RGBE late
RGBE dense
RGBE dense + foreground
```

### Tabla 6 — low-label

Filas scratch/JEPA; columnas 1/5/10/25/100 %.

### Tabla 7 — zero-shot Tabla VI

Exactamente las tres secuencias, promedio, runtime, protocolo de bbox y modalidad.
Incluir Garl paper, Garl checkpoint reproducido, scratch y JEPA.

### Tabla 8 — robustez

Degradación por corrupción/intensidad y cambio de incertidumbre.

### Tabla 9 — eficiencia

Preprocessing, inference, total, p50/p95, VRAM, params, FLOPs y ONNX size.

Todas se generan desde artefactos. Ninguna cifra se copia manualmente a Markdown.

---

## 16. Criterios de rechazo inmediato

Rechazar una arm o run si ocurre cualquiera:

- usa TTC/depth/3D en SSL-Pure;
- usa EvTTC para seleccionar el candidato zero-shot;
- mezcla train/validation por secuencia o track;
- no cubre todas las secuencias esperadas;
- descarta silenciosamente muestras;
- no admite TTC negativo;
- JEPA loss es cero por máscara/flag inválido;
- target encoder recibe gradiente;
- predictor ve target futuro;
- dense tokens se promedian antes del readout en el arm dense;
- la rama high-resolution se construye haciendo upsample de la cache 160×90;
- el tokenizer recorta bordes para conseguir divisibilidad o no propaga padding mask;
- KDA recibe patches aplanados en orden raster como si fueran pasos temporales;
- se afirma complejidad lineal aunque quede atención global sobre todos los tokens R4;
- se atribuye a KDA una comparación que también cambia resolución, window attention
  o token merge;
- el checkpoint oficial no se carga exactamente;
- compara event-only con RGBE sin declarar modalidad;
- llama bbox-free a un modelo con cajas GT;
- reporta RTE sin count/failures;
- selecciona checkpoint por test;
- una sola seed se presenta como resultado principal;
- bootstrap por ventanas correlacionadas;
- NaN se reemplaza por cero;
- un artifact no pasa schema/hash;
- CI roja;
- no se puede reanudar o reproducir el run.

---

## 17. Definición de terminado por hitos

### Hito H0 — repositorio ejecutable

- [ ] CI verde;
- [ ] migración de artifacts real probada;
- [ ] runner resumible;
- [ ] configs/schemas congelados.

### Hito H1 — Garl reproducido

- [ ] release audit verde;
- [ ] preprocessing parity;
- [ ] checkpoint parity;
- [ ] submission reference válida;
- [ ] Tabla VI reproducida dentro de tolerancia.

### Hito H2 — datos v4

- [ ] cache full con cobertura y hashes;
- [ ] TTC firmado;
- [ ] sampler balanceado;
- [ ] JEPA pair válido;
- [ ] RGB/calibración/masks trazables.

### Hito H3 — JEPA demostrado

- [ ] SSL-Pure sin fuga;
- [ ] no colapso;
- [ ] linear probe mejora;
- [ ] same-architecture downstream mejora;
- [ ] tres seeds.

### Hito H4 — arquitectura candidata

- [ ] Tubelet dense block-causal supera global/scratch;
- [ ] auditoría de resolución y buckets de tamaño regenerable;
- [ ] ningún borde se pierde y masks de padding pasan tests;
- [ ] R1/R2 comparados; R4 window+merge evaluado bajo presupuesto;
- [ ] KDA temporal promovido o rechazado con control S4 idéntico y T sweep;
- [ ] RGBE comparable;
- [ ] bbox-only shortcut cuantificado;
- [ ] validation eAP mejora frente a Garl local;
- [ ] runtime aceptable.

### Hito H5 — zero-shot

- [ ] freeze previo;
- [ ] predict/score separados;
- [ ] RTE por tres secuencias;
- [ ] promedio comparado con 10.60 %;
- [ ] Garl checkpoint ejecutado con mismo adapter.

### Hito H6 — evaluación oficial

- [ ] submission validada;
- [ ] CodaBench ejecutado con autorización;
- [ ] score archivado;
- [ ] claim ajustado a evidencia.

### Hito H7 — entrega

- [ ] ONNX/TorchScript;
- [ ] streaming demo;
- [ ] robustness/calibration/latency;
- [ ] report regenerable;
- [ ] README/model card/dataset card/limitations;
- [ ] todos los resultados negativos preservados.

---

## 18. Primer lote de implementación exacto

El agente debe comenzar por este lote y no por entrenar 8 epochs:

1. Corregir `scripts/repair_eap_geo2_provenance.py` para consumir el artifact real
   de `src/e_jepa_ttc/training/eap_jepa.py`.
2. Añadir integración de migración con un mini-run real.
3. Corregir Ruff, whitespace y el test heurístico de evidencia.
4. Añadir Pyright y dejar CI verde.
5. Implementar `garlttc_calibration.py` y eliminar la dependencia imposible de
   `row['K_event']` en `garlttc_lhr_cache.py`.
6. Implementar sampler balanceado antes de materialización.
7. Validar `second_error` y exigir `frame_ttc[t2]`.
8. Separar `jepa_pair_valid` de `precontext_motion_valid`.
9. Implementar métricas Garl firmadas y sustituir la selección positiva.
10. Crear input schema v4 y hacer fallar manifests EvTTC incompatibles antes de GPU.
11. Ejecutar cache smoke de 12 muestras con cobertura 12/12.
12. Ejecutar cache piloto 4.096 y verificar distribución.
13. Reproducir preprocessing/modelo/checkpoint oficial.
14. Solo entonces implementar el nuevo `EJEPATubeletLHR` dense block-causal.
15. Ejecutar `analyze_patch_resolution.py` y congelar
    `artifacts/audits/patch_resolution_v1/patch_resolution_audit.json`.
16. Eliminar truncado implícito, propagar `valid_patch_mask` e implementar R1
    256×144/p16 antes de aumentar más la resolución.
17. Tras los gates dense, implementar S3/S4 (window + merge); KDA S5/S7 es el último
    cambio y nunca se mezcla en el mismo commit experimental que la pirámide.

Salida esperada del primer lote:

```text
artifacts/audits/recovery_v4/readiness.json
```

con:

```json
{
  "artifact_type": "e_jepa_garl_readiness_v1",
  "ci_green": true,
  "artifact_contract_green": true,
  "cache_smoke_green": true,
  "signed_metrics_green": true,
  "official_preprocessing_parity_green": true,
  "official_model_parity_green": true,
  "long_training_authorized": true
}
```

`long_training_authorized` debe calcularse a partir de los gates; no puede pasarse
manualmente como `true`.

---

## 19. Decisión arquitectónica final de este plan

El candidato principal es:

```text
E-JEPA Tubelet LHR RGBE

eAP event streams (SSL-Pure)
  -> Event Tubelet Transformer online encoder
  -> dense spatio-temporal patches
  -> structured masked multi-horizon JEPA
  -> EMA target encoder (training only)

Garl supervised train
  -> official event time-volume20 endpoint ROIs
  -> pretrained event Tubelet encoder
  -> RGB endpoint encoder
  -> dense block-causal temporal/cross-modal fusion
  -> object attention query
  -> learned height-ratio TTC head
  -> optional bounded residual and uncertainty
  -> optional foreground decoder (training only)
```

El predictor JEPA, target EMA y foreground decoder no forman parte de inferencia.
Los patches densos sí forman parte del estimador TTC; de lo contrario no se puede
atribuir la mejora al diseño Patch Policy/dense.

KDA no se incluye de inicio en el candidato principal, y el resultado negativo del
Object-KDA actual se conserva. Sí se incorpora como vía condicional de escalado una
vez reproducido el candidato simple: atención espacial windowed a resolución alta,
merge 2×2 space-to-depth y KDA exclusivamente temporal cuando T sea suficientemente
largo. La Gated MLA/global attention solo puede entrar después del merge o sobre
queries fijas; no se permite global attention sobre los 4.800 tokens de R4. MoE e
intent-to-action permanecen fuera salvo ablation posterior. La vía más prometedora
es combinar correctamente:

1. representación oficial y benchmark comparable;
2. pretraining JEPA causal realmente sin TTC;
3. tokens densos que sobreviven hasta el readout;
4. atención temporal block-causal;
5. resolución suficiente mediante ventanas y pirámide sin perder bordes;
6. KDA temporal solo si demuestra ahorro a precisión igual contra S4;
7. geometría LHR firmada y estable;
8. RGB+eventos para igualdad de modalidad con Garl-TTC;
9. evaluación zero-shot congelada y CodaBench oficial.

Solo después de completar H6 se puede escribir que el modelo bate Garl-TTC. Hasta
entonces, el lenguaje correcto es “candidato diseñado para superar Garl-TTC” y las
conclusiones deben limitarse al gate más alto alcanzado.
