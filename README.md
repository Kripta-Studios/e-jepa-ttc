# E-JEPA-TTC

Pipeline reproducible para estimar Time-to-Contact/Time-to-Collision a partir de
cámaras de eventos, con una ruta event-only high-resolution y una futura ablación
RGB-E multimodal.

Estado: núcleo causal-scale v5 validado solo sobre foreground sintético ideal; la
hipótesis científica todavía no está demostrada. No existe claim SOTA ni resultado
oficial eAP/CodaBench. Consulta el
[estado operativo](STATUS.md) antes de ejecutar experimentos largos.

## Qué produce

- TTC continuo firmado en segundos;
- logits de riesgo por horizontes configurables;
- incertidumbre cuando la cabeza correspondiente está activa;
- embeddings globales/densos y diagnósticos de colapso;
- métricas macro por secuencia, robustez, calibración y latencia;
- checkpoints auditables, export ONNX, demo offline e informe regenerable.

## Instalación

Requisitos recomendados: Python 3.11, `uv`, PyTorch con CUDA para entrenamiento y
una GPU de consumo con aproximadamente 12 GiB de VRAM.

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

Validación completa:

```powershell
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q
```

## Datos

Las rutas locales nunca se hardcodean en el paquete. Las fuentes principales son:

```text
EAP_ROOT=E:\eAP_dataset
GARLTTC_ROOT=E:\GarlTTC_dataset
GARLTTC_RELEASE_ROOT=E:\Garl-TTC
```

- EvTTC-32 local se usa para desarrollo, grouped CV y evaluación controlada.
- El benchmark EvTTC permanece sellado hasta congelar candidato.
- eAP/Garl se lee bajo demanda desde HDF5/parquet.
- CARLA se retiró del camino activo tras una transferencia negativa.

## Object Event TTC v4

V4 corrige los fallos falsados por las auditorías v3:

```text
eventos t0/t1/t2
  -> una sola ROI cuadrada común (unión temporal + margen)
  -> 12 canales activos, sin el tail de 9 canales constantes
  -> encoder event-only online + target encoder EMA
  -> predicción JEPA local de tokens futuros, sin cajas ni motion embedding
  -> cabeza de expansión event-only exactamente antisimétrica
  -> cabeza motion-only independiente
  -> fusión tardía con gate event mínimo
  -> TTC firmado derivado de g = delta_t / TTC
```

El trainer aplica warm-up event-only y dropout de modalidad. Un checkpoint no se
selecciona si los eventos pueden ponerse a cero o barajarse sin degradación
medible. V4 conserva scratch, Level-transfer y cada rama por separado en las
métricas; no permite presentar el atajo geométrico como aprendizaje visual.

Screen completo:

```powershell
uv run --no-sync python scripts/run_e_jepa_object_event_v4.py `
  --profile screen `
  --stages preflight cache scratch level `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --pretrained artifacts\runs\level_dynamics_pilot256\pretrain\level\seed-7\checkpoint.pt `
  --device cuda
```

Consulta [el contrato y los gates de v4](docs/object_event_v4.md) antes de
escalar a full o abrir EvTTC.

## Causal Scale TTC v5

V4.31 falsó el matcher congelado. V5 reemplaza su mecanismo por un contrato común
para event-only, RGB-only y fusión tardía RGB-E:

```text
foreground causal -> altura visible -> log-ratio firmado
                   -> TTC + incertidumbre + riesgo derivados físicamente
```

El primer core event-only está implementado sin coordenadas bbox, categoría o ID de
secuencia como inputs. Su residual es antisimétrico y acotado; la cabeza TTC libre es
solo auxiliar. El gate ideal-foreground versionado pasó. Nueve diagnósticos de
aprendizaje train/validation llevaron el candidato a Pearson `.9560`, pendiente
`.9686`, signo `.9957`, IoU `.8640` y error TTC simétrico `.2639`; sigue no promovido
porque leakage de traslación `.02399` falla el gate `.02`. El test sintético se abrió
una vez desde el commit limpio `d9d20af` y confirmó el fallo: Pearson `.92135` y
translation `.02749`. Todos los datos reales permanecen cerrados.

V6 no reutiliza ese test: una cabeza foreground separable sin strides, evaluada solo
en train/validation 401/502, reduce translation leakage a `.00462` y alcanza IoU
`.89323`. Pearson `.92042` sigue por debajo de `.95`, por lo que test 603 no se abre.
La comparación firmada está documentada en [Causal Scale v6](docs/causal_scale_v6.md).

V7 transporta causalmente el TTC del par anterior y lo combina con el par actual sin
añadir parámetros. En validation 502 pasa todos los gates: Pearson `.96126`, TTC
`.24345` y translation `.00351`. El test limpio 603 posterior falló Pearson `.92014`;
todos los demás gates pasaron. V7 queda cerrado y real-data continúa sellado.

```powershell
uv run --no-sync python scripts/evaluate_causal_scale_v5_operator.py --require-clean
```

Consulta [el contrato, resultado y siguiente gate](docs/causal_scale_v5.md).

## Entrenamiento high-resolution histórico

Preflight del perfil full, sin reservar GPU:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --dry-run
```

Screen corto y acotado:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile screen --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_screen_v1
```

El perfil full exige un commit limpio. Omitir `--max-samples-per-split` es parte
del contrato: usa todas las filas válidas del split firmado. `--resume` restaura
el checkpoint y el orden de muestras se deriva de seed+época.

## EvTTC Tabla VI y submission

El runner mantiene separados:

```text
train -> freeze -> evttc-predict -> evttc-score -> submission-validate
```

`evttc-predict` no puede abrir TTC/depth/categoría/máscaras. `evttc-score` recibe
los targets en otro proceso. La creación del manifest EvTTC real y el paquete
eAP/CodaBench siguen pendientes.

## Evidencia actual

| Evidencia | Resultado | Decisión |
|---|---:|---|
| BASE histórico EvTTC | 8,1554 % error relativo | ancla, no SOTA |
| A0 grouped CV 5x3 | 30,25 % ± 0,52 | arquitectura EvTTC histórica |
| A1 Dense grouped CV | 30,55 % ± 0,06 | rechazado |
| bbox-ROI / AttnRes / KDA | regresión o gate fallido | rechazados |
| high-res raw smoke 16/16 | MiD macro 1868,3186 | integración solamente |
| Object Expansion v3 | usa casi exclusivamente motion | falsado como event-TTC |
| Object Event v4 | v4.30 OOF negativo; v4.31 train-only estable pero no físicamente equivariante | no promocionado; full cerrado |
| Causal Scale v5 | test sintético: Pearson .92135, translation .02749; gates fallidos | no promovido; real-data cerrado |
| Causal Scale v6 | validation: Pearson .92042, translation .00462, IoU .89323 | test 603 sellado; no promovido |
| Causal Scale v7 | test: Pearson .92014 (falla), TTC .24576, translation .00338 | no promovido; seed 603 consumida |

El smoke high-resolution valida integración, no precisión. Los resultados v3
muestran que la expansión firmada y la supervisión de ratio son útiles, pero
que el crop independiente, la ausencia de t0 y el atajo de cajas impiden
atribuir el resultado a eventos. V4 existe precisamente para falsar o corregir
esa ruta.

## Arquitectura base

```text
eventos raw HDF5
  -> voxel temporal causal [T,C,H,W]
  -> patch embedding high-resolution
  -> atención espacial por ventanas
  -> merge 2x2 opcional
  -> mixer temporal block-causal
  -> query pooling / tokens densos
  -> cabeza TTC firmada
```

KDA permanece como resultado negativo. La configuración RGB-E existe como
contrato de investigación, pero el trainer event-only la rechaza hasta que la
fusión causal esté implementada y probada.

## Estructura útil

- `src/e_jepa_ttc/`: datos, modelos, pérdidas, evaluación e inferencia;
- `scripts/build_eap_object_event_v4_cache.py`: caché común t0/t1/t2;
- `scripts/train_e_jepa_object_event_v4.py`: trainer y gates v4;
- `scripts/run_e_jepa_object_event_v4.py`: orquestación screen/full;
- `configs/experiment/e_jepa_garl_object_event_{screen,full}_v4.yaml`: perfiles v4;
- `data/protocols/garl_evttc_table_vi_v1.yaml`: frontera predict/score;
- `artifacts/metrics/`: evidencia compacta versionada;
- `PLAN.md`: plan científico y gates;
- `STATUS.md`: handoff operativo actual.

`artifacts/runs`, `artifacts/features`, datasets y caches son locales/ignorados y
se pueden regenerar. No deben subirse a Git.

## Integridad científica

- splits por secuencia, nunca aleatorios por ventana;
- selección de modelo macro por secuencia;
- pretraining SSL sin etiquetas TTC;
- EvTTC no se usa para seleccionar el supuesto zero-shot;
- resultados negativos y fallos relevantes se conservan;
- no se declara SOTA desde smokes, una semilla o cifras copiadas de artículos;
- v4 exige dependencia observable de eventos antes de congelar candidato.

## Documentación

### v4.30 authoritative negative result and v4.31 next action

The authoritative full v4.30 SHA256 is `9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689`, status `completed_oof_gate_failed`. Stabilization passed `.0010116798/.0423071422/.1308624286`; rank-only winner `stable_multiscale_similarity` has no champion. Its best-arm Pearson `.4791568608`, negative accuracy `0`, balanced `.5`, std ratio `.3731916487`, slope `.1788173388`, high-bucket Pearson `-.1972577670`, and ratios `.92439/.58893/.48926/.30467` failed the frozen objective; both arms failed with no sealed data opened. The target-free saved-NPZ post-hoc audit (not preregistered) found forward-vs-swap `log_eta` correlation `+.53338`, zero sign flips, and 95.8% coverage at `|log_eta| >= .005`. The next action after Sol's rethink is a TTC-label-free but train-box-conditioned common-object-ROI v4.31 redesign: TTC/sign/bucket-independent selection, immutable sequence/time-disjoint train-only stabilization/audit pools, sanitized event/ROI-only artifacts, exact physical reversal controls, and no development/test/EvTTC. The direct full-frame v4.31 draft was rejected before execution and is not evidence.

The v4.31 implementation and negative 512-row diagnostic are documented in [the causal-audit handoff](docs/object_event_v4_31.md).  Cache preflight passed and no sealed data were opened.  Stability passed, but the operator failed analytic zoom, slope, sign, oddness, translation leakage and swap coverage; stage 2 was absent and the recorded worktree was dirty, so the result is explicitly non-selectable and non-authoritative.  Full remains closed.

### Superseded historical v4.30 executable protocol

The following is superseded diagnostic history, not current v4.30 state. A 96-row post-fix train-only diagnostic was `diagnostic_only`: JS median `.010237284936010838` passed; JS p95 `.19495552778244019` and BASE-pixel displacement p95 `.5500071191315064` failed. The earlier `D9DE07…` diagnostic is superseded. The authoritative completed v4.30 summary is the SHA and negative result stated above; historical diagnostics cannot modify it.

- [plan de ejecución](PLAN.md)
- [estado actual](STATUS.md)
- [protocolo experimental](docs/experimental_protocol.md)
- [Object Event TTC v4](docs/object_event_v4.md)
- [Object Event TTC v4.29 preregistration](docs/object_event_v4_29.md)
- [Object Event TTC v4.30 stable similarity preregistration](docs/object_event_v4_30.md)
- [Object Event TTC v4.31 causal audit handoff](docs/object_event_v4_31.md)
- [Causal Scale TTC v5](docs/causal_scale_v5.md)
- [Causal Scale TTC v6](docs/causal_scale_v6.md)
- [Causal Scale TTC v7](docs/causal_scale_v7.md)
- [ADR-0001: geometry-bound causal scale](docs/decisions/ADR-0001-causal-scale-v5.md)
- [dataset card](docs/dataset_card.md)
- [model card](docs/model_card.md)
- [limitaciones](docs/limitations.md)
- [reproducibilidad](docs/reproducibility.md)
- [informe técnico](docs/technical_report.md)
- [informe PDF](docs/e_jepa_ttc_paper.pdf)

El sistema es experimental y no está validado para control de seguridad.
