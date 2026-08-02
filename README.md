# E-JEPA-TTC

Pipeline reproducible para estimar Time-to-Contact/Time-to-Collision a partir de
cámaras de eventos, con una ruta event-only high-resolution y una futura ablación
RGB-E multimodal.

Estado: infraestructura validada, hipótesis científica todavía no demostrada. No
existe claim SOTA ni resultado oficial eAP/CodaBench. Consulta el
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
- CARLA se retiró del camino activo tras una transferencia negativa; no hace falta
  descargarlo para reproducir el pipeline actual.

No se construye una caché Garl high-resolution completa: se estimó en unos
455 GiB. Los shards pequeños son diagnósticos opcionales; el trainer final consume
solo el batch activo.

## Entrenamiento high-resolution

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

Full event-only, seeds predeclaradas 7/13/23 y freeze por validation Garl:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_full_v1 `
  --resume
```

El perfil full exige un commit limpio. Omitir `--max-samples-per-split` es parte del
contrato: usa todas las filas válidas del split firmado. `--resume` restaura el
checkpoint y el orden de muestras se deriva de seed+época.

También existen aliases Make:

```text
make setup
make smoke-data
make train-baseline
make pretrain-jepa
make finetune-ttc
make garl-full-dry-run
make garl-full
make jepa-shortcut-audit
make evaluate
make demo
make report
```

## EvTTC Tabla VI y submission

El runner mantiene separados:

```text
train -> freeze -> evttc-predict -> evttc-score -> submission-validate
```

`evttc-predict` no puede abrir TTC/depth/categoría/máscaras y requiere:

- checkpoint congelado;
- protocolo Tabla VI congelado;
- configuración de inferencia portable;
- manifest NPZ label-free con cobertura exacta y hashes.

`evttc-score` recibe los targets en otro proceso. `submission-validate` comprueba
formato y cobertura local, pero no sube archivos. La creación del manifest EvTTC
real y el paquete eAP/CodaBench siguen pendientes.

## Evidencia actual

| Evidencia | Resultado | Decisión |
|---|---:|---|
| BASE histórico EvTTC | 8,1554 % error relativo | ancla, no SOTA |
| A0 grouped CV 5x3 | 30,25 % ± 0,52 | arquitectura EvTTC histórica |
| A1 Dense grouped CV | 30,55 % ± 0,06 | rechazado |
| bbox-ROI / AttnRes / KDA | regresión o gate fallido | rechazados |
| high-res raw smoke 16/16 | MiD macro 1868,3186 | integración solamente |
| JEPA actual, shortcut lento sintético | R² dinámica 0,15; acc. shortcut 0,84 | expone colapso semántico |
| residual temporal, mismo control | R² dinámica 0,72; MAE log-TTC 0,29 | candidato condicional |
| R²-lite, mismo control | MAE log-TTC 0,36 | rechazado para producción |

El smoke high-resolution valida que datos, GPU, pérdida firmada, checkpointing y
métricas encajan. Su precisión es muy mala y no justifica un entrenamiento full.
La prioridad es un JEPA denso compatible y una mejora clara en screens pareados.
La auditoría sintética falsable concluye que varianza/VISReg no bastan contra un
shortcut constante por secuencia. El residual temporal es la única solución que
pasó ese gate, pero falla cuando la señal cambia por frame; por ello solo se
probará como ablación `level` frente a `level+temporal_residual` sobre las mismas
filas eAP. No se incorpora R²/HSIC/CMI al modelo final.

## Arquitectura activa

```text
eventos raw HDF5
  -> voxel temporal causal [T,C,H,W]
  -> patch embedding high-resolution
  -> atención espacial por ventanas
  -> merge 2x2 opcional
  -> mixer temporal block-causal
  -> query pooling
  -> cabeza TTC firmada
```

KDA permanece como resultado negativo. La configuración RGB-E existe como contrato
de investigación, pero el trainer event-only la rechaza hasta que la fusión causal
esté implementada y probada.

## Estructura útil

- `src/e_jepa_ttc/`: librería de datos, modelos, pérdidas, evaluación e inferencia;
- `scripts/train_e_jepa_tubelet_lhr.py`: trainer raw cache-free;
- `scripts/run_e_jepa_garl_final.py`: orquestación screen/full/evaluación;
- `configs/experiment/e_jepa_garl_event_{screen,full}_v1.yaml`: perfiles activos;
- `data/protocols/garl_evttc_table_vi_v1.yaml`: frontera predict/score;
- `artifacts/metrics/`: evidencia compacta versionada;
- `PLAN.md`: plan científico y gates;
- `STATUS.md`: handoff operativo actual.

`artifacts/runs`, `artifacts/features`, datasets y caches son locales/ignorados y se
pueden regenerar. No deben subirse a Git.

## Integridad científica

- splits por secuencia, nunca aleatorios por ventana;
- selección de modelo macro por secuencia;
- pretraining SSL sin etiquetas TTC;
- EvTTC no se usa para seleccionar el supuesto zero-shot;
- resultados negativos y fallos relevantes se conservan como resúmenes compactos;
- no se declara SOTA desde smokes, una semilla o cifras copiadas de artículos.

## Documentación

- [plan de ejecución](PLAN.md)
- [estado actual](STATUS.md)
- [protocolo experimental](docs/experimental_protocol.md)
- [dataset card](docs/dataset_card.md)
- [model card](docs/model_card.md)
- [limitaciones](docs/limitations.md)
- [reproducibilidad](docs/reproducibility.md)
- [informe técnico](docs/technical_report.md)
- [informe PDF](docs/e_jepa_ttc_paper.pdf)

El sistema es experimental y no está validado para control de seguridad.
