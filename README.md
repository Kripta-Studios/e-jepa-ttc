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
| Object Event v4 | pendiente de screen | candidato condicionado por gates |

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

- [plan de ejecución](PLAN.md)
- [estado actual](STATUS.md)
- [protocolo experimental](docs/experimental_protocol.md)
- [Object Event TTC v4](docs/object_event_v4.md)
- [dataset card](docs/dataset_card.md)
- [model card](docs/model_card.md)
- [limitaciones](docs/limitations.md)
- [reproducibilidad](docs/reproducibility.md)
- [informe técnico](docs/technical_report.md)
- [informe PDF](docs/e_jepa_ttc_paper.pdf)

El sistema es experimental y no está validado para control de seguridad.
