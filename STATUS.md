# Estado del repositorio

Actualizado: 2026-08-02.

Branch activa: `scientific-recovery-v3-hardening`.

Último commit publicado: `1d9e23a` (`Document JEPA capacity audit and current
research status`). `7d33989` contiene el auditor de shortcuts, `cbdf54c` el runner
cache-free full y `7ec2b90` el saneamiento principal y el trainer high-resolution
raw.

## Objetivo

Construir un estimador TTC por objeto que supere a Garl-TTC bajo sus protocolos
eAP y EvTTC, primero event-only y después RGB-E multimodal. El candidato combina
eventos high-resolution, tokens densos y predicción temporal JEPA, pero solo puede
llamarse SOTA después de una comparación reproducible con el mismo split, métrica,
modalidad y benchmark oficial.

No existe actualmente un resultado SOTA, una evaluación oficial eAP/CodaBench ni
un checkpoint final promovido.

## Estado ejecutivo

Funciona y está validado:

- contratos de entrada Garl, calibración, timestamps, sampling balanceado y
  métricas TTC firmadas;
- protección SSL-pure contra TTC, profundidad, categoría, máscaras y EvTTC;
- preprocessing raw/resized con diferencia máxima cero frente al contrato local;
- error de paridad de modelo acotado a `7,629e-6` en altura y `1,8358e-5` en TTC;
- padding high-resolution, máscaras de validez, merge 2x2, KDA temporal y guard
  teórico de memoria;
- trainer event-only raw/on-demand, sin caché densa, con BF16, acumulación de
  gradiente, resume por época determinista, selección macro por secuencia y
  checkpoints `best`/`last`;
- runner canónico con perfiles `screen` y `full`, seeds full `7/13/23`, freeze
  previo a EvTTC, separación predict/score y validación offline de submission;
- CLI de entrenamiento, evaluación, robustez, export ONNX, demo y reporte;
- reporte regenerable con JSON/JSONL/CSV/Parquet y validación de hashes.
- auditor de capacidad semántica JEPA con cinco brazos, tres semillas, control
  de shortcut por secuencia/fráme y artefactos compactos reproducibles.

No funciona todavía o está bloqueado:

- pretraining JEPA high-resolution compatible con el encoder denso downstream;
- RGB-E en el trainer nuevo; una configuración RGB-E se rechaza explícitamente;
- generación del manifest label-free EvTTC Tabla VI desde datos reales;
- seis secuencias eAP ausentes y test oficial eAP/CodaBench;
- full multisemilla, robustez/calibración reales, export y demo del checkpoint
  final.

### Preflight y commitment boundary Level–Dynamics

El 2026-08-02 se completó el preflight Sol Advisor sobre el árbol real:

- sesión primaria observada `gpt-5.6-sol`, esfuerzo `high`;
- agentes companion byte-exactos y los tres roles nativos expuestos;
- host observado: RTX 5070 Ti Laptop 12 GiB, 31,18 GiB RAM y Ryzen 9 16C/32T;
- un primer reviewer exigió corregir NCE degenerado, geometría post-merge,
  transferencia de backbone, residual target-only, límites de memoria, frontera
  Parquet label-free, ROI downstream y el conflicto de seeds;
- la especificación corregida está en
  `docs/dense_level_dynamics_jepa_spec_v1.md`, SHA-256
  `D9E0BAAADEA9EAB52193EC9293C5F22D9983B7CF132B149DD8CEFE21ADE52C6A`;
- un segundo Sol fresco devolvió `proceed`. El reviewer fue observado como
  `sol_advisor_sol_reviewer`, `gpt-5.6-sol`, esfuerzo `high`, sandbox efectivo
  `danger-full-access` y permiso `disabled`; hashes de repo/artefactos idénticos
  antes/después prueban que permaneció conductualmente read-only.

La implementación aún no está hecha: este veredicto autoriza delegar Phase 1, no
promociona ninguna arm ni checkpoint.

## Evidencia válida y resultados negativos

### Anclas EvTTC históricas

- `B0_HISTORICAL_BASE_EXACT`: reproducción byte a byte en su split histórico,
  MAE `0,322892 s`, RMSE `0,584432 s`, error relativo `8,1554 %`.
- `CPLA-high is diagnostic only`: no puede reutilizarse como split final ni como
  evidencia de test, porque ya intervino en diagnóstico/validación histórica.
- Grouped CV 5 folds x 3 seeds seleccionó `A0_MATCHED_GLOBAL`: score
  `0,58452 ± 0,00853`, error relativo `30,25 % ± 0,52`, MAE
  `1,011 ± 0,039 s`.
- A1 Dense, bbox-ROI, AttnRes y Object-KDA no pasan sus gates completos. No deben
  recombinarse por intuición sin una nueva hipótesis predeclarada.

### Screen high-resolution actual

`artifacts/metrics/e_jepa_tubelet_lhr_trainer_smoke_current_v1.json` registra un
smoke real de una época, 16 muestras train y 16 validation, seed 7 y BF16 en la
RTX 5070 Ti Laptop:

| Métrica | Valor |
|---|---:|
| tiempo | 17,16 s |
| MiD validation macro por secuencia | 1868,3186 |
| MiD validation global | 1941,2832 |
| RTE ponderado | 119,2892 % |
| failure rate | 0 % |

Este run prueba el flujo end-to-end y el contrato del checkpoint, no calidad. Está
marcado `claim_eligible=false`; su error está órdenes de magnitud por encima del
objetivo y no es comparable con un entrenamiento completo.

El rerun Phase-0 sobre el HEAD `1d9e23a`, conservado en
`artifacts/metrics/e_jepa_tubelet_lhr_phase0_head1d9e23a_v1.json`, completó también
16/16, una época, seed 7 y BF16 en `17,99 s`. Su MiD validation macro fue
`2117,5968`, peor que el smoke histórico, con config hash
`ac153d62e0fad81cb7a47958f879bd290b427e06e79d584c3527bb1c02ab67ec` y checkpoint
hash `fb39f4f4028e824f2aea9dbd348d3d5b775f59f0520c22742fef2d5ff7506488`.
Sigue siendo `integration_only`, `claim_eligible=false`; la diferencia respecto al
smoke previo no puede atribuirse a una sola causa porque cambió el commit/config.

### Colapso de rango y shortcut semántico

El smoke SSL eAP conservado no activa el guard estadístico: solo `3,125 %` de
dimensiones contextuales caen bajo el umbral. Sin embargo, en un embedding de 192
dimensiones presenta rango efectivo `2,255` para contexto, `1,095` para predictor
y `5,105` para target. Esto demuestra deficiencia de rango, pero el artefacto
compacto no contiene embeddings emparejados con nuisances y por tanto no permite
afirmar qué variable monopoliza el latente.

`artifacts/metrics/jepa_semantic_capacity_audit_v1.json` agrega un falsador
sintético de cinco brazos y tres semillas. La representación se entrena sin TTC ni
bits del shortcut; esas variables se usan solo en probes congelados:

| Brazo, shortcut fijo por secuencia | R² dinámica | MAE log-TTC | shortcut | duplicación | rango efectivo |
|---|---:|---:|---:|---:|---:|
| varianza actual | 0,15 | 0,39 | 0,84 | 1,93 | 11,41 |
| VISReg | 0,20 | 0,38 | 0,92 | 1,63 | 11,91 |
| residuo temporal | **0,72** | **0,29** | **0,65** | 1,06 | 4,28 |
| R² rate+dependencia | 0,29 | 0,36 | 0,88 | 1,92 | 10,83 |
| residuo+R² | 0,48 | 0,34 | 0,68 | **0,68** | 7,49 |

El objetivo actual satisface varianza/rango razonable mientras codifica el
shortcut y pierde dinámica: el fallo semántico queda reproducido en el control.
VISReg solo no lo corrige. R²-lite reduce redundancia cuando se combina con
residuo, pero falla el gate predeclarado de mejora log-TTC y queda rechazado para
producción.

El residuo temporal pasa todos sus gates en el shortcut lento. En el control donde
el shortcut cambia cada frame ocurre lo contrario: el objetivo actual alcanza R²
dinámica `0,74`/MAE `0,19`, mientras el residuo cae a `-0,05`/`0,40`. Por tanto no
se debe sustituir globalmente el nivel futuro por su residuo. La candidata mínima
es separar `nivel/escala` de `dinámica/expansión` y comparar
`level` frente a `level+temporal_residual` en el JEPA high-resolution real.

Este resultado es mecanístico sintético, no demuestra mejora eAP/EvTTC ni SOTA.
INTACT no aplica al protocolo actual porque no existen acciones expertas; R²
completo con CMI/HSIC no se implementará antes del gate real y batch 4 no ofrece
estadística suficiente para esos estimadores.

### Evidencia negativa preservada

- KDA/Object-KDA, FlowMimic e inverse-TTC global fueron rechazados por regresión.
- CARLA SSL y TTC sintético empeoraron la transferencia EvTTC; sus resúmenes
  compactos siguen en `artifacts/metrics`, pero el dataset, caches y checkpoints
  locales fueron eliminados.
- una caché Garl de 256 muestras se materializó, verificó por SHA, consumió y
  eliminó correctamente;
- una prueba de 4.096 muestras se detuvo cerca de 11 GiB de RAM;
- la caché densa full se estima en aproximadamente 455 GiB y queda fuera del
  pipeline activo.

## Datos y almacenamiento local

| Fuente | Estado | Uso |
|---|---|---|
| `datasets/evttc` | 32 secuencias, conservado | desarrollo/grouped CV |
| `datasets/evttc_official_benchmark_sealed` | conservado y sellado | evaluación final |
| `E:\eAP_dataset` | 40/46 secuencias disponibles | eventos raw bajo demanda |
| `E:\GarlTTC_dataset` | solo lectura | labels y protocolo Garl |
| `E:\Garl-TTC` | solo lectura | release oficial |
| CARLA DVS Looming | eliminado | ruta rechazada; solo métricas compactas |

También se eliminaron por ser regenerables `artifacts/runs` (16,87 GiB),
`artifacts/features` (12,81 GiB) y 2,25 GiB adicionales de cachés locales. Tras
toda la limpieza, C: dispone de más de 315 GiB libres. Los nuevos runs recrean
sus directorios de salida.

## Comandos canónicos

Preflight completo sin entrenar:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --dry-run
```

Screen parcial:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile screen --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_screen_v1
```

Full event-only y freeze multisemilla, solo desde un commit limpio:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_full_v1 `
  --resume
```

La evaluación EvTTC usa etapas separadas `evttc-predict` y `evttc-score`; exige
un config de inferencia y un manifest label-free verificados. La validación de
submission usa `submission-validate`. El runner nunca sube resultados a un
servicio externo.

Auditor de shortcut semántico completo, sin datasets ni checkpoints:

```powershell
make jepa-shortcut-audit
```

El JSON de decisión canónico es
`artifacts/metrics/jepa_semantic_capacity_audit_v1.json`.

## Cuello de botella para SOTA

El cuello de botella principal ya no es el disco ni un fallo de plumbing. Es la
calidad/identidad de la representación aprendida:

1. no hay pretraining JEPA high-resolution compatible con los tokens del modelo
   final; por tanto el candidato actual es, en la práctica, supervisado desde cero;
2. el smoke real produce MiD muy malo, señal de que el readout y la dinámica de
   aprendizaje aún no extraen una señal TTC útil con poco presupuesto;
3. el SSL eAP existente tiene predictor casi unidimensional y el falsador muestra
   que varianza/VISReg pueden conservar shortcuts lentos; el residual solo es una
   candidata condicional y aún no está validado en eAP real;
4. falta la modalidad RGB-E, que es la referencia fuerte de Garl-TTC;
5. falta una ruta geométrica causal aprendida (expansión/FoE) que supere A0 sin usar
   bbox/depth oracle;
6. no existe todavía evaluación comparable EvTTC Tabla VI ni eAP oficial, así que
   tampoco sabemos la brecha real bajo benchmark.

La optimización correcta es iterar con shards raw balanceados y gates baratos:
overfit pequeño, mejora de validation macro, transfer probe y solo después escalar.
Ejecutar ahora tres full runs consumiría muchas horas sin evidencia de que el
objetivo aprendido sea competitivo.

## Verificación actual

En un árbol sin `artifacts/runs` ni `artifacts/features`:

- Ruff check y format: verde;
- Pyright: 0 errores y 0 warnings;
- Pytest completo: verde, con siete skips de evidencia/entorno opcional;
- `git diff --check`: verde;
- hash de `src/e_jepa_ttc/data/garlttc_lhr_cache.py`:
  `D0268908E1877B7D034F29C440ED1BC1159963B88A3CB52C5314F757C5819A7C`.

## Orden recomendado para el siguiente agente

1. construir el pretrainer JEPA denso compatible con dos heads latentes explícitos:
   nivel/escala y dinámica/expansión;
2. en 256–2.048 filas raw, comparar `level` frente a
   `level+temporal_residual` con los mismos seeds/filas y probes congelados de
   expansión, event rate, ID de secuencia y TTC; no añadir R²/CMI/HSIC;
3. comparar la mejor inicialización JEPA frente a random en el mismo screen Garl;
4. implementar RGB-E como ablación aislada y medir ganancia marginal;
5. construir el manifest EvTTC label-free y cerrar predict/score Tabla VI;
6. solo si los gates mejoran, ejecutar full 7/13/23, congelar y preparar la
   submission eAP/CodaBench;
7. mantener Benchmark-10 sellado hasta el freeze final.
