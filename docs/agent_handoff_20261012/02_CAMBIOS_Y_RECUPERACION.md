# 02 — Cambios, causa de los fallos y conservación del entrenamiento

## Auditoría de la receta RGB, 9 de octubre

Fuente primaria: [ADR-0002](../decisions/ADR-0002-rgb-port-audit-corrections.md).
La petición del usuario fue conservar BF16 y los checkpoints existentes.

| Problema | Cambio y código | Evidencia / alcance |
|---|---|---|
| Router C2F trataba verde/azul como estadísticas de eventos | Rama RGB con luminancia y log1p de gradiente espacial en `src/e_jepa_ttc/models/causal_scale_ttc.py` | Rama event conserva matemática; RGB cambia semántica de entrada del router |
| Reunir pérdidas T2/T3 solo por muestras ponderaba mal posiciones válidas | `operational/rgb_port_revision/loss.py`: EffectiveBatch/Source, denominadores efectivos, top-k global | Una frontera backward/optimizer; longitudes reales, sin padding; gradientes comprobados |
| NPZ event releídos tras expulsar shards completos | `rgb_port_revision/cache.py::GroupRowCache`, retener filas pendientes del grupo de 256 | 131→21 lecturas; 85,74→51,01 s en CPU; tensores/metadatos iguales |
| Extremos de intervalo TTC invertidos incorrectamente | `rgb_port_revision/outputs.py` | Invierte extremos en una rama; cruce por fase cero produce estado no disponible; no promete cuantiles TTC calibrados |
| Historia externa dependía de las consultas de evaluación | `rgb_port_revision/transfer_inputs.py` | Frames reales, causalidad, deltas medidos, 100 ms con tolerancia 1 ms, límite 650 ms |
| Costes omitían construcción cruda y reporte leía esquema incorrecto | `rgb_port/cost_routes.py`, `rgb_port_revision/raw_events.py`, `rgb_port/report.py` | HDF5→voxels/rutas completas y lectura de warm_ms; RSS CPU antes/después no es pico |
| Modificación de código congelado invalidaba continuidad | `rgb_port_revision/migration.py`, manifests de migración | Solo transición de hashes enumerada; código anterior y freezes conservados |

La restauración original se comprobó en E_A5=15.303 y E_C2F=6.685 updates,
incluidos AdamW, scheduler, cursor y RNG. Es un hito histórico: ambos fits
terminaron después en 30.330. Las anotaciones `typing.cast` no convierten
tensores ni cambian precisión.

## Continuidad, I/O y carga paralela, 10–11 de octubre

`operational/rgb_port_continuity/`, `rgb_port_loader_v3/`, los wrappers RGB y
sus contratos conservan el orden del sampler y esperan a la restauración antes
de anticipar batches. Los reintentos I/O son acotados; una corrupción no se
convierte en una muestra distinta. Tras una pausa se vuelve al estado durable,
no se da por entrenado el trabajo especulativo.

El margen de commit de Windows se redujo a 1 GiB por solicitud del usuario.
RAM física libre y margen de memoria comprometida son diferentes; un proceso
puede sufrir presión aunque la GPU parezca libre. Más procesos no aportan una
aceleración automática y pueden multiplicar descompresión, buffers y memoria.

La historia y recibos están en [loader_v3](../loader_v3_20261010/README.md),
[RGB loader](../rgb_loader_20261011/README.md) y
[RGB overlap](../rgb_overlap_20261011/README.md). Se conservaron los fallos
intermedios y mediciones negativas. El informe 06 separa preparación CPU,
cómputo GPU y entrenamiento completo.

## Bloqueo H_R_RAW por arrays NumPy

`_observation_signature` recibía arrays de Parquet y `json.dumps` fallaba.
`operational/rgb_port_signature_fix/signature.py` normaliza recursivamente
arrays/escalares NumPy solo en los siete campos originales de identidad.
No redondea relojes, no añade labels ni altera filas. Los hashes nativos se
conservan; NaN, infinitos y objetos inválidos siguen siendo errores.

La recuperación reactivó H_R_RAW y sus descendientes bloqueados, conservando
las fases completas y el log del primer intento. Se verificaron los historiales
H/V y 40 archivos de los ocho fits ya terminados. Fuente:
[informe y evidencia](../rgb_signature_fix_20261011/README.md).

## Bloqueo de fusión por mediana CUDA determinista

F_TRUE/F_ZERO fallaron antes de su primer update: la mediana CUDA con índices
no tenía implementación determinista admitida. El wrapper
`operational/rgb_port_fusion_fix/fusion.py` usa el elemento central de una
ordenación estable de tres fases. Conserva valores, parámetros, objetivo y
determinismo, sin editar la implementación congelada.

Los expertos de esta receta están congelados; la equivalencia del gradiente
hacia entradas empatadas requeriría una nueva comprobación en entrenamiento
conjunto. Se verificaron pérdida y gradientes reales antes de reanudar.
Commit de corrección `17e029b`; [informe](../fusion_deterministic_20261011/README.md).
Los dos fits terminaron posteriormente; no están pendientes de esta reparación.

## Máscara nativa en transferencia y costes

`known` tiene forma `[B]` y `support` forma `[B,T]`. El validador externo exigía
incorrectamente igualdad de forma. `operational/rgb_port_external_fix/` aplica
el contrato nativo validando la expansión temporal de la máscara, sin cambiar
los tensores que consume el modelo. Se verificó igualdad de salidas reales y
se pudieron completar Dev32 y los perfiles. Commit `0f6204a`;
[evidencia](../external_mask_20261012/README.md).

## FCWD RGB y scoring sin etiquetas

`operational/fcwd_rgb_registration/` añade registro local causal con intrínsecos,
gates de calidad y fallback. Commit `0fedf67`. Es una ruta nueva explícita:
no convierte los extrínsecos LiDAR→RGB en extrínsecos event→RGB.
El informe 05 documenta geometría, rechazos y cobertura.

Tras sellar todas las predicciones, ambos scorers fallaron al ejecutar
`float('')` sobre GT ausente. Los adaptadores
`operational/fcwd_rgb_score_fix.py` y
`operational/dev32_missing_truth_score.py` decodifican vacío como NaN y filtran
solo GT no finito para métricas que necesitan etiqueta. No eliminan fallos de
predicción ni imputan TTC=0. Las 630/1.024 filas permanecen en los CSV.

`fcwd_rgb_resume.py` y `external_missing_truth_queue.py` enrutan los scorers
corregidos. Se conservaron sellados, hashes y logs originales. `REPORT` se
regeneró en su segundo intento. Commit `642bd1d`; publicación de métricas
`133002a`. La QA final registra 26 tests previos + 3 nuevos, Ruff aprobado,
Pyright sin errores ni advertencias y 72 comprobaciones independientes de MiD.

## Qué conserva la trazabilidad

Cada capa adicional tiene admisión de código/evidencia y se vincula a sus
contratos anteriores. Una nueva publicación Git debe pasar
`operational.rgb_port_continuity.git_admission.inspect`; no basta con cambiar
el hash a mano. Los checksums históricos no deben sobrescribirse para hacer
pasar una validación. Los checkpoints originales y la receta de los fits
finales permanecen como referencia para futuras investigaciones.
