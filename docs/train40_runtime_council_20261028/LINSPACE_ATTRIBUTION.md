# Atribución del residuo numérico de linspace

Resultado regenerado desde [el recibo JSON](../../artifacts/train40_runtime_council_20261028/compiled_pipeline/linspace_attribution_v1/summary.json): **12/12 casos exactos**, 3888 escalares comparados en ambos sentidos y 18 features por par. Los controles original y `native_weighted_mean` reprodujeron exactamente sus salidas archivadas; se preservaron 51 archivos.

Consultas 488, 505; productores E_A5_MATCHED, E_C2F_MATCHED, R_A5 y R_C2F, con escalas fine/coarse donde corresponden. Entradas congeladas, con sus shapes, strides y dtypes originales. No se cargaron RAW, checkpoints ni normalizadores.

El control anterior daba 8/12 casos exactos y 202 escalares diferentes, exclusivamente en divergencia x/y/isotrópica global y de foreground. El candidato mantuvo matching original y las fronteras opacas anteriores de muestreo inverso y media ponderada. La única intervención adicional sustituyó las dos construcciones x/y de `linspace(-0.5, 0.5, ...)` en una copia privada de `_summarise_one` por el operador ATen opaco, conservando dtype y device. Un guard AST y hashes vinculan esos dos callsites; no se alteraron globals científicos.

| Escala | Casos | Control anterior exacto | Candidato exacto | Diferencias antes → después |
|---|---:|---:|---:|---:|
| fine | 8 | 8 | 8 | 0 → 0 |
| coarse | 4 | 0 | 4 | 202 → 0 |

El ensayo aislado de coordenadas CUDA comparó ATen original, linspace compilado y ATen opaco compilado:

| Tamaño | Índices distintos en compilado | Máximo error absoluto | ATen opaco exacto |
|---:|---|---:|---|
| 16 | 6, 7, 8, 9 | 1.4901161193847656e-08 | sí |
| 32 | ninguno | 0 | sí |

Esta intervención atribuye el residuo coarse observado a la construcción compilada de coordenadas, condicionada a las fronteras anteriores ya preservadas. El patrón 16/32 concuerda entre el ensayo aislado y los resúmenes completos. No prueba que linspace sea la única fuente de discrepancia de un pipeline totalmente compilado; el mecanismo concreto de redondeo o instrucciones de máquina no se aisló aquí.

La cobertura se limita a estos tensores y a este entorno. No hay admisión de TTC final, modelos completos, otra población, 50 Hz ni latencia por actualización. Una integración futura requiere paridad y medición completas; el coste de las fronteras opacas tampoco se midió en este diagnóstico.

Entorno: commit `3f64ccf63fe053e9758cd1455a532631d0df1d85`, PyTorch `2.11.0+cu128`, CUDA `12.8`, NVIDIA GeForce RTX 5070 Ti Laptop GPU, `FP32_TF32_OFF`, seed 7. Ejecución registrada: 2026-10-28T10:17:49.040117+00:00 → 2026-10-28T10:18:25.523937+00:00.

SHA-256 del JSON fuente: `c39cd79515109d35af9c7c75926d85fd60bf3e8ad0d22f02478d1b5e753860e6`. Opciones de compilación: `{"eager_numerics.division_rounding": true, "emulate_precision_casts": true, "force_same_precision": true, "triton.cudagraphs": false, "use_fast_math": false}`.

Regeneración desde la raíz del repositorio (sólo JSON, sin GPU):

```powershell
python docs/train40_runtime_council_20261028/build_linspace_attribution.py
python docs/train40_runtime_council_20261028/build_linspace_attribution.py --check
```
