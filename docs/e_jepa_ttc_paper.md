# E-JEPA-TTC: informe científico local

> **Desactualizado para Scientific Recovery V7.** Este paper no contiene
> resultados V7. No debe actualizarse hasta que exista un agregado firmado según
> [`CODEX_HANDOFF.md`](../CODEX_HANDOFF.md).

Actualizado: 2026-08-02.

## Resumen

E-JEPA-TTC estudia estimación TTC con cámaras de eventos mediante tokens densos
high-resolution y pretraining JEPA multihorizonte. La revisión actual no declara
SOTA. Reconstruye un baseline EvTTC histórico, ejecuta grouped CV, audita el
contrato Garl/eAP, proporciona un trainer raw cache-free reproducible y añade una
auditoría falsable de capacidad semántica sin dataset real.

El BASE histórico obtiene `0,322892 s` MAE, `0,584432 s` RMSE y `8,1554 %` de
error relativo en su validation original. Dense Patch mejora un split matched,
pero pierde grouped CV de cinco folds y tres seeds frente a A0. El primer smoke
del nuevo trainer high-resolution completa end-to-end, aunque obtiene MiD macro
`1868,3186`; es evidencia de integración, no de calidad.

## Pregunta y frontera del claim

La hipótesis es que JEPA denso sobre eventos mejora la estimación TTC respecto a
entrenamiento supervisado desde cero y Garl-TTC. Event-only y RGB-E se evalúan por
separado. Un claim exige protocolo/modalidad iguales, tres seeds, freeze previo,
EvTTC comparable y eAP/CodaBench oficial.

Ninguna de esas evaluaciones externas está cerrada.

## Evidencia histórica

| Modelo | Protocolo | Error relativo | MAE |
|---|---|---:|---:|
| BASE exacto | split histórico | 8,1554 % | 0,3229 s |
| A0 global | grouped CV 5x3 | 30,25 % ± 0,52 | 1,011 ± 0,039 s |
| A1 Dense | grouped CV 5x3 | 30,55 % ± 0,06 | 1,007 ± 0,013 s |

A1 mejora solo `0,41 %` el MAE, pero empeora score/error relativo y duplica
aproximadamente la latencia. AttnRes, Object-KDA y bbox-ROI también fallan sus
gates. A0 queda como arquitectura EvTTC histórica.

## Datos

- EvTTC-32: supervisión/evaluación local con splits por secuencia.
- Benchmark EvTTC: sellado hasta freeze.
- eAP/GarlTTC: 40/46 secuencias locales, lectura raw bajo demanda.
- CARLA: ruta negativa retirada; se conservan métricas compactas.

No existe split aleatorio por ventanas. TTC, profundidad, altura 3D, categoría y
máscaras no entran al encoder event-only.

## Arquitectura high-resolution

```text
eventos [B,T,21,H,W]
-> patch embedding
-> atención espacial por ventanas
-> merge 2x2 opcional
-> mixer temporal block-causal
-> query pooling
-> cabeza TTC firmada
```

Padding y máscaras conservan píxeles/tokens. Un guard previo estima memoria. KDA
temporal se rechazó por regresión. Se corrigió el dtype BF16 del query pooling.

## Objetivo y selección

El trainer optimiza Smooth L1 sobre TTC transformado como
`sign(x) * log1p(abs(x))`. Esta transformación conserva eventos de separación
con TTC negativo. El checkpoint se selecciona por MiD macro por secuencia.

El pretraining JEPA high-resolution compatible todavía no existe. El alias pooled
legacy falla explícitamente para evitar una transferencia arquitectónicamente
inválida.

## Auditoría de colapso semántico

El artefacto SSL real de dimensión 192 tiene rangos efectivos contexto/predictor/
target `2,255/1,095/5,105`. El predictor es casi unidimensional, pero estos
estadísticos no revelan si representa TTC, expansión o un shortcut.

Un benchmark sintético de cinco brazos x seeds 7/13/23 entrenó representaciones sin
etiquetas y ajustó probes después de congelarlas:

| Objetivo | R² dinámica | MAE log-TTC | acc. shortcut | duplicación |
|---|---:|---:|---:|---:|
| varianza repo | 0,15 | 0,39 | 0,84 | 1,93 |
| VISReg | 0,20 | 0,38 | 0,92 | 1,63 |
| residual temporal | **0,72** | **0,29** | 0,65 | 1,06 |
| R² rate+dependencia | 0,29 | 0,36 | 0,88 | 1,92 |
| residual+R² | 0,48 | 0,34 | 0,68 | **0,68** |

La varianza saludable no impide que domine una nuisance lenta. VISReg no lo
corrige y R²-lite falla el gate TTC, por lo que se rechaza. El residual temporal
es prometedor solo para nuisances lentas: cuando el shortcut cambia cada frame,
su R² cae a -0,05 frente a 0,74 del control. La propuesta mínima conserva
`z_level` y añade un `z_delta` residual; no sustituye el nivel ni incorpora
R²/HSIC/CMI antes de una prueba real pareada.

## Pipeline cache-free

El trainer abre HDF5 mediante índices temporales y materializa solo el batch. El
perfil screen usa una seed y hasta 2.048 muestras/split. El perfil full usa todas
las filas, seeds 7/13/23, BF16, acumulación, early stopping y Git limpio.

Una caché full se estimó en ~455 GiB. El shard 256 fue correcto; el intento 4.096
alcanzó ~11 GiB RAM. El 2026-08-02 se eliminaron CARLA, runs y features locales,
recuperando ~101 GiB en esa pasada.

## Screen real actual

| Propiedad | Valor |
|---|---:|
| train / validation | 16 / 16 |
| épocas / seed | 1 / 7 |
| tiempo | 17,16 s |
| MiD macro validation | 1868,3186 |
| RTE ponderado | 119,2892 % |

La failure rate es cero, por lo que el problema no es una excepción encubierta. El
error demuestra que la formulación todavía no extrae señal suficiente con ese
presupuesto.

## Evaluación y submission

El runner separa `train`, `freeze`, `evttc-predict`, `evttc-score` y
`submission-validate`. Predict rechaza campos target antes de cargar el checkpoint.
Score abre targets en otro proceso. La validación de submission es offline y no se
presenta como envío.

## Limitaciones

- no hay JEPA denso compatible ni RGB-E high-resolution;
- falta probar `level` frente a `level+temporal_residual` y diagnosticar la
  semántica real con probes congelados;
- faltan seis secuencias eAP y evaluación oficial;
- falta manifest EvTTC Tabla VI label-free;
- geometría causal bbox-free/FoE no supera A0;
- faltan full 7/13/23, robustez, calibración, ONNX y demo final;
- el modelo no está validado para seguridad.

## Próximos gates

1. overfit/no-collapse JEPA denso de nivel y residual en 256 muestras;
2. `level` frente a `level+temporal_residual` en screen pareado, con probes;
3. RGB-E como ablación aislada;
4. EvTTC predict/score real;
5. full multisemilla solo tras una mejora clara;
6. eAP/CodaBench tras freeze.

## Referencias

Las referencias completas están en `docs/references.bib`.
