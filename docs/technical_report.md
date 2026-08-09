# Informe técnico

Actualizado: 2026-08-09.

## Addendum — cambio mecanístico Causal Scale v5

La auditoría v4.31 rechazó el matcher congelado aunque su representación fuese
estable. El reemplazo v5 liga la salida principal a foreground y razón de altura:
`r=log(h_t/h_t-1)` e `inverse_TTC=expm1(r)/delta_t`. El mismo contrato se usará para
event-only, RGB-only y fusión tardía, sin coordenadas bbox como features.

El gate clean-tree sobre foreground sintético ideal pasó Pearson `1.0`, slope
`.9999995`, sign `1.0`, oddness `0/0`, translation `0`, square-rotation `.0017103` y
zero-event unknown `1.0`. Este control valida la matemática, no el aprendizaje visual;
por tanto solo autoriza el siguiente experimento sintético con máscaras predichas. No
modifica ninguna tabla eAP/EvTTC ni soporta un claim frente a Garl-TTC.

El aprendizaje sintético event-only posterior ejecutó nueve diagnósticos únicamente
sobre seeds train 101 y validation 202. El artefacto comparativo firmado
`causal_scale_v5_diagnostic_comparison_v1.json` muestra la mejora de Pearson `.2678`
a `.9560`, slope `.1932` a `.9686`, signo `.6207` a `.9957`, IoU `.3619` a `.8640`
y error TTC simétrico `1.2969` a `.2639`. La calibración escalar ajustada solo en
validation logra cobertura 80% `.7974`. El candidato aún falla translation leakage
`.02399 > .02`, por lo que no se promovió ni abrió test sintético o datos reales.
Huber adicional y resize-conv quedan como ablations negativas.

## 1. Resumen

E-JEPA-TTC investiga TTC con eventos mediante tokens espaciales high-resolution y
predicción temporal JEPA. La auditoría actual ha cerrado contratos de datos,
causalidad, métricas firmadas, memoria y ejecución sin cache global. Todavía no ha
demostrado mejora científica ni SOTA.

El resultado nuevo más importante es negativo: el primer smoke raw 16/16 completa
el pipeline, pero obtiene MiD macro `1868,3186`. El cuello de botella ya no es
almacenamiento; es aprender una representación útil y comparable.

## 2. Evidencia histórica EvTTC

`B0_HISTORICAL_BASE_EXACT` reproduce byte a byte el checkpoint histórico:

| MAE | RMSE | error relativo |
|---:|---:|---:|
| 0,322892 s | 0,584432 s | 8,1554 % |

En la comparación matched, Dense Patch ganó un split, pero grouped CV 5 folds x 3
seeds seleccionó A0:

| Candidato | Score | RTE | MAE | Decisión |
|---|---:|---:|---:|---|
| A0 global | 0,58452 ± 0,00853 | 30,25 % ± 0,52 | 1,011 ± 0,039 s | seleccionar |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | 1,007 ± 0,013 s | rechazar |

AttnRes, Object-KDA, bbox-ROI y FlowMimic tampoco pasaron sus gates. Estos
resultados evitan reintroducir complejidad sin una hipótesis nueva.

## 3. Auditoría Garl/eAP

Se corrigieron o verificaron:

- join de cinco claves y selección previa a medios;
- resolución de calibración y timestamps;
- time volumes/event voxels y resize;
- TTC firmado y métricas Garl;
- separación train/validation por secuencia;
- exclusión de TTC/depth/category/masks del encoder;
- paridad raw/resized y paridad numérica de outputs;
- checkpoint y provenance.

La cobertura local eAP es 40/46 secuencias. Seis secuencias y el test oficial
siguen ausentes.

## 4. Arquitectura high-resolution

El modelo factoriza espacio y tiempo:

```text
[B,T,C,H,W]
-> patches
-> window spatial attention
-> optional 2x2 merge
-> block-causal temporal mixer [B,T,P,D]
-> query pooling
-> signed TTC head
```

Padding y máscaras impiden perder píxeles/tokens por divisibilidad. El guard OOM
estima la atención antes de reservar. Se corrigió una incompatibilidad BF16 en
query pooling. KDA temporal fue medido y rechazado por regresión.

## 5. Trainer raw cache-free

`scripts/train_e_jepa_tubelet_lhr.py` abre HDF5 por worker y materializa solo el
batch activo. Implementa:

- sampling balanceado determinista;
- BF16/FP16/FP32;
- gradient accumulation;
- clipping;
- orden reproducible por seed+época;
- resume best/last;
- selección MiD macro por secuencia;
- hashes de config, datos, split y checkpoint.

El screen usa dim 32/patch 8. El full candidate usa dim 192/patch 16, batch 4,
acumulación 6, hasta 30 épocas y seeds 7/13/23. Full exige un árbol Git limpio.

## 6. Auditoría de representación semántica

El smoke SSL real de dimensión 192 registra rango efectivo de contexto 2,255,
predictor 1,095 y target 5,105. Esto confirma un predictor casi unidimensional,
pero no permite atribuir ese eje a TTC, expansión o ID de secuencia porque no se
conservaron embeddings y etiquetas de nuisance compatibles.

Se añadió por ello un falsador sintético sin dataset real ni entrenamiento largo.
Las cinco variantes usan las mismas observaciones y seeds 7/13/23; las etiquetas
solo entran en probes posteriores sobre encoders congelados.

| Objetivo, shortcut fijo | R² dinámica | MAE log-TTC | acc. shortcut | duplicación |
|---|---:|---:|---:|---:|
| varianza repo | 0,15 | 0,39 | 0,84 | 1,93 |
| VISReg | 0,20 | 0,38 | 0,92 | 1,63 |
| residual temporal | **0,72** | **0,29** | 0,65 | 1,06 |
| R² rate+dependencia | 0,29 | 0,36 | 0,88 | 1,92 |
| residual+R² | 0,48 | 0,34 | 0,68 | **0,68** |

La distribución puede parecer sana mientras el shortcut domina. R²-lite reduce
alguna redundancia, pero no alcanza la mejora TTC predeclarada; se rechaza para
producción. El residual temporal pasa el caso lento, pero en el control donde el
shortcut cambia cada frame cae a R² -0,05 y MAE 0,40, frente a R² 0,74 y MAE 0,19
del objetivo repo. Es una intervención condicional, no una solución universal.

## 7. Almacenamiento

Una caché full se estimó en ~455 GiB. El shard de 256 muestras pasó SHA,
consumo y borrado; 4.096 muestras llegaron a ~11 GiB RAM sin finalizar. La ruta de
caché completa se retiró.

El 2026-08-02 se eliminaron:

- CARLA DVS Looming extraído (~71,64 GiB), tras resultado negativo;
- `artifacts/runs` (~16,87 GiB);
- `artifacts/features` (~12,81 GiB);
- archivos/launchers/caches obsoletos adicionales de auditorías previas.

Los resúmenes negativos compactos siguen versionados. C: quedó con más de 315 GiB
libres.

## 8. Orquestación final

`scripts/run_e_jepa_garl_final.py` implementa una única ruta:

```text
screen o full train
-> multiseed freeze
-> EvTTC label-free predict
-> separate score
-> offline submission validation
```

El runner no construye la cache full, no abre labels durante predict y no sube
submissions. El full dry-run ya verifica las tres seeds y rutas.

## 9. Bloqueos científicos

1. No existe pretraining JEPA high-resolution compatible. El alias legacy falla
   explícitamente.
2. El predictor SSL real es casi unidimensional y todavía no se sabe qué semántica
   conserva; faltan probes reales de expansión, event rate, secuencia y TTC.
3. RGB-E no está implementado en el trainer nuevo.
4. La geometría causal bbox-free/expansión/FoE no supera A0.
5. Falta materializar inputs EvTTC Tabla VI label-free.
6. Faltan eAP oficial/CodaBench, tres seeds efectivas, robustez y calibración.

Estos son los cuellos de botella para superar tanto Garl-TTC como los métodos
geométricos EvTTC; más épocas sobre la formulación actual no garantizan resolverlos.

## 10. Ruta experimental recomendada

1. JEPA denso compatible con dos canales: nivel y residual temporal.
2. Comparar `level` frente a `level+temporal_residual` en 256/2.048 muestras con
   probes congelados y mismo compute; no añadir R²/HSIC/CMI.
3. Solo si mejora MiD/RTE macro, aumentar presupuesto.
4. Añadir RGB-E como ablación aislada.
5. Validar expansión/FoE causal contra A0 y geometría oracle.
6. Construir EvTTC predict/score real.
7. Ejecutar full 7/13/23 y congelar.
8. Enviar a eAP/CodaBench con número de submissions registrado.

## 11. Integridad

No se afirma SOTA desde smokes, validation local, una semilla o cifras del paper.
Toda tabla final debe regenerarse desde artefactos firmados. El sistema no es apto
para control de seguridad.
