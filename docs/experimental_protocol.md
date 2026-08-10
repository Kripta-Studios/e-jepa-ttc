# Protocolo experimental

## Event-only screen A0 / release reference (2026-08-10)

A0 usa 2.048 train y 2.048 validation con secuencias disjuntas. La tabla release
usa los mismos tokens de validation, pero queda separada porque el checkpoint vio
las tres secuencias durante train y usó 88.744 filas/50 épocas. Los intervalos del
comparador resamplean secuencias completas (10.000 iteraciones), nunca ventanas.
La tabla matched desactiva la inicialización desde el checkpoint release.
Usa el cache oficial firmado `92af2810…f564b52`, seed 7, batch 32, máximo 18
épocas, selección elegible desde época 8 y desempate por failure. Preprocessing,
training, validation e inferencia se cronometran por separado. Resume rechaza
cualquier cambio de config, cache, seed, épocas, batch, paciencia o guard temporal.

El resultado matched seed 7 es el baseline principal a batir en este screen: MiD
global `203.0982270`, macro-secuencia `203.6341709`, failure `0%`. El checkpoint
seleccionado es época 11 de 16 ejecutadas. La tabla release (`117.4282` macro)
permanece separada por presupuesto desigual y exposición de validation. A1 se
compara causalmente contra matched, no usa la release como gate.

### A1 congelado antes de ejecución y resultado

A1 cambia exclusivamente `weak_box -> bbox_geometry`. La bbox se transforma en
`h,w,cx,cy` visibles sin crear una máscara. El model config/hash, 344.591 parámetros,
prediction path y todo el protocolo de optimización son idénticos a A0. BCE/Dice y
pair-ratio pesan cero; `h,w,cx,cy` pesan efectivamente 1.25 cada uno. Los diagnósticos
se reportan por época, globales y macro por secuencia, pero no alteran selección.
Un resultado distinto posterior requerirá una nueva identidad A1-R/A2; no se
reescribirá A1 tras observar validation.

A1 seed 7 completó 18 épocas y seleccionó la 18 bajo ese contrato inmutable:
MiD macro `346.8294571`, failure `9.9609375%`, Pearson log-ratio `.1108212`.
Frente a Garl matched queda `+143.1953` MiD y el IC95% de tres secuencias completas
es `[115.1042,166.6705]`. El resultado no autoriza promoción ni escalado. Dado que
anchura y centros permanecen en Pearson `<=.079` y `delta log h`/bbox en `.0591`,
el siguiente protocolo debe modificar una sola vez la representación densa
event-native conservando la cabeza geométrica. A1-R solo se reconsiderará si una
representación futura mide bien los endpoints pero falla su diferencia temporal.

La auditoría posterior es diagnóstica, no un sweep: mide por separado t1/t2 y los
momentos de actividad derivados solo de events. Confirma target width variable y
fallo en ambos endpoints. La siguiente identidad cambiará una sola línea del model
config: `equivariant_separable -> equivariant_fullres`. Todos los pesos de loss,
seed, filas, schedule, selección y conversiones físicas quedan iguales a A1. No se
activará pair-ratio ni se usará RGB teacher en ese control.

El diagnóstico de fallo no selecciona hiperparámetros ni abre test. Descompone el
checkpoint ya seleccionado sobre la misma validation pública en bbox-ratio,
extensión analítica, residual, ratio combinado y ratio efectivo. A1 queda
preregistrado antes de ejecutarse y solo cambia el tipo de supervisión bbox. Sus
gates diagnósticos incluyen correlación analítica-bbox y analítica-TTC; las métricas
recortadas/saturadas no pueden usarse como gate principal.

Actualizado: 2026-08-10.

## Gate previo Causal Scale v5

Antes de abrir eAP/EvTTC, el brazo event-only debe aprender foreground y razón de
escala sobre dinámicas sintéticas con seeds 101/202/303 disjuntas. Train y validation
pueden usarse para arquitectura, selección y una calibración escalar de varianza. El
test se abre una sola vez desde Git limpio después de congelar código, configuración y
gates. Fallar cualquier gate mantiene cerrados todos los datos reales. Un pass solo
autoriza diseñar un screen eAP train-only; no autoriza test, Garl-TTC ni claim SOTA.

V5 consumió seed 303 una vez en `d9d20af` y falló correlación/traslación. Esa seed
queda congelada para auditoría y no puede entrar en selección de v5 o sucesores. Un
nuevo protocolo debe declarar grupos test nuevos antes de entrenar.

V7 consumió seed 603 una vez desde el commit limpio publicado `0bc781f` y falló
solo el gate de Pearson (`.9201432 < .95`). La seed 603 queda igualmente cerrada. El
sucesor debe seleccionar sobre varios grupos train/validation, agregar por grupo y
declarar nuevas seeds test antes de cualquier entrenamiento; no puede ajustar gates
ni arquitectura con 303 o 603.

V8 satisface ese requisito con train 701/702/703, validation 801/802/803 y test
901/902/903. Mantiene 1536/384 muestras train/validation, selecciona con media macro
y peor grupo, y exige pass individual de todos los tests. Los grupos test no se
instancian durante `diagnostic`.

Tras los diagnósticos `.10/.15`, el usuario autorizó abrir eAP exclusivamente para
un screen exploratorio train/validation aunque el gate sintético siga fallido. Esta
excepción no permite ajustar con test, ejecutar CodaBench/EvTTC test ni afirmar
superioridad sobre Garl-TTC. Toda comparación debe usar el mismo split y presupuesto.

### Protocolo congelado del screen eAP causal-scale v1

- cache identity `36c12d75...f1fac309`, file SHA256 `bba9ff9b...557eb72`;
- 2.048 train de 9 secuencias y 2.048 validation de 3 secuencias disjuntas;
- seed 7, máximo 18 épocas, BF16, batch 32, máximo total 6 h;
- warm-up foreground 3 épocas;
- pérdida física NLL + CVaR top 10% peso 2;
- checkpoint selection: MiD macro por secuencia, desempate failure rate;
- early stopping no antes de época 8, paciencia 5;
- cajas t1/t2 solo como weak supervision; t0 proxy inválido; ninguna caja entra al
  forward;
- Garl primario: checkpoint oficial event-only sobre los mismos 2.048 tokens;
- test privado, CodaBench, EvTTC test y seeds V8 901/902/903 permanecen cerrados;
- una seed/validation local nunca autoriza claim SOTA.

La ejecución completa no se había iniciado al congelar este documento. El benchmark
128+128 es solo throughput (`5.289 s`, `395.6 MiB`), no métrica científica.

## Pregunta

¿Un encoder event-only high-resolution con pretraining JEPA multihorizonte mejora
TTC frente a entrenamiento supervisado desde cero y frente a Garl-TTC bajo el
mismo protocolo? RGB-E es una segunda comparación aislada, no una mezcla con el
resultado event-only.

## Jerarquía de evidencia

```text
unit/synthetic smoke
  contrato matemático; nunca promoción

raw screen
  datos reales, 256–2.048 muestras/split, una seed

semantic representation gate
  level vs level+residual, probes congelados, igual compute

paired confirmation
  random vs JEPA, mismo split/seed/trainer

full candidate
  todas las filas válidas, seeds 7/13/23, commit limpio

freeze
  selección solo con validation Garl; EvTTC permanece cerrado

EvTTC Table VI predict
  inferencia label-free, cero updates

EvTTC Table VI score
  targets abiertos en proceso separado

eAP/CodaBench
  submission congelada y número de intentos registrado
```

## Datos

- GarlTTC/eAP: train y validation del candidato; lectura raw bajo demanda.
- EvTTC-32: evaluación grouped y Tabla VI; nunca selecciona el supuesto zero-shot.
- Benchmark-10: sellado hasta freeze.
- CARLA: resultado negativo histórico, fuera del camino activo.

## Screen

El screen activo usa `configs/experiment/e_jepa_garl_event_screen_v1.yaml`:

- event-only, 320x192, cinco pasos;
- máximo 2.048 muestras por split;
- seed 7;
- hasta ocho épocas;
- selección MiD macro por secuencia;
- `claim_eligible=false`.

Antes de ampliar debe pasar:

1. loss finita y descenso respecto al inicio;
2. ninguna fuga de target al input;
3. cobertura de todas las secuencias del split;
4. mejora pareada frente a random/control;
5. coste compatible con el presupuesto del host.

## Gate semántico de representación

El benchmark sintético fija la hipótesis, no promueve el modelo. En eAP se
compararán únicamente:

```text
JEPA level
JEPA level + temporal residual
```

Ambos deben compartir filas, inicialización del backbone, predictor, batch/accum,
seeds y tiempo aproximado. Tras congelar el encoder se ajustan probes por secuencia
para TTC/log-TTC, dirección/tasa de expansión, event rate e ID de secuencia. El
residual solo se promociona si mejora señal dinámica/TTC sin aumentar memorización
de secuencia y no empeora MiD macro. No se añaden rate, HSIC, CMI, MMD ni INTACT
antes de superar esta comparación mínima.

## Full candidate

`configs/experiment/e_jepa_garl_event_full_v1.yaml` congela:

- encoder event-only dim 192, patch 16 y block-causal;
- todas las filas válidas del split 32/8;
- seeds 7/13/23;
- BF16, batch 4, acumulación 6;
- máximo 30 épocas, mínimo 10, paciencia 6;
- Git limpio obligatorio;
- resume determinista por seed+época.

El freeze exige que los tres runs compartan commit, config y dataset hashes. El
seed elegido minimiza MiD macro en validation Garl. El freeze no convierte el run
en resultado oficial.

## EvTTC Tabla VI

`predict` recibe solo un checkpoint congelado y shards label-free con identidades:

```text
sequence_id, sample_token, track_id, timestamp_us
```

Se rechazan TTC, depth, category, bbox/masks privilegiadas o cobertura incompleta.
El score recibe un payload de targets separado y no puede retroalimentar training,
normalización o selección.

Protocolos bbox:

- `P0_oracle_bbox_roi`: diagnóstico asistido, nunca raw/bbox-free;
- `P1_predicted_bbox_roi`: deployable solo si el detector es causal y congelado;
- `P2_raw_fullframe`: candidato sin cajas GT.

## Comparaciones mínimas

1. trivial mean/median y geometría analítica;
2. Garl event-only oficial/local comparable;
3. high-res supervisado random;
4. high-res con JEPA denso compatible, `level` vs `level+residual`;
5. RGB-E como ablación posterior;
6. geometría causal bbox-free solo si supera al control.

No ejecutar el cartesiano completo. Se promociona un único finalista por gate.

## Métricas

- eAP/Garl: MiD firmado, RTE ponderado, failure rate y macro por secuencia;
- EvTTC: RTE, MAE/RMSE y métricas declaradas de Tabla VI;
- estadística: media/desviación entre seeds y bootstrap por secuencia;
- eficiencia: preprocessing e inferencia separados, RAM, VRAM y throughput;
- incertidumbre: NLL, ECE/cobertura y aumento bajo corrupción.

No se hace bootstrap por ventanas correlacionadas.

## Claim boundary

Un claim SOTA requiere al menos:

- protocolo igual al baseline;
- modalidad declarada event-only o RGB-E;
- tres seeds y freeze previo;
- resultado EvTTC comparable;
- submission oficial eAP/CodaBench;
- hashes de commit/config/dataset/checkpoint;
- latencia y memoria del candidato real;
- resultados negativos y número de submissions visibles.

Actualmente ninguno de esos gates externos está cerrado.
