# Protocolo experimental

Actualizado: 2026-08-09.

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
