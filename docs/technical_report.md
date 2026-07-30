# Informe técnico v6

Actualizado: 2026-07-30.

## 1. Resumen

E-JEPA-TTC estudia representación predictiva de eventos para estimar
Time-to-Collision. La revisión v6 elimina del camino activo las pérdidas
FlowMimic globales, reconstruye exactamente `BASE`, implementa una comparación
matched entre pooling global y tokens densos, y añade una réplica auditable de
Garl-TTC.

Conclusión actual:

> Dense Patch supera a A0 en la confirmación histórica de un split, pero no
> sobrevive grouped CV de cinco folds y tres seeds. A0 es la arquitectura
> final; A1 conserva valor como hipótesis para pretraining object-centric, no
> como ganador actual. AttnRes y Object-KDA permanecen rechazados.

## 2. Evidencia histórica

### 2.1 BASE exacto

La auditoría carga el checkpoint y cache originales y reproduce todas las
predicciones.

| Propiedad | Valor |
|---|---|
| encoder | EventTubeletTransformer |
| input | 21 canales |
| dim/depth/heads | 192 / 6 / 6 |
| patch | 16 |
| salida | log-TTC |
| train/validation | 7.835 / 2.040 |
| best downstream | epoch 26/30 |
| best SSL | epoch 6/30 |

| Métrica validation | Valor |
|---|---:|
| MAE | 0,3228917687 s |
| RMSE | 0,5844324448 s |
| error relativo medio | 8,1553575311 % |

La equivalencia de arrays es byte a byte. Se conserva como
`B0_HISTORICAL_BASE_EXACT`.

### 2.2 FlowMimic

El experimento multisemilla histórico no justifica sus pérdidas globales:

| Variante | MAE medio | Degradación |
|---|---:|---:|
| BASE | 0,332715 s | - |
| alignment | 0,369604 s | +11,1 % |
| inverse-TTC | 0,432909 s | +30,1 % |
| ambas | 0,435869 s | +31,0 % |

El resultado negativo se conserva, pero el código FlowMimic de uso único se
retira del pipeline activo.

### 2.3 Confirmación matched

| Brazo | Best epoch | Error rel. macro | Score | MAE macro | ms/ventana |
|---|---:|---:|---:|---:|---:|
| A0 global | 17 | 16,129 % | 0,32523 | 0,701 s | 8,98 |
| **A1 Dense** | **20** | **15,210 %** | **0,30543** | **0,628 s** | 17,15 |
| A2 AttnRes | 10 | 16,136 % | 0,32503 | 0,653 s | 16,70 |
| K1 Object-KDA | 7 | 16,960 % | 0,34139 | 0,731 s | 16,99 |

El máximo fue 40 épocas para todos, con mínimo 10 y paciencia 6. A1 completó
26 épocas; A0, 23; A2, 16; K1, 13. El screen de ocho épocas subestimó A1
porque su mejor checkpoint aparece en la época 20. A1 mejora el error relativo
un 5,70 % frente a A0, pero aproximadamente duplica la latencia.

## 3. Protocolo de comparación

`BASE` histórico y `A0_MATCHED_GLOBAL` tienen roles distintos:

```text
B0 histórico
    reproducción exacta del resultado anterior

A0 matched
    control entrenado con el mismo cache y trainer que A1/A2/K1
```

La auditoría matched exige igualdad en:

- IDs de ventanas;
- hash del backbone inicial;
- hash de la cabeza común;
- train/validation count;
- épocas máximas;
- batch y acumulación;
- learning rate y weight decay;
- loss y early stopping.

Core y Garl disponen de caches y resúmenes separados. Esta separación evita que
una segunda ejecución reemplace `matrix_summary.json` de la primera.

## 4. Arquitecturas Core

### 4.1 A0 global

Mantiene el encoder BASE y reduce tokens antes de la ruta temporal. Es el
control matched, no el resultado histórico.

### 4.2 Dense Patch / Patch Policy

Cada instante ejecuta interacción espacial bidireccional entre patches. La
causalidad se aplica únicamente entre instantes:

```text
patches del mismo frame  ↔  atención espacial
frames pasados          →  frame actual
frame futuro            ✕  frame pasado
```

El pooling se retrasa hasta después del mezclador temporal.

### 4.3 Attention Residuals

Cada tarea aprende pesos sobre las salidas de profundidad:

```text
mask, motion, geometry, risk
```

Las claves se RMS-normalizan para que una capa no gane por escala; los valores
mezclados conservan su magnitud original. Se registra el peso máximo por tarea
y se rechaza colapso persistente a una sola capa.

### 4.4 Object-KDA

La implementación usa recurrencia delta temporal. No aplana los patches de un
frame como si fueran pasos causales. Primero resuelve el espacio y después
actualiza memoria temporal de objeto/región.

KDA se promociona solo si mejora precisión, reduce memoria de forma material o
permite duplicar resolución/horizonte sin degradación relevante.

## 5. Réplica Garl-TTC

La implementación fue contrastada con el repositorio público
`NAIL-HNU/Garl-TTC`, commit
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`.

Contrato:

```text
t0, t1, t2 a 100 ms
RGB: t0 y t2
eventos: t0→t1 y t1→t2
20 planos por intervalo, 40 canales
ROI común de endpoint, 128x128
ResNet-50 RGB y eventos
direct o Learned Height Ratio
early/late fusion
foreground decoder training-only
```

La adaptación EvTTC usa altura visible después del crop común. No se presenta
como reproducción del MiD oficial eAP.

## 6. Geometría

Para escala aparente \(s\), el inverse-TTC en el endpoint actual de un par es:

\[
q_t = \frac{s_t/s_{t-\Delta t}-1}{\Delta t}.
\]

Se implementan:

1. height ratio;
2. raíz/área;
3. expansión afín con traslación y rotación separadas;
4. alineamiento por contraste de eventos.

Los expertos usan el último par válido. Promediar TTC de pares que representan
endpoints distintos queda prohibido.

También se implementó un baseline de expansión causal de bbox con hasta 21
observaciones pasadas y calibración log-afín ajustada solo en train. Cubre
311/314 ventanas. Al usar A1 únicamente en las tres filas inválidas obtiene
14,790 % de error relativo macro, pero el score empeora de 0,30543 a 0,31144
por RMSE; no pasa el gate del 5 %. Además, usa más contexto que A1 y debe
reportarse como track bbox-assisted.

El port source-traceable de STRTTC implementa NLTS, contornos, ajuste local de
planos, normal flow, RANSAC y el solver lineal de tres parámetros. En 40
muestras del split histórico solo resuelve 27; sus éxitos tienen 112,96 % de
error relativo macro. Los 13 fallos se conservan, por lo que la métrica
success-only no es comparable con un candidato de cobertura completa.

## 7. Compensación ego

### 7.1 Contrato del HDF5

El formato oficial declara posición, actitud y velocidad en el mundo. La
auditoría local confirma:

```text
velocity = [north, east, up]
attitude = [roll, pitch, heading_degrees]
```

La velocidad se transforma al frame óptico del evento con:

\[
T_{\text{event}\leftarrow\text{nav}} =
T_{\text{event}\leftarrow\text{RGB}}
T_{\text{RGB}\leftarrow\text{LiDAR}}
T_{\text{LiDAR}\leftarrow\text{nav}}.
\]

También se incluye la velocidad producida por el brazo rígido cuando cambia el
heading.

### 7.2 Warp físico

Para un punto observado en el pasado:

\[
P_{\mathrm{current}} =
R_{\mathrm{current}\leftarrow\mathrm{past}}P_{\mathrm{past}}
- \Delta C_{\mathrm{current}}.
\]

Los puntos se reproyectan con intrínsecos de la cámara de eventos. El componente
de cierre ego se expresa como:

\[
q_{\mathrm{ego}} = v_z / Z.
\]

La profundidad \(Z\) es imprescindible. Usar la distancia oficial EvTTC crea un
oracle, no una entrada de inferencia. Una versión desplegable requiere
profundidad predicha y un gate separado.

## 8. Evidencia ejecutada

Se ejecutaron 38 ventanas train, 10 validation y dos épocas. El smoke confirmó:

- forward/backward;
- causalidad;
- paridad de inicialización;
- checkpointing;
- ejecución de Garl G0–G7;
- ejecución de expertos geométricos.

Los errores del smoke son demasiado inestables para promoción. El screen Core
de ocho épocas también fue insuficiente para Dense. La confirmación larga
promovió A1 únicamente a grouped CV y rechazó A2/K1.

El grouped CV cerró 30/30 runs A0/A1. La desviación se calcula entre las tres
medias por seed, no tratando ventanas como réplicas independientes:

| Variante | Score ± sd | Error rel. ± sd | MAE ± sd | ms/ventana |
|---|---:|---:|---:|---:|
| **A0** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s | 4,54 |
| A1 | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** | 9,82 |

A1 empeora 1,47 % el score y 0,99 % el error relativo, mejora solo 0,41 % el
MAE, consume 1,58× entrenamiento y 2,16× latencia. Los bootstrap OOF pareados
por secuencia cruzan cero para las tres seeds. A0 queda seleccionado.

Una ablación posterior probó si la localización resolvía por sí sola la
debilidad de A1. `R1_MATCHED_BBOX_ROI` aplica la bbox GT al pooling de tokens,
sin añadir geometría. En cinco folds seed 7 obtiene score 0,59814, error
relativo 30,99 % y MAE 1,0100 s. Frente al A0 de la misma seed empeora 2,90 %,
2,74 % y 4,55 %, con 1,67× tiempo. Se descarta: la región correcta no aporta
TTC si la cabeza no mide explícitamente la expansión.

Después del freeze de arquitectura se compararon dos perfiles A0 sobre el
mismo cache 19/5/8. `matched` mejoró el score medio un 10,95 % frente a
`throughput`, con 4,02× tiempo. El checkpoint matched seed 13 fue el mejor en
validation y se congeló antes de abrir family-OOD.

| Evaluación | Secuencias / ventanas | Score | Error rel. macro | MAE macro |
|---|---:|---:|---:|---:|
| validation | 5 / 314 | 0,28992 | 14,46 % | 0,541 s |
| family-OOD reutilizado | 8 / 481 | 0,53784 | 30,56 % | 0,805 s |

La degradación OOD es 85,5 % en score, 111,4 % en error relativo y 48,8 % en
MAE. El intervalo bootstrap 95 % de MAE OOD es 0,593–1,128 s. No es un test
externo virgen ni Benchmark-10.

El screen Garl ResNet-50 ejecutó G0–G7 con batch efectivo 24. G5 RGBE-LHR early
fue el mejor brazo local con 36,52 % de error relativo macro. No se declara
paridad: el repositorio oficial usa 50 épocas y late fusion parte de ramas LHR
preentrenadas.

## 9. Screens y gates

Perfil Screen:

```text
train/validation       304 / 80 ventanas
épocas máximas         8
early stopping         patience 2, mínimo 3 épocas
precision              BF16
Core batch             24
Garl effective batch   24
```

Orden:

1. Core y Garl en split histórico: completado.
2. Confirmación larga Core: completada; A1 promovido a grouped CV.
3. Grouped CV A0/A1 cinco folds × tres seeds: completado; gana A0.
4. Ajuste final A0 matched/throughput y family-OOD: completado.
5. Repetición Garl con presupuesto oficial antes de cualquier claim.

TargetQuery, máscara predicha, refiner, router, residual e incertidumbre quedan
bloqueados hasta que `A4_GT_GEOMETRY` supere el gate frente a BASE.

## 10. Datos externos

eAP train-40 está completo (40 secuencias, 216 archivos, 536,64 GiB). El
inventario firmado selecciona 12 secuencias sin consultar EvTTC: nueve train y
tres validation. El loader usa `ms_to_idx`, lee solo eventos bajo demanda y no
abre los 118 GiB RGB. Se comparan:

```text
sin eAP
eAP-SSL sin TTC
eAP-Geo con bbox proyectada, expansión/cierre y objectness, sin target TTC
```

Los dos brazos usan el mismo encoder EventTubelet, seed, ventanas y presupuesto.
El análisis usa 1.024/256 muestras, máximo tres épocas y early stopping 2/1;
después entrena A0/A1 en fold 0/seed 7. Se escala de 12 a 40 solo si mejora RTE
y MAE en al menos dos folds.
El smoke SSL real completó best/last en 4,9 s efectivos con validation loss
0,06474 y sin colapso; es una loss latente, no error TTC.

CARLA DVS Looming se añadió como fuente sintética explícita. El archivo local
coincide con el MD5 oficial y contiene 1.406 secuencias/7.692 millones de
eventos. Con 100 ms de contexto se aceptan 1.395: 759 colisiones con TTC
positivo y 636 negativos censurados. El loader usa mmap y nunca habilita
pickle. Los bloques de 25 IDs producen 803/298/294 secuencias
train/validation/test. Este holdout solo es out-of-sample dentro de CARLA; la
transferencia a EvTTC mide el cambio de dominio real.

CARLA no contiene bbox temporales, está cuantizado a 10 ms y cubre TTC positivo
solo hasta aproximadamente 3,85 s. Por tanto sirve para pretraining de
percepción, expansión y riesgo, no para sustituir la supervisión o el benchmark
EvTTC.

El loader JEPA lazy genera 12.020 pares train, 4.457 validation y 4.297 test
con un máximo de 16 ventanas por secuencia. Un smoke de dos épocas redujo la
loss de validation de 0,02563 a 0,02247 sin dimensiones colapsadas. La
evaluación de contrato sobre 16 pares del test sintético dio loss 0,02195;
esta cifra no es TTC y no se usa para seleccionar arquitectura EvTTC.

En probes de 640–960 observaciones, batch 24/acumulación 2/ocho workers fue el
perfil más rápido (8,46 observaciones/s, ~688 MiB peak VRAM). Batch
16/32/48/96 y perfiles de 6/12 workers no lo superaron por contención de
lectura/voxelización.
El full usa BF16, AdamW fused, warm-up/cosine, EMA, clipping y early stopping
8/6 con máximo 30 épocas. Se proyectan 32,5 min por época y un máximo de
16,2 h. Los pilotos pareados EvTTC muestran que CARLA-SSL empeora A0 en RTE un
1,72 % y el auxiliar TTC sintético un 17,3 %; no se ejecuta el full mientras no
cambie la hipótesis.

## 11. Eficiencia y almacenamiento

- caches comprimidos y acotados;
- sin cache global de voxels;
- eAP leído por `ms_to_idx`, sin abrir RGB ni duplicar sus 418 GiB de eventos;
- CARLA leído por ventanas mmap, sin cache voxel ni segunda copia;
- logs JSONL, best/last y resume atómico para evitar repetir épocas CARLA;
- sin extracción completa de TAR RGB;
- SAM almacenado como máscara comprimida;
- DINO reducido a tokens objeto/borde;
- máximo tres checkpoints por run;
- BF16 y acumulación de gradiente;
- solver geométrico en FP32;
- Benchmark-10 sin teachers ni lectura de checkpoints durante la latencia.

## 12. Limitaciones actuales

- A1 solo gana en un split histórico y pierde grouped CV multisemilla;
- R1 demuestra que bbox-ROI sin geometría explícita tampoco mejora A0;
- Garl ResNet-50 tiene screen corto, no la reproducción de 50 épocas;
- la geometría bbox no pasa el score compuesto y STRTTC tiene cobertura
  incompleta;
- compensación traslacional final depende de profundidad predicha;
- bbox-free no está promocionado;
- eAP está completo y su piloto SSL/Geo→EvTTC está en curso;
- CARLA SSL y TTC sintético empeoran el screen cross-domain;
- family-OOD muestra degradación material;
- no existe evaluación Benchmark-10 oficial ni claim SOTA.

## 13. Fuentes primarias

- EvTTC: https://arxiv.org/abs/2412.05053
- Formato EvTTC: https://nail-hnu.github.io/EvTTC/download/data_format/
- Garl-TTC/eAP: https://arxiv.org/abs/2603.16303
- Código Garl-TTC: https://github.com/NAIL-HNU/Garl-TTC
- Patch Policy: https://arxiv.org/abs/2607.18236
- Kimi K3: https://arxiv.org/abs/2607.24653
- Event-Aided TTC: https://arxiv.org/abs/2407.07324
- Código Event-Aided TTC/STRTTC:
  https://github.com/NAIL-HNU/event_aided_ttc
