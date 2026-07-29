# Informe técnico v6

Actualizado: 2026-07-30.

## 1. Resumen

E-JEPA-TTC estudia representación predictiva de eventos para estimar
Time-to-Collision. La revisión v6 elimina del camino activo las pérdidas
FlowMimic globales, reconstruye exactamente `BASE`, implementa una comparación
matched entre pooling global y tokens densos, y añade una réplica auditable de
Garl-TTC.

Conclusión actual:

> La infraestructura está preparada para una selección justa, pero ningún
> módulo nuevo ha demostrado todavía una mejora válida sobre BASE.

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

## 8. Smoke ejecutado

Se ejecutaron 38 ventanas train, 10 validation y dos épocas. El smoke confirmó:

- forward/backward;
- causalidad;
- paridad de inicialización;
- checkpointing;
- ejecución de Garl G0–G7;
- ejecución de expertos geométricos.

Los errores del smoke son demasiado inestables para promoción. En particular,
la señal favorable de KDA y LHR debe repetirse en Screen.

## 9. Screens y gates

Perfil Screen:

```text
train/validation       304 / 80 ventanas
épocas máximas         8
early stopping         patience 2, mínimo 3 épocas
precision              BF16
Core batch             24
Garl effective batch   128
```

Orden:

1. Core en split histórico.
2. Garl ResNet-50 en split histórico.
3. Descartar brazos que no superen su referencia.
4. Grouped CV cinco folds, seed 7.
5. Tres seeds para BASE y máximo dos finalistas.

TargetQuery, máscara predicha, refiner, router, residual e incertidumbre quedan
bloqueados hasta que `A4_GT_GEOMETRY` supere el gate frente a BASE.

## 10. Datos externos

eAP train-40 no interviene en la selección arquitectónica actual. Después de
congelar el mejor modelo EvTTC se podrá comparar:

```text
sin eAP
eAP SSL sin TTC
eAP SSL + pseudo-TTC 0,05
```

El último brazo solo se conserva si mejora OOF de forma reproducible. El
pseudo-TTC nunca se denomina ground truth.

## 11. Eficiencia y almacenamiento

- caches comprimidos y acotados;
- sin cache global de voxels;
- sin extracción completa de TAR RGB;
- SAM almacenado como máscara comprimida;
- DINO reducido a tokens objeto/borde;
- máximo tres checkpoints por run;
- BF16 y acumulación de gradiente;
- solver geométrico en FP32;
- Benchmark-10 sin teachers ni lectura de checkpoints durante la latencia.

## 12. Limitaciones actuales

- no hay resultado Screen todavía;
- no hay grouped CV del pipeline v6;
- Garl ResNet-50 solo tiene verificación de topología y smoke compacto;
- compensación traslacional final depende de profundidad predicha;
- bbox-free no está promocionado;
- eAP continúa descargándose;
- no existe evaluación oficial ni claim SOTA.

## 13. Fuentes primarias

- EvTTC: https://arxiv.org/abs/2412.05053
- Formato EvTTC: https://nail-hnu.github.io/EvTTC/download/data_format/
- Garl-TTC/eAP: https://arxiv.org/abs/2603.16303
- Código Garl-TTC: https://github.com/NAIL-HNU/Garl-TTC
- Patch Policy: https://arxiv.org/abs/2607.18236
- Kimi K3: https://arxiv.org/abs/2607.24653
- Event-Aided TTC: https://arxiv.org/abs/2407.07324
