# OGE-JEPA-TTC v6: informe científico local

Actualizado: 2026-07-30.

## Resumen

OGE-JEPA-TTC estudia si una representación densa, object-centric y
geométricamente restringida puede mejorar la estimación TTC basada en eventos.
La revisión v6 no declara SOTA. Primero reconstruye exactamente el BASE
histórico, replica las decisiones arquitectónicas de Garl-TTC y define gates
que impiden promover componentes usando smokes o el benchmark externo.

El BASE reproducido obtiene `0,322892 s` MAE, `0,584432 s` RMSE y `8,1554 %`
de error relativo en la validación histórica. Las predicciones coinciden byte a
byte. En una confirmación matched de hasta 40 épocas, Dense Patch reduce el
error relativo macro de `16,129 %` a `15,210 %`; AttnRes (`16,136 %`) y
Object-KDA (`16,960 %`) no mejoran Dense. Sin embargo, en grouped CV de cinco
folds y tres semillas, A1 empeora el score un `1,47 %` y el error relativo un
`0,99 %`, mientras que solo mejora el MAE un `0,41 %`. A0 es la arquitectura
final; no existe claim SOTA.

## Motivación

El head global actual puede descartar bordes, escala y movimiento local antes
de identificar el objeto. OGE retrasa el pooling, separa interacción espacial de
causalidad temporal y obliga a que la predicción pase por inverse-TTC físico
antes de cualquier residual.

## Datos y separación

- EvTTC-32: única supervisión TTC oficial.
- Benchmark-10: inferencia final sellada.
- eAP train-40: 40 secuencias completas; piloto firmado 9/3 sobre 12 y full 32/8
  secuencias, exclusivamente pretraining no-TTC.
- CARLA DVS Looming: 1.395 secuencias sintéticas utilizables para SSL,
  expansión y riesgo; TTC positivo y negativos censurados.

No existe split aleatorio por ventanas. La selección final usa grouped CV de
cinco folds y bootstrap por secuencia.

## Controles

`B0_HISTORICAL_BASE_EXACT` es una reproducción, mientras que
`A0_MATCHED_GLOBAL` es el control de la nueva matriz. A0/A1/A2/K1 comparten
muestras, backbone, cabeza, trainer y early stopping.

| Brazo | Mejor época | Error relativo macro | MAE macro |
|---|---:|---:|---:|
| A0 global | 17 | 16,129 % | 0,701 s |
| **A1 Dense** | **20** | **15,210 %** | **0,628 s** |
| A2 AttnRes | 10 | 16,136 % | 0,653 s |
| K1 Object-KDA | 7 | 16,960 % | 0,731 s |

El máximo de épocas no favoreció a A0: A1 completó 26 épocas frente a 23 de
A0 bajo la misma regla de parada. El screen de ocho épocas era demasiado corto
para observar el mejor checkpoint de A1 en la época 20.

La promoción histórica se sometió después al gate predeclarado:

| Brazo | Score CV ± sd seeds | Error rel. CV ± sd | MAE CV ± sd |
|---|---:|---:|---:|
| **A0 global** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** |

A0 gana 10/15 comparaciones pareadas en score/error relativo y 8/15 en MAE.
A1 cuesta 1,58× entrenamiento y 2,16× latencia. Los intervalos bootstrap OOF
pareados por secuencia cruzan cero, por lo que la mejora de un split no se
generaliza.

La ablación `R1_MATCHED_BBOX_ROI` comprueba si basta con seleccionar tokens
dentro de la bbox GT. En cinco folds de seed 7 obtiene score `0,59814`, error
relativo `30,99 %` y MAE `1,0100 s`; frente a A0 de la misma seed empeora
`2,90 %`, `2,74 %` y `4,55 %`, respectivamente, con `1,67×` tiempo. Se
rechaza: localizar la región no equivale a medir su expansión.

## Arquitectura

```text
eventos / RGB opcional / navegación
→ Event-JEPA dense tokens
→ Task-Specific Attention Residuals
→ Spatial Patch Mixer
→ block-causal o Object-KDA
→ query/ROI solo tras gate
→ height + area + affine + event contrast
→ mezcla geométrica
→ TTC
```

Patch Policy garantiza comunicación bidireccional entre patches del mismo
instante. KDA solo opera temporalmente.

## Garl-TTC

La réplica local usa tres timestamps a 100 ms, dos intervalos de eventos, 40
planos, RGB endpoint, ROI 128x128, ResNet-50, LHR, late fusion y foreground
training-only. La adaptación EvTTC usa altura visible; no se presenta como una
reproducción del MiD eAP oficial.

## Geometría y ego-motion

Height, area y affine se evalúan en el endpoint actual. La geometría bbox
causal cubre 311/314 ventanas; con fallback A1 mejora ligeramente el error
relativo, pero empeora el score compuesto y no pasa el gate. El port causal
STRTTC cubre 27/40 muestras y falla el gate de cobertura y precisión.

La navegación se
convierte desde norte/este/arriba a la cámara de eventos mediante extrínsecas.
El warp traslacional necesita profundidad; si usa distancia EvTTC se etiqueta
como oracle.

## Evidencia actual

El screen Garl G0–G7 y la confirmación Core están ejecutados. El mejor Garl
local corto es G5 RGBE-LHR early con 36,52 % de error relativo macro; todavía
no reproduce las 50 épocas del protocolo oficial.

Tras seleccionar A0 por grouped CV, tres seeds compararon perfiles de ejecución.
`matched` obtiene score medio `0,30400` frente a `0,34139` de `throughput`, a
cambio de 4,02× tiempo de entrenamiento. Validation selecciona el checkpoint
A0 matched seed 13 antes de abrir el diagnóstico familiar:

| Split | Secuencias / ventanas | Score | Error rel. macro | MAE macro |
|---|---:|---:|---:|---:|
| validation | 5 / 314 | 0,28992 | 14,46 % | 0,541 s |
| family-OOD reutilizado | 8 / 481 | 0,53784 | 30,56 % | 0,805 s |

Family-OOD degrada el score un 85,5 %, el error relativo un 111,4 % y el MAE
un 48,8 %. Es disjunto respecto al ajuste actual, pero no es un test virgen del
proyecto. Benchmark-10 permanece sellado.

CARLA DVS Looming fue auditado sin pickle ni duplicación de eventos. El
manifest contiene 1.406 secuencias y 7.692 millones de eventos; 1.395 secuencias
superan el requisito de contexto de 100 ms. El split bloqueado usa 803/298/294
secuencias train/validation/test. Es out-of-sample dentro del simulador, no OOD
real. En el screen pareado, CARLA-SSL empeora el RTE de A0 un 1,72 % y el
brazo con TTC sintético lo empeora un 17,3 %; no se promocionan.

El smoke JEPA de dos épocas redujo validation loss de 0,02563 a 0,02247, sin
colapso. Un holdout de contrato de 16 pares sintéticos obtuvo 0,02195. Son
métricas de predicción latente e integración, no error TTC. El perfil full
genera 12.020/4.457/4.297 pares, usa BF16, batch 24, acumulación 2, ocho
workers y early stopping 8/6. Esos diagnósticos latentes no predijeron una
mejora TTC cross-domain.

El piloto eAP usa eventos HDF5 bajo demanda mediante `ms_to_idx`, sin abrir RGB
ni construir cache masivo. eAP-SSL predice embeddings futuros; eAP-Geo añade
bbox 3D proyectada, cierre/expansión y objectness por patch, pero nunca target
TTC. En tres épocas, SSL obtuvo loss `0,002358` y Geo loss `0,087108` e IoU
patch `0,2867`, sin colapso. La utilidad se decide transfiriendo el encoder a A0
y A1 con el mismo fold, seed, cabeza, trainer y early stopping que el control.

En folds 0/1, seed 7, Geo mejora A0 en RTE/MAE en 2/2
(+3,66 %/+4,30 % agregado) y A1 en RTE en 2/2 (+6,57 %), aunque MAE solo 1/2.
SSL es inconsistente. A0-Geo abre el gate al split full de las 40 secuencias en
32/8 (16.384/4.096 ventanas), pero los bootstrap RTE aún cruzan cero.

Los próximos pasos son:

1. ejecutar eAP-Geo full-40 con early stopping;
2. confirmar A0/A1-Geo en cinco folds × tres seeds;
3. implementar expansión/FoE explícita y un residual acotado sobre A0;
4. Garl con 50 épocas y pretraining por ramas;
5. Benchmark-10 únicamente tras congelar una candidata que pase OOF.

TargetQuery, máscaras predichas, refiner, router, residual e incertidumbre
permanecen bloqueados hasta que la geometría bbox-GT supere su gate.

## Limitaciones

No hay bbox-free promovido, profundidad predicha ni evaluación sobre el
Benchmark-10. Family-OOD muestra una degradación grande; eAP no incluye TTC
oficial y CARLA tiene cambio de dominio, reloj de 10 ms y TTC positivo limitado
a aproximadamente 3,85 s. El sistema no es apto para control de seguridad.

## Referencias

Las referencias completas están en `docs/references.bib`.
