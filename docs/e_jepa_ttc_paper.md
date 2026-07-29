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
byte. Dense Patch, AttnRes, Object-KDA y la geometría multi-experto están
implementados, pero sus screens comparables siguen pendientes.

## Motivación

El head global actual puede descartar bordes, escala y movimiento local antes
de identificar el objeto. OGE retrasa el pooling, separa interacción espacial de
causalidad temporal y obliga a que la predicción pase por inverse-TTC físico
antes de cualquier residual.

## Datos y separación

- EvTTC-32: única supervisión TTC oficial.
- Benchmark-10: inferencia final sellada.
- eAP train-40: pretraining no-TTC posterior.

No existe split aleatorio por ventanas. La selección final usa grouped CV de
cinco folds y bootstrap por secuencia.

## Controles

`B0_HISTORICAL_BASE_EXACT` es una reproducción, mientras que
`A0_MATCHED_GLOBAL` es el control de la nueva matriz. A0/A1/A2/K1 comparten
muestras, backbone, cabeza, trainer y early stopping.

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

Height, area y affine se evalúan en el endpoint actual. La navegación se
convierte desde norte/este/arriba a la cámara de eventos mediante extrínsecas.
El warp traslacional necesita profundidad; si usa distancia EvTTC se etiqueta
como oracle.

## Evidencia actual

Los smokes ejecutados validan integración, no precisión. Los próximos pasos son:

1. screen Core 304/80;
2. screen Garl ResNet-50;
3. grouped CV de los promovidos;
4. tres seeds para máximo dos finalistas;
5. Benchmark-10 después del freeze.

TargetQuery, máscaras predichas, refiner, router, residual e incertidumbre
permanecen bloqueados hasta que la geometría bbox-GT supere BASE.

## Limitaciones

No hay resultado Screen, grouped CV v6, bbox-free promovido, profundidad
predicha ni evaluación externa. El sistema no es apto para control de seguridad.

## Referencias

Las referencias completas están en `docs/references.bib`.
