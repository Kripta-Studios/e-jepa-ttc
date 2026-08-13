# Metodología

> La metodología histórica inferior sigue siendo válida para sus experimentos,
> pero la unidad experimental vigente de Scientific Recovery V5 es una cadena
> A4→A6/A8 y Garl autocontenida por fold. Véase
> [Scientific Recovery V5](SCIENTIFIC_RECOVERY_V5_STATUS.md).

Actualizado: 2026-08-02.

La especificación completa está en [PLAN.md](../PLAN.md). Este documento resume
la ruta activa.

## Entrada raw y representación

Las ventanas eAP se localizan mediante el índice HDF5 y se leen solo cuando el
batch las necesita. Cada muestra se subdivide causalmente en cinco pasos y cada
paso se codifica como voxel de cinco bins por polaridad más canales auxiliares
observables, total 21 canales.

No se materializa un cache global. La misma selección balanceada se aplica antes
de abrir medios para evitar gasto y sesgo de I/O.

## Encoder high-resolution

El encoder conserva tokens espaciales mediante atención local por ventanas,
padding con máscara y merge 2x2 opcional. El mixer temporal block-causal opera
sobre `[B,T,P,D]`; query pooling produce el embedding del readout TTC.

KDA existe como ablación negativa y no es el default. El guard de memoria estima
tokens/atención antes de reservar tensores incompatibles con la GPU.

## Objetivo TTC

El target Garl es firmado. La pérdida aplica Smooth L1 a:

```text
signed_log1p(x) = sign(x) * log(1 + abs(x))
```

La inferencia devuelve TTC firmado directamente; no se aplica `exp` ni clamp
positivo que destruya el régimen negativo. El checkpoint se elige por MiD macro
por secuencia, no por promedio de ventanas.

## JEPA

La ruta científica prevista usa encoder online, encoder target EMA y predicción
multihorizonte de tokens futuros. Ningún target TTC/depth/categoría/máscara puede
entrar al pretraining SSL.

El pretrainer high-resolution aún no está implementado. El encoder pooled legacy
se rechaza porque transferir sus pesos como si fueran equivalentes mezclaría
arquitecturas y produciría una comparación inválida.

## Auditoría de capacidad semántica

La salud estadística no se usa como certificado semántico. El auditor
`semantic_shortcuts.py` entrena cinco objetivos sin etiquetas TTC ni shortcut y
después ajusta probes sobre encoders congelados. Ejecuta seeds 7/13/23 con un
shortcut de 12 bits constante por secuencia y un control donde cambia por frame.

En el caso lento, el objetivo repo de varianza conserva rango efectivo 11,41 y
cero dimensiones colapsadas, pero solo alcanza R² dinámica 0,15 mientras el
shortcut se decodifica con accuracy 0,84. VISReg no lo corrige. El residual
temporal alcanza R² 0,72 y reduce MAE log-TTC de 0,39 a 0,29. R²-lite no supera el
gate predeclarado y queda rechazado.

El control frame-varying invierte el resultado: el residual cae a R² -0,05 frente
a 0,74 del control. Por tanto no se sustituye el embedding de nivel. La propuesta
mínima es mantener `z_level` para escala/contenido y añadir `z_delta` para
expansión/movimiento. La única prueba real autorizada es
`level` frente a `level+temporal_residual`, con igual encoder, filas, seeds y
presupuesto. INTACT requiere acciones expertas y no aplica al dataset TTC actual.

## Geometría

Height ratio, area rate, affine expansion y event contrast se conservan como
baselines/teachers. Los fallos cuentan como falta de cobertura. Bbox/depth GT son
oracles; un candidato desplegable necesita localización y expansión causales
predichas.

## Selección y evaluación

- split por secuencia;
- sampling determinista y balanceado;
- screen barato antes de full;
- random vs JEPA con idéntico trainer;
- seeds full 7/13/23;
- freeze con validation Garl;
- predict EvTTC sin labels y score en proceso separado;
- benchmark oficial solo después del freeze.

Los resultados smoke prueban implementación, nunca SOTA.
