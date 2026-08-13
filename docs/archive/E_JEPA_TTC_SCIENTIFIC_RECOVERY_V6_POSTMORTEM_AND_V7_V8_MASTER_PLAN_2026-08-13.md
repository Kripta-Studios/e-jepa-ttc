> **Propuesta histórica, reemplazada el 2026-08-13.** El contrato operativo y
> científico vigente es [`CODEX_HANDOFF.md`](../../CODEX_HANDOFF.md). El cuerpo de
> este documento se conserva sin reescritura para mantener la trazabilidad.

# E-JEPA-TTC — Scientific Recovery V6 Postmortem y Plan Maestro V7/V8

**Fecha:** 2026-08-13  
**Repo:** `Kripta-Studios/e-jepa-ttc`  
**Estado científico de partida:** Scientific Recovery V6 cerrado sin promoción  
**Rama V6:** `scientific-recovery-v6-oof-diagnostics`  
**HEAD V6 reportado:** `28c1efb50622255719f239622ba07858ce704535`  
**Documento canónico V6:** `docs/SCIENTIFIC_RECOVERY_V6_STATUS.md`  
**Private/test:** cerrado  
**Public validation para selección V5/V6:** no usado  
**Candidato sealed:** ninguno  

---

# 0. Resumen ejecutivo: qué sabemos y qué haría ahora

La recuperación científica ha acotado el problema hasta un punto donde seguir añadiendo
bloques arquitectónicos a ciegas tendría poco valor.

Los resultados grouped-dev limpios de V6 son:

| Modelo | Parámetros | MiD macro ↓ | Failure ↓ | Pearson ↑ | Geometría |
|---|---:|---:|---:|---:|---|
| Garl matched | 24,674,178 | **144.353** | **0%** | 0.042 | N/A |
| A5 causal | 424,274 | **155.472** | 4.761% | **0.360** | degradada |
| V6.1 radius=2 | 627,827 | 194.122 | 6.689% | 0.216 | exacta |
| A8.0 radius=1 | 627,827 | 197.691 | 7.019% | 0.225 | exacta |
| A6 | 498,130 | 211.509 | 7.849% | 0.201 | exacta |

La secuencia de evidencia es:

```text
A4
  geometría útil
  TTC débil
    ↓
A5
  el encoder se adapta libremente
  TTC mejora muchísimo
  geometría se degrada
    ↓
A6
  geometría congelada
  transport adapter aprende parte del TTC
  pero queda muy limitado
    ↓
A8
  geometry encoder frozen + transport encoder separado
  mejora A6 de forma estadísticamente clara
  pero queda lejos de A5/Garl
    ↓
V6.1
  radius 1 → radius 2
  mejora solo 3.57 MiD de media
  IC95% cruza cero
  ayuda en flujo alto y perjudica flujo bajo
```

La conclusión principal es:

> **La capacidad de aprender TTC existe en la familia E-JEPA —A5 llega a 155.472—,
> pero todavía no sabemos cuánto de la brecha restante con Garl procede de capacidad
> de modelo y cuánto de arquitectura, representación temporal, objetivo, foreground
> supervision o contrato de coverage.**

Por ello, la siguiente fase recomendada no es todavía una “V7 arquitectónica”.
Primero debe ser una:

> **Scientific Recovery V7 — Capacity Attribution Study**

Su función es aislar una incertidumbre enorme que aún no se ha medido: Garl tiene
24.67 M parámetros frente a 0.424 M de A5 y 0.628 M de V6.1, aproximadamente 58× y
39× más respectivamente.

V7 debe responder:

1. ¿Garl sigue siendo casi igual de bueno cuando se reduce a ~0.6–1 M parámetros?
2. ¿A5 causal mejora cuando se escala a ~2–3 M y ~6–8 M?
3. ¿La rama geometry-preserving mejora de forma material al aumentar capacidad?
4. ¿Aumentar parámetros mejora OOF o solo reduce train y amplía overfitting?
5. ¿La ventaja de Garl es principalmente capacidad o persiste a capacidad matched?

La recomendación es un **estudio secuencial**, no un grid masivo:

```text
V7.0 preflight / parameter counting
        ↓
V7.1
Garl-small ~0.6–1M
A5-medium ~2–3M
geometry-preserving-medium ~2–3M
        ↓
evaluar capacity effect
        ↓
 ┌──────────────┴────────────────┐
 |                               |
efecto pequeño              efecto grande
 |                               |
STOP capacity             V7.2 ~6–8M
 |                               |
V8 arquitectura          solo si sigue escalando:
                         considerar 20–25M
```

La V7 no es un candidato SOTA ni autoriza abrir private/test.

---

# 1. Qué se está intentando resolver realmente

No estamos intentando responder simplemente:

> “¿Qué modelo tiene menor MiD?”

Eso ya se conoce:

```text
Garl < A5 << V6.1 < A8 < A6
```

La pregunta científica es más precisa:

> **¿Qué mecanismo impide que una arquitectura E-JEPA causal, event-only y
> geométricamente coherente alcance o supere a Garl bajo la evaluación matched?**

Hay varias hipótesis vivas:

### H1 — capacidad

Garl tiene 24.67 M parámetros.

A5 tiene 0.424 M.

V6.1/A8 tienen 0.628 M.

La diferencia es demasiado grande para ignorarla, pero todavía no está aislada.

### H2 — representación temporal

Garl y E-JEPA no reciben la misma representación.

Según la auditoría local:

- Garl usa una representación de dos intervalos con 20 planos temporales por intervalo;
- E-JEPA usa tres pasos y 12 entradas por paso;
- en E-JEPA, 10 de esas entradas son bins de eventos separados por polaridad y 2 son
  event-count/event-rate;
- el modo de alineamiento temporal y ROI también difiere.

No describir esto como “40 canales por endpoint vs 12”. El contraste correcto es:

> **detalle temporal, polaridad, intervalos y alineamiento distintos.**

### H3 — objetivo/inductive bias

Garl está construido alrededor de una parametrización TTC fuertemente alineada con
cambio de escala/altura.

A5 demuestra que el encoder E-JEPA puede especializarse en TTC, pero esa
especialización degrada los probes geométricos.

### H4 — foreground / boundary supervision

La estimación TTC por expansión aparente es muy sensible a contorno, altura y borde del
objeto.

La historia A1/A3/A4 y el ROI stress ya mostraron que la representación de forma importa.

### H5 — geometry freeze demasiado restrictivo

A8/V6.1 exigen que la rama geométrica permanezca idéntica al parent.

A5 alcanza 155.472 precisamente cuando esa restricción desaparece.

Esto sugiere que:

```text
preservación bitwise ≠ necesariamente preservación científica óptima
```

Podría ser mejor una preservación blanda y verificable.

### H6 — transport scale

V6.1 prueba que radius=2 ayuda en flujo alto, pero perjudica flujo bajo.

Por tanto:

```text
r=2 global
```

no es la solución.

Sí queda apoyada la idea:

```text
routing r=1/r=2 condicionado por régimen
```

pero todavía no debe ejecutarse como conclusión confirmatoria.

### H7 — failure/coverage contract

Garl siempre devuelve predicción.

E-JEPA puede marcar samples unknown/failure por guards físicos.

Parte de la diferencia en failure rate es contractual, no necesariamente incapacidad
de producir una estimación puntual.

Una futura comparación debe separar:

```text
point prediction coverage
confidence / abstention
risk-coverage
```

---

# 2. Restricciones científicas que NO deben romperse

## 2.1. Private/test continúa cerrado

Hasta que exista un candidato final frozen:

```text
private_test_opened = false
```

debe seguir siendo cierto.

No abrir el test para:

- comparar tamaños;
- elegir width/depth;
- elegir radio;
- ajustar loss;
- seleccionar confidence threshold;
- decidir early stopping;
- escoger seed.

## 2.2. Public validation histórica no es test independiente

Aunque V5/V6 no la usaron para seleccionar A8/V6.1, la public validation ya se reutilizó
adaptativamente durante recovery anterior.

No debe presentarse ahora como confirmación independiente.

## 2.3. Los nueve train sequences ya son development evidence

Los grouped folds V5/V6 son mucho más limpios que ajustar sobre public validation, pero
V6 ya utilizó sus resultados para formular la siguiente hipótesis.

Por tanto:

> V7 sobre esos mismos folds es **desarrollo/atribución**, no confirmación independiente.

Esto es aceptable siempre que se declare.

## 2.4. No reescribir artifacts históricos

No modificar:

```text
artifacts/scientific_recovery_v5/
artifacts/scientific_recovery_v6/
```

para hacerlos encajar con V7.

V7 debe usar directorios nuevos.

## 2.5. No cambiar múltiples mecanismos en una misma ablation de capacidad

Para aislar capacidad:

- representación fija;
- losses fijas;
- folds fijos;
- preprocessing fijo;
- selector fijo;
- seed policy fija;
- arquitectura topológica fija en lo posible;
- cambia únicamente el tamaño.

Si se cambia además:

```text
bins + foreground + optimizer + width + objective
```

ya no es una capacity ablation.

---

# 3. Reconstrucción de resultados: de V4 a V6

## 3.1. A4: geometría sin TTC competitivo

A4 estableció que la representación podía conservar señal geométrica útil.

En el recovery histórico:

```text
A4 legacy seed7 MiD ≈ 262.825
```

y las correlaciones geométricas eran sustancialmente mejores que en A5.

La lectura de A4 no es “buen TTC”, sino:

> existe una representación geométrica observable que puede usarse como anchor.

## 3.2. A5: breakthrough TTC y conflicto geométrico

A5 añadió transporte/adaptación y consiguió:

```text
legacy seed7 ≈ 163.211 MiD
```

En grouped-dev causal V6:

```text
A5 causal = 155.472 MiD
failure   = 4.761%
Pearson   = 0.360
```

Esto es fundamental.

A5 causal es aproximadamente:

```text
38.65 MiD mejor que V6.1
42.22 MiD mejor que A8.0
56.04 MiD mejor que A6
```

y solo:

```text
11.119 MiD peor que Garl
```

A5 demuestra que el cuello no es simplemente:

- eventos insuficientes;
- causalidad insuficiente;
- imposibilidad de aprender TTC.

La familia tiene información suficiente para TTC fuerte.

El coste es geométrico:

```text
slope vs bbox   ~0.16  → ~0.027
slope vs física ~0.27  → ~0.041
```

y también caen ratios de dispersión predicha/objetivo.

Por tanto:

> A5 aprende una representación supervisada especializada en TTC que ya no conserva
> la representación geométrica requerida por el árbol científico.

## 3.3. A6: separación parcial

A6 congela geometry y permite adaptación transport/TTC.

Grouped-dev:

```text
MiD     = 211.509
failure = 7.849%
Pearson = 0.201
```

Es mejor que A4, pero no recupera A5.

Conclusión:

> un corrector pequeño sobre una geometría congelada es insuficiente.

## 3.4. A8: dual-stream correcto conceptualmente

A8 separa:

```text
geometry encoder frozen
transport encoder trainable
```

Resultado:

```text
A8 = 197.691
A6 = 211.509
delta ≈ -13.818 MiD
```

Mejora relativa aproximada:

```text
6.53%
```

El paired OOF V5 demostró una mejora clara de A8 respecto a A6.

Por tanto la separación dual-stream sí sirve.

Pero:

```text
A8 - Garl ≈ +53.34 MiD
```

y el gate `<=175` falla.

## 3.5. V6.1: radio 2 no resuelve el problema

V6.1:

```text
194.122 MiD
```

vs A8:

```text
197.691 MiD
```

Mejora:

```text
-3.570 MiD
IC95% [-8.190, +0.999]
```

No hay evidencia suficiente para declarar una mejora robusta.

Más importante:

```text
flow Q1/Q2: r2 empeora
flow Q3/Q4: r2 mejora
```

Así que la conclusión no es:

> aumentar radius.

Es:

> el radio óptimo depende del régimen de movimiento.

---

# 4. Dónde pierde A5 contra Garl

La brecha total A5−Garl es:

```text
+11.119 MiD
IC95% [4.271, 17.527]
```

Garl sigue siendo significativamente mejor.

Descomposición:

| Bucket | A5 − Garl |
|---|---:|
| crucial 0–3 s | +11.003 |
| small 3–6 s | +5.263 |
| large 6–10 s | -0.917 |
| negative | -4.231 |
| total | +11.119 |

El bucket `crucial` por sí solo equivale aproximadamente al 99% del gap neto.

`small` añade déficit, mientras que `large` y `negative` compensan una parte.

Esto implica:

> **No necesitamos mejorar A5 indiscriminadamente en todo el espacio TTC.**

La principal oportunidad está en:

```text
positive TTC 0–6 s
```

Especialmente:

```text
0–3 s
```

Pero hay que evitar convertir esta observación en tuning directo contra el OOF ya usado.

Debe traducirse en una hipótesis preregistrada para una fase posterior.

---

# 5. Sobreajuste de Garl: qué demuestra y qué NO demuestra

Los folds Garl muestran train-dev gap en el checkpoint seleccionado:

| Fold | Train MiD | Dev MiD | Gap |
|---|---:|---:|---:|
| F0 | 119.22 | 130.35 | +11.13 |
| F1 | 97.56 | 163.22 | +65.66 |
| F2 | 102.78 | 139.49 | +36.71 |

Y al continuar después del mejor checkpoint:

| Fold | Último train | Último dev |
|---|---:|---:|
| F0 | 108.61 | 142.34 |
| F1 | 79.22 | 175.00 |
| F2 | 91.37 | 145.68 |

Esto sí es evidencia de:

```text
epoch-level overfitting
```

especialmente F1/F2.

Pero no explica la victoria de Garl.

En los checkpoints seleccionados:

| Fold | Garl | A8 |
|---|---:|---:|
| F0 | 130.35 | 180.80 |
| F1 | 163.22 | 201.55 |
| F2 | 139.49 | 210.73 |

Garl gana en los tres dev folds no vistos durante entrenamiento.

Por tanto:

> un modelo puede sobreajustar después de su checkpoint óptimo y aun así tener una
> representación que generaliza mejor que la del competidor.

La pregunta correcta no es:

> “¿Garl overfittea?”

Sí.

La pregunta importante es:

> “¿Por qué su checkpoint seleccionado sigue generalizando mejor?”

---

# 6. La hipótesis de capacidad es importante y aún no está aislada

Ratios:

```text
Garl / A5   ≈ 58.16× parámetros
Garl / V6.1 ≈ 39.30× parámetros
```

Esta diferencia es suficientemente grande para ser una incertidumbre científica P1.

Pero no se puede inferir:

```text
39× params → 53 MiD de ventaja
```

porque cambian simultáneamente:

- representación;
- backbone;
- losses;
- parametrización TTC;
- temporal discretization;
- foreground supervision;
- coverage contract;
- freeze policy.

La capacity ablation debe separar estas variables.

---

# 7. Decisión: V7 debe ser un Capacity Attribution Study

Nombre recomendado:

```text
scientific-recovery-v7-capacity-attribution
```

Documento:

```text
docs/SCIENTIFIC_RECOVERY_V7_CAPACITY_ATTRIBUTION_PLAN.md
```

Config de protocolo:

```text
configs/protocol/scientific_recovery_v7_capacity_attribution.json
```

Artefactos:

```text
artifacts/scientific_recovery_v7/
    manifests/
    configs/
    runs/
    diagnostics/
    results/
    audits/
```

V7 no debe presentarse como:

```text
nuevo candidato SOTA
```

sino como:

```text
capacity attribution study
```

---

# 8. Qué familias escalar

## 8.1. Garl — curva descendente

Objetivo:

> determinar cuánto empeora Garl al reducir capacidad manteniendo su familia
> arquitectónica y representación.

Puntos objetivo:

```text
G-full   = 24.67M existente
G-large  ≈ 6–8M
G-medium ≈ 2–3M
G-small  ≈ 0.6–1.0M
```

No hace falta empezar entrenando todos.

Primero construir configs y contar parámetros.

### Restricciones

Mantener:

- mismos bloques conceptuales;
- mismo depth si es posible;
- escalar widths/channels de forma uniforme;
- mismo input representation;
- mismo preprocessing;
- misma loss;
- mismo training budget;
- misma selección de checkpoint.

Si para llegar a 0.6M hay que eliminar módulos enteros, deja de ser una ablation
puramente de capacidad.

En ese caso usar:

```text
G-small ≈ mínimo alcanzable preservando topología
```

y registrar el valor real.

## 8.2. A5 causal — curva ascendente principal

Esta es la curva E-JEPA más importante.

¿Por qué A5 y no A8 primero?

Porque A5 es el mejor E-JEPA TTC:

```text
155.472
```

y está solo 11.119 detrás de Garl.

Puntos:

```text
A5-base   = 0.424M existente
A5-medium ≈ 2–3M
A5-large  ≈ 6–8M
A5-xl     ≈ 20–25M solo si la curva sigue mejorando
```

Pregunta:

> ¿el mejor estimador TTC de nuestra familia está capacity-limited?

## 8.3. Geometry-preserving — curva secundaria

Usar A8/V6.1 como familia de control:

```text
GP-base   = ~0.628M
GP-medium = ~2–3M
GP-large  = ~6–8M
```

No ir directamente a 25M.

Pregunta:

> ¿la diferencia A5 vs geometry-frozen se debe a la restricción conceptual o
> simplemente a que la rama independiente tiene muy poca capacidad?

---

# 9. Cómo escalar sin cambiar accidentalmente la arquitectura

Crear un helper nuevo:

```text
scripts/plan_scientific_recovery_v7_capacity.py
```

Responsabilidades:

1. cargar modelo base;
2. recibir un target band;
3. construir candidate width multipliers;
4. contar parámetros exactos;
5. comprobar que los módulos requeridos siguen presentes;
6. escribir config frozen;
7. escribir manifest;
8. no entrenar.

Ejemplo de manifest:

```json
{
  "artifact_type": "scientific_recovery_v7_capacity_plan_v1",
  "family": "a5_causal",
  "target_parameter_band": [2000000, 3000000],
  "actual_parameter_count": 2487312,
  "base_config_sha256": "...",
  "generated_config_sha256": "...",
  "topology_preserved": true,
  "input_contract_unchanged": true,
  "loss_contract_unchanged": true,
  "private_test_opened": false
}
```

Tests:

```text
tests/unit/test_scientific_recovery_v7_capacity.py
```

Debe verificar:

- parameter bands;
- deterministic config generation;
- no cambio de input channels cuando el experimento es capacity-only;
- no cambio de losses;
- no cambio de ROI protocol;
- no cambio de sample manifest;
- no acceso a public validation/private;
- required modules preserved.

---

# 10. Optimizer y fairness: detalle importante

“No cambiar nada excepto parámetros” parece ideal, pero hay un riesgo:

> un learning rate adecuado para 0.4M puede ser incorrecto para 8M o 25M.

Hay dos diseños válidos.

## Diseño A — pure fixed-recipe capacity ablation

Mantener exactamente:

```text
optimizer
LR
weight decay
schedule
epochs / update budget
batch
```

Ventaja:

- atribución muy limpia.

Desventaja:

- un modelo grande puede parecer malo porque la recipe no escala.

## Diseño B — preregistered size-aware training rule

Antes de ver resultados:

- mantener optimizer family;
- definir una regla determinista de LR/batch scaling según tamaño;
- aplicarla a todos los tamaños;
- no retocar tras observar folds.

Este diseño es más justo si el tamaño varía 10–50×.

### Recomendación

Para V7 inicial:

1. usar la recipe original sin tuning para `small` y `medium`;
2. si se ejecuta `large`, aplicar una regla size-aware congelada en el preregistro;
3. nunca hacer per-fold tuning.

No usar la misma hiperparametrización entre Garl y E-JEPA solo por “fairness”.
La fairness inter-family debe estar en:

```text
samples
targets
folds
privilege
metric
selection budget
```

no en obligar a dos arquitecturas distintas a usar un LR idéntico.

---

# 11. V7 debe ser secuencial

No ejecutar 4 tamaños × 3 familias × 3 folds × 3 seeds de entrada.

## Stage V7.0 — preflight

Sin entrenamiento:

- parameter counting;
- FLOPs si puede medirse de forma fiable;
- peak activation estimate;
- configs;
- manifests;
- test suite;
- sample identity;
- Garl/A5/GP input contract.

Salida:

```text
artifacts/scientific_recovery_v7/manifests/capacity_preflight.json
```

Gate:

```text
PASS únicamente si todos los configs preservan su familia y protocolo.
```

## Stage V7.1 — screen informativo mínimo

Ejecutar:

```text
G-small    ~0.6–1M
A5-medium  ~2–3M
GP-medium  ~2–3M
```

Los tres en los mismos 3 folds, seed 7.

Por tanto:

```text
9 entrenamientos
```

No 36.

### Por qué estos tres

`G-small` pregunta:

> ¿Garl necesita realmente 24.67M?

`A5-medium` pregunta:

> ¿el mejor TTC E-JEPA todavía mejora con capacidad?

`GP-medium` pregunta:

> ¿la restricción geometry-preserving puede compensarse con más capacidad?

---

# 12. Estadística V7

Reusar la infraestructura paired ya endurecida.

Clustering:

```text
sequence_id + track_id
```

Resamples:

```text
5000
```

Comparaciones:

```text
G-small - G-full
A5-medium - A5-base
GP-medium - GP-base
A5-medium - G-small
GP-medium - G-small
```

También:

```text
train MiD
OOF/dev MiD
generalization gap
```

por tamaño.

Output:

```text
artifacts/scientific_recovery_v7/results/capacity_screen.json
```

---

# 13. Gate de “capacity effect”

No seleccionar únicamente por point estimate.

Definir antes:

## Efecto material

```text
|ΔMiD| >= 5
```

y preferiblemente IC95% que no cruce cero.

## Efecto fuerte

```text
|ΔMiD| >= 10
```

con IC95% consistente.

Estos thresholds no son claims SOTA; son criterios de si merece seguir gastando compute.

---

# 14. Árbol de decisión V7

## Caso 1 — Garl-small sigue casi igual

Ejemplo:

```text
G-full  144
G-small 149
```

con diferencia pequeña.

Conclusión:

> la ventaja de Garl no procede principalmente de 24M parámetros.

Acción:

```text
STOP Garl capacity branch
```

No entrenar G-medium/G-large.

V8 debe atacar:

- objetivo;
- representación temporal;
- foreground;
- preserve-vs-adapt geometry.

## Caso 2 — Garl-small empeora mucho

Ejemplo:

```text
G-full  144
G-small 180
```

Conclusión:

> capacidad importa de forma grande dentro de Garl.

Entonces ejecutar:

```text
G-medium ~2–3M
```

para obtener la curva.

Solo si G-medium sigue lejos de full:

```text
G-large ~6–8M
```

## Caso 3 — A5-medium mejora fuertemente

Ejemplo:

```text
A5-base   155
A5-medium 145
```

Conclusión:

> el mejor E-JEPA TTC es capacity-limited.

Entonces ejecutar:

```text
A5-large ~6–8M
```

y no construir todavía una V8 compleja.

## Caso 4 — A5-medium apenas cambia

Ejemplo:

```text
155 → 153
```

Conclusión:

> capacidad por sí sola no cierra la brecha.

Moverse a V8 arquitectónica.

## Caso 5 — GP-medium mejora mucho y A5-medium no

Conclusión:

> la rama dual geometry-preserving estaba under-capacity.

Escalar GP merece prioridad.

## Caso 6 — A5 mejora y GP no

Conclusión:

> el cuello es la restricción de preservación/fusión, no simplemente capacidad total.

Esto apoya una V8 con preservación blanda.

## Caso 7 — ambos mejoran

Entonces capacidad sí importa, pero todavía hay que decidir qué familia escala mejor en
eficiencia.

Comparar:

```text
ΔMiD / log(params)
ΔMiD / million params
OOF gap
```

---

# 15. Qué NO significa una capacity curve

Incluso si A5 8M llega a 145:

No significa automáticamente:

> E-JEPA ha superado Garl.

Necesitará:

- paired comparison;
- CI;
- mismo scope;
- failure/coverage;
- causalidad;
- geometry claim explícito;
- freeze;
- sealed test.

Y si la geometría de A5 sigue degradada:

> puede ser el mejor TTC arm, pero no el candidato científico geometry-preserving.

---

# 16. V8 si capacity NO es la explicación principal

Si V7 indica que capacidad satura pronto, la siguiente arquitectura debe integrar lo que
V6 ya enseñó.

Nombre provisional:

```text
scientific-recovery-v8-soft-geometry-multiscale-ttc
```

Arquitectura conceptual:

```text
events
   │
   ├── geometry trunk
   │     lower layers frozen / strongly distilled
   │     upper layers softly adaptable
   │
   ├── TTC / transport trunk trainable
   │       ├── transport r=1
   │       └── transport r=2
   │
   ├── foreground / boundary auxiliary training-only supervision
   │
   └── regime features
           magnitude
           confidence
           entropy
           cycle-error
             │
             ▼
        causal router
        r1 / r2 / geometry
             │
             ▼
     height-ratio / inverse-TTC
             │
             ▼
        point prediction
             +
       separate confidence
```

---

# 17. Preservación geométrica: pasar de bitwise a científica

V6 demuestra un dilema:

```text
bitwise geometry → TTC limitado
free encoder      → TTC fuerte, geometry cae
```

La siguiente pregunta lógica es si existe un punto intermedio.

En vez de:

```text
encoder.requires_grad = False
```

usar:

```text
lower geometry layers frozen
upper layers trainable
+
feature distillation against parent
+
geometry probes / losses
```

Loss conceptual:

```text
L =
    λ_ttc      * L_TTC
  + λ_geo      * L_geometry
  + λ_ratio    * L_height_ratio
  + λ_distill  * L_parent_features
  + λ_fg       * L_foreground
```

Los pesos deben preregistrarse usando únicamente train-development evidence.

No ejecutar un sweep cartesiano.

Probar primero una sola configuración basada en magnitudes de losses normalizadas.

---

# 18. Foreground supervision

El TTC corto depende fuertemente de estimar expansión aparente correctamente.

Una rama training-only de foreground/boundary puede ayudar a:

- altura;
- contorno;
- cambio de escala;
- robustez de crop.

No usar bbox como feature del neural forward si el claim sigue siendo event-only.

Puede usarse como:

```text
training supervision
```

si queda explícitamente documentado.

El decoder no necesita existir en inference.

---

# 19. Multi-scale routing

V6.1 ya proporcionó la evidencia necesaria para formular esta hipótesis:

```text
r2:
  malo en low flow
  bueno en high flow
```

Por tanto el siguiente modelo no debe escoger un radio global.

Ejemplo:

```python
gate = sigmoid(router(regime_features))
transport = (1 - gate) * transport_r1 + gate * transport_r2
```

`regime_features` deben ser observables causalmente:

- motion magnitude;
- confidence;
- entropy;
- event density;
- cycle consistency.

No incluir TTC target/bucket como input.

---

# 20. Short-TTC calibration

La brecha A5–Garl está concentrada en 0–6 s.

Una V8 puede dar más importancia a errores críticos, pero debe evitar una trampa:

> entrenar explícitamente para reproducir las contribuciones exactas del OOF observado.

En vez de usar las secuencias como lookup o ajustar weights post hoc, preregistrar una
loss genérica de short-TTC:

```text
L_short
```

basada en el target train-only.

Ejemplo de buckets conceptuales:

```text
0–3
3–6
6–10
negative
```

si son parte del protocolo oficial.

Objetivo:

- reducir error crítico;
- no destruir large/negative, donde A5 ya es competitivo.

---

# 21. Coverage y failure deben separarse

E-JEPA puede declarar unknown.

Garl siempre predice.

Una próxima fase debe producir siempre:

```text
point_prediction
```

y separadamente:

```text
confidence
abstention_score
```

Así pueden reportarse dos cosas:

## Full coverage

```text
MiD a 100% de samples
```

## Selective

```text
risk vs coverage
```

No convertir unknowns actuales en “correctos”.

El fallback puede ser:

- analytic geometry TTC;
- direct inverse-TTC head;
- previous safe estimate;

pero debe estar preregistrado.

---

# 22. Longer history / SSM: no todavía

No es una mala idea.

Pero V6 indicó que el factor observable más útil era `motion_scale`, y radius=2 produjo
mejora en high-motion.

Esto prioriza:

```text
multi-scale spatial/temporal transport
```

antes de introducir recurrencia larga.

Un SSM/recurrent branch debería activarse solo si un diagnóstico futuro muestra que
los errores restantes se concentran en:

- baja densidad de eventos;
- aceleración;
- cambios de dirección;
- historia insuficiente.

No añadirlo simplemente porque sea moderno.

---

# 23. Fuentes externas que apoyan, pero NO sustituyen, la evidencia local

## Garl-TTC / eAP

Referencia principal del comparator local y de la utilidad de height-ratio /
foreground-style inductive biases.

La evidencia local del repo sigue siendo la autoridad para nuestras cifras.

## V-JEPA 2.1 — arXiv:2603.14482

Apoya una idea relevante:

- dense predictive loss;
- deep self-supervision;
- dense spatial/temporal features;
- scaling de capacidad y datos.

No demuestra que vaya a mejorar TTC event-only, pero sí apoya la estrategia de preservar
estructura mediante supervisión densa en lugar de congelar todo el encoder.

## Event-Aided TTC — arXiv:2407.07324

Su enfoque coarse-to-fine es conceptualmente consistente con el hallazgo V6 de que
un único radio fijo no cubre todos los regímenes.

No es evidencia de que nuestro exacto router r1/r2 vaya a funcionar.

---

# 24. Git: cerrar V6 correctamente antes de V7

El usuario reporta:

```text
branch = scientific-recovery-v6-oof-diagnostics
HEAD   = 28c1efb50622255719f239622ba07858ce704535
tracked worktree clean
18 untracked preexistentes
push = NO
```

No usar:

```text
git add .
git clean
git clean -fd
```

Los untracked deben preservarse.

## 24.1. Verificación

```powershell
cd "C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc"

$Git = "C:\Program Files\Git\cmd\git.exe"

& $Git branch --show-current
& $Git rev-parse HEAD
& $Git status -sb
& $Git diff --check
```

Debe verse:

```text
scientific-recovery-v6-oof-diagnostics
28c1efb50622255719f239622ba07858ce704535
```

## 24.2. Push V6

```powershell
& $Git push -u origin scientific-recovery-v6-oof-diagnostics
```

## 24.3. Tag V6

```powershell
& $Git tag -a scientific-recovery-v6 `
  28c1efb50622255719f239622ba07858ce704535 `
  -m "Scientific Recovery V6 closed: radius-2 diagnostic failed promotion; A5 causal best E-JEPA TTC"
```

```powershell
& $Git push origin scientific-recovery-v6
```

## 24.4. Comprobar relación con main

```powershell
& $Git fetch origin --prune

& $Git rev-list --left-right --count `
  origin/main...origin/scientific-recovery-v6-oof-diagnostics
```

### Si devuelve:

```text
0    N
```

`main` es ancestro directo y puede avanzarse por fast-forward:

```powershell
& $Git push origin `
  scientific-recovery-v6-oof-diagnostics:main
```

Después:

```powershell
& $Git fetch origin
& $Git branch -f main origin/main
```

### Si el primer número NO es cero

No hacer merge/rebase automáticamente.

Inspeccionar:

```powershell
& $Git log --oneline `
  origin/scientific-recovery-v6-oof-diagnostics..origin/main

& $Git log --oneline `
  origin/main..origin/scientific-recovery-v6-oof-diagnostics
```

Resolver explícitamente.

## 24.5. Crear V7

Una vez congelado V6:

```powershell
& $Git switch -c scientific-recovery-v7-capacity-attribution scientific-recovery-v6
```

Si ya se avanzó `main` y apunta al mismo commit:

```powershell
& $Git switch main
& $Git switch -c scientific-recovery-v7-capacity-attribution
```

---

# 25. Primer commit V7: solo preregistro, no código experimental

Crear:

```text
docs/SCIENTIFIC_RECOVERY_V7_CAPACITY_ATTRIBUTION_PLAN.md
configs/protocol/scientific_recovery_v7_capacity_attribution.json
```

Actualizar explícitamente:

```text
README.md
STATUS.md
PLAN.md
CODEX_HANDOFF.md
```

Commit:

```powershell
& $Git add -- `
  docs/SCIENTIFIC_RECOVERY_V7_CAPACITY_ATTRIBUTION_PLAN.md `
  configs/protocol/scientific_recovery_v7_capacity_attribution.json `
  README.md `
  STATUS.md `
  PLAN.md `
  CODEX_HANDOFF.md

& $Git diff --cached --check
& $Git diff --cached

& $Git commit -m "research: preregister V7 capacity attribution study"
```

Solo después implementar.

---

# 26. Segundo commit V7: capacity config factory

Propuesta:

```text
scripts/plan_scientific_recovery_v7_capacity.py
tests/unit/test_scientific_recovery_v7_capacity.py
```

Si hace falta modificar model config plumbing:

```text
src/e_jepa_ttc/models/causal_scale_ttc.py
```

pero evitar cambios científicos colaterales.

Commit:

```text
feat: add deterministic V7 capacity config planning
```

---

# 27. Tercer commit V7: runner fail-closed

Crear:

```text
scripts/run_scientific_recovery_v7_capacity.py
```

Debe:

1. verificar clean tracked tree;
2. verificar commit/config hashes;
3. comprobar sample/fold identities;
4. validar private/test closed;
5. ejecutar folds secuencialmente;
6. no reutilizar outputs;
7. escribir progress JSON, no leer checkpoints para monitorización;
8. parar tras cada stage para aplicar gate.

Commit:

```text
feat: add fail-closed V7 capacity runner
```

---

# 28. Cuarto commit V7: summarizer

Crear:

```text
scripts/summarize_scientific_recovery_v7_capacity.py
```

Output:

```text
artifacts/scientific_recovery_v7/results/capacity_screen.json
```

Debe incluir:

```text
parameter count
train MiD
dev MiD
train-dev gap
failure
Pearson
per-sequence MiD
per-bucket contribution
paired delta
CI95
bootstrap probability
runtime
peak VRAM
```

Además generar:

```text
capacity_curve.csv
```

para plots posteriores.

---

# 29. Tests mínimos V7

## Test 1 — determinismo

Mismo base config + target band → mismo config/hash.

## Test 2 — exact sample contract

Los tres modelos del screen usan exactamente los mismos folds.

## Test 3 — no sealed access

Ningún config/path contiene private/test.

## Test 4 — capacity-only

Entre tamaños de la misma familia no cambian:

```text
input schema
loss weights
ROI protocol
temporal representation
prediction contract
```

salvo variables explícitas de capacity.

## Test 5 — parameter monotonicity

```text
small < medium < large
```

y cada uno dentro de target band.

## Test 6 — topology preservation

Todos los módulos requeridos siguen presentes.

## Test 7 — stale artifact fail-closed

Output existente con hash distinto → abort.

## Test 8 — paired bootstrap provenance

Los comparisons contienen hashes actuales.

## Test 9 — checkpoint selection

Selector idéntico dentro de cada familia.

## Test 10 — failed/incomplete fold

No agrega un resultado incompleto.

---

# 30. Comandos de validación antes de GPU

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  .\scripts\plan_scientific_recovery_v7_capacity.py `
  .\scripts\run_scientific_recovery_v7_capacity.py `
  .\scripts\summarize_scientific_recovery_v7_capacity.py

.\.venv\Scripts\python.exe -m pytest -q `
  .\tests\unit\test_scientific_recovery_v7_capacity.py `
  .\tests\unit\test_scientific_recovery_v5_grouped_dev.py `
  .\tests\unit\test_scientific_recovery_v3.py

.\.venv\Scripts\ruff.exe check `
  .\scripts\plan_scientific_recovery_v7_capacity.py `
  .\scripts\run_scientific_recovery_v7_capacity.py `
  .\scripts\summarize_scientific_recovery_v7_capacity.py `
  .\tests\unit\test_scientific_recovery_v7_capacity.py

git diff --check
```

Adaptar la ruta Ruff al entorno real si el executable se invoca mediante `python -m ruff`.

---

# 31. Gate después de V7.1

Generar una decisión explícita:

```json
{
  "garl_capacity_effect": "...",
  "a5_capacity_effect": "...",
  "geometry_preserving_capacity_effect": "...",
  "decision": "STOP_CAPACITY | RUN_MEDIUM_CURVE | RUN_LARGE_CURVE | OPEN_V8_ARCHITECTURE"
}
```

No permitir una decisión manual no registrada.

---

# 32. Cuándo ejecutar 3 seeds adicionales

No hacer 3 seeds en todos los tamaños.

Primero:

```text
3 folds × seed7
```

Si un tamaño produce una conclusión material:

- efecto fuerte;
- posible promoción;
- o resultado sorprendente que cambia la interpretación;

replicar ese punto con:

```text
seed13
seed23
```

Esto maximiza información por GPU.

---

# 33. Qué haría si A5-medium ya bate Garl local

Ejemplo:

```text
A5-medium = 140
```

No abrir private.

Primero:

1. paired OOF A5-medium vs Garl;
2. IC95% completo;
3. failures/full coverage;
4. geometry audit;
5. seed13/23;
6. freeze.

Si sigue degradando geometría:

> sigue siendo un breakthrough TTC, no automáticamente el candidato geometry-preserving.

Esto podría obligar a decidir explícitamente si la preservation constraint sigue siendo
un requisito del objetivo final o una propiedad deseable secundaria.

No cambiar esa definición post hoc.

---

# 34. Qué haría si GP-medium bate A5

Este sería el resultado ideal:

```text
geometry exacta
+
TTC ~A5 o mejor
```

Entonces:

- replicar seeds;
- evaluar r1/r2 routing solo si todavía existe gap;
- no continuar capacity curve automáticamente;
- congelar el primer modelo suficiente.

Principio:

> parar cuando la hipótesis queda resuelta; no seguir buscando un número menor solo
> porque aún queda compute.

---

# 35. Qué haría si nada mejora al escalar

Si:

```text
G-small ≈ G-full
A5-medium ≈ A5-base
GP-medium ≈ GP-base
```

entonces capacity queda prácticamente descartada como explicación principal.

Abrir V8 inmediatamente sobre:

1. soft geometry preservation;
2. A5-like trainable TTC encoder;
3. training-only foreground/boundary;
4. conditional r1/r2;
5. full-coverage point prediction;
6. short-TTC calibration.

No entrenar 8M/25M.

---

# 36. Qué NO haría a partir de V6

1. No ejecutar V6.2 retrospectivamente.
2. No cambiar radius a 3 “a ver qué pasa”.
3. No saltar A8 directamente a 25M.
4. No declarar que Garl gana “porque es 39× mayor”.
5. No declarar que overfitting invalida Garl.
6. No abrir public validation para elegir V7.
7. No abrir private/test.
8. No mezclar capacity + temporal bins + foreground en el mismo experimento.
9. No seleccionar el mejor seed post hoc.
10. No usar Pearson como proxy suficiente de MiD.
11. No optimizar únicamente aggregate MiD ignorando crucial/small y failure.
12. No mantener bitwise geometry como dogma si V7/V8 preregistra una definición de
    preservación científica más apropiada.
13. No usar `git add .`.
14. No borrar los 18 untracked preexistentes.
15. No reutilizar directorios de runs antiguos.

---

# 37. Boundary checks / sanity checks

## Check A — “A5 casi bate Garl, por tanto ya está”

Falso.

```text
A5 - Garl = +11.119
IC95% [4.271,17.527]
```

El intervalo está completamente por encima de cero.

Garl gana.

## Check B — “V6.1 mejora A8, por tanto multi-scale confirmado”

Demasiado fuerte.

```text
V6.1 - A8 = -3.570
IC95% [-8.190,+0.999]
```

Hay señal, pero el CI cruza cero.

Lo confirmado es un patrón de régimen, no una mejora global robusta.

## Check C — “Garl overfittea, así que no generaliza”

Falso.

Los selected checkpoints ganan los tres dev folds.

## Check D — “más parámetros explican Garl”

No probado.

Hay demasiadas variables confundidas.

## Check E — “A5 destruye geometría porque tiene menos parámetros”

No soportado.

A5 tiene menor capacidad que A8 y aun así TTC mucho mejor.

El cambio de training freedom/objective es una explicación más directa.

## Check F — “frozen geometry es necesariamente el objetivo correcto”

No está demostrado.

Es una restricción científica elegida.

V6 muestra que puede ser demasiado fuerte.

Una futura fase puede preregistrar soft preservation sin contradecir la historia.

## Check G — “0% failures Garl significa que nunca se equivoca”

No.

Significa que su contrato devuelve una predicción para todos los samples.

No equivale a cero error.

## Check H — “si A5-medium mejora 10 MiD ya es SOTA”

No.

V7 sigue siendo development evidence.

---

# 38. Error de interpretación más plausible

El error más probable ahora sería leer:

```text
Garl 24.67M
E-JEPA 0.4–0.6M
```

y concluir:

> “hagamos todo 25M”.

Eso no es una ablation.

La forma científica de resolverlo es una curva secuencial controlada.

El segundo error más probable sería observar que A5 está solo 11.1 MiD detrás y abandonar
la geometría sin declarar que cambió el objetivo científico.

No.

Primero hay que separar:

```text
best TTC arm
vs
best geometry-preserving candidate
```

y decidir explícitamente cuál claim se persigue.

---

# 39. Decisión final recomendada

## Fase inmediata

```text
CERRAR/PUBLICAR V6
        ↓
PREREGISTRAR V7 CAPACITY ATTRIBUTION
        ↓
G-small
A5-medium
GP-medium
        ↓
paired OOF + train/dev gap
```

## Después

### Si capacity explica una parte grande

```text
V7.2 size curve
→ 6–8M
→ solo después considerar 20–25M
```

### Si capacity no explica la brecha

```text
V8
soft geometry preservation
+ A5-like TTC adaptation
+ foreground/boundary supervision
+ conditional r1/r2
+ short-TTC calibration
+ full-coverage prediction
```

---

# 40. Criterio final para gastar el sealed test

No abrirlo simplemente cuando un point estimate sea 143.

Antes debe existir:

```text
frozen git commit
frozen model config
frozen training config
frozen checkpoint hashes
frozen metric implementation
frozen preprocessing manifest
frozen Garl comparator
frozen failure/coverage contract
replicated seeds
paired OOF evidence
prefix causality PASS
no leakage PASS
claim scope explicit
```

Y para afirmar una victoria comparable:

```text
paired Δ(E-JEPA − Garl) < 0
95% CI completely below 0
```

más una cobertura/failure comparación honesta.

---

# 41. Resultado deseado de V7

V7 no tiene que producir un nuevo récord para ser un éxito.

Será exitoso si permite escribir una de estas conclusiones con evidencia:

### Resultado A

> Garl conserva casi toda su ventaja al reducirse a capacidad comparable; la explicación
> principal no es parameter count.

### Resultado B

> La ventaja de Garl degrada de forma monotónica al reducir capacidad; el tamaño explica
> una parte importante del gap.

### Resultado C

> A5 escala fuertemente con capacidad y alcanza la región Garl; E-JEPA estaba
> under-capacity.

### Resultado D

> A5 satura pronto, pero geometry-preserving escala; la rama dual estaba under-capacity.

### Resultado E

> Ninguna familia E-JEPA mejora materialmente al escalar; la siguiente inversión debe
> ser representación/objetivo, no parámetros.

Cualquiera de estas respuestas reduce significativamente la incertidumbre científica.

---

# 42. Checklist operacional

Antes de V7:

```text
[ ] V6 HEAD confirmado
[ ] tracked worktree clean
[ ] V6 pushed
[ ] tag scientific-recovery-v6 pushed
[ ] relation with main comprobada
[ ] V7 branch creada desde V6
[ ] private/test closed
[ ] public validation no usada
[ ] V7 protocol document committed
[ ] capacity config manifest frozen
[ ] parameter counts verified
[ ] folds exactos verified
[ ] paired infrastructure reused with provenance
[ ] tests PASS
[ ] Ruff PASS
[ ] git diff --check PASS
```

Antes de cada training:

```text
[ ] config hash
[ ] code commit hash
[ ] input manifest hash
[ ] fold id
[ ] seed
[ ] output dir new/nonexistent
[ ] no concurrent CUDA trainer
```

Después:

```text
[ ] summary complete
[ ] checkpoint hash
[ ] predictions hash
[ ] sample tokens exact
[ ] target equality
[ ] paired bootstrap
[ ] train/dev gap
[ ] failure
[ ] per-bucket
[ ] per-sequence
[ ] decision artifact
```

---

# 43. Estado final de conocimiento a 2026-08-13

```text
Garl
  mejor TTC local matched
  144.353
  0% failure contract
  24.67M params
  overfit tardío observable
  pero mejor OOF en los 3 folds

A5 causal
  mejor E-JEPA TTC
  155.472
  solo +11.119 vs Garl
  gana large/negative
  pierde principalmente 0–6 s
  degrada geometría

V6.1
  mejor geometry-exact observado
  194.122
  radius=2 ayuda high-flow
  no mejora global con evidencia suficiente

A8
  dual-stream funciona
  197.691
  pero underpowered/restricted

Capacidad
  plausible
  NO aislada

Siguiente acción
  V7 Capacity Attribution Study
```

---

# 44. Conclusión

La recuperación ya no está en una fase de “buscar cualquier mejora”.

Ha producido una triangulación científica fuerte:

1. **A5 demuestra capacidad representacional para TTC.**
2. **A8 demuestra que separar geometry y transport es mejor que un adapter pequeño.**
3. **V6 demuestra que aumentar el radio fijo no resuelve el problema y que el régimen de
   movimiento importa.**
4. **Garl demuestra que la brecha restante es real OOF, no un artefacto obvio de
   memorización.**
5. **El enorme ratio de parámetros sigue siendo la incertidumbre más barata y limpia
   que falta aislar antes de diseñar otra arquitectura compleja.**

Por ello, la mejor continuación es:

> **cerrar V6 de forma inmutable y ejecutar una V7 de atribución de capacidad pequeña,
> secuencial y preregistrada.**

Si la capacidad explica la brecha, escalar de forma controlada.

Si no la explica, la V8 debe atacar exactamente el conflicto que ya observamos:

> **adaptabilidad TTC de A5 + preservación geométrica blanda + transport multi-scale
> condicionado + foreground/boundary supervision + calibración short-TTC + full
> coverage.**

Ese camino reduce incertidumbre en cada etapa y mantiene intacta la validez del eventual
sealed test.
