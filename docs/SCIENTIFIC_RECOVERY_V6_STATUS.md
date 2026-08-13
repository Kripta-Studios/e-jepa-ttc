# Scientific Recovery V6: estado canónico

Fecha de corte: 2026-08-13. Rama:
`scientific-recovery-v6-oof-diagnostics`. Código de agregación y diagnóstico:
`2e792453b84fed37824c606d12b9721e366c23f9`.

V6 queda cerrado sin promoción. Se analizaron los 8192 resultados OOF V5 antes de
entrenar, se seleccionó `motion_scale` y se congeló V6.1 como un único cambio:
radio de transporte 1 a 2. Después se entrenaron sus tres folds y, como comparador
diagnóstico, tres folds A5 causales desde cero. No se abrió public validation,
private ni test.

## Resultado

| Modelo | Parámetros | F0 | F1 | F2 | Macro 9 secuencias | Failure | Pearson | Alcance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A6 | 498130 | 180,682 | 218,518 | 235,326 | 211,509 | 7,849% | 0,201 | baseline frozen |
| A8.0 radio 1 | 627827 | 180,795 | 201,548 | 210,731 | 197,691 | 7,019% | 0,225 | geometry exacta |
| V6.1 radio 2 | 627827 | 181,351 | 198,032 | 202,982 | 194,122 | 6,689% | 0,216 | geometry exacta |
| A5 causal | 424274 | 120,424 | 176,187 | 169,805 | 155,472 | 4,761% | 0,360 | diagnóstico sin geometry constraint |
| Garl | 24674178 | 130,350 | 163,219 | 139,490 | 144,353 | 0% | 0,042 | comparador matched |

V6.1 mejora A8.0 por 3,570 MiD, pero el intervalo pareado incluye cero. A5 causal
es claramente mejor que las ramas geometry-frozen y queda a 11,119 MiD de Garl.

| Comparación, primero menos segundo | Delta MiD | IC95% | P(primero menor) | Delta failure | IC95% failure |
|---|---:|---:|---:|---:|---:|
| V6.1 - A8.0 | -3,570 | [-8,190; 0,999] | 0,9354 | -0,330 pp | [-1,084; 0,398] |
| V6.1 - Garl | +49,769 | [40,708; 58,572] | 0,0000 | +6,689 pp | [6,126; 7,277] |
| A5 causal - V6.1 | -38,650 | [-46,052; -31,250] | 1,0000 | -1,929 pp | [-2,640; -1,242] |
| A5 causal - A8.0 | -42,219 | [-49,401; -35,418] | 1,0000 | -2,258 pp | [-2,964; -1,562] |
| A5 causal - Garl | +11,119 | [4,271; 17,527] | 0,0008 | +4,761 pp | [4,232; 5,278] |

El gate V6.1 exige mejorar A8.0, MiD no mayor que 175, geometría exacta,
causalidad de prefijo y coverage. Pasa mejora media, geometría, causalidad y
coverage; falla `MiD <= 175` y el objetivo fuerte `<=160`. La decisión es `FAIL`.

## Qué explica la brecha con Garl

La descomposición de la métrica oficial localiza la mayor parte del déficit en TTC
positivo corto:

| Modelo menos Garl | Crucial 0-3 s | Small 3-6 s | Large 6-10 s | Negative | Total |
|---|---:|---:|---:|---:|---:|
| V6.1 | +40,326 | +12,425 | +0,031 | -3,012 | +49,769 |
| A5 causal | +11,003 | +5,263 | -0,917 | -4,231 | +11,119 |

A5 ya supera a Garl en la contribución `large` y `negative`, pero pierde donde el
paper asigna el 80% del peso: `crucial` y `small`. Garl también mantiene 100% de
coverage y cero failures; A5 falla 4,761% y V6.1 6,689%.

A5 supera a Garl en cuatro de nueve secuencias y pierde especialmente en
`OBneIVg4Cw` (+54,61), `6h5yRW2LGc` (+40,78) y `WbCh1DRerJ` (+23,94). Su Pearson
global mayor no contradice el resultado: Pearson no mide calibración en eta ni
aplica los pesos por bucket de MiD.

El radio 2 se comporta como predijo D0, pero su efecto es selectivo. Frente a A8,
el cambio diagnóstico de MiD crudo es -10,68 en el cuartil de mayor cambio de
escala, frente a -2,13 en el menor. Por magnitud de transporte, radio 2 empeora
Q1/Q2 (+4,24/+3,24) y mejora Q3/Q4 (-12,43/-15,42). Esto apoya transporte
multi-escala condicionado por régimen, no un radio mayor aplicado siempre. La
mejora OOF débil y su IC95% impiden tratarlo como solución confirmada.

## ¿Sigue siendo A5 el mejor?

Sí, con una precisión necesaria: A5 causal es el mejor E-JEPA por TTC en el
grouped-dev limpio actual, con MiD 155,472. También mejora la cifra histórica A5
de 163,211, pero ambas proceden de poblaciones y protocolos distintos y no deben
compararse como una réplica directa.

A5 no es el mejor candidato científico completo. Al descongelar el encoder reduce
la fidelidad de la dinámica geométrica. Frente a A8/V6.1, su ratio de desviación
predicha/objetivo cae aproximadamente de 0,66 a 0,16 contra bbox y de 0,77 a 0,18
contra física; las slopes caen de 0,16 a 0,027 y de 0,27 a 0,041. La mejora TTC
procede de especialización supervisada incompatible con el requisito actual de
preservar la geometría del parent.

Por tanto:

- mejor E-JEPA TTC limpio: A5 causal;
- mejor rama con geometría exacta: V6.1, aunque su ventaja sobre A8 es incierta;
- mejor TTC global: Garl;
- candidato promocionable: ninguno.

## Siguiente hipótesis

No se continúa V6.1 retrospectivamente. Una nueva V7 debería conservar la rama
geométrica frozen y añadir una rama TTC entrenable con libertad comparable a A5,
en vez de limitarla al corrector local actual. La fusión debería depender de
magnitud/confianza y poder elegir radio 1 o 2, porque radio 2 perjudica los casos de
flujo bajo. El objetivo primario debe priorizar `crucial` y `small`, con failure
como gate explícito. Esta propuesta es una hipótesis derivada del outer-dev V6 y
necesita un nuevo preregistro; el mismo OOF no puede presentarse después como
confirmación independiente.

## Integridad y artefactos

- D0: `artifacts/scientific_recovery_v6/diagnostics/a8_oof_failure_modes.json`,
  artifact `32ec156f6dab87645e0272af8bc006e5d9a741bdbc1bca535937e5fca5130bcd`.
- Agregado: `artifacts/scientific_recovery_v6/results/aggregate.json`, artifact
  `ed6da1c77a211870406810d4d6b446d450845b021f211f45d0485b257945e77a`.
- Diagnóstico de brecha:
  `artifacts/scientific_recovery_v6/diagnostics/oof_garl_gap.json`, artifact
  `f0ebc082d06571c645c42542d53a39324c22723f346b17dcac2e49d9ae646b9c`.
- Las tres auditorías de geometría V6.1 prueban igualdad exacta con el parent.
- V6.1 y A5 causal pasan el probe de causalidad de prefijo a tolerancia `1e-6`.
- Bootstrap: 5000 remuestreos pareados por `sequence_id+track_id`, 422 clusters.
- Garl está matched en samples, targets, budget, métrica y privilegio oracle-ROI;
  su preprocessing no es idéntico.
- `public_validation_used_for_selection=false`, `private_test_opened=false` y no
  existe candidato sealed.
