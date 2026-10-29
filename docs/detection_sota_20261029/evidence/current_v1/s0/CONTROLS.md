# Controles S0 del detector bajo evaluación v2

Los controles usan GT v2 idéntico, un prior ajustado solo con train y una permutación prefijada de listas completas. El resultado es local y no oficial.

- AP observado: `0.019821723110847754`.
- AP prior constante: `0.003839390946110635`.
- AP permutación: `0.003654753769763242`.
- Clusters de bootstrap: `2`.
- Réplicas: `100`.

## Diferencias emparejadas (observado − control)

- constant_prior: mediana `0.020248878259559355`, IC95 `[0.01598233216473712, 0.02087538046330463]`.
- frame_derangement: mediana `0.019701849655445147`, IC95 `[0.016166969341084515, 0.021077573610685837]`.

Con solo dos secuencias, los intervalos por cluster son débiles y no acreditan generalización. Las réplicas sin soporte de una clase permanecen `null`.
