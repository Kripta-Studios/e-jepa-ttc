# Protocolo experimental v6

Actualizado: 2026-07-30.

## Datos

- desarrollo: EvTTC-32;
- selección: split histórico para screen y grouped CV para confirmación;
- externo: Benchmark-10 sellado;
- pretraining futuro: eAP train-40 sin TTC.

## Jerarquía de evidencia

```text
Smoke
  integración; nunca promoción

Screen
  304/80 ventanas, seed 7, hasta 8 épocas

Confirmación matched
  1.208/314 ventanas, hasta 40 épocas, early stopping compartido

Grouped CV
  cinco folds completos, seed 7

Multisemilla
  folds completos, seeds 7/13/21, máximo dos finalistas

Benchmark-10
  una inferencia después del freeze
```

## Comparación Core

`A0_MATCHED_GLOBAL`, `A1_MATCHED_DENSE_BLOCK`,
`A2_MATCHED_DENSE_ATTNRES` y `K1_OBJECT_KDA` comparten samples, inicialización,
trainer y criterio de checkpoint.

La confirmación histórica promueve únicamente A1. A0 y A1 se vuelven a
comparar en grouped CV desde una inicialización aleatoria común explícita para
no reutilizar un checkpoint SSL que haya visto eventos de las secuencias
validadas. Esa prueba responde a la arquitectura; la confirmación histórica
separada responde a la inicialización BASE.

## Comparación Garl

G0–G7 usan el mismo cache Garl y batch efectivo 24 en Screen. G6/G7 inicializan
sus ramas desde los mejores checkpoints G3/G4 del mismo fold y seed.

El screen local no reproduce todavía las 50 épocas del repositorio público y
no se usa como claim de paridad.

## Métrica de selección

```text
score =
mean_relative_error
+ 0,25 normalized_RMSE
+ 0,25 high_TTC_error
+ 0,25 low_TTC_safety_error
```

Desempate: peor secuencia, RMSE, variación entre seeds y latencia.

## Reglas

- ninguna ventana cruza secuencia;
- ninguna decisión se toma con Benchmark-10;
- resultados smoke se etiquetan `integration_only`;
- el oracle con distancia GT no compite como modelo desplegable;
- una métrica calculada solo sobre éxitos debe informar también cobertura y
  no puede compararse con candidatos de cobertura completa;
- todo resultado debe señalar commit, config hash y manifest hash.
