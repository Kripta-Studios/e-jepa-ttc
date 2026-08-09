# Causal Scale v8: selección multigrupo

Actualizado: 2026-08-10.

## Motivación

V7 pasó validation 502 con Pearson `.9612550`, pero cayó a `.9201432` en el único
test 603. Como traslación, foreground, signo, slope, TTC y calibración sí pasaron, el
fallo observado es de selección/generalización de correlación, no de la identidad
física ni del soporte espacial.

V8 no modifica gates ni usa 303/603 para desarrollo. Conserva la arquitectura V7 y
el mismo presupuesto total de muestras, pero distribuye train y validation entre
grupos independientes. Así se puede seleccionar un checkpoint estable sin atribuir
una realización aleatoria favorable a la arquitectura.

## Contrato congelado antes del entrenamiento

| Split | Grupos | Muestras totales | Uso |
|---|---|---:|---|
| train | 701, 702, 703 | 1536 | optimización |
| validation | 801, 802, 803 | 384 | selección y calibración |
| test | 901, 902, 903 | 510 | sellado hasta publicación |

La puntuación de selección por época es:

```text
0.5 * mean(validation_group_selection_score)
+ 0.5 * max(validation_group_selection_score)
```

La función por grupo mantiene las mismas penalizaciones preregistradas de V7. No se
pooléan predicciones entre grupos para ocultar un grupo débil: el summary conserva
métricas y scores por grupo. En `full`, cada grupo test se evalúa una vez y el pass
global exige que los tres pasen todos los gates congelados.

## Controles de integridad

- las seeds deben ser únicas dentro y entre splits;
- `diagnostic` solo instancia train/validation;
- `full` exige estado de código limpio;
- 303 y 603 permanecen consumidas e inmutables;
- real data, eAP, EvTTC y etiquetas TTC siguen cerrados;
- un pass sintético solo autoriza diseñar un screen eAP train-only.

Configuración: `configs/experiment/e_jepa_garl_event_causal_scale_synthetic_v8.yaml`.
Los resultados siguientes son solo diagnósticos validation; no autorizan SOTA.

## Diagnóstico base

El run publicado `46f9d61` seleccionó epoch 16 y mantuvo test cerrado. Macro Pearson
fue `.8063141`; por grupo, 801/802/803 obtuvieron `.6426862/.8916795/.8845764`.
IoU macro `.8901637`, slope `.9311736`, signo `.9886764`, TTC `.2820616` y
translation `.0046902` sí pasaron sus umbrales.

El análisis por muestra localizó 1–3 endpoints catastróficos por grupo. En 801, al
retirar solo el 5% de mayor error para diagnóstico (no como métrica seleccionable),
Pearson sube de `.6426` a `.9661`. El caso `synthetic-801-115` convierte una altura
real `[20,21,22]` en extensiones blandas `[.308,.735,.331]`: el decoder aislado no
puede estabilizar un endpoint de bajo soporte aunque los vecinos sean correctos.

Se habilitan dos brazos validation-only con consenso temporal simétrico de logits,
pesos `.10` y `.15`. El operador usa padding de borde y kernel
`[w, 1-2w, w]`; por ello es equivariante a reversión temporal, no usa eventos
posteriores al contexto actual y no añade parámetros. Sus resultados aún están
pendientes y los tests 901–903 continúan sellados.

Los brazos entrenados obtienen Pearson macro `.9338758` (`w=.10`) y `.9380405`
(`w=.15`). El segundo equilibra los grupos en `.94338/.93594/.93479`, pero TTC
`.3019122` falla por `.0019122`; ninguno se promueve. Como última ablation sintética
se congela CVaR sobre el 10% de mayor error de ratio, peso `2.0`, junto a `w=.15`.

Por autorización explícita del usuario, el fallo sintético ya no impide un screen
exploratorio eAP train/validation. No autoriza eAP test, CodaBench, EvTTC test ni un
claim frente a Garl-TTC. El resultado sintético permanece fallido y visible.
