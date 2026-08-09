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
Todavía no existen resultados V8 ni autorización para afirmar SOTA.
