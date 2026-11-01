# 04. Resultados, regresiones y alcance estadístico

## Reglas de lectura

La RAE de las tablas es el promedio del error absoluto relativo, expresado como
porcentaje. Para los targets positivos del replay final:

```text
RAE = 100 × mean(abs(prediction - target) / target)
MAE = mean(abs(prediction - target))
```

La media micro pondera cada consulta; la macro por grabación da el mismo peso a
cada grabación. Un error relativo puede mejorar mientras empeora MAE si aumenta
el error en TTC largo. No se usan ventanas de una misma secuencia como unidades
independientes del bootstrap. Tampoco se cuentan tres semillas como tres nuevas
adquisiciones. Cada tabla mantiene su población y papel.

## F1: ajuste completo y coste en el dominio fuente

Fuente: [RESULTS.json del refit](../evttc_transfer_v2/evidence/phase_adapt_final/RESULTS.json).
Las siguientes filas son **ajuste sobre los propios datos usados para entrenar**.

| Dominio | Filas | Variante | RAE % | MAE s |
|---|---:|---|---:|---:|
| EvTTC nativo | 3.571 | F0 | 40,7063 | 1,094082 |
| EvTTC nativo | 3.571 | F1 seed 7 | 36,8129 | 1,051901 |
| EvTTC nativo | 3.571 | F1 seed 13 | 37,2912 | 1,055066 |
| EvTTC nativo | 3.571 | F1 seed 21 | 37,1122 | 1,059053 |
| eAP/TRAIN40 | 88.744 | F0 | 27,3657 | 1,416030 |
| eAP/TRAIN40 | 88.744 | F1 seed 7 | 26,9172 | 1,433496 |
| eAP/TRAIN40 | 88.744 | F1 seed 13 | 27,1869 | 1,438726 |
| eAP/TRAIN40 | 88.744 | F1 seed 21 | 27,1382 | 1,433922 |

El pequeño avance relativo en eAP no oculta la regresión de MAE. Las dos épocas
no se presentan como convergencia demostrada. Los tres checkpoints se conservan.

## F1: validación agrupada en 24 grabaciones

Fuente: [campaña agrupada y robustez](../evttc_transfer_v2/CAMPAIGN_RESULTS.md),
con [resumen regenerable](../evttc_transfer_v2/evidence/phase_campaign/SUMMARY.json).
Son 2.531 consultas, siete familias y tres semillas, con las ocho grabaciones
solapadas excluidas. La exposición histórica permanece.

| Modelo | Macro RAE % | Micro RAE % | MAE s | Cobertura |
|---|---:|---:|---:|---:|
| F0 | 30,600 | 27,664 | 0,8494 | 100 % |
| F1 seed 7 | 25,468 | 24,703 | 0,8585 | 100 % |
| F1 seed 13 | 26,188 | 25,501 | 0,8925 | 100 % |
| F1 seed 21 | 25,938 | 25,183 | 0,8653 | 100 % |

Los intervalos del cambio macro por familias incluyen cero: seed 7
`[-12,289; 0,741]` pp, seed 13 `[-11,891; 1,810]` pp y seed 21
`[-12,135; 1,043]` pp. Los valores centrales favorecen RAE, pero ninguna semilla
mejora el MAE global frente a F0. La cobertura de intervalos nominales del 80 %
queda alrededor del 66–68 %; reescalar con residuos del propio train solo la
eleva aproximadamente al 69–70 %. No se incorporó una garantía de cobertura.

El desglose de seed 7 explica parte del conflicto. En 92 consultas por debajo de
1 s, RAE baja de 39,516 a 28,435 %. En las 1.724 consultas entre 1 y 4 s, de
31,520 a 26,497 %. En las 715 por encima de 4 s empeora de 16,841 a 19,896 % y
MAE pasa de 1,2542 a 1,5496 s. El agregado no debe borrar esta cola.

## Slider-500 cuando aún era externo al ajuste actual

Los pesos y las 77 consultas se congelaron antes de abrir sus valores TTC.
F0 produjo 76 salidas evaluables: RAE 83,887 %, MAE 1,1782 s. F1 seed 7 obtuvo
87,578 % y 1,1657 s sobre 76 salidas; seed 13, 91,022 % y 1,2047 s; seed 21,
87,330 % y 1,1600 s. La bajada de MAE de dos semillas no confirma una mejora
relativa. Se mantiene la consulta sin salida.

Los observadores de caja fueron referencias útiles: ancho obtuvo 24,870 % sobre
74 salidas, área 27,949 % sobre 76 y altura 47,815 % sobre 75. Esas coberturas
diferentes obligan a consultar las comparaciones pareadas; no basta ordenar los
porcentajes. Los controles CMax/STRTTC locales abarcaron doce consultas fijadas
y no se presentan como reproducciones oficiales de sus implementaciones.

## Gate de calidad robusto

Fuente: [informe robusto](../evttc_transfer_v2/evidence/phase_robust/report/REPORT.md).

| Población y rol | Referencia RAE % / MAE s | Gate RAE % / MAE s |
|---|---:|---:|
| Carretera OOF, 1.747 consultas | F0: 27,199 / 0,6963 | Calidad: 12,834 / 0,4224 |
| Carretera OOF, control de igual capacidad | Control: 22,081 / 0,7544 | Calidad: 12,834 / 0,4224 |
| CCRm, reserva expuesta, 580 | F0: 16,749 / 1,0830 | 15,662 / 1,0197 |
| Slider-750 local, 51 | F0: 282,476 / 4,5395 | 168,038 / 3,3157 |
| Slider-1000 local, 18 | F0: 156,876 / 0,8212 | 38,419 / 0,2757 |
| Refit completo, 2.677 | F0: 34,986 / 0,9289 | 13,129 / 0,4779 |

La última fila es fit. La mejora en CCRm tiene un intervalo bootstrap por
grabaciones de `[-2,061; 0,290]` pp, que incluye cero. El observador geométrico
aislado obtiene 18,077 % en Slider-750 con 49/51 salidas y 7,694 % en Slider-1000
con 16/18: superar a la geometría no quedó demostrado para la población completa.

En la cola OOF de 439 consultas por encima de 4 s, MAE pasa de 0,8947 a 0,8825 s,
pero RMSE empeora de 1,2275 a 1,4351 s. El gate reduce mucho el error medio de
carretera sin dominar todos sus extremos.

La calibración evaluated usa CPNAO separado, radio simétrico de 1,1318 s y cobertura
de CCRm de 79,83 %. Es una cobertura descriptiva sobre datos expuestos/correlacionados.
El refit all-data entrenó con esas etiquetas y no conserva un intervalo validado.

## Router de arranque: mejora pequeña y regresiones conservadas

Fuente: [resultados del router](../evttc_transfer_v2/evidence/phase_startup/RESULTS.json)
y [su informe](../evttc_transfer_v2/evidence/phase_startup/REPORT.md).

| Prueba | Gate robusto previo | Router inverso | Lectura |
|---|---:|---:|---|
| RAE OOF carretera, 1.747 | 12,8336 % | 12,6513 % | Mejora pequeña |
| MAE OOF carretera | 0,422359 s | 0,404882 s | Mejora en ese agregado |
| RAE CCRm, 580 | 15,6624 % | 15,1910 % | Mejora relativa |
| MAE CCRm | 1,019652 s | 1,035545 s | Regresión |
| RAE ajuste carretera all-data | 11,289 % | 10,9488 % | Fit, no generalización |
| RAE ajuste Slider all-data | 45,036 % | 19,7372 % | Fit y exposición ampliada |

El bootstrap emparejado por las cinco familias de carretera, con 5.000 réplicas,
da un cambio medio por consulta (micro) de −0,1824 pp y CI95 `[-1,5176; 0,8001]`. El ensayo QA
en segundos se rechazó: 15,2286 % y 0,541610 s en carretera, peor que el gate
anterior. El estrés crudo limpio del router inverso fue 13,9528 % frente a
12,3776 % del anterior; con dropout del 70 %, 14,9805 % frente a 12,8970 %.
Por eso la entrega no sustituyó automáticamente el default robusto.

Un test de pérdida de cajas alcanzó 46,4918 % y reveló el uso de historia demasiado
antigua. La corrección de frescura vuelve a F0, 28,2881 %, en esa condición.
El test ya influyó en el diseño; no se vuelve a presentar como confirmación ciega.

## Nuevo envío: comparación local de las 434 consultas

Fuente: [LOCAL_EVALUATION.json](../evttc_submission_20261031/evidence/LOCAL_EVALUATION.json).
Las salidas se generaron desde sensores y se sellaron antes de esta evaluación.
Ambas columnas aplican la interpolación de negativos descrita en el correo.

| Población | Targets / consultas | RAE original % | RAE adaptado % | MAE original s | MAE adaptado s |
|---|---:|---:|---:|---:|---:|
| CCRs1-low | 50/50 | 36.3885 | 35.9026 | 1.516138 | 1.481718 |
| CCRs1-medium | 32/43 | 70.5582 | 68.1662 | 0.993157 | 0.967736 |
| CCRs1-high | 30/30 | 13.1178 | 9.4401 | 0.175780 | 0.137442 |
| CCRs2-low | 46/46 | 15.2812 | 9.6247 | 0.439735 | 0.294427 |
| CCRs2-medium | 55/55 | 9.3429 | 7.1012 | 0.196866 | 0.143918 |
| CCRs2-high | 59/59 | 7.9857 | 6.4514 | 0.148254 | 0.118841 |
| CCRm-low | 42/42 | 21.0100 | 6.9778 | 0.800198 | 0.279679 |
| CCRm-medium | 40/40 | 15.0546 | 10.1449 | 0.365496 | 0.248933 |
| Slider-750 | 51/51 | 283.9311 | 76.0714 | 4.544831 | 1.407064 |
| Slider-1000 | 18/18 | 156.8813 | 48.9130 | 0.821204 | 0.356766 |
| road | 354/365 | 21.5915 | 17.4363 | 0.567491 | 0.450145 |
| slider | 69/69 | 250.7877 | 68.9866 | 3.573450 | 1.133073 |
| all | 423/434 | 58.9781 | 25.8452 | 1.057825 | 0.561545 |

La columna de cobertura cuenta targets disponibles, no predicciones entregadas.
El candidato contiene 434/434 salidas finitas; el promedio local usa 423 targets.
El original conserva dos negativos y el nuevo uno en Slider-750. La vista raw
del candidato da 26,3949 % y 0,572904 s; la vista procesada, 25,8452 % y 0,561545 s.
Se publican ambas, sin reescribir los ficheros originales.

El original procesado sobre los targets locales da 58,9781 %. No se toma el
32,04 % derivado del PDF oficial y se combina con 25,8452 % para anunciar una
mejora: serían denominadores y correspondencias de targets distintos. La
comparación local emparejada da una reducción de RAE del 56,18 % y de MAE del
46,92 %. El organizador aún debe medir el resultado oficial del nuevo candidato.

## Robustez del candidato final y ajuste

La validación final reutiliza los outputs nativos ya congelados de perturbaciones
crudas de 24 grabaciones. Los productores no han cambiado; repetir su decodificación
no aporta otra realización del experimento. Se evalúa el nuevo router sin tuning.

| Condición | F0 RAE % | Adaptado RAE % |
|---|---:|---:|
| Limpia | 28,2881 | 11,3983 |
| Dropout 70 % | 29,9054 | 12,0180 |
| Pérdida polaridad positiva | 30,4897 | 12,0047 |
| Pérdida polaridad negativa | 31,1322 | 11,8184 |
| Jitter 1.000 μs | 28,2651 | 11,4088 |
| Fondo 5 % | 28,2366 | 11,3854 |

Son 144 inferencias de fusión sobre puntos medios preseleccionados y expuestos,
no un nuevo test independiente de 144 grabaciones. El ajuste sobre las 2.608
filas usadas para entrenar alcanza 11,1586 % y 0,452961 s. Ninguna de estas dos
tablas demuestra precisión de un sistema de seguridad fuera de la colección.
