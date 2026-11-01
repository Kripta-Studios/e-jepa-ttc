# Resultado oficial eAP 3D Detection — 1 de noviembre de 2026

Generado desde los ZIP devueltos por el scorer. No se leen ni reconstruyen etiquetas test.

**AP oficial: 0.05129508; eAP score: 0.13980148.**
ATE: 0.92764309; ASE: 0.56968339; AOE: 0.64073705.

El scorer terminó con exitCode 0. Coinciden los 35,940 frames y las 1,796,950 predicciones del envío.
La entrega contiene un frame vacío por un PNG público truncado, documentado antes del scoring.
prediction_result.zip solo contiene metadata de ingestión nula; no contiene predicciones ni un error de scoring.

| Clase | GT test | AP oficial (%) | Recall alcanzado a 2 m, rejilla (%) |
|---|---:|---:|---:|
| car | 126,744 | 32.237 | 53 |
| bicycle | 603 | 0.000 | 8 |
| truck | 37,230 | 1.016 | 14 |
| pedestrian | 8,428 | 0.000 | 6 |
| motorcycle | 15,266 | 4.914 | 28 |
| bus | 4,768 | 2.870 | 20 |
| tricycle | 1,174 | 0.000 | 9 |
| other_vehicle | 522 | 0.000 | 2 |

El recall se lee en la rejilla de 1 % devuelta; no es un recuento exacto de matches ni un oracle del pool.
AP cero no implica ausencia absoluta de aciertos: las curvas de esas clases contienen precisión positiva en algún umbral.

## Diferencia de métricas y población

La configuración oficial fija min_recall=0.1 y min_precision=0.1. Sobre cada curva de 101 puntos:
`AP = mean(max(precision[11:] - 0.1, 0)) / 0.9`.
Esta reconstrucción coincide con los 32 AP por clase/distancia del scorer. La macro de ocho clases coincide con AP.
La fórmula está también documentada en el [devkit público nuScenes](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py).
Esa coincidencia no acredita todos los detalles del matching o los filtros privados del servidor.

La media de las curvas test devueltas, sin esos recortes, es 0.08624620. Es un diagnóstico
aritmético; no reproduce nuestro AP101 con envolvente ni constituye otra puntuación oficial.

La evaluación local usó 107,308 anotaciones elegibles e ignoró 83,161
de 190,469 anotaciones (43.66 %) por profundidad fuera del protocolo.
El perfil local limita profundidad de cámara a 0,1–60 m y usa reglas de dominio/ignorados para GT y predicciones.
La selección local y el test son además secuencias distintas. No se puede atribuir toda la diferencia a generalización
ni toda a la métrica; faltan igualar población y scoring sobre TRAIN/selección públicos.

## Lectura de los errores y del score compuesto

Las cuatro clases con recall insuficiente a 2 m reciben 1.0 en los tres errores TP; el promedio global incluye
esos valores de penalización. ATE no es simplemente la distancia media de todas las cajas, ni ASE un porcentaje
de error de longitud. La orientación es angular; convertir la macro a grados no elimina las penalizaciones.

`eAP = (5*mAP + (1-ATE) + (1-ASE) + (1-AOE)) / 8` para estos errores, todos inferiores a uno en la macro.
La regla general usa max(0,1-error). La reconstrucción coincide con el score devuelto.
El score compuesto 0.1398 no significa mAP 13.98 %. El JSON indica velocity_evaluated=false.

## Conclusión y siguiente decisión

El envío es válido y demuestra señal útil principalmente en coches. El resultado no acredita un detector robusto
para las ocho clases ni liderazgo oficial: no se ha verificado un leaderboard comparable.
Truck tiene 37.230 anotaciones test; su debilidad no puede explicarse por falta de soporte test.
No hay un envío oficial baseline sin ranker: este resultado no permite atribuirle una mejora o un perjuicio en test.
Las 50 propuestas por frame explican el total enviado, pero no prueban que bajar un umbral o aplicar NMS mejore AP.

Antes de gastar más entrenamiento, alinear en TRAIN/selección el cálculo con la configuración observada y reportar
por separado población raw y dominio local. Reevaluar las predicciones guardadas de baseline y ranking sin inferir
otra vez, manteniendo separado el perfil histórico. Después diagnosticar recall por clase, rango y confusiones.
Elegir las siguientes intervenciones con validación pública, no ajustando umbrales, modelos o geometría contra test.
Este análisis no lanza reentrenamiento ni nuevas participaciones y no modifica el artefacto enviado.
