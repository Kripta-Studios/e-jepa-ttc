# 02 — Datos, métricas y condiciones de comparación

## Poblaciones distintas

| Nombre | Dataset y función | Población | Interpretación correcta |
|---|---|---:|---|
| eAP TRAIN40 | Cuarenta secuencias de entrenamiento | 88.744 consultas | Diagnóstico dentro de entrenamiento; no holdout |
| EvTTC Dev32 | Inventario interno de desarrollo de 32 secuencias EvTTC | 3.947 consultas; 3.640 elegibles | Transferencia ya expuesta al desarrollo |
| FCWD | Forward Collision Warning Data; FCWD1/2/3 | 630 consultas; 597 elegibles | Dataset externo pequeño; ya examinado, no test ciego |
| eAP test12 | Split público de entradas GarlTTC con etiquetas privadas | 6.762 sample_token de 12 secuencias | Evaluación oficial pendiente de scoring privado |

Dev32 no significa 32 frames ni el test oficial de Garl. FCWD no es un subconjunto de eAP ni de Dev32. Su [página de dataset](https://nail-hnu.github.io/EventAidedTTC/) pertenece al trabajo Event-Aided TTC. La [página eAP](https://nail-hnu.github.io/eAP_dataset/) y el [artículo GarlTTC](https://arxiv.org/abs/2603.16303) describen otra fuente de datos. Estos enlaces identifican las fuentes; las tablas del handoff proceden de resultados locales publicados, no de cifras copiadas del paper.

Las consultas sin GT finito o con GT cero se registran como no elegibles; el criterio no depende del error del modelo. Son 307 en Dev32 y 33 en FCWD. Una predicción inválida sobre GT elegible no autoriza a eliminar la consulta para mejorar el promedio.

## Qué mide MiD

Para este scorer, con Δt=0,1 s:

```text
phi(T) = -log(1 - 0.1 / T)
MiD(T, P) = 10000 * abs(phi(T) - phi(P))
```

La transformación exige un argumento del logaritmo positivo y finito. Cerca de TTC positivo corto, un mismo error en segundos causa más MiD que para TTC largo. El código local de admisión del scorer está en [`operational/garl_comparison/mid.py`](../../operational/garl_comparison/mid.py); exige SHA-256 `f3bbf089ba6e47edfce1de522c92ddc969db2260fac7544e82bc017fedf82dce` antes de ejecutar el scorer externo. El replay público implementa únicamente el cálculo necesario para las filas FCWD válidas y lo compara con la evidencia; no se presenta como sustituto general del programa CodaBench.

| Banda | GT TTC | Peso oficial |
|---|---|---:|
| c | (0,3] s | 0,5 |
| s | (3,6] s | 0,3 |
| l | (6,10] s | 0,1 |
| n | (-10,0] s | 0,1 |

```text
overall_MiD = 0.5*MiDc + 0.3*MiDs + 0.1*MiDl + 0.1*MiDn
```

`mean_MiD` promedia consultas individuales y conserva GT fuera de estas cuatro bandas cuando su MiD es válido. `overall_MiD` combina medias de bandas con los pesos fijados. Son métricas diferentes. FCWD tiene 201/189/108 ejemplos c/s/l y 99 con GT>10; Dev32 tiene 1.945/1.293/293 y 109 fuera de bandas. Ninguno tiene negativos. **No se rellena MiDn con cero ni se renormalizan los otros pesos**: overall queda no disponible.

TRAIN40 sí cubre las cuatro bandas: c=9.741, s=24.251, l=17.709 y n=37.043. Puede calcularse overall, pero sigue siendo entrenamiento. No existe un único «overall de EvTTC y eAP» que pueda obtenerse mezclando poblaciones sin un protocolo adicional.

## Fallos, cobertura y métricas que no deben confundirse

- El scorer conserva promedios sobre MiD finito. Por eso la evidencia añade `invalid_mid_total`, contadores por banda y `strict_mean_MiD`; este último queda no disponible si hay MiD inválido.
- Direct en Dev32 tiene una fila MiD inválida. Su mean_MiD 137,323 no puede describirse como una media estricta sobre las 3.640 consultas.
- El FR del scorer detecta la regla de fallo TTC del benchmark (predicción no finita o magnitud inferior a 0,1 s). No representa la tasa de aviso tardío o ausente. FR=0 no demuestra que el modelo detecte el peligro a tiempo.
- El fallo de aviso urgente de la revisión usa GT positivo≤1 s y predicción≤0 o>1 s. Es una medida distinta, documentada en [`score.py`](../../operational/ttc_revision/score.py).
- RTE de informes antiguos es error relativo TTC; no es MiD. MAE está en segundos; MiD tiene el escalado 10.000 anterior. No comparar sus magnitudes numéricas directamente.

## Comparación nativa y sensibilidad al clipping

H8 limita arquitectónicamente su salida a ±60 s. Garl puede producir valores finitos extremos. La vista principal conserva ambas predicciones nativas; la vista `equal_clip_60_sensitivity` aplica el mismo clipping a todas las predicciones finitas. No reemplaza fallos no finitos ni convierte la sensibilidad en el resultado oficial.

Esto importa en Dev32 RGB+eventos: un MAE muy alto de Garl convive con un mean_MiD competitivo. MiD comprime errores de TTC largo; el outlier afecta de forma distinta a las métricas. La sensibilidad con clipping ayuda a entenderlo, pero no autoriza a ocultar la vista nativa.

## Condiciones de entrada y entrenamiento

Se conservan entradas nativas: H8 utiliza más historia temporal que Garl; Garl full añade RGB. «Mismo conjunto de consultas» no significa «misma información» ni «mismo presupuesto de contexto». Las ROI son anotadas y su disponibilidad temporal en la adaptación se registra. Aplicar la ROI de la consulta a historia pasada no prueba que una ROI equivalente estuviese disponible en cada instante pasado.

En FCWD falta un mapping espacial certificado para ejecutar Garl full. Los sensores tienen resoluciones diferentes y la calibración MAT requiere interpretación específica; no se improvisó un escalado para completar la tabla. Garl event-only sí tiene la comparación publicada. La procedencia exacta de secuencias de entrenamiento del checkpoint público Garl no se ha verificado por un manifest completo.

## Estadística y selección

Los intervalos se calculan remuestreando secuencias completas y conservando el emparejamiento entre modelos. No se consideran miles de ventanas correlacionadas como miles de experimentos independientes. FCWD tiene solo tres secuencias: incluso un intervalo que excluya cero no garantiza generalización a escenas nuevas.

Las revisiones tienen endpoints fijados y guardan resultados negativos. Dev32 y FCWD han sido observados repetidamente: cualquier investigación guiada por este handoff debe tratarlos como desarrollo. Una nueva mejora necesita validación por secuencia y un conjunto externo no usado para decidir la receta. test12 privado no debe convertirse en un bucle de selección de hiperparámetros.
