# 03 — Experimento prospectivo de cabezas: masa × objetivo

**Hipótesis:** una parte del sesgo corto/largo puede reducirse modificando la presión de entrenamiento por rango y el objetivo, sin añadir compute de inferencia ni nuevos productores. No se supone que ésta sea la causa única ni que el experimento vaya a ganar FCWD.

## Prerrequisitos exactos

Sistema inicial: E_A5_MATCHED, E_C2F_MATCHED y PAIR_E_MATCHED de V13, completos y congelados, entrenados sólo en P. La cabeza se entrena sólo con la caché de rol H y normalizador H registrado. V sirve exclusivamente como desarrollo interno, ya expuesto a la investigación; no es confirmación externa nueva. No utilizar los productores V12 TRAIN40 para fingir exclusión de V.

E_CTX_MATCHED V13 no se denomina automáticamente H8 V12: conserva sus observaciones realmente disponibles y sus tiempos. El protocolo V13 de transferencia puede tener cuatro observaciones a 100 ms bajo un soporte de 650 ms, aunque la cabeza admita ocho slots. No rellenar ni densificar el contexto para imitar V12. El control causal del estudio es el E_CTX nativo del mismo sistema, no la cifra histórica de otra población o la mediana V12 de tres cabezas.

Los padres RGB no son requisito de esta rama. Cuando la cola nativa publique las cachés H finales y sus evaluaciones V, reutilizarlas en sólo lectura. Si falta una caché V, implementar una exportación estrictamente label-free de inputs y una evaluación separada con GT V. No inferir desde padres parciales.

El objetivo es una comparación nueva dentro de la receta V13. No afirmar que haya identificado la causa retrospectiva del resultado V12 si mejorase.

## Matriz cerrada, semilla 7

| ID | Masa | Objetivo |
|---|---|---|
| RT-M0-L0-S7 | Nativa, byte a byte | Nativo |
| RT-M1-L0-S7 | Grupos y bandas | Nativo |
| RT-M2-L0-S7 | Bandas con subrangos acotados | Nativo |
| RT-M0-L1-S7 | Nativa | Nativo + término TTC relativo |
| RT-M1-L1-S7 | Grupos y bandas | Nativo + término TTC relativo |
| RT-M2-L1-S7 | Bandas con subrangos acotados | Nativo + término TTC relativo |

6 fits × 2.500 updates = 15.000. El M0-L0 puede reutilizar E_CTX_MATCHED seed7 completo únicamente si modelo, pérdida, población, sampler, precisión, normalizador, batch, scheduler e inicialización coinciden; la prueba de identidad es obligatoria. La reutilización no libera permiso para otro brazo.

## Factor masa: no confundir con sampler

M0 copia exactamente el vector nativo, sin redondearlo/normalizarlo de nuevo.

M1 conserva la masa total nativa de cada grupo. Dentro del grupo asigna los pesos relativos 0,5/0,3/0,1/0,1 a las bandas c/s/l/n presentes y divide cada masa entre sus filas. Esto es una distribución de entrenamiento; jamás sustituye la agregación oficial de evaluación.

M2 conserva cada masa grupo×banda de M1. En c busca masa igual para subrangos (0,1;1], (1;2], (2;3]; en l para (6;8], (8;10], sólo entre subrangos presentes. Cada masa por fila se limita a [0,25;4] veces M1 mediante una proyección conservativa. Si una subbanda muy escasa alcanza el cap, no se consigue igualdad perfecta; se informa. s y n permanecen idénticos a M1.

Se mantienen todos los IDs y draws uniformes con reemplazo, los mismos por seed entre brazos. Cambiar masas no aumenta la frecuencia de selección de las filas: modifica su contribución al objetivo. Un sampler estratificado sería otro factor y NO está autorizado aquí. Registrar exposición de episodios; más peso no crea ejemplos independientes.

Dominios TRAIN admitidos: `-10<T<0` o `0.1<T<=10`. Si H contiene targets fuera, no eliminarlos para ejecutar: documentar la dependencia semántica y dejar este factorial sin fit hasta que exista un protocolo explícito para ellos. Las otras tareas siguen viables.

## Factor objetivo: no cambiar el output

L0 llama a `simplex_t.model.training_loss` nativo con sus términos de fase, cuantiles y costes. No se elimina la cabeza auxiliar ni se reemplaza la red.

L1 añade únicamente:

```text
z_i = (T_hat_i - T_i) / max(|T_i|, 0.5 s)
L1 = L0 + 0.25 * mean_batch(N * mass_i * SmoothL1(z_i, 0; beta=1))
```

`T_hat` es el TTC emitido por el mapping nativo aplicado a `raw_location`. El target T se reconstruye de la fase etiquetada sin imponerle el cap de salida. Esta fórmula trata targets negativos válidos y no hace log de un TTC firmado. Los coeficientes son decisiones prospectivas, no óptimos estimados de FCWD. No se ajustan tras medir gradientes o scores; se documenta una escala poco útil si ocurre.

Se conservan arquitectura, 17 features, tiempos, normalizador, cap, política de cero y saturación. Por ello una región de gradiente nulo no desaparecerá mágicamente. Este ensayo NO es la cabeza Direct histórica ni una solución probada al rango largo.

## Receta y control de implementación

GRU 160 de dos capas, FP32 CPU, batch 128, AdamW con weight_decay=1e-3, warmup100, pico lr3e-4, final3e-5, cosine, clipping1, endpoint 2500. Comparar la receta ejecutada nativa completa; no usar defaults incompletos. Conservar parámetros optimizer/sampler/RNG de la receta confirmada.

Todos parten de la misma inicialización por seed. No se hace fine-tuning de la cabeza ganadora anterior. Guardar cada 100 y en pausa. Congelar los seis endpoints antes de ver los primeros scores de esta matriz V. No early stopping sobre V ni Dev32/FCWD. No usar una semilla favorable como salida canónica.

## Selección prospectiva y réplica

Primaria para selección: reducción de MiDc en V respecto a M0-L0, bajo salvaguardas. Criterios propuestos, no umbrales de seguridad de automoción:

1. MiDc al menos 10% menor y mejora en al menos 60% de los grupos V con c.
2. Agregado V de bandas con la receta nativa no empeora. No crear overall si falta alguna banda: en ese caso no hay selección de ganador global y la salida será `LIMITED_VALIDATION_SUPPORT`, con resultados descriptivos completos.
3. En s/l/n, deterioro por banda no mayor de `max(2 MiD, 5% de la referencia)`; MAE global no supera 1,02× la referencia.
4. Cobertura de predicción/MiD completa en soporte elegible; error de signo no aumenta más de 0,5 puntos porcentuales. p95 de sobreestimación positiva crucial no sube más de 5%.
5. Tasas de aviso tardío y falsa alerta reportadas; si cada denominador tiene al menos 30 casos, aumento permitido máximo de 1 punto porcentual. Si no, declarar soporte limitado, sin inferir seguridad.

Pueden reducirse estas reglas a resultados descriptivos si V no permite estimarlas, pero no se aflojan para producir un ganador. Los intervalos se publican; no se exige significación para invertir en réplica.

Entre candidatos habilitados: escoger una única receta por menor MiDc V, después menor agregado y finalmente menor índice ID en empates exactos. Replicar **ese candidato y M0-L0**, seeds 13/23: 4 fits y 10.000 updates. Selección conjunta antes de esas réplicas; no cambiar candidato según las seeds nuevas. Si no hay candidato, termina con 15.000 o menos.

En réplicas se promedian pérdidas por query y se remuestrean grupos completos; no se promedian predicciones TTC para fabricar un ensemble. Las réplicas siguen siendo de cabezas sobre padres comunes. Un resultado incierto se conserva; no añadir seeds.

## Transferencia y comparadores

Una vez congelada la identidad candidata, ejecutar a lo sumo una evaluación de transferencia de los endpoints de la matriz/replicación sobre las cohortes ya admitidas, con todos los resultados publicados. No usarla para elegir entre brazos. Mantener las predicciones Garl nativas y sensibilidad al cap por separado. FCWD RGB sigue no disponible sin su mapping.

Evaluar mean_MiD, overall cuando exista, cada banda, MAE, mediana, RTE, signo, colas, sobreestimación y cobertura. Población fija; sin filas excluidas por error. FCWD/Dev32 permanecen desarrollo históricamente observado aunque no se usen en estos updates.
