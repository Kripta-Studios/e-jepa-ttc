# Comparación de familias antes del refit

Revisión del 17-10-2026 sobre código, recibos y resultados existentes. No ejecuta
entrenamiento, inferencia ni selección de nuevas variantes. V18 conserva
`NO_CANDIDATE`. No hay un campo universal `best_model`.

**Instrucción posterior del usuario:** la primera evaluación oficial prevista
será la del modelo final entrenado sobre todo TRAIN40; no se enviará antes el
V16 histórico. Se conserva la comparación de familias de este documento, pero
queda sustituido su orden anterior de evaluación. Continúa la preparación y
ejecución autorizada del refit, sin usar Test12 para escoger la receta.

## Decisión

La nueva información cambia la justificación y exige comparar las dos cadenas
antes de comprometer el presupuesto. Mantengo **V16 como recomendación
provisional para una única familia final** por la interfaz física de su stacking
y la integración disponible. No lo elijo por disponer de un ZIP, ni afirmo que
vaya a superar a V18 después del refit. El presupuesto OOF completo no está
cerrado y el runner secuencial existente no implementa todavía las rotaciones.

Las identidades permanecen separadas:

| Función | Sistema |
|---|---|
| Candidato oficial ya congelado | V16 PHASE_ONLY_EVENT_S7, RGB+eventos |
| Mejor agregado puntual en la comparación interna V publicada | V18 C1 + R_CTX |
| Referencia individual event-only de agregado/crucial H | FULL/L0 |
| Referencia event-only de equilibrio negativo | LW, misma arquitectura y distinto objetivo |
| Control de transferencia | F_TRUE |
| Selector exploratorio no admitido | R0 NATIVE/FULL |

Los resultados de V y de H no se ordenan en una tabla de rendimiento común.
La media de métricas entre semillas tampoco describe un ensemble desplegado.
La comparación local ya ejecutada está en
[REPORT.md](../post_v18_transfer_20261017/REPORT.md); sus pequeñas cohortes
observadas no resuelven qué refit tendrá mejor generalización.

## Cadenas realmente consumidas

V16:

```text
E_A5 + E_C2F -> PAIR_E -> E_CTX --+
                                +-> PHASE_ONLY
R_A5 + R_C2F -> PAIR_R -> R_CTX --+
```

V18:

```text
E_A5 -> adaptación C1 ----------------+
                                      +-> cabeza de fusión C1/R_CTX
R_A5 + R_C2F -> PAIR_R -> R_CTX -------+
```

C1 adapta encoder y torre espacial. No depende de B1 ni necesita E_C2F,
PAIR_E o E_CTX en esa ruta de fusión. No existe un C1 TRAIN40 admitido para
reutilizar como endpoint final. El A5 TRAIN40 existente sólo sería su padre.

La genealogía histórica C1 está comprobada directamente en
`artifacts/joint_measurement_20261012/adaptive/C1_geometry/REGISTRATION.json`:
padre `9f779a0376d6988d1e3f70ef566797f840d1fe4dbf204ce2f9ae1dad5322d512`.
El `RUN_PROVENANCE.json` de `E_A5_MATCHED` en la rama hermana V13 declara
`P_only_fixed_epoch18`, 53.890 consultas y 30.330 updates. **No es el A5 V12
entrenado con todo TRAIN40.** Se descarta una estimación preliminar que confundía
estos padres y prohibía por ello reutilizar la rotación histórica C1.
La posible reutilización requiere comprobar el resto de bindings y exclusiones;
el hash de un padre por sí solo no admite toda una rotación.

## Comparación secuencial con exposición equivalente

Esto compara dos refits finales secuenciales, sin coste auxiliar OOF. Se
conservan 18 épocas para productores y exposición proporcional para otras etapas:
`ceil(updates_originales * N_nuevo / N_original)`. TRAIN40 tiene 88.744 consultas;
P histórico 53.890 y H histórico 18.422.

| Etapa nueva | V16 | V18 |
|---|---:|---:|
| Dos productores RGB, 18 épocas, batch 32 | 99.864 | 99.864 |
| PAIR_R | 11.264 | 11.264 |
| R_CTX | 12.044 | 12.044 |
| PAIR_E | 11.264 | — |
| E_CTX | 12.044 | — |
| C1, desde A5 TRAIN40 | — | 19.762 |
| Cabeza final correspondiente | 12.044 | 12.044 |
| **Total de updates nuevos** | **158.524** | **154.978** |
| **Ajustes nuevos** | **7** | **6** |

V16 reutiliza A5/C2F event TRAIN40 ya admitidos; V18 reutilizaría A5 después
de comprobar su interfaz C1. Estos padres completos no son inicializaciones
válidas para folds que deban excluir grupos que ya vieron.

Los updates no son unidades comparables de cómputo: C1 usa imágenes,
correspondencias y adaptación completa FP32 con microbatch 4; PAIR_E/E_CTX
trabajan sobre características guardadas. Los perfiles históricos sugieren
aproximadamente 15,4 horas para los dos productores RGB comunes. Sumando sólo
etapas con tiempos disponibles, el subtotal secuencial V16 ronda 16,8 horas,
y V18 17,9–21,1 horas. **Ambos excluyen su cabeza final, extracción, preparación,
recuperación y coordinación: no son presupuestos completos ni promesas de plazo.**
El rango C1 distingue el throughput de réplicas maduras del coste cargado al
primer ajuste; no se convierte esa diferencia en una aceleración nueva medida.

## El esquema OOF no se traslada automáticamente

V16 permite reunir PHASE17 sin normalizar y las fases/tiempos que consume
PHASE_ONLY. Sus significados están definidos entre rotaciones; esto no garantiza
que sus distribuciones sean idénticas. Los normalizadores finales deben ajustarse
sobre las características OOF correspondientes y todos los padres de una entrada
deben excluir su grupo.

En V18, `operational/joint_measurement/fusion_pair_extract.py` declara 196
coordenadas: 96 medias espaciales, 96 máximos y cuatro valores de fase/tiempo.
Las 192 coordenadas aprendidas no tienen correspondencia garantizada entre
encoders C1 ajustados independientemente. La rama RGB incorpora también
características normalizadas por cada modelo. **No está admitido concatenar
estos vectores entre folds como si procedieran de un único productor.**
Esto no demuestra que tal entrenamiento sea imposible; exige justificar y
validar un contrato adicional.

Entrenar cabezas locales por fold y reunir sus predicciones permite evaluar un
procedimiento V18, pero no proporciona por sí solo las entradas OOF compatibles
para una única cabeza final sobre el C1 TRAIN40. Una cabeza final entrenada sobre
sus propios productores sigue siendo stacking in-sample. Cambiar la fusión para
usar sólo coordenadas físicas sería otra arquitectura, fuera de este refit.

Por ello no se compara el refit V18 secuencial de seis ajustes con el V16 OOF
de 39 ajustes como si tuviesen el mismo contrato. Para V16, los 709.458 updates
prospectivos se calculan en `NESTED_ROTATION_REVIEW.json`; no constituyen una
cola iniciada. No hay todavía un presupuesto OOF V18 equivalente admitido.

## Consecuencia operativa

Se conserva el candidato V16 para evaluación separada y V18 como alternativa.
No se entrenan ambos por defecto ni se reabren V18/V19. La siguiente preparación
debe cerrar el coste total y la ejecución recuperable del diseño elegido antes
de iniciar sus updates científicos. El score oficial no se utilizará para elegir
la receta. Cachés compartidas sin pérdida siguen siendo útiles para ambas rutas.

El fallo técnico de memoria impide habilitar CUDA Graphs con la evidencia actual;
no demuestra que esa técnica sea inviable en general. No se ha medido una nueva
aceleración válida. La continuidad usa ejecución ordinaria y límites de memoria.
No se ha enviado ningún candidato a CodaBench en este trabajo.
