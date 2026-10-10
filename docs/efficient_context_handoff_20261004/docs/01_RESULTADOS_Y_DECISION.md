# 1. Evidencia, interpretación y decisión

## Resultado cerrado: no repetir trabajo
T6 conserva 72 fits y 180.000 updates científicos históricos; su cierre no añadió
actualizaciones. La campaña nocturna completó 24 cabezas nuevas y 60.000 updates
científicos, con 80 técnicos y hasta 100 repetidos o inciertos. El perfilado terminó
las 1.728 solicitudes de 64 consultas TRAIN, nueve rutas y tres modos. Las 614
mediciones que faltaban están completadas. No relanzar ninguna de estas tareas.

| Sistema | MiD OLD_DEV | Alcance |
|---|---:|---|
| RISK17 | 146,039451 | Referencia histórica |
| SIMPLEX17 | 144,021187 | Referencia histórica |
| H1 | 133,089985 | Media de pérdidas de tres seeds de cabeza |
| H8 | 121,784022 | Media de pérdidas de tres seeds de cabeza |
| H16 | 119,241356 | Media de pérdidas de tres seeds de cabeza |
| FULL_C0 | 121,404706 | Seed 7 |
| A5_ONLY_C0 | 127,785376 | Seed 7 |
| C2F_ONLY_C0 | 126,704974 | Seed 7 |
| A5_PAIR_C0 | 124,668732 | Seed 7 |
| SET_AGE_C0 | 122,442604 | Seed 7 |
| SET_NOTIME_C0 | 123,686600 | Seed 7 |

H16 mejora 2,542666 MiD, aproximadamente un 2,09 %, frente a H8. Su intervalo
jerárquico agregado es [−5,213398; −0,111241]. Usando sólo las nuevas seeds 13/23,
el delta es −2,423702 y el intervalo [−5,244259; +0,041091]. OBneIVg4Cw empeora en
las tres seeds. Hay estabilidad local entre inicializaciones, pero no uniformidad
entre escenas ni confirmación independiente. No confundir medias de pérdidas con
un ensemble de predicciones. Las unidades de evaluación siguen siendo nueve
secuencias OLD_DEV reutilizadas, no 27 escenas independientes.

Las reducciones no pasan el margen de no-inferioridad registrado. Eso no demuestra
que todos los expertos sean imprescindibles en cualquier sistema; significa que
estas reducciones, con esta receta y este margen, no quedaron validadas. Tampoco
se demuestra equivalencia porque un intervalo incluya cero.

H8 supera H1, REPEAT_CURRENT y la EWMA fija. REVERSED, entrenado con inversión,
no identifica la dependencia causal del orden. SET_AGE se aproxima, pero no
establece equivalencia. La afirmación defendible es que el contexto pasado y una
corrección aprendida son útiles. No se ha demostrado que la GRU reconozca una
aceleración o corrija un retardo físico. Los diagnósticos contienen casi sólo
consultas aisladas, dos cold starts, cero cambios rápidos según el umbral
congelado y una transición de signo. No justifican una ventaja de reacción AEB.

## Coste: medir el sistema que realmente se ejecuta
El p95 de la cabeza preparada ronda 19,47 ms para H8 y 31,45 ms para H16. La ruta
medida desde HDF5 y ROI suministrado alcanza p95 calientes de 8,48–9,35 s para H8
y 13,20–15,30 s para H16. Son segundos, no milisegundos. El sistema de replay
actual no demuestra operación en tiempo real. Estos números tampoco son una
latencia intrínseca de la cámara: incluyen preparación offline, GPU compartida
y el backend registrado. No se comparan directamente con módulos ONNX de Garl
medidos en otro dispositivo y con entradas preparadas.

En warm_block2, las fracciones de las SUMAS de tiempos son:

| Tramo | H8 | H16 |
|---|---:|---:|
| Lectura | 30,27 % | 29,38 % |
| ROI y voxelización | 55,20 % | 60,98 % |
| Productores y transferencias | 14,06 % | 9,16 % |
| Cabeza y emisión | 0,45 % | 0,47 % |

El pequeño resto corresponde a normalización. No son descomposiciones del p95.
Aplicando Amdahl a esa carga, eliminar todo el coste de la cabeza H8 sólo daría
aproximadamente 1,00455×. Acelerar diez veces lectura y preparación daría alrededor
de 4,33×; hacer gratis ese tramo daría como máximo 6,88× para la misma carga.
Son cálculos explicativos, no predicciones de velocidad ni límites de una futura
arquitectura streaming. Por eso no conviene optimizar ahora otra GRU marginalmente
más rápida sin resolver primero la preparación.

## Hallazgo independiente: las ventanas casi nunca son duplicados exactos
Esta revisión leyó los índices `history/{D0,D1,DENSE_OLD}/query_context_index.npz`
del ZIP T6 físicamente disponible, cuyo hash empieza por `84e250` y termina en
`7e8a`. Son índices históricos congelados, no una reproducción de los eventos.
D1 tiene 27.307 filas de índice; ese número no equivale a los ejemplos utilizados
por cada fold.

En D1/H16 hay 1.308.075 usos de ventanas, pero 1.308.019 ventanas únicas sumando
la unicidad dentro de cada consulta. Sólo 56 usos se ahorrarían mediante esa
reutilización exacta: un 0,004281 %. En D0/H16 es un 0,017823 %. Las duraciones
nominales cercanas a 100 ms difieren ligeramente entre timestamps. **Deduplicar
ventanas exactas no es el primer proyecto de aceleración.** Redondear límites
para generar coincidencias cambiaría las observaciones y exigiría otro experimento.

La historia entre anchors y el soporte total de eventos tampoco son lo mismo.
Las medianas de la unión temporal son H1 = 300,004 ms, H8 = 650,004 ms y H16 =
1.050,004 ms. H16 añade los 750 ms de separación entre anchors a unos 300 ms de
la observación experta más antigua. Esto no obliga a esperar un segundo nuevo en
cada consulta si existe un búfer, pero exige retener y procesar ese pasado.

En todos los índices inspeccionados, la última ventana actual termina en el
anchor registrado. La disponibilidad del ROI es posterior. Ninguna de estas dos
observaciones acredita por sí sola la semántica ancestral del GT o la latencia
real de un detector. `evidence/WINDOW_REDUNDANCY.json` conserva los resultados y
`evidence/audit_windows.py` permite regenerarlos desde el ZIP original.

## Decisión
Mantener H8 y H16; acelerar primero la preparación sin cambiar sus predicciones.
Ejecutar una sola prueba H8-WIDE para separar alcance y número de observaciones.
Obtener comparadores Garl ligados a TRAIN, ROI y disponibilidad antes de atribuir
una superioridad metodológica. No reabrir LATENT, escalar el router por defecto,
repetir los trabajos cerrados o reemplazar las etiquetas por contacto físico.

Fuentes: informes preservados en `evidence/`, archivos fijados en `SOURCE_PIN.json`
y referencias primarias en `docs/08_SOURCES.md`. Los recibos de verificación
publicados y las comprobaciones ejecutadas aquí están diferenciados.
