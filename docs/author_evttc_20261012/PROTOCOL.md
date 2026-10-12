# Protocolo prospectivo de reconstrucción

## Identificación de secuencias

Se utiliza el manifiesto EvTTC previo solo como inventario para localizar medios,
sin reutilizar sus queries ni targets. CCRs2 se corresponde localmente con
`CCRs-2/{medium,high}-100/overlap-100`. Para CCRm-medium existen variantes de
solapamiento 0/50/100. Solo la de 100 tiene anotaciones en 21,4–22,2 s; la de
50 ni siquiera alcanza ese intervalo. Esta asociación es una inferencia de
soporte temporal de los archivos, no una confirmación adicional del autor.

## Selección previa a TTC

Se toma el primer frame RGB dentro de cada intervalo y una rejilla de 100 ms
anclada en él. Se usa el frame más próximo, empates hacia el anterior y tolerancia
1 ms. Un slot ausente invalida su tripleta; no se cierra el hueco saltando a la
siguiente anotación. Se forman tripletas enteramente dentro del intervalo.

Se obtienen 49 instantes: seis son arranque de las tres secuencias y 43 generan
queries (20 CCRs2-medium, 17 CCRs2-high, 6 CCRm-medium). No se usa TTC para
determinar esos números. Todos los candidatos quedan en el manifiesto/audit.
El último frame de cada tripleta define el instante de consulta.

La anotación del frame está disponible en ese timestamp bajo un protocolo de
ROI oracle; no se añade la demora artificial de 50 ms de Dev32. El primer
objeto válido de la anotación inicializa el seguimiento. Luego se selecciona
el centro más próximo al anterior, con orden de anotación para desempatar.

## Mapping y preparación

Se usan intrínsecos/distorsión izquierda del HDF5 y la rotación de la relación
calibrada RGB→event adoptada en el adaptador previo. Se proyectan cuatro esquinas
y se obtiene la envolvente. **La traslación se ignora por falta de profundidad
disponible como entrada**. El correo no especifica la aproximación de profundidad
ni resuelve la convención del nombre legado `T_to_left_bfs`; no se afirma
paridad geométrica exacta con el código privado de los autores.

Garl usa cuadrados enteros de dimensión máxima, factor 1,0, eventos centrados
en cada endpoint y un crop RGB común centrado en el último. Se conserva la
normalización ImageNet y el muestreo bilineal de la implementación pública.
Los dos segmentos event son [t0,t1) y [t1,t2) reales; sus deltas se registran.
El checkpoint convierte alturas a TTC con su delta fijo 0,1 s. Se exige que
ambos segmentos y sus ROI tengan eventos; un fallo no elimina al candidato.

V13 y V12 reciben el cuadrado sin expansión de esa misma consulta, constante
en sus historiales para conservar expansión aparente. Conservan sus propias
representaciones, normalización y contextos nativos. V13 usa cuatro observaciones
event y hasta las tripletas RGB disponibles en 650 ms; V12 usa H8 a 50 ms.
Garl usa 200 ms de eventos. **Igualdad de queries/labels no es igualdad del
presupuesto temporal de entrada.** El crop 1,0 también es una adaptación de
entrada de los checkpoints propios, no un entrenamiento nuevo.

## Checkpoints y sellado

Se cargan todos los endpoints V13, las tres cabezas V12 sobre productores
compartidos, Garl event-only y Garl full. Los cargadores verifican estados y
hashes existentes. `paper_ours_full.pth` tiene SHA256
`e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e`.
No se optimiza ni se hace fine-tuning. Se publica la mediana H8 además de cada
cabeza; no son tres réplicas independientes de productores.

Cada predicción se guarda de forma durable con identidad/hash; los CSV y
fragmentos se sellan antes de leer `ttc.csv`. Las reanudaciones rechazan cambios
de código/configuración/población. Los fits y evaluaciones anteriores no se
sobrescriben. El código nuevo está separado de los módulos congelados.

## Labels y denominadores

El scorer busca el vecino temporal más próximo en la columna de timestamps
de `ttc.csv`, tolerancia ≤0,05 s. Si el vecino es no finito o cero, no busca
otro más favorable. No interpola. Para reflejar un pipeline que descarta
observaciones inválidas antes de formar tripletas, se exige GT válido en los
tres frames; la etiqueta evaluada es la del último. Esta decisión queda
declarada porque no existe lista oficial de exclusiones con la que contrastarla.

Se reportan RTE/MAE por escenario, promedio macro de escenarios y pooled por
consulta; disponibilidad individual, intersección de todos los modelos y
parejas frente a Garl full. Fallos permanecen en denominadores de cobertura.
MiD por banda es diagnóstico adicional con conteos inválidos en RESULT.json;
no sustituye RTE ni se imputa banda negativa para inventar un overall oficial.

## Recursos y tiempo

CPU, FP32, un hilo PyTorch; GPU ocupada por otro trabajo independiente. Se
registran preparación/forward/total Garl y coste de la familia V13 completa
(seis cabezas, productores compartidos) por consulta. No se mezclan esos
perímetros ni se comparan directamente con el runtime del paper. No hay
aislamiento del host ni un benchmark de despliegue en esta campaña.
