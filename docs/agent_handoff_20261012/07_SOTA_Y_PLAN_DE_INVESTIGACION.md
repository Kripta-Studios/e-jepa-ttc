# 07 — Por qué no se puede afirmar SOTA y qué falta

## Afirmación que sí sostiene la evidencia

«En transferencia exploratoria Dev32, la fusión V13 reduce MAE un 26,75%
respecto a las predicciones reproducidas del checkpoint Garl RGB+eventos y
mejora MiD en las tres bandas positivas disponibles. En FCWD reduce MAE un
33,38% respecto a Garl event-only, utilizando además RGB registrado cuando
está disponible. El sistema conserva limitaciones de error crucial,
disponibilidad RGB y coste.»

La aritmética está en
`../fcwd_rgb_registration_20261012/SUMMARY_COMPARISONS.json` y se puede
recalcular con los CSV. No se sustituyen estas condiciones por «vence a Garl
en todo», «SOTA event-only» o «listo para conducción».

## Obstáculos concretos

1. **Falta test oficial.** test12 no tiene labels públicos ni score CodaBench
   acreditado. El paquete de submission valida formato/tokens, no precisión.
2. **Falta parte de la métrica.** Dev32/FCWD no tienen TTC negativos y no
   permiten overall_MiD oficial completo. El overall TRAIN40 está en train.
3. **Hay derrotas observadas.** Garl gana MiD crucial FCWD; R_CTX gana MAE a
   F_TRUE Dev32; V13 event no domina V12; estudiante/H2 no mejoran todo.
4. **No siempre se compara la misma información.** FCWD propio añade RGB,
   Garl disponible es event-only; contextos y preparación también difieren.
5. **Los conjuntos ya están expuestos.** Diagnosticar y diseñar sobre ellos
   convierte una evaluación posterior en exploratoria, salvo nuevo holdout.
6. **Falta generalidad estadística.** No hay tres semillas completas de V13.
   FCWD solo tiene tres secuencias; ventanas no son unidades independientes.
7. **La latencia no demuestra dominancia.** Garl optimizado gana el piloto V12;
   los costes V13 finales se midieron con carga concurrente. No hay evaluación
   completa en hardware de despliegue ni detección/tracking contabilizados.
8. **Persisten aproximaciones de entrada.** Registro local sin extrínsecos
   GT, ROI retrospectiva, rechazo de RGB y fallback. Deben medirse y declararse.

La falta de evidencia SOTA no implica que los avances no sirvan. Hay una
contribución posible de arquitectura compacta, continuidad reproducible,
reutilización temporal y análisis multimodal, con claims acotados y resultados
negativos publicados. Un paper defendible necesita protocolo y evidencia;
adoptar el estilo narrativo de otro trabajo no cambia esos requisitos.

## Plan priorizado para otro agente

### A. Reconstruir y auditar sin entrenar

Ejecutar `replay.py`, comprobar manifests, leer los dos CSV y reproducir las
cohortes completa/RGB disponible/fallback. Mantener ID, secuencia, timestamp,
modalidad, TTC, clipping y GT faltante. Auditar las diferencias entre los
comparadores Dev32 de los informes históricos y la campaña final antes de
mezclar sus métricas o atribuir causalmente una mejora.

### B. Obtener trazas discriminantes

Para las consultas problemáticas y controles TRAIN fijados por hash, guardar:
alturas y expansión analítica → residual productor → fases/TTC A5/C2F/PAIR →
mediana base → residual de contexto/fusión → fase/TTC final. Acompañar ROI,
relojes, soporte, confianza, normalización y gate RGB. Una traza de la salida
final sola no identifica la capa culpable.

Si los expertos ya fallan, intervenir en productor/entrada. Si los expertos
son razonables y el residual empeora, estudiar objetivo, pesos y normalización.
Si el fallo está en relojes/ROI, corregir el pipeline y volver a ejecutar todos
los comparadores afectados; no entrenar una red para compensar un bug.

### C. Auditar geometría FCWD sin optimizar contra TTC

Seleccionar correspondencias de imagen/eventos y controles de soporte sin
consultar errores TTC. Medir error de reproyección, cobertura y rechazo por
textura, movimiento, profundidad aparente y bordes. Probar otras familias de
transformación con validación independiente. Aplicar una transformación
admitida también a Garl multimodal y presentar cobertura común y total.
No bajar NCC después de ver qué ejemplos perjudican la métrica.

### D. Ablation de masa y objetivo en una campaña nueva

Con padres finales congelados y caché H admitida, medir masa materializada,
exposición por secuencia/episodio y contribución al gradiente por banda.
Comparar igual masa de grupo con distintos repartos por banda usando las
mismas semillas, presupuesto, datos y cabeza. Registrar efectos sobre
signo, fase, MAE, MiD, colas y calibración. Repetir ventanas no crea episodios.
No modificar retrospectivamente loss/sampler/normalizadores del run final V13.
V14 ya existe: coordinar esa investigación allí sin alterar sus archivos ajenos.

### E. Eficiencia tras fijar precisión aceptable

Priorizar reutilización exacta de entradas/productores y control de estado;
después medir aproximaciones de ROI/voxels, estudiante y precisión reducida.
Cada variante debe superar un presupuesto conjunto de error y latencia fijado
antes del test. Incluir Garl optimizado contemporáneo. Una mejora CPU local
debe verificarse en el camino completo y con límites de memoria reales.

### F. Cerrar evidencia externa

Congelar modelo, datos, protocolo y criterios con TRAIN/validation; realizar
evaluación nueva y ciega por secuencias. Cuando exista acceso, enviar test12
con tokens exactos y conservar recibo/score del servidor. Añadir réplicas
completas, bootstrap por secuencia, incertidumbre y robustez. Solo después
del resultado evaluar si procede una afirmación SOTA, delimitada al benchmark,
métrica, modalidad y presupuesto realmente comparados.
