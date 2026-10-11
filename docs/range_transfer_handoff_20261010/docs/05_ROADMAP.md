# 05 — Continuación hacia precisión, RGB y coste

## El orden que mantendría

A. Completar V13 en su rama actual y evaluar E, R y fusión con sus propios pesos finales.
B. Ejecutar el estudio masa×loss event-only pequeño, sin cambiar los productores.
C. Según atribución, registrar UNA intervención de productor/datos o de distilación; no una búsqueda simultánea de backbones.
D. Replicar el sistema completo y realizar refit/declaración de entrenamiento final, más evaluación oficial o independiente.

Este paquete autoriza sólo diagnóstico y B. C/D son roadmap, no presupuesto oculto. Ganar Garl en una tabla no equivale a dominar todos los regímenes ni a SOTA universal.

## Si la masa/pérdida mejora el rango crítico

Confirmar la mejora en V y la ausencia de regressiones en los otros rangos. Medir si la misma intervención merece estudio propio en R_CTX/F_TRUE. No insertarla retroactivamente en V13. Incorporarla al refit final sólo con una nueva receta registrada y repetir el control correspondiente.

El beneficio potencial de este ensayo es que cambia cómo aprende, no el tamaño del forward: no agrega encoders. La corrección puede seguir dependiendo de todas las features en inferencia, pero el entrenamiento recibe una distribución de coste distinta.

## Si el sesgo ya está en los productores o la información es insuficiente

Primero comprobar geometría, soporte, crop y clock. Un bug de input se corrige y se publica para todos los comparadores afectados. Si no hay bug:

1. Probar un readout que conserve evidencia de correspondencia espacial antes de PHASE17, comparando el mismo encoder congelado con resumen/denso y control sin nueva evidencia. No trasladar embeddings de distintos productores como coordenadas universalmente alineadas.
2. Sólo si la prueba lo justifica, fine-tuning de un productor event con objetivo TTC más una referencia congelada de su versión original. Reconstruir PAIR y cachés del nuevo padre; pesos/normalizadores antiguos no se heredan sin admitirlos.
3. Ampliar episodios independientes con TTC positivo 1–3 y ≤1. Las tres FCWD existentes no se convierten en una colección masiva repitiendo ventanas.

La modificación temporal de TRAIN requiere semántica coherente: si se define t'=a*t, duraciones y TTC' se escalan por a. Eso es una transformación temporal etiquetada, no prueba de que simule exactamente un sensor de eventos bajo otra velocidad/iluminación. Respetar exposición RGB, discretización y límite de inputs; validar separadamente antes de autorizar augmentation.

## Distilación orientada a tarea, no a copiar cualquier error del teacher

El estudiante actual infiere cinco features a partir de seis A5 y perjudica Dev32. La siguiente comparación posible sería estudiante de features frente al mismo estudiante con supervisión adicional de fase/TTC, mismos datos y updates. Congelar teacher y cabeza durante el estudio, verificar gradientes a través del readout y mantener originales como controles. El teacher RGB+eventos puede ayudar al estudiante event-only durante TRAIN, pero no le transmite información irrecuperable en su entrada durante inferencia.

No entrenar esa rama hasta que existan teacher/padres finales y resultados V13. No incluirla en el saldo de 30.000.

## Rendimiento: actualizar la prioridad tras los graphs

El piloto actual tiene10 consultas calientes de2 secuencias: H8 warp+graphs82,345 ms frente a Garl optimizado36,806 ms, con arranques distintos y p95 H8≈203,49 ms. Es trabajo real de ingeniería, no una demostración de tiempo real vehicular.

Medir `head_and_commit` con sub-timers de cabeza, materialización/clonado de outputs de graph, sincronización y escritura/gestión de estado. Las 31,594 ms no prueban que la GRU necesite rediseño. Mantener outputs independientes antes del siguiente replay de graph.

Separar exacto, warp y estudiante; evitar etiquetar hits aproximados como paridad exacta. Si el nuevo ROI incluye soporte fuera del recorte antiguo, reconstruir o declarar la pérdida, sin suponer que interpolación recupera eventos descartados. Comparar Garl con las optimizaciones exactas aplicables.

Evaluación de coste futura: un modelo canónico y mediana de tres por separado; ingesta/resets/cold/warm; p50/p95/p99; más de un objeto; memoria máxima; loopslabel-free con misma población de coste seleccionada sin GT. No comparar tiempos HDF5 de uno con forward preparado del otro. Sólo CPU ligero hasta que el dueño V13 ceda un slot GPU; no detener un fit científico para perseguir un microbenchmark.

## RGB en FCWD y evaluación oficial

La falta de mapping FCWD RGB bloquea ese comparador, no la investigación event-only. Resolver calibración numérica portable: identificar intrínsecos, distorsión, extrínsecos, convención de coordenadas, timestamp/exposición y fuente de cajas. El MAT MCOS puede requerir herramientas oficiales de MATLAB o export de los autores. No aproximar con un escalado de1280×720 a1920×1200 ni usar profundidad real del test como input oculto. Una caja RGB anotada manualmente sería otro contrato oracle, que exigiría declaración y comparadores simétricos.

test12 ya dispone de6.762 predicciones H8 y Garl EO empaquetadas. No repetir esa exportación para creer que hay un score. La subida o contacto con autores sigue siendo acción del usuario/autorización distinta. Una puntuación oficial obtenida después de escoger el candidato sería una medida real; no usar retornos sucesivos para ajustar hiperparámetros y seguir llamándolos test ciego.

La página pública FCWD contiene tres secuencias; más timestamps de Dev32 no añaden nuevos grupos. Si no hay datos externos independientes, el resultado publicable puede ser un estudio reproducible de desarrollo con límites explícitos, no una promesa de generalización.
