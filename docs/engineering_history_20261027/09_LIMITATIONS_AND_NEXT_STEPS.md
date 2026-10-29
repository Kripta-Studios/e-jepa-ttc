# Limitaciones, decisiones conservadas y siguientes pasos

## Qué terminó y qué significa

La cadena final TRAIN40 y sus entradas OOF están preservadas. El candidato tiene pesos, normalizadores, fuentes, predicciones y un sello de identidad. La comparación local posterior produjo predicciones para 554 consultas EvTTC con ambos sistemas congelados. Publicar estos resultados no requiere volver a entrenar, recalcular inferencias ni repetir los miles de tests de campañas históricas.

V18 permanece `NO_CANDIDATE`: ninguna variante cumplió conjuntamente sus requisitos de promoción. Sus mejoras parciales, las regresiones negativas de FULL y la alternativa LW siguen siendo resultados útiles. El refit posterior tiene su propia genealogía y no altera esos gates.

## Precisión: ámbito de las afirmaciones

El sistema final obtiene menor error relativo medio que Garl en las dos cohortes EvTTC publicadas, pero no gana en todas las secuencias. R_CTX también supera a la fusión final en varias métricas agregadas. No se elimina ese control ni se selecciona otro checkpoint retrospectivamente para cambiar el candidato según el test.

Las 46 consultas históricas y las 468 consultas extendidas con GT admisible no representan un test oficial ciego. El dataset ya se observó durante investigación; sólo hay TTC positivo en esta evaluación. Las cifras no validan negativos, peatones de cualquier dominio, nuevas cámaras, ROI obtenidas por un detector ni condiciones de despliegue no ensayadas.

Los productores finales usan TRAIN40. Los antiguos P/H/V pasan a formar parte del ajuste final y dejan de ser una validación independiente del modelo completo. Las entradas OOF controlan la exposición de cada grupo en su genealogía; no convierten los diagnósticos sobre TRAIN40 en test externo.

La comparación con Garl conserva pesos oficiales e implementa un contrato documentado, pero no reconstruye una lista oficial exhaustiva de consultas de la Tabla VI. Cada sistema utiliza su contexto nativo. Debe citarse como comparación reproducida bajo este protocolo, no como reproducción exacta de toda la publicación original.

## Latencia: lo que falta medir para un producto

La ejecución local secuencial sobre las 554 consultas favorece a Garl en tiempo medio total. Sus tiempos incluyen entrada/salida y preprocesamiento. La cifra histórica de 200 FPS del artículo y un forward sobre caché preparada no se comparan directamente con esa ruta.

Una medición de despliegue posterior necesita fijar hardware, potencia, precisión, batch, calentamiento, concurrencia, estado de caché, origen de ROI y frontera temporal. Debe separar adquisición, representación, productores, PAIR, contextos y cabeza. Los percentiles de latencia y la capacidad sostenida importan además de la media.

CUDA Graphs sólo se consideran activos donde lo acredita la ruta y su recibo. Los probes parciales, los speedups de tensores preparados y los cambios rechazados no son una promesa de aceleración de la cadena final. Compresión, prefetch y staging pueden desplazar el cuello de botella; eliminar una espera de lectura no garantiza saturar la GPU.

## Recuperación y archivo

El checkpoint cada 100 updates limita el intervalo normal no persistido; no garantiza pérdida cero ante cualquier fallo entre escrituras. Journals, identidades de procesos, RNG, scheduler y estado del optimizador deben acompañar la reanudación. Un checkpoint de pesos sin esos estados sirve para inferencia, pero no equivale a una continuación científica exacta.

Los archivos preservados permiten recuperar resultados y fuentes sin volver a ejecutar entrenamientos. No se ha duplicado todo el medio RAW público de cientos de gigabytes: se conservan revisiones, inventarios y medios derivados necesarios, y el RAW puede descargarse de nuevo si hace falta. La existencia de copias locales no acredita por sí sola que el usuario haya eliminado recursos remotos ni detenido su facturación.

## Siguientes acciones, en orden

1. Conservar el candidato y sus hashes publicados. Mantener separados el ZIP V16 histórico y el ZIP TRAIN40 final.
2. Comprobar las reglas y el historial de envíos de la cuenta antes de enviar. La disponibilidad pública de un historial más largo no demuestra por sí sola su admisibilidad competitiva.
3. Enviar el candidato final sólo con autorización expresa para esa acción externa. Esta publicación Git no es una submission.
4. Archivar el recibo y la puntuación oficial sin escoger nuevas semillas, clips o checkpoints a partir del test privado.
5. Usar la documentación de precisión, latencia, coste y fallos para el informe técnico. Cualquier optimización de despliegue posterior tendrá una prueba de paridad propia y una medición comparable.

No existe aquí una autorización implícita para otra campaña de arquitecturas ni un compromiso de alcanzar SOTA. El siguiente resultado decisivo es una evaluación oficial trazable del candidato final, seguida de una interpretación que respete su modalidad y protocolo.
