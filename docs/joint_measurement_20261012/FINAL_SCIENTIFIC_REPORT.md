# V18: coste completo de fusión y cierre científico

El programa científico autorizado está ejecutado y termina sin una receta que cumpla todos los gates originales. La entrega no está certificada por este informe: el ZIP y su replay se verifican después, con recibos externos ligados a su SHA-256.

## Coste desde sensores crudos

64 consultas originales de V, cuatro rutas emparejadas y 256 salidas. Se conservan 13 consultas históricas con hash y se midieron las 51 restantes. No se descartó ninguna consulta por su coste o error. Incluye HDF5 event, TAR RGB, ROI, ventanas, historial, productores necesarios, normalizadores, transferencias y TTC firmado.

| Ruta | n | Media s | Mediana s | p95 s | p99 s | Consultas/s |
|---|---:|---:|---:|---:|---:|---:|
| FUSION-TRUE-S7 | 64 | 3.2084 | 3.0123 | 5.6105 | 7.7761 | 0.3117 |
| FUSION-ZERO-S7 | 64 | 3.2509 | 3.2202 | 5.5228 | 7.7412 | 0.3076 |
| F_TRUE | 64 | 7.2856 | 6.5882 | 14.1426 | 17.1799 | 0.1373 |
| R_CTX | 64 | 1.9189 | 1.9650 | 3.8960 | 4.3412 | 0.5211 |

Diferencia absoluta máxima frente a las predicciones verificadas: 0.000793457031 s. Fallos del criterio estricto original: 1. Todas las salidas cumplen dominio, signo y la aceptación absoluta de 0,001 s autorizada después del diagnóstico. No se afirma igualdad bit a bit. Esta enmienda afecta a la admisión de coste, no a las predicciones científicas ni a sus gates.

Son tiempos warm con el estado de cachés declarado por el runtime; no demuestran lectura física fría desde disco. Carga, primer forward y cabeza preparada se conservan por separado en los recibos. Los milisegundos de una cabeza o del forward preparado no describen estos segundos del pipeline completo. TRUE resulta más rápido que F_TRUE en esta medición y más lento que R_CTX. ZERO conserva el pipeline del control de igual capacidad; no representa un RGB mínimo optimizado.

Para reproducir la aritmética histórica se conserva el tamaño de batch original de cada productor, repitiendo la consulta cuando corresponde y cobrando todo ese trabajo. No equivale al mínimo coste posible de una implementación batch1 rediseñada. Las cabezas son CPU FP32 y los productores event/RGB usan CUDA, con las transferencias incluidas. Las rutas y el hardware son los declarados en REGISTRATION.json; no se extrapolan a otro dispositivo.

## Qué queda demostrado y qué no

- La matriz primaria entrenó realmente los encoders DIRECT/CORR y mantuvo el control congelado. NATIVE fue comparado como sistema y no obtuvo promoción.
- RGB P2 mejora regiones críticas pero conserva regresiones negativas y de signo, también en P. No se abrió V para rescatar sus recetas rechazadas.
- C1 aporta una lectura geométrica útil frente al readout libre en H, aunque no resuelve todas las bandas ni el MAE. Su uso como complemento event se investigó por separado; no fue una promoción standalone retrospectiva.
- D1 no respalda aumentar la anchura como solución en esta receta. B1 aporta señal frente al control TTC de igual número de updates, pero no pasa la banda negativa.
- A1 FULL supera LAST-BLOCK en las métricas puntuales H. LAST-BLOCK retiene mejor la correspondencia conocida, pero eso no produce mejor TTC. FULL cumple signo y falla la protección negativa frente a A5. Véase [A1_RESULTS.md](A1_RESULTS.md).
- La fusión tiene señal reproducida frente al control ZERO en tres semillas. No alcanza el gate crucial original del 10%; el intervalo agregado frente a V16 incluye cero. No se establece superioridad final frente a V16.
- El patrón en entrenamiento hace razonable investigar pérdida, masas, salida y encoder conjuntamente. No hemos aislado causalmente una nueva pérdida en A1, ni demostrado que el dataset sea la única limitación.

C1/D1/B1/A1 agotan las cuatro comparaciones adaptativas. La única familia de réplica autorizada fue C1/fusión. No hay receta final admitida que habilite refit event o multimodal ni un nuevo export test12 V18. No se consumen H/V como entrenamiento adicional para fabricar un candidato. La decisión explícita está en `DECISION_AFTER_A1.json`; los históricos V16 y V17 permanecen preservados.

## Coste acumulado y reproducción

183,200 updates científicos, 858 técnicos y 273 de recuperación superior: 184,331 físicos superiores, con 0 pendientes. Los científicos incluyen 200 updates de diagnósticos de runtime sobre P. Los controles reutilizados no se suman otra vez.

Scopes GPU propios: 196739.722 s (54.650 h), sobre 216000 s autorizados. Se retienen siete reservas históricas de 900 s como cotas conservadoras: no equivalen a procesos vivos. Incluye carga/guardado y evaluación dentro de scopes; no es tiempo puro de kernels. La actividad CUDA accidental del CI se declara aparte en su informe, sin presentarla como pruebas CPU ni como fits V18.

El ZIP esencial incluye código, informes, curvas, cohortes completas, recibos y pesos portables. Excluye sensores/cachés, padres externos y estados completos de optimizador, que permanecen locales y enumerados. El replay recalcula métricas y ejecuta ejemplos portables; no reentrena ni reproduce todos los eventos crudos. Los testigos cuyos payloads se omiten se enumeran como no reproducidos.

P es entrenamiento; H y V son desarrollo, V ya expuesto. Las tres semillas comparten padres históricos. Los exports de cabeza requieren productores. Garl conserva su preprocesado y comparación descriptiva, sin intercambiar sus cifras oficiales con nuestros grupos. No hay score oficial nuevo ni afirmación SOTA.

Regenerar en el entorno documentado: `python -B -m operational.joint_measurement.final_report`. Exige quiescencia y verifica bindings, decisiones, población/paridad/coste y contabilidad; no realiza updates. El empaquetado y replay posterior se describen en [DELIVERY_REPRODUCTION.md](DELIVERY_REPRODUCTION.md).

Evidencia del programa SHA-256: `c3de3b1eff301aafb58028c2ca0b59ec05c91ed26d94df78a99a451e90da0e3e`.
Coste completo SHA-256: `bc19f28d6d334264fb48c2d30e830b790946ce8201b397c8860f4ac6b2178f55`.
