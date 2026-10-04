Perfilado autorizado completado: 1.728/1.728 mediciones, 64 consultas TRAIN, nueve rutas y tres bloques. Se conservaron sin cambios los 1.114 fragmentos anteriores. Cero actualizaciones de optimizador.

La extracción independiente verifica 4.929 miembros del manifiesto. Se regeneraron 576 entradas de cabeza y 27 filas de coste con diferencia cero.

Lectura y preparación del contexto acumulan aproximadamente el 85–86 % del tiempo de H8 y el 90–91 % de H16. Las cabezas representan menos del 0,5 % en ambos bloques calientes. El resultado orienta la propuesta posterior hacia raw/ROI; esa optimización no se ejecuta aquí.

Los tiempos corresponden a GPU compartida, ROI suministrado y modelos residentes: no son reacción AEB ni un benchmark aislado. H8 conserva su registro y las conclusiones de precisión anteriores. No quedan entrenamientos ni mediciones pendientes en esta cola.

[Síntesis completa y tablas](https://github.com/Kripta-Studios/e-jepa-ttc/blob/simplex-t-local-results-20261004-complete/docs/simplex_t_results_20261004_complete/README.md).

Esta release añade el bundle completo de la ruta y sus recibos. [Los cinco bundles históricos y la entrega parcial preservada](https://github.com/Kripta-Studios/e-jepa-ttc/releases/tag/simplex-t-local-results-20261004) siguen disponibles; RELEASE_ASSETS.json enlaza cada archivo con su SHA-256.
