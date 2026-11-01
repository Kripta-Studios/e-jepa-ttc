# Recuperación de la captura interrumpida

El 1 de noviembre se comprobó que habían desaparecido los procesos de captura
y continuación, sin COMPLETE ni FAILED. El último log alcanzaba RGB 1474/5805;
la captura de eventos había terminado. No se atribuye una causa no observada.
Los directorios anteriores se conservan como ejecuciones interrumpidas.

Se permite recuperar exclusivamente fragmentos con CALLS.json y PAIR_INPUTS.npz
completos. Antes de usarlos se verifica cada fuente, modelo y entrada del FREEZE
anterior; cada CALLS debe coincidir con el manifest público por identidad y
metadatos, y los tensores deben ser finitos y tener el inventario esperado.
Los hashes de esos fragmentos se sellan en un inventario nuevo: no se afirma
que existiera un manifiesto final antes de la interrupción.

Se recalculan desde medios originales el primer y último fragmento disponible
de cada secuencia/modalidad y se exige igualdad exacta de todos sus arrays
PAIR133/raw_phase/raw17/expert3/delta antes de admitir reutilización. La
reconstrucción posterior verifica además PAIR/raw17 en todos los fragmentos.
Un centinela que difiera aborta; no permite aumentar tolerancias ni seleccionar
otros ejemplos. Fragmentos incompletos se registran y recalculan. El resto de
observaciones pendientes se obtiene mediante el mismo capturador congelado.

La recuperación se ejecuta en un directorio nuevo. No cambia consultas, pesos,
normalizadores, clocks, históricos, precisión ni protocolos de comparación.
Los tiempos de copiar fragmentos no son tiempos de inferencia. Se conserva la
distinción entre paridad histórica fallida y sensibilidad local. Las mediciones
de latencia posteriores vuelven a ejecutar el predictor desde sus entradas.
