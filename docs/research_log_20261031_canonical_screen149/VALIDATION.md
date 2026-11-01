# Validación del screen

49 tests correctos; Ruff check sin errores; Pyright con cero errores.
Ruff format pasa para los ocho archivos nuevos. La comprobación ampliada
señaló formato pendiente en tests/unit/test_phase_canonical_pair.py, idéntico
al cierre f1f9c8ed tras normalizar finales de línea. Se conserva el recibo
FAIL inicial y el recibo PASS_CHANGED_SCOPE; no se alteran fuentes históricas.

El único lote rechazado original es event:410 en Slider-500:4. La captura
termina en E_A5_MATCHED antes de invocar PAIR; la intervención no cambia ese
productor. No es un rechazo nuevo del candidato.

La referencia exacta de este screen es expanded149 y la ejecución original
actual. Los estados nativos históricos de training/POPULATION.npz no son
idénticos: se observó una diferencia máxima de 0,024908066 s en el contexto
de eventos de Slider-500:2. Sus estados nativos no se usaron como control de
las métricas del screen; solo se reutilizaron geometría, calidad y, después
del sello, targets. No atribuir aquella deriva histórica a CanonicalPair.
