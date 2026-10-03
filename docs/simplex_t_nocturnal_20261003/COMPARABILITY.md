# Comparabilidad: inventario acotado N0

El inventario reutiliza exclusivamente resultados de desarrollo ya publicados y verifica sus archivos contra hashes del cierre o manifiestos históricos. No entrena, no ejecuta forwards, no reabre T6 y no importa Torch.

La población verificada tiene 8.192 queries, nueve secuencias y folds 0/1/2. Se exige igualdad exacta de token/sequence/track/fold, target float64 y masa: nueve secuencias macro y buckets con masas 0,5/0,3/0,1/0,1. MiD es 10.000 veces el error absoluto PHASE17 firmado. Se conserva cobertura completa; no se comparan scores selectivos con scores completos.

| Familia publicada | MiD | Admisión |
|---|---:|---|
| SIMPLEX_CURRENT_A5_NESTED | 162.209480368 | mismo contrato |
| SIMPLEX_CURRENT_C2F_NESTED | 158.942028615 | mismo contrato |
| SIMPLEX_CURRENT_PAIR_NESTED | 158.953683016 | mismo contrato |
| TPR-D1-H1-C160@7 | 132.831619143 | mismo contrato |
| TPR-D1-H1-C160@13 | 133.378988372 | mismo contrato |
| TPR-D1-H1-C160@23 | 133.059346588 | mismo contrato |
| TPR-D1-H8-C160@7 | 121.646332258 | mismo contrato |
| TPR-D1-H8-C160@13 | 121.840952046 | mismo contrato |
| TPR-D1-H8-C160@23 | 121.864780635 | mismo contrato |
| TPR-D1-H16-C160@7 | 118.865738910 | mismo contrato |
| SELECTOR-D1-H8-C160@7 | 143.503730240 | mismo contrato |
| RISK17 | 146.039451139 | mismo contrato |
| SIMPLEX17 | 144.021186529 | mismo contrato |
| CURRENT_MEDIAN | 157.954041218 | mismo contrato |
| EWMA_0P3S_H8 | 159.278795431 | mismo contrato |
| OFFICIAL_V7_A5 | 158.448579309 | evidencia insuficiente |
| GARL_LOCAL_FROZEN | 144.353027166 | evidencia insuficiente |
| OFFICIAL_V7_C2F | 158.573140450 | evidencia insuficiente |
| HISTORICAL_NESTED_ROUTER_A5 | 162.199841801 | evidencia insuficiente |
| HISTORICAL_NESTED_ROUTER_C2F | 158.924561891 | evidencia insuficiente |
| PROSPECTIVE_V8_ROUTER_R | 153.876799517 | evidencia insuficiente |
| STAGE70_PUBLISHED_DEVELOPMENT_CANDIDATES | ausente | evidencia insuficiente |

Las filas H1/H8/H16 y controles SIMPLEX comparten población, contrato de evaluación y productores congelados; cambian deliberadamente contexto o corrección. Esto permite comparación local de precisión, pero no convierte el control actual/H1 en el mismo sistema de información temporal que H8. H16 seed7 sigue siendo exploratorio observado. Seeds nuevas no son escenas nuevas.

Los expertos actuales registrados son los outputs de la observación actual compilada utilizada por H8, no las tablas históricas nested originales. Sus original_cost se verifican contra expert_phase/target_phase sin reconvertir el TTC FP32. A5/C2F oficiales V7 son productores distintos; los MiD 162,199841801/158,924561891 del nested histórico y 158,448579309/158,573140450 oficiales no se intercambian. PAIR comparte encoder A5 y no se cuenta como segundo encoder independiente.

A5/C2F oficiales y Garl local: se verificaron físicamente hashes de predicciones publicadas, población completa y score. La genealogía, privilegios de información, ROI/availability y selección completa no se revalidaron con esta inspección limitada. Por ello quedan como evidencia insuficiente para declarar un ganador de sistema, aunque sus cifras locales estén sustentadas. Garl no queda automáticamente clasificado como RGB o privilegiado: esa propiedad falta en la evidencia inspeccionada.

El router V8 y constituyentes nested históricos tienen aquí evidencia documental explícita; no se ejecuta otra arqueología para convertirla en admisión completa. Sus scores no se presentan como recomputados en N0.

Stage70: el directorio compartido sólo contiene ACK/revisiones de interfaces; no se encontró en ese índice una publicación de candidato de desarrollo con score y contrato. No se accedió a resultados privados, avance de fits ni confirmación. La fila queda ausente/insuficiente, sin bloquear experimentos independientes.

Ninguna latencia de sistema se ha inventado. Las celdas vacías significan no medido. N4 debe añadir medidas con scope explícito; una cabeza aislada no acredita sensor-to-decision, ahorro de expertos ni caching reutilizable bajo cambios del ROI actual.

Este inventario no selecciona ni promueve modelos: los comparadores primarios prospectivos permanecen los registrados para H16/N2/N3. La mejor cifra de otra cohorte o de un contrato incompleto no se convierte en ganador comparable.

Rutas absolutas y SHA-256 de cada evidencia figuran en BASELINE_REGISTRY.csv y BASELINE_EVIDENCE.json. Los hashes de población son una serialización específica de N0, descrita en el recibo; no reemplazan los hashes históricos científicos.
