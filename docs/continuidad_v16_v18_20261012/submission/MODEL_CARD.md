# PHASE_ONLY_EVENT_S7 — candidato congelado

Estado: preparado, no enviado y sin puntuación oficial. Identidad científica:
V16 publicado en `71ec9d900189a120b06c3956029496e6fd704f6f`; congelación del
12 de octubre de 2026 a las 14:05:52 UTC.

El sistema utiliza eventos y RGB. La cabeza tiene 1.955 parámetros y toma veinte
coordenadas de fases y relojes de los padres V13. Conserva R_CTX como ancla y
añade una corrección aprendida de fase. Los encoders, PAIR, contextos y
normalizadores de los padres siguen siendo necesarios y están identificados
en MODEL_INPUT_FREEZE.json. El número de parámetros de la cabeza no representa
el tamaño ni el coste del sistema completo.

Productores/PAIR originales entrenados en P; refinadores y cabeza supervisados
en H. V fue desarrollo expuesto y se utilizó durante una búsqueda adaptativa.
Las semillas 7/13/23 replican la cabeza con los mismos padres. La semilla 7 fue
la primera registrada; no se eligió por el score test12. No hay refit TRAIN40.

El historial es causal y del mismo objeto: hasta ocho observaciones por modalidad,
soporte admitido de 650 ms y soporte máximo observado de 600.006 microsegundos
event / 600.000 RGB. Se mantiene la ROI y el contrato nativo de cada observación.
Los 6.762 tokens corresponden a 5.805 observaciones físicas: 957 aliases se
deduplican al construir contexto, pero todos reciben salida. Test12 tiene RGB
disponible para las 6.762 predicciones. En transferencia, cuando falta RGB,
el fallback es exactamente E_CTX de la misma ruta.

Salida: TTC firmado en segundos, con conversión nativa de fase y límite de
magnitud de 60 s. Los negativos forman parte válida de la tarea. No se aplican
abs(TTC), selectores por target ni cambios posteriores a la congelación.

La cabeza pasa los criterios internos originales del 2 % en las tres semillas.
En FCWD ofrece mejor MiDc que F_TRUE, pero peor MAE y mean_MiD. Dev32 no muestra
mejora consistente de MiD frente a las referencias propias. Las cohortes
externas usadas aquí no aportan el componente negativo del score oficial.
No se afirma SOTA, calibración de seguridad ni funcionamiento en tiempo real.

Archivo a enviar, sin modificar:
`PHASE_ONLY_S7_GarlTTC_submission.zip`

SHA-256: `60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1`.
La verificación local comprueba formato, tokens, valores y fuentes; no precisión.
La admisibilidad del contexto largo queda por confirmar con el benchmark.
