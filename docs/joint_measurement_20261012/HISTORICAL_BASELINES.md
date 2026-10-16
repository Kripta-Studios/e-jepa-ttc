# Comparadores históricos conservados en V18

La matriz V18 utiliza A5-P2 como control de productor, E_CTX como sistema event
completo y R_CTX/F_TRUE como referencias RGB y multimodales. Sus identidades,
normalizadores y poblaciones están ligados a las admisiones previas. V16
PHASE_ONLY se contrasta por separado sobre V, sin convertir ese contraste
descriptivo en un criterio nuevo para seleccionar V18. La comparación de medias
de tres semillas no demuestra superioridad agregada de V18 sobre V16.

## Garl y coste ya ejecutado por V16

El 16 de octubre se incorporaron nueve archivos históricos desde la publicación
de continuidad V16. Cada archivo se contrastó en tamaño y SHA-256 contra su
`PUBLICATION_MANIFEST.json`, y la copia conserva los mismos bytes. Se recalcularon
los cuatro resúmenes de coste Garl desde sus 128 mediciones: n, disponibilidad
RGB, p50, p95, p99 y media. Diferencia máxima: 1,82e-12 ms. No se repitió inferencia
ni entrenamiento y no se consultaron etiquetas para elegir una receta V18.

Los archivos están en `references/V16_native_cost` dentro de la campaña y se
incluyen como metadatos/tablas en el ZIP compacto. El recibo
`IMPORT_RECEIPT.json` tiene SHA-256
`2a13ba6c6a5912fa21e9ae6ef447a0f8d57c448e4cd5336d950083c39cb626f6`.
El manifiesto histórico declara el commit fuente
`71ec9d900189a120b06c3956029496e6fd704f6f`.

| Evidencia reutilizada | Qué prueba | Qué no prueba |
|---|---|---|
| Registro de candidatos event/full/visual | Identidad y modalidad de los comparadores conservados | Procedencia TRAIN certificada para una nueva evaluación held-out |
| Binding Garl full y fallback event | Pesos completos, carga estricta y representación nativa; no voxels A5 | Que un modelo V18 sea mejor |
| 128 costes Garl sobre 64 consultas, dos pasadas | Ruta CPU FP32, un hilo, con inputs nativos previamente optimizados | Paridad GPU o un benchmark nuevo V18/Garl |
| 128 controles residentes Garl | Paridad bit a bit en aquella ejecución CPU | Igualdad en otro batch, dispositivo o precisión |
| 256 rutas PHASE native/optimized | Coste del adaptador CPU histórico, con sus controles | Coste del nuevo pipeline C1/RGB de V18 |

La ruta full conserva 46 canales: dos RGB de tres canales y dos ventanas event
de veinte, con sus tiempos y normalización propios. Cuando falta RGB se conserva
el fallback event nativo. El perfil de Garl se ejecutó después del de PHASE y
con V18 y el escritorio activos; la carga entre perfiles no estuvo controlada.
No se infiere una ventaja causal de latencia a partir de esas cifras.

El registro conserva la evaluación histórica full/visual FCWD y la limitación
de geometría independiente pendiente. Es transferencia ya expuesta; no autoriza
ajustar V18 con FCWD/Dev32 ni convierte el preprocesado registrado en calibración
extrínseca certificada. No se ha repetido esa campaña ni sustituido Garl por una
implementación reducida.

La petición de coste nativo y optimizado debe conservar esta distinción: el
perfil reutilizado mide Garl nativo con preprocesado previamente optimizado, no
dos variantes nuevas de Garl medidas aquí. Tampoco completa el coste sensorial
de fusión V18, que sigue sin admisión por su discrepancia numérica documentada en
[FUSION_COST_DIAGNOSTIC.md](FUSION_COST_DIAGNOSTIC.md).

## Comparación oficial

Las predicciones locales test12 de V16 permanecen congeladas y sin puntuación
oficial. No se han enviado al servidor. V, FCWD y Dev32 no sustituyen ese test;
los resultados internos no acreditan SOTA. Un eventual refit V18 requiere una
selección admitida y una receta congelada antes de consumir el pool TRAIN.
