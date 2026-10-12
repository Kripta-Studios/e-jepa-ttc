# V18: decisiones anteriores a los primeros updates

Autorización del usuario: «haz todo lo del zip no pares». Se ejecuta el programa
recibido, con techo nuevo de 1.000.000 updates físicos y 64.800 segundos en scopes
GPU propios. V17 queda cerrado. No se autorizan publicaciones, submissions,
pagos ni modificaciones de otras campañas.

## Contratos fijados

- Matriz primaria y recetas: `configs/joint_measurement/primary.json`.
- Comparador común para admisión exploratoria: A5 matched P2 congelado, medido
  con las mismas ventanas y población. Se publican además todos los contrastes
  CORR/DIRECT, CORR/FROZEN y NATIVE/A5-P2. El control congelado no bloquea CORR.
- Cobertura completa, signo (+0,5 pp máximo) y cada banda (+10% máximo) son
  requisitos conjuntos para cualquiera de las tres vías exploratorias del ZIP.
- Máximo dos candidatos de integración, ordenados por agregado, MiDc e ID.
  La promoción final mantiene los gates numéricos V14 contra E_CTX matched V,
  además de coste E2E comparable; nunca equivale a confirmación oficial.
- H/V mantienen su condición de desarrollo expuesto. Ningún resultado externo
  se usa para escoger entre recetas de esta campaña.

## Correcciones previas a entrenamiento

La interpolación bilineal del volumen coarse se divide por la interpolación de
su soporte. La máscara de soporte sigue siendo la registrada, sin extenderla.
Así un descriptor constante no pierde similitud por mezclarse con padding cero.
La regresión reproduce 416 valores afectados en la referencia y verifica que
la implementación corregida conserva coseno uno en todas las posiciones válidas.
Es una decisión prospectiva V18; no altera ninguna salida V17.

DIRECT construye las mismas máscaras geométricas sin evaluar correlaciones de
features que después anularía. Su arquitectura y pesos iniciales siguen siendo
idénticos a CORR. La política de inicialización, datos, dropout y pérdida no cambia.

P2 conserva el sufijo real de T3 y ambas anotaciones de geometría válidas. No se
arrastra la máscara histórica que invalidaba el t0 proxy excluido. La ruta nativa
conserva pesos de pérdidas y DINO P-only; los modelos de par no reciben teachers.
Se comprueba igualdad de la reducción efectiva y top-k global. El checkpoint de
activaciones preserva RNG y gradientes del dropout con `use_reentrant=False`.
Referencia API: https://docs.pytorch.org/docs/stable/checkpoint.html . El conector
Context7 no estaba disponible; se consultó la documentación primaria de PyTorch.

## Datos y coste

Los packs nuevos son copias lossless de entradas sensoriales FP32, no embeddings
congelados. Cada registro conserva sus tokens, ROI, delta, target y teacher aparte.
Se verifican hashes de origen y replay exacto de serialización antes de publicarlo.
Sólo el forward de eventos y delta entra en la predicción. El muestreo sigue siendo
uniforme con reemplazo, pareado por seed; N*w no se renormaliza por minibatch.

Preparación CPU y lectura de documentación no cuentan como GPU. QA con updates
sintéticos sí se registra en el ledger técnico. No se rebaja batch, precisión,
población ni endpoint ante falta de recursos; se guarda estado y se reanuda.

## Campañas ajenas

V16 mantiene inferencia test12 congelada viva al inicio; no se modifica ni se
duplica. Sus resultados se leen posteriormente como comparadores con procedencia.
Stage70 sí tiene endpoints reales S72-DIRECT/PPS-PAIR/PPS-TRAJ de 8.000 updates.
El ejemplo PPS-PAIR outer0 usa seis grupos TRAIN y tres excluidos, no los roles
P24/H8/V8 ni el padre A5 matched de esta campaña. No se presenta V18 como primera
investigación histórica de pares ni se reutilizan esos pesos como padres P-only.
Su arquitectura/procedencia se conserva en el inventario antes de nuevos fits.
