# Estudio posterior de SIMPLEX-T

Esta ejecución implementa el plan autorizado después del cierre: diagnóstico de
predicciones congeladas, admisión acotada de comparadores y tres bloques de coste
de cabeza. Autoriza cero fits y cero actualizaciones de optimizador. H8 conserva
su identidad histórica; los resultados nuevos no abren confirmación ni holdouts.

Desde el worktree original, con el entorno histórico:

```powershell
$env:PYTHONUTF8='1'
..\e-jepa-ttc\.venv\Scripts\python.exe -B operational/simplex_t_post_campaign/run.py diagnose
..\e-jepa-ttc\.venv\Scripts\python.exe -B operational/simplex_t_post_campaign/run.py profile
..\e-jepa-ttc\.venv\Scripts\python.exe -B operational/simplex_t_post_campaign/deliver.py results
```

El perfilado reusa cada fragmento completo de 25 medidas tras validar protocolo,
runtime, pesos, inputs y queries. Rechaza bindings distintos. No selecciona otro
checkpoint ni repite el entrenamiento. `prepare` se reservó a la admisión inicial;
para recuperar esta ejecución se usan los comandos anteriores y su protocolo
persistido, no una selección de «la última campaña».

El bundle usa `regenerate.py --root . --output REGENERATION.json --heads` para
reconciliar pérdidas, estratos, tiempos archivados e inferencia de nueve cabezas.
`--create-reference` sólo pertenece a la construcción del bundle y rechaza
sobrescribir referencias existentes. No debe usarse para verificar una entrega.

La ruta integral permanece bloqueada mientras la interfaz compartida no conceda
un slot exclusivo GPU/lectura pesada. `segmented_route` es una interfaz probada
con fixtures, condicionada a ROI externo; no integra por sí sola productores
reales ni constituye una medida de eAP o de reacción AEB.

`route_policy.execute_producers` limita los callbacks al conjunto registrado de
cada cabeza, reutiliza la codificación A5 para PAIR y rechaza dependencias de
historia/validez/normalización en un productor excluido. Su prueba con fixtures
no certifica una ruta raw real: esa integración requiere admisión y bindings
canónicos. La disponibilidad upstream heredada de una caché completa no se
convierte automáticamente en disponibilidad independiente de un sistema reducido.

El diagnóstico inicial conserva el recibo acotado sin aliases. La entrega final
resuelve aliases CSV mediante `deliver.py results` y publica
`COMPARATOR_REVIEW_RECONCILED.json`, que sustituye esos flags preliminares.
