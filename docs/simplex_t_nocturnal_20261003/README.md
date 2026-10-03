# SIMPLEX-T: réplica H16 y estudio acotado de precisión/coste

Campaña completa: 24 fits, 60.000 updates científicos y cero fits pendientes.
H16 mejora frente a H8 en las tres seeds de cabeza; el intervalo jerárquico de
las dos seeds nuevas todavía incluye cero. Las tres reducciones de expertos y
los dos agregadores no pasan el cribado prospectivo de precisión. Hay nueve
perfiles de cabeza; el coste integral queda bloqueado por la falta del slot
exclusivo GPU/lectura pesada del propietario. H8 permanece registrado.

Los resultados y estados de entrega se encuentran en
`INFORME_NOCTURNO_SIMPLEX_T.md`, `NEXT_DECISION_NOCTURNA.json` y
`ARTIFACT_MANIFEST.json`. Las tablas se generan desde predicciones, contrastes
y recibos físicos; los números del informe conservan sus fuentes JSON/CSV.

La autorización científica comprende 24 fits y 60.000 updates: seis H16 con
seeds13/23 y dieciocho fits C0 con seed7. H8 conserva su identidad histórica
`TPR-D1-H8-C160`; ninguna alternativa recibe promoción automática. Los controles
H8 y H16seed7 se reutilizan, sin reentrenamiento. Las seeds replican cabezas;
las unidades independientes siguen siendo nueve secuencias.

El código científico histórico es
`0c1a7285b6b5af869b0bf5a13a9629f08997e7a2` y el freeze histórico tiene SHA-256
`2eb3fff9ea24145c4af16c0c8645eb65893144580e41880790e169239fa8f3a7`.
Los protocolos independientes y source pins registran los bindings locales y
las recetas ejecutadas. T6 permanece cerrado. No se abrieron holdouts ni se
entrenaron expertos, encoders o LATENT.

El bundle esencial permanece como artefacto local identificado por SHA-256.
Git contiene código, informes, tablas y recibos; no incluye datos crudos,
checkpoints grandes ni cachés de entrenamiento. La ampliación operativa mantiene
RAM disponible mínima2GiB, RSS propio máximo4GiB y margen10GB después de reservas;
el usuario amplió el presupuesto propio de artefactos a10GB decimales.

Para reproducir la inferencia incluida, extraer el ZIP y ejecutar de forma
secuencial, usando el entorno Python/Torch registrado:

```powershell
python -B '<extraccion>/verify_checkpoints.py' --root '<extraccion>' --output '<recibos>/CHECKPOINTS.json'
python -B '<extraccion>/regenerate.py' --root '<extraccion>' --output '<recibos>/C0_REGENERATION.json'
python -B '<extraccion>/profile_cached.py' --root '<extraccion>' --output '<recibos>/HEAD_COST_REPLAY.json'
```

La entrega completa exige18 cabezas C0 regeneradas y24 checkpoints admitidos.
El bundle también incluye `h16/E_JEPA_TTC_H16_REPLICATION_20261003.zip`, idéntico
al archivo independiente H16 ya regenerado, con SHA-256
`08ee2aae60d56006a73fdb8b249a6467cee0704b0a258bb9ad5659f863b3e348`.
Extraer ese ZIP interno y ejecutar su `regenerate.py` con `--root` y `--output`
para reproducir las seis cabezas H16 nuevas. Son comprobaciones de inferencia:
cero updates de optimizador. Los recibos identifican el alcance comprobado y
separan comprobación nueva de evidencia H16 heredada del mismo archivo.

Los inputs normalizados incluidos permiten reconstruir las predicciones de las
cabezas. No permiten reconstruir datos raw, fuentes TRAIN o expertos desde cero.
El perfilador incluido usa 64 inputs TRAIN por cabeza, sin targets, y mide nueve
cabezas con la misma receta batch1; los tiempos dependen del estado del host.
Una repetición se publica por separado de los perfiles primarios y no sustituye
sus valores ni selecciona el mejor tiempo.
La latencia con entradas preparadas mide la cabeza; la ruta completa requiere
fuentes y un slot exclusivo concedido por el propietario de recursos. Su
ausencia se declara como dependencia externa, sin estimar tiempos inexistentes.
No se afirma reacción AEB más rápida, tracking persistente ni incertidumbre
calibrada. Los contrastes usan OLD_DEV reutilizado y son exploratorios.
H16 consulta aproximadamente 750 ms de pasado del ROI actual; ese horizonte
describe información disponible y no impone 750 ms de latencia por consulta.
Preparar ese contexto puede requerir recalcular features retrospectivas: no se
supone caché gratuita entre consultas ni estado GRU persistente.

Los adaptadores de recuperación están documentados en
`operational/simplex_t_io_recovery/README.md`. Tras alcanzar el presupuesto
científico no se debe relanzar el trainer. La publicación y sus verificadores
pueden reanudarse sin entrenamiento mediante el comando específico documentado.

Los informes del ZIP son una instantánea anterior al push. El recibo de entrega
externo registra la regeneración posterior y el manifiesto local identifica el
archivo final. El commit usa `[skip ci]` para evitar entrenamientos sintéticos
adicionales del workflow general; la QA focalizada y sus recibos se entregan.
Los atributos locales de Git conservan los bytes y hashes de la evidencia.
