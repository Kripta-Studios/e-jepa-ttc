# 4. Cola cerrada, presupuesto y decisiones

Esta autorización nueva se activa al enviar el prompt. No reutiliza el saldo del
techo antiguo. No hay obligación de consumir todo el presupuesto ni deadline de horas.
Un fit parcial es recuperable, no un endpoint comparable.

## M0 / preparación sin entrenamiento

Regenerar la entrega compacta una sola vez; contrastar hashes, source_head, datos/pesos
locales y procesos activos. Ejecutar `scripts/audit_saved_predictions.py` sobre el CSV
publicado. Publicar `native` y `common_signed_cap60` con mismos IDs y sin eliminar colas.
No usar el resultado de ese sidecar para elegir un cap o umbral nuevo.

Descubrir estado R1 y Stage70 por manifiestos/propietarios concretos. R1 puede cerrar
con su propio supervisor mientras M0 prepara cache RGB; las tareas pesadas comparten
un slot. No volver a lanzar su protocolo viejo desde V13 sin resolver sus raíces.

Resolver partición D por metadata; cachear pares de eventos y RGB deduplicados por
identidad completa. Congelar población y modelos antes de puntuar candidatos. La prueba
TRAIN técnica exige forward/backward real, rango, normalización, gradients y cobertura;
no consultar DEV para decidir microbatch.

## B / puente y controles de comparador

B_TRUE, B_REPEAT y B_ZERO, 2.500 updates cada uno, seed7, TRAIN40. H8 y encoder RGB
genérico congelados; se entrenan el readout de pareja y el refiner. Effective batch128,
AdamW lr3e-4, wd1e-3, warmup100, cosine final3e-5, clipping1, FP32, CPU o GPU fijada
antes del primer update con equivalencia admitida. No cambia device al reanudar.

Garl-PHI-CAL-E y Garl-PHI-CAL-ER: 2.500 updates cada uno, seed7, sobre el checkpoint
público correspondiente congelado y TRAIN40. Calibrador diagnóstico simple y fijo:
MLP(2→32→32→1) con LayerNorm/SiLU, entradas fase nativa saneada y flag de dominio;
prior de fase TRAIN, salida canónica. No consulta imágenes extra ni utiliza los targets
en forward. El conversor no debe crashear con TTC nativo fuera del dominio: máscara de
feature, fallback registrado; conservar como baseline el TTC nativo sin tocarlo.
Son **derivados calibrados de Garl**, no Garl publicado y no encoders independientes.
No usar sus pesos en D. Si faltan predicciones TRAIN públicas, generarlas con el loader
nativo ya disponible, por lotes recuperables. No reentrenar Garl completo.

Evaluar endpoints B una vez en EvTTC desarrollo ya abierto, y en TRAIN40 como diagnóstico.
B no decide D, no es confirmación y no bloquea la arquitectura si no mejora.

## D / primera ronda, todos obligatorios si técnicamente viables

| Brazo | Inicialización | Inputs | Updates |
|---|---|---|---:|
| EV_DIRECT_P2@7 | genérica ImageNet | dos ventanas nativas | 24.000 |
| EV_CORR_P2@7 | misma genérica | dos ventanas nativas | 24.000 |
| EV_CORR_T4@7 | misma genérica | cuatro ventanas | 24.000 |

Se entrenan todos los parámetros del encoder/FPN/readout, salvo estadísticas BN fijas.
No reutilizar TRAIN40 task checkpoints. Effective batch32; microbatch4 y acumulación8
como punto inicial. Si no cabe, admisión técnica 2×16 o1×32, fijada antes del primer
update y usada en todos los brazos comparados. GroupNorm/LayerNorm y BN eval evitan
atribuir equivalencia de BatchNorm grande a gradient accumulation.

AdamW: backbone lr1e-4, heads/FPN lr3e-4, wd1e-4; warmup500; cosine a10% del pico,
grad clip1. BF16 autocast del backbone sólo si el hardware/QA lo permite; correlación,
normalización de similitudes, pérdidas y emisión FP32; scoring float64. Alternativa
FP32 se fija una vez antes de entrenar, nunca por resultados. No hay early stopping
sobre DEV. Guardar endpoint final24.000 y comparar los tres después de congelarlos.

E_BASE_P2 = menor MiD de DIRECT/CORR en D, con guardrails y desempate por menor coste.
E_BASE_EXT = T4 sólo si mejora ≥3% frente a E_BASE_P2, cumple guardrails y no viola el
contrato elegido para despliegue; de lo contrario E_BASE_EXT = E_BASE_P2. No se exige
significación para investigar o replicar, pero tampoco se proclama superioridad por eso.
La comparación local extendida y la candidata oficial P2 permanecen identificadas.

La fase R usa un único E_BASE elegido por esta regla y el contrato objetivo admitido.
Si el benchmark oficial permite T4 puede usarse E_BASE_EXT; si no, usar E_BASE_P2.
Se conserva T4 como investigación local aunque no se elija.

## R / RGB propio, no condicionado al éxito de B

R_TRUE y R_REPEAT sobre E_BASE seed7, 12.000 updates cada uno. E congelado; RGB backbone,
FPN/readout/fusión entrenables desde genérico, no desde Garl full. Effective batch32,
misma política BF16/FP32, lr1e-4 backbone y3e-4 nuevos módulos, wd1e-4, warmup500,
cosine10%, clip1.0. Dropout RGB25%, pérdida `L_fused + .25 L_rgb`.

Su comparador de precisión es E_BASE, además de R_REPEAT. R_TRUE no tiene que ganar
para cerrar un resultado: conserva fallos y confirma que la rama RGB realmente recibe
gradientes. Una mejora sólo sobre B o entrenamiento no abre réplicas R.

## Replicación integral

Un candidato E se replica si mejora al menos3% frente a EV_DIRECT_P2 en D, o al menos5%
frente a Garl E público en D, con guardrails. Para DIRECT la primera vía no puede
cumplirse contra sí mismo: necesita la segunda. Garl puede haber visto esos grupos,
por lo que esa vía justifica inversión de réplica, no una conclusión held-out comparada.
Si no hay Garl admisible para ese scoring ni mejora sobre DIRECT, se conservan resultados
sin forzar réplica. La rama R seed7 sigue siendo una pregunta independiente habilitada.
No se usa la derrota de un predictor constante como criterio suficiente de competitividad.

R_TRUE se replica con E si mejora ≥3% en MiD frente al E de su seed, gana al menos60%
de los grupos DEV y cumple guardrails; si sólo mejora el ajuste TRAIN no basta.
La evidencia estadística negativa no se oculta; CI cruzando cero no prohíbe aclarar
optimización con estas dos semillas preautorizadas.

Máximo semillas13/23: 2×(E24.000 + R12.000) =72.000. Cada seed E se entrena desde
inicialización genérica y luego su propia rama R. No son sólo seeds del refiner.
Si R no pasa, se ejecutan sólo los E pertinentes; el presupuesto R no se reasigna.
No rechazar una réplica por ser peor ni sustituirla por una cuarta seed.

## KD / alternativa adicional event-only

Sólo si R_TRUE seed7 pasa su gate, entrenar E_KD y E_EXTRA desde el mismo E seed7,
10.000 updates cada uno, TRAIN D. LRbackbone3e-5, heads1e-4, cosine10%, wd1e-4,
effectivebatch32; teacher congelado. Mismo presupuesto y sampler para ambos.
Si KD mejora al menos2% frente a EXTRA y cumple guardrails, repetir ese par para
seeds13/23 cuyo teacher/E técnicamente válido exista (no se exige que esa seed gane para conservarla). Total máximo60.000. No transformar el resultado
KD en entrenamiento sin RGB; E_extra es la comparación indispensable.

## F / refit y exportación

Elegir arquitectura y receta sobre D, conservar todos los scores. E y ER se seleccionan
**por separado**; no se obliga a usar el mismo ganador si KD sólo mejora E. El refit
principal usa seed7 fijada ahora (no la mejor seed de EvTTC). Entrenar una copia nueva
sobre las 40 secuencias, misma inicialización genérica y receta: E24.000 + R12.000.
KD final10.000 sólo si fue elegido sobre D; su teacher es el R final congelado y
se entrena una copia E separada, sin modificar R ni su padre E.

La réplica integral13/23 se hizo en D; un refit final seed7 no se presenta como tres
refits finales. Prepara artefactos event-only y full separados, model cards con
modalidades TRAIN/inferencia, tests de ausencia RGB y exportador de submission.
Mantener también H8 TRAIN40 original como fallback y referencia, no sobrescribirlo.

Si los candidatos no son competitivos, entregar evaluación negativa y propuesta única;
no fabricar una victoria ajustando la definición del TTC, las filas, los caps o el test.

## Contabilidad máxima (no permiso para otros experimentos)

| Rama | Updates científicos máximos |
|---|---:|
| B: tres puentes | 7.500 |
| Dos calibradores Garl de control | 5.000 |
| D: tres encoders/readouts | 72.000 |
| R: dos ramas RGB | 24.000 |
| Réplicas E/R seeds13/23 | 72.000 |
| KD/EXTRA y réplicas condicionales | 60.000 |
| F: refit E/R y KD seleccionado | 46.000 |
| Total científico máximo | 286.500 |
| Técnica real/sintética | 500 |
| Recuperación/replay conservador | 13.000 |
| **Techo físico** | **300.000** |

Preservar fits parciales y accounting confirmado/incierto. No extender a otro brazo la
reserva que una rama no consumió. No acortar recetas para que entren en un deadline.
