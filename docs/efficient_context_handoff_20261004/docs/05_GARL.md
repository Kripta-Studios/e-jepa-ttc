# 5. E3 — comparación Garl con procedencia y contexto verificables

## Referencias externas
La búsqueda del 4 de octubre de 2026 recuperó Garl/eAP como referencia directa.
No estableció otro ganador publicado estrictamente comparable de TTC de objetos
en eAP; no es una demostración de que no exista. El paper reporta 66,2 MiD para
event-only+LHR y 45,0 para el modelo RGB+event completo en su test. También publica
4,55 ms para sus módulos ONNX en Orin NX. Ni esas cifras se comparan con 119 OLD_DEV,
ni esos milisegundos con segundos de replay HDF5 en un portátil.

La configuración oficial `event_lhr.yaml` utiliza ResNet50, 40 canales, resolución
128×128 y dos frames; 50 épocas, Adam con lr 1e-3, wd 0 y reducciones en épocas 10/20/30/40
con gamma 0,5. Batch 128. Usar el upstream fijado en el commit
`256661242b8a7f5e56aa3c1c02348b30f6e89de6`, verificando también el blob de la
configuración antes de entrenar.
El encoding nativo de Garl no se sustituye por los voxels de A5 para forzar igualdad.

EV-TTC (IEEE, 2025) es una referencia de TTC denso con otro dataset y representación;
reporta 9,5 ms en Orin NX. REACT (arXiv:2609.19204, septiembre) reporta 9,59% de error
relativo en EvTTC y 4,6 ms, sin ROI. Aquí se recuperó su abstract, no una reproducción
ni todo su protocolo. Son referencias de sistemas, no scores intercambiables con
MiD. No implican por sí solos mejor frenado que un AEB comercial.

## E3a — recuperar lo ya existente, sin otra arqueología indefinida
Reutilizar `COMPARATOR_REVIEW_RECONCILED`: consultas y folds sí coinciden. Los
últimos bits de serialización de targets no constituyen otra población. Garl local
144,353027 y C2F oficial 158,573140 son referencias descriptivas con contrato incompleto.
Faltan vínculos de TRAIN, selección de checkpoint, ROI, modalidad y disponibilidad.
Hacer una sola revisión acotada de los paths manifestados y reportes V5/V7.

Consultar el estado real de Stage70/Stage76 y reutilizar productores compatibles.
No volver a entrenarlos porque otro agente no dejó su estado en el README raíz.
Si un comparador sigue incompleto, registrar qué falta y continuar E1/E2.

## E3b — nuevos productores Garl sólo donde falten los admisibles
Autorizados hasta 3 outer y 9 inner, seed 7. Cada outer excluye sus grupos OLD; cada
inner excluye tanto el outer holdout como su inner holdout. Pool D1 permitido del
fold, sin añadir test. El objetivo es una comparación local, no reproducir el paper
con un conjunto de entrenamiento distinto y llamarlo réplica exacta.

Mantener arquitectura, encoding y pérdida LHR del upstream. Usar las etiquetas
geométricas TRAIN que requiera su receta y declararlas; nunca en el forward.
Congelar todos los endpoints de los productores antes de la primera evaluación
OLD_DEV de esta familia. No inicializar desde checkpoints Garl entrenados en eAP que hayan visto los holdouts.
La configuración apunta a `paper_event_only_lhr.pth`: auditar y deshabilitar cualquier
carga de esos pesos. Toda inicialización externa debe tener procedencia permitida.

Endpoint fijo: época 50, sin elegir checkpoints con OLD_DEV. Antes del primer fit,
perfilar una microbatch viable usando datos sintéticos. Intentar128; si no cabe,
escoger el mayor tamaño de [64,32,16,8] compatible con la reserva de recursos.
Acumular gradientes hasta batch efectivo128. **BatchNorm a microbatch 8 no equivale a
BatchNorm a 128**: declarar la adaptación hardware, no una réplica exacta. Si 8 no
cabe, bloquear esta rama; no bajar resolución o capacidad para fabricar una baseline
débil. El perfil, el estado de BN y la elección quedan fijados antes del entrenamiento.

Calcular los updates exactos de 50 épocas de todos los fits, considerando filtros,
últimos batches y acumulación, antes de ejecutar. Máximo conjunto 200.000. Si excede,
entregar el presupuesto preciso; no truncar 50 épocas y evaluar un comparador a medio
entrenar. Mantener un único trainer GPU pesado sin interferir con Stage70.

La caché densa de todos los ejemplos Garl puede exceder el presupuesto de disco.
Usar lectura acotada y caché LRU de shards; no materializar silenciosamente decenas
de GB. Registrar coste de extracción. Esta fase puede ser mucho más larga que
entrenar cabezas, por lo que no se promete completarla en una noche.

## E3c — el mismo contexto también para Garl
Construir Garl-H1 y Garl-H8: GRU160 y corrección en fase con la misma familia de
cabeza y receta de SIMPLEX-T, `lambda_cost=0`. Tres features por observación:
fase Garl, log1p del conteo y log1p de la tasa de eventos del ROI. Añadir los mismos
cuatro tiempos y máscara. Anchor: fase Garl actual. No incorporar A5/PAIR en esta
baseline. Usar el encoding nativo de 40 canales para producir Garl y declarar
exactamente de qué ventana procede el conteo común.

Seis fits H1/H8 × tres folds, seed 7, 2.500 updates. Predicciones de entrenamiento
INNER-OOF; evaluación con productor OUTER correcto. Normalización exclusivamente
TRAIN. No mezclar predicciones OOF provenientes de modelos que hayan visto el
holdout actual. No reutilizar embeddings de coordenadas no alineadas como solución
rápida a la preparación de las features.

Comparar H8 completo con Garl-H8 es comparar sistemas de representaciones distintas
con contexto, datos y privilegios declarados. No aísla únicamente la GRU. Dar además
Garl-H8−Garl-H1 y Garl-H1−Garl nativo para distinguir contexto y recalibración.
Declarar la supervisión RGB/DINO de los antiguos expertos aunque su inferencia sea
event-only. «Mismo contexto» debe verificarse por timestamps/ROI, no sólo por el
nombre H8. Si las ventanas nativas no admiten igualdad exacta, publicar la diferencia
y una comparación de presupuesto de información, no inventar equivalencia.

## Evaluación y confirmación
Tres ámbitos: precisión en OLD_DEV con la misma métrica y masas; módulos con inputs
preparados en el mismo hardware/runtime; ruta sensores+ROI de ambos sistemas.
No transplantar el preprocesado A5 a Garl. Reportar parámetros, BN, microbatch,
span de eventos, disponibilidad del ROI, teachers y grupos de entrenamiento.

No abrir Stage76, public validation, private test, EvTTC test ni CodaBench. Preparar
una única confirmación coordinada para una autorización posterior, con candidaturas,
comparadores y endpoints fijados. Si no existe un grupo verdaderamente no expuesto,
documentarlo: no llamar «nuevo» a D1 ni a OLD_DEV. Una mejor cifra local sobre datos
adaptativamente reutilizados no demuestra SOTA oficial.

Si Garl con contexto supera al sistema actual, conservarlo como candidato de
sistema: el objetivo es resolver mejor la tarea, no mantener A5/PAIR por identidad
del proyecto. Esa selección seguiría siendo desarrollo hasta una confirmación
independiente y no cambia la identidad histórica de H8.
