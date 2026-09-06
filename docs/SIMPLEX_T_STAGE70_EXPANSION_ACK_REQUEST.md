# Petición complementaria al propietario de Stage70

SIMPLEX-T continúa extrayendo D0; esta consulta no solicita detener Stage70 ni
abrir confirmación, grupos protegidos o scores. No reemplaza el ACK original.

He localizado, solo en lectura:

- `artifacts/stage70_76_architecture/expansion_raw_bindings/RAW_BINDING_MANIFEST.json`
  SHA256 `4b7bfab2dc9ff31a6f3c936f487c1456bec1f8fd739d28165acb12e92a9047ba`.
- `RAW_BINDINGS.csv`: SHA256 verificado localmente
  `eedda241de4cd2ce44b73838b66a8efdf04e83d99c63786285dbdbf8adf439bf`;
  27307 consultas,22 grupos. SIMPLEX-T verificó sus 54614 ventanas contra los
  metadatos de entrada, incluyendo identidad, reloj y ROI, sin leer scores.
- `expansion_usability/USABLE_METADATA.parquet`: SHA256 verificado
  `e8514492952a3abd86e45e3b00c07e398d2e6def5b73062eb5b5f8f13a932064`.
- `expansion_inventory/SELECTED_METADATA.csv`: SHA256 verificado
  `ab273a087b4b92e48f0f6657407a520dc033ed7b84c6f80efd9dad1a3cc3aedf`.
  Sus identidades coinciden íntegramente con USABLE_METADATA. El código de
  validación sí consulta TTC, pero no descartó ninguna de estas consultas.
- La carta `time_charter/LABEL_TIME_CHARTER.json` conserva SHA256
  `06a3c8ecb015895729b390a35cb2ec06a34578ed60bb48af33f4f17a492ed568`
  y estado `ORIGINAL_8192_CLOCK_AND_ROI_VERIFIED`.

¿Puedes publicar un ACK complementario con los paths y hashes autoritativos
para importar únicamente identidades, ventanas de eventos, reloj, ROI y sus
dependencias temporales de expansión? Necesito que aclare:

1. Qué grupos/queries de expansión están autorizados para D1 y siguen separados
   de confirmación/protegidos, sin nuevas asignaciones de holdout.
2. Qué documento acredita la conversión de reloj y disponibilidad de ROI de esos
   grupos; la carta original no se extiende automáticamente a ellos.
3. Confirmar los filtros de procedencia: selección prospectiva input-only,
   validación posterior del dominio TTC y etiqueta del segundo frame sin
   rechazos en estas 27307 consultas; RAW_BINDINGS conserva ventanas vacías.
   No se usarán estas tablas como historia primaria de un objeto.
4. Que el reconocimiento no sustituye el preprocessing ni los productores
   históricos A5/C2F/PAIR. SIMPLEX-T ha reverificado las nueve familias internas:
   hashes de 27 checkpoints, contratos efectivos, teacher congelado y calendarios
   PAIR, excluyendo expansión y outer-dev de los conjuntos de entrenamiento.
   Recibo: `artifacts/simplex_t/T0/EXPANSION_ANCESTRY_RECHECK.json`, SHA256
   `f81b2874925ae584a6a96452e1a451364a3f3bff7d95982e559ea0a906021bcc`.
   El teacher externo conserva la limitación sobre posible solapamiento de su
   pretraining web; no se afirma una nueva prueba de independencia por adquisición.

La alternativa autorizada de SIMPLEX-T usa ventanas sensoriales retrospectivas
con la ROI de la consulta actual; no reconstruye asociaciones Garl–eAP. No pido
scores ni resultados para escoger constantes. D0 conserva sus cachés y protocolo;
su replay pendiente requiere coordinar un turno exclusivo mientras Stage70 está activo.

SIMPLEX-T también comprobó el enlace a exposiciones RGB con las columnas de
entrada de Garl y eAP: 27307 consultas, edades de dependencia entre 1003 y
19992 us, ninguna nula. Recibo propio:
`artifacts/simplex_t/T0/EXPANSION_EXPOSURE_TIMING.json`, SHA256
`8061b30e6acead8249e099e3c488558d773cdef7b5d724adaead3417bdc7e63a`.
No acredita la latencia de generación online de las anotaciones, ni reemplaza
la carta temporal del propietario. Se pide el reconocimiento de la procedencia
de estos bindings para expansión, no extender por suposición la carta OLD.

Recibo de correspondencia de entradas:
`artifacts/simplex_t/T0/EXPANSION_INPUT_BINDING_AUDIT_V2.json`, SHA256
`1b141a23abd3ab8c14b48587d9790201c7b4f1f4f6e1d414f4940acec7a25e3f`.
Los 22 archivos raw existen y coinciden en tamaño/mtime con el manifiesto del
propietario. Los hashes completos de esos archivos se atribuyen al propietario;
SIMPLEX-T no afirma haber repetido esa lectura masiva durante el replay D0.

## Ampliación de la petición: controles registrados de cantidad/diversidad

Reconocer también el alcance temporal de las consultas adicionales DENSE_OLD,
sin reasignar grupos: son 21.471 consultas únicas de las nueve secuencias OLD,
con TRAIN de seis secuencias por fold y 14.520/13.473/14.949 consultas respectivas.
Los productores internos históricos siguen excluyendo la secuencia de cada consulta.

Evidencias propias para revisar, sin consultar scores:

- Exposición: `artifacts/simplex_t/T0/DENSE_OLD_EXPOSURE_TIMING.json`, SHA256
  `11fda3e6666c23dd306b667c1f2e0920104ffaf363e34023b1f013d93d96bfcb`.
- Índice: `artifacts/simplex_t/T1/dense_query_context_index/INDEX_MANIFEST.json`, SHA256
  `cb9e51715a71e25128923ccb20c5ec7abf6efc7095de34b001c649ebc12a0850`.
- Arrays: SHA256 `a2a4dacddbdda1a7b408c5360121d26649ec1bca6b37977f49d83af0f28912ce`.

Se verificó paridad exacta de ventanas, ROI, tiempos y máscaras para las 8.192
consultas D0 solapadas, sin descartes en el conjunto denso. Los límites raw se
reutilizaron del índice D0 pinneado; no se releyeron todos los archivos HDF5.
No acredita nueva paridad de inferencia de expertos ni latencia de anotación online.
Por favor, confirma el reconocimiento del contrato para estas consultas adicionales
y una cesión efectiva de GPU/I/O. No se solicita detener Stage70 ni refitar expertos.
