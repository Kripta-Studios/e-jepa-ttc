# Petición complementaria al propietario de Stage70

SIMPLEX-T continúa extrayendo D0; esta consulta no solicita detener Stage70 ni
abrir confirmación, grupos protegidos o scores. No reemplaza el ACK original.

He localizado, solo en lectura:

- `artifacts/stage70_76_architecture/expansion_raw_bindings/RAW_BINDING_MANIFEST.json`
  SHA256 `4b7bfab2dc9ff31a6f3c936f487c1456bec1f8fd739d28165acb12e92a9047ba`.
- Su referencia a `RAW_BINDINGS.csv`: SHA256 declarado
  `eedda241de4cd2ce44b73838b66a8efdf04e83d99c63786285dbdbf8adf439bf`;
  27307 consultas,22 grupos. El payload CSV no se ha importado ni interpretado.
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
3. Qué filtros de elegibilidad produjeron USABLE_METADATA y RAW_BINDINGS: si usan
   TTC, profundidad, velocidad o descarte por eventos, indícalo expresamente.
   No se usarán estas tablas como historia primaria de un objeto.
4. Que el reconocimiento no sustituye el preprocessing ni los productores
   históricos A5/C2F/PAIR. SIMPLEX-T auditará sus propias exclusiones transitivas.

La alternativa autorizada de SIMPLEX-T usa ventanas sensoriales retrospectivas
con la ROI de la consulta actual; no reconstruye asociaciones Garl–eAP. No pido
scores ni resultados para escoger constantes. D0 sigue en ejecución independiente.
