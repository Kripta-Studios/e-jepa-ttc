# Revisión de cierre V18, 16 de octubre

La campaña permanece abierta. Esta revisión identifica evidencia pendiente;
no certifica una entrega final ni modifica recetas, poblaciones o gates.
Todas las rutas de artefactos siguientes son relativas a
`artifacts/joint_measurement_20261012/`.

| Trabajo pendiente | Evidencia necesaria | Estado de esta revisión |
|---|---|---|
| A1 LAST-BLOCK y FULL | `adaptive/A1_adaptation_depth/ENDPOINTS.json`, ambos checkpoints de 12.000 updates, journals durables y estados completos coherentes | LAST-BLOCK activo; FULL posterior; no hay dos endpoints |
| Evaluación A1 | `adaptive/A1_adaptation_depth/evidence/COMPLETE.json`, cuatro tablas P/H completas, métricas por grupo, bootstrap, diagnóstico de correspondencias, coste preparado y dos exports | Registrada e implementada; sin resultados todavía |
| Decisión científica A1 | Informe basado en ambos endpoints; señal frente a FULL separada de admisión frente a A5 y de promoción del sistema | Pendiente de P/H; V no se abre para rescatar un rechazo |
| Coste raw de fusión | `fusion/cost/raw_e2e_millisecond_v1/COMPLETE.json`, 64 consultas originales, cuatro rutas por consulta, tiempos y paridad por fila | 13 consultas auditadas bajo aceptación de 1 ms; faltan 51 |
| Refit y export local condicionados | Decisión explícita desde gates originales; si procede, registro previo, genealogía, features nuevas y evaluación correspondiente | No habilitados automáticamente por una señal de mecanismo |
| Informe y contabilidad final | Conclusión regenerada, registro de experimentos, curvas, grupos, costes con alcance, todos los fallos y límites, `NEXT_DECISION` final | Hay informes parciales; A1 y coste aún abiertos |
| Bundle final | ZIP ≤100.000.000 bytes, SHA-256, manifiesto exacto y replay desde extracción independiente | El ZIP histórico de progreso no cumple la función de entrega final |

La matriz primaria, RGB, C1, D1, B1 y la fusión de tres semillas tienen informes
y evidencia consolidada enlazados en [el índice](README.md). Sus resultados no
se reinterpretan como promoción: la fusión no alcanza el gate crucial del 10 %
y el intervalo agregado frente a V16 incluye cero. La comparación de coste Garl
es CPU FP32 con preprocesado nativo; no es una comparación de precisión oficial
ni un ranking de CPU frente a CUDA.

## Orden de continuación y reanudación

1. Mantener el coordinador A1 vivo: LAST-BLOCK y FULL comparten la receta
   registrada y se ejecutan de forma sucesiva. El runtime conserva caché y
   grafos entre scopes. Una interrupción exige restaurar el checkpoint completo,
   no empezar desde pesos aislados ni reiniciar contadores.
2. Tras ambos endpoints y sin otro escritor GPU, ejecutar:

   ```powershell
   ./scripts/Resume-JointMeasurement.ps1 -Stage A1Evaluation
   ```

   La evaluación valida primero los endpoints, no sólo la existencia de archivos.
   Su [contrato](A1_EVALUATION_CONTRACT_20261016.md) mantiene P/H, costes y exports.
   El modo Auto elige esa etapa al invocarlo; no es un supervisor que la arranque
   automáticamente al terminar el entrenamiento.
3. Completar el coste raw en una ventana exclusiva de GPU. Desde el worktree,
   con el entorno compartido y `PYTHONPATH=src;.`, ejecutar:

   ```powershell
   ../e-jepa-ttc/.venv/Scripts/python.exe -u -B -m operational.joint_measurement.fusion_cost_millisecond run
   ```

   Una salida `PARTIAL` conserva filas y permite repetir el mismo comando. No
   quitar consultas ni sustituir los cuatro caminos. El criterio autorizado es
   absoluto ≤1 ms, con signo y dominio válidos; la salida conserva también el
   resultado del criterio estricto y declara la enmienda posterior al diagnóstico.
   Las 13 filas reutilizadas llevan su origen y hash; no se presentan como nuevas
   mediciones. Véase [FUSION_COST_DIAGNOSTIC.md](FUSION_COST_DIAGNOSTIC.md).
4. Interpretar A1 y registrar la decisión sobre V/refit/export. Como máximo hay
   un refit event y uno multimodal habilitados por evidencia. Si se usa TRAIN40,
   sus grupos H/V pasan a entrenamiento de esos pesos; no cuentan como evaluación
   independiente. La familia de réplica ya utilizada es C1/fusión; no se abre
   otra familia ni un rescate de semillas para obtener una cifra favorable.
5. Con trabajos finalizados y sin escritores, regenerar informes y contabilidad,
   publicar el código e informes autorizados y construir la entrega compacta.
   Verificar la extracción, hashes, tablas completas y exports mediante el replay
   independiente existente. No etiquetar como final un bundle con trabajo requerido
   pendiente. Documentar expresamente los datos y checkpoints de optimizador que
   no caben en el ZIP y las dependencias externas necesarias para reconstruirlo.

## Límites que permanecen

Las cuatro comparaciones adaptativas son C1, D1, B1 y A1. La ampliación de recursos
permite completar el programa registrado; no crea una búsqueda ilimitada sobre
V o FCWD/Dev32. El límite acumulado operativo vigente es 216.000 segundos de
scopes GPU y 1.000.000 updates físicos, con historia y recuperación preservadas.

Los grafos de pérdida han pasado QA exacta y restauración en ambos brazos. La
reducción medida del 40,7 % corresponde al forward/backward preparado, no al
tiempo completo de campaña. Un mayor ritmo no demuestra una mejora del modelo.

No hay score oficial nuevo. Se puede preparar el export local admisible después
de congelar candidatos, pero el envío requiere autorización separada. La falta
de score impide afirmar SOTA; no impide cerrar honestamente los experimentos
autorizados, incluyendo sus negativos y limitaciones.
