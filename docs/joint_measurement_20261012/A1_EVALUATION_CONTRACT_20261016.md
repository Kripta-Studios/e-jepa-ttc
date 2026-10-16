# Evaluación A1: adaptación parcial frente a completa

La evaluación está implementada para los dos endpoints de 12.000 updates:
`A1-LAST-BLOCK-S7` y `A1-FULL-S7`. Su registro se fija antes de terminar el
entrenamiento y antes de generar predicciones A1 en H. Este documento describe
el contrato; **no contiene resultados A1 ni una promoción del sistema**.

Registro: `adaptive/A1_adaptation_depth/evaluation/REGISTRATION.json`, SHA-256
`fe5e1cc6696e941b31fc438c9a4ad7f6d6c90ff5bf427f456ced85b673e410b3`.
Se publica una copia exacta en `evidence_A1_persistent_20261016`.

## Qué compara

Ambos brazos parten del endpoint P de correspondencias de B1. LAST-BLOCK permite
actualizar únicamente el último bloque del encoder y el predictor; FULL permite
actualizar todo el encoder y el mismo predictor. Conservan el objetivo, las
masas, el sampler y el presupuesto registrados. El contraste investiga si una
adaptación limitada conserva mejor las correspondencias aprendidas. No demuestra
de antemano que adaptar menos sea mejor.

El evaluador separa tres preguntas:

1. Si LAST-BLOCK aporta una señal frente al control FULL, con los límites
   registrados de agregado, crítico, otras bandas, signo y cobertura.
2. Si cualquiera de los dos satisface la admisión de desarrollo frente a
   A5-P2-FROZEN, incluyendo la ruta de coste preparada.
3. Si existe evidencia para promover un sistema completo. El evaluador no hace
   esa promoción: requiere una decisión posterior y sus comparaciones propias.

Los criterios proceden del registro científico A1. Una señal del mecanismo no
anula las restricciones frente a A5; tampoco las anula una gran aceleración.
Los candidatos admisibles se ordenan por agregado, MiDc e identificador, como
se registró antes de entrenar. No se evalúa V en este módulo.

## Evidencia obligatoria

- Ambos endpoints completos, con checkpoint y versión durable coincidentes,
  hashes, identidad, journal y cero updates pendientes.
- Reconstrucción de los 384.000 IDs de muestreo de cada brazo; comprobación de
  pasos y momentos de Adam, scheduler, sampler y longitud de las curvas.
- Comparación de los pesos reales con el padre: debe cambiar el último bloque
  y el readout, y ningún tensor del prefijo congelado de LAST-BLOCK.
- Predicciones completas en P y H de ambos brazos, conservando los relojes,
  poblaciones y scorer originales. Los fragmentos ya verificados son reanudables.
- Recálculo independiente de métricas y verificación de las tablas A5 reutilizadas.
- Bootstrap emparejado de 2.000 remuestreos por grupo de adquisición: cada brazo
  frente a A5 y LAST-BLOCK frente a FULL, tanto en P como en H.
- Diagnóstico de las mismas 24 consultas P con transformaciones conocidas,
  comparando el padre de correspondencias y los dos endpoints. No se interpreta
  como precisión de correspondencias temporales reales.
- Coste emparejado sobre las 128 consultas P y 20 warmups del protocolo existente.
  Mide entrada event preparada en GPU hasta TTC; **excluye sensores y transferencias**.
- Dos exports portables con las cinco entradas P predefinidas, incluidos los
  resultados negativos. Paridad eager exacta y TorchScript con la tolerancia
  ya utilizada por el adaptador geométrico.

## Optimización de inferencia y recuperación

El grafo de inferencia y la precarga sin pérdida se prueban primero con ambos
endpoints sobre 66 consultas P: 16 batches de cuatro y un tail de dos. Se exigen
igualdad exacta de pixels, relojes, fase y TTC; pesos y RNG intactos; rechazo de
relojes inválidos. La mediana del ratio de tiempo graph/eager debe ser ≤0,90.

Si esa optimización se rechaza, se conserva el diagnóstico y se utiliza el
evaluador eager original. No se cambia la tolerancia ni la población. Esta
optimización no altera el benchmark de coste preparado ni las reglas científicas.

```powershell
./scripts/Resume-JointMeasurement.ps1 -Stage A1Evaluation
```

El modo Auto selecciona esta etapa sólo cuando existen ambos endpoints; cuando
ya existe evidencia completa exige interpretar la siguiente decisión. Los
bloqueos de procesos y recursos impiden iniciarla encima del entrenamiento.
El observador dispone de la etapa `adaptation_evaluation` para comprobar progreso
en las horas en punto. No reinicia procesos automáticamente.

## Verificación de la implementación

Código: commit `469537e`. Ruff y Pyright aprobados; 27 tests CPU aprobados,
15 específicos de evaluación A1 y 12 del observador. Cubren control ausente,
durabilidad incompleta, pesos/versiones alterados, identidad modificada, prefijo
congelado alterado, rechazo de V, pruebas de inferencia obsoletas, ordenación y
separación entre mecanismo, admisión absoluta y coste.

Estas pruebas no sustituyen la ejecución sobre endpoints reales. La paridad
CUDA de inferencia, las tablas completas, el coste y los exports de A1 siguen
pendientes hasta que finalicen ambos entrenamientos. No se han añadido updates
científicos ni técnicos para preparar esta evaluación.
