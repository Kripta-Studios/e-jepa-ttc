# PHASE-TTC: integración del handoff y continuación local

Fecha: 2026-10-29.

> Informe histórico de la primera integración local. La continuación con
> entrenamiento, inferencia y puente PHASE está en
> [RGB_TRAINING_RESULTS.md](RGB_TRAINING_RESULTS.md). Los estados siguientes
> corresponden a aquella etapa y se conservan para mantener su procedencia.

**El paquete está integrado y comprobado en un worktree aislado. G0 sigue
bloqueado, G1 sigue bloqueado y G2 no se ha ejecutado.** La continuación local
recupera las etiquetas train, demuestra que Torch puede ejecutar CUDA con
gradientes en esta GPU y aporta un diagnóstico reproducible de correspondencias
eAP/GarlTTC. No hay todavía un detector eAP entrenado ni inferencia conjunta
detector–PHASE.

El [plan recibido](../../DEVELOPMENT_PLAN.md) y los
[informes del agente anterior](../detection_g0_g2_20261029/EXECUTION_REPORT.md)
se conservan como evidencia de aquella sesión. Este informe actualiza únicamente
lo que se ha observado y ejecutado aquí.

## Entrega e integridad

Se trabaja en la rama `dev/eap-detection-local-20261029`, dentro del directorio
hermano `phase-ttc-detection-local`. La carpeta original permanece en `main`.

| Elemento | Procedencia comprobada |
|---|---|
| Base de la entrega | `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef` |
| Commit declarado por el otro agente | `327dba42dbd1e028ee7d6b239f487163b1e6e0d7` |
| Commit local de importación del parche | `c15eb936` |
| Corrección local de recuperación en Windows | `227c66a0` |
| ZIP SHA-256 | `75ec2ebd7fabb814f1150ad8fcf64ff3cdcdbf5d31903327cdbe9d02936bf55d` |

El ZIP contiene 46 entradas; las 45 enumeradas por su manifiesto coinciden en
tamaño y SHA-256. Sus rutas se comprobaron antes de extraer. Los cinco documentos
sueltos coinciden con sus copias empaquetadas. El parche aplica sobre la base
declarada y añade exactamente 43 archivos: se comprobaron sus bytes en el commit
de importación contra el manifiesto recibido. El commit local difiere del commit
declarado porque aquí se aplicó el parche; no se presenta como una reproducción
de la identidad Git del otro agente.

Véanse el [recibo de entrada](evidence/intake_receipt.json), el
[manifiesto original](evidence/delivery_manifest.json) y la
[verificación local](evidence/final_verification.json). Los archivos que ya
existían en la base, incluidos `pyproject.toml`, `uv.lock`, el código histórico,
el artículo y sus evidencias, se mantienen sin cambios. Los recibos nuevos
permanecen separados de los de la sesión remota. No se ha fusionado esta rama
con `main`, publicado en GitHub ni enviado una submission durante esta
continuación.

## G0: datos locales y recuperación

La revisión eAP utilizada es `a814a946e60d37f7242358af5f344542177e50c5`.
Se recuperaron exclusivamente las 40 etiquetas públicas train que faltaban en
`E:/eAP_dataset`: **128.416.244 bytes**. Los dos manifests ya estaban presentes.
Los archivos recuperados se verificaron mediante los hashes fijados; no se
descargaron medios RGB/eventos ni etiquetas test.

El primer intento descubrió una incompatibilidad real: el disco E: usa exFAT y
rechaza `os.link`. La publicación del archivo descargado usa ahora `os.rename`
en Windows, cuya semántica rechaza un destino existente, y mantiene hard links
en POSIX. El temporal está en el directorio de destino. Dos regresiones verifican
el caso sin hard links y una escritura concurrente del destino: el archivo ajeno
se conserva y el temporal propio se limpia. No se sustituyó por una operación
que sobreescriba datos existentes.

| Comprobación local | Resultado | Límite |
|---|---|---|
| Manifests | 118.247 frames / 40 secuencias train; 35.940 / 12 test | Test usado como metadatos estructurales |
| Etiquetas train | 40 archivos, 804.510 filas, ocho clases; controles estructurales sin fallos | Relaciones numéricas, no convención física completa |
| Medios train declarados | 40 HDF5 y 136 TAR presentes, sin tamaño inválido | Sin hash completo ni verificación de todos los miembros TAR |
| Medios test declarados | 12 HDF5 y 36 de 45 TAR presentes | Nueve TAR ausentes |
| Impacto de TAR test ausentes | 4.132 frames de siete secuencias | Ausencia comprobada ahora en este equipo |
| Template de submission | 35.940 tokens exactos, listas vacías | No prueba semántica geométrica ni comportamiento del scorer |

El [recibo de recuperación](evidence/g0_recovery_receipt.json) distingue el
inventario anterior a la descarga de los recibos de publicación. La
[auditoría posterior](evidence/g0_local_audit.json) contiene el estado observado
tras recuperarlos. No debe interpretarse la presencia de un TAR/HDF5 como prueba
de integridad o disponibilidad física de cada muestra.

### Correspondencias GarlTTC/eAP

El [diagnóstico adicional](GARL_ALIGNMENT.md), con
[código reproducible](../../scripts/audit_eap_garl_alignment.py), selecciona
tres consultas repartidas temporalmente por cada una de las 40 secuencias train:
120 consultas y 240 extremos. No consulta etiquetas test.

Las imágenes se emparejan mediante secuencia y miembro RGB, comprobando además
shard y archivo de eventos. Los timestamps de anotación y del manifest de medios
difieren; no se debe hacer un join directo suponiendo que comparten reloj. El
acuerdo de una proyección de corners con una ROI almacenada tampoco demuestra
que los píxeles RGB estén rectificados al plano del evento.

La igualdad textual entre `public_track_id` de GarlTTC e `instance_id` de eAP no
establece identidad de objeto. La muestra contiene correspondencias con grandes
diferencias geométricas. El vecino por centro se registra solo como diagnóstico;
no constituye una asignación aprobada ni un conversor. También se conservan las
discrepancias entre `box3d_h` y la extensión vertical inferida de las corners.
Los valores exactos y el alcance de cada comparación figuran en el informe y su
[recibo](evidence/garl_alignment.json).

Persisten los bloqueos de equivalencia original/HF/submission, rectificación
RGB/eventos real, semántica de velocidad, causalidad y verificación de medios.
No se ha alterado `eap_contract_v1.json` para dar por resueltas estas cuestiones.

## G1: GPU accesible, detector pendiente

El [preflight local](G1_LOCAL_PREFLIGHT.md) documenta una RTX 5070 Ti Laptop,
capability 12.0 y 12.820.480.000 bytes de memoria total. El entorno existente
`../phase-ttc-main-integration/.venv` dispone de Torch 2.11.0+cu128: un producto
matricial CUDA y su backward producen pérdida y gradientes finitos. Se reutilizó
para esta comprobación sin modificar sus paquetes.

Eso resuelve la falta de acceso a GPU de la sesión anterior, pero no G1. Las
fuentes fijadas de MonoDGP y MonoDETR se comprobaron por revisión y 20 hashes.
Ambos comprobadores fallan al importar `MultiScaleDeformableAttention`: la paridad
del operador y el forward/loss/backward/update/decode del detector siguen sin
ejecutarse.

El toolkit 12.4 seleccionado no anuncia `compute_120`. Los toolkits 13.1 y 13.2
sí lo anuncian, pero el precheck real de este Torch cu128 rechaza la diferencia
de versión mayor. MSVC existe, aunque `cl.exe` no está en PATH en la sesión
auditada. No se intentó compilar ni se eludieron esas comprobaciones. Tampoco se
arrancó WSL ni se instaló un nuevo toolkit. El informe G1 distingue estas
observaciones de lo que todavía requiere ensayo.

## Comprobaciones y reproducción

Se creó `.venv-audit` con Python 3.12.13 y las dependencias de auditoría fijadas
en `configs/detection/g0_requirements.lock.txt`. Este entorno no tiene Torch y
no se presenta como entorno del detector. Su
[inventario instalado](evidence/audit_environment_freeze.txt) incluye la
dependencia de plataforma Windows `colorama`.

La comprobación focalizada final obtuvo **77 tests correctos, sin fallos,
errores ni skips**: 63 recibidos, dos regresiones de recuperación en Windows y
12 casos del diagnóstico eAP/GarlTTC. Ruff y formato pasaron sobre 14 archivos
Python. La [verificación final](evidence/final_verification.json) conserva comandos,
códigos de salida, salida textual y hashes del código ejecutado. Incluye los
tests focalizados, Ruff y formato sobre los 14 archivos Python de auditoría,
preservación de los archivos de la base y coincidencia de los 43 archivos
importados con el ZIP. El [JUnit final](evidence/final_targeted_tests.xml)
permite consultar cada caso. Esto no es la suite completa de PHASE ni una prueba
del detector.

La revisión independiente local fue exclusivamente de lectura y no dejó
hallazgos materiales abiertos. Contrastó el informe con los recibos, comprobó
sus enlaces, los 14 hashes de código y el JUnit final; mantuvo expresamente los
bloqueos G0/G1 y la ausencia de ejecución de G2. Ese dictamen acepta la entrega
de auditoría, no certifica la geometría física ni el detector.

`paper/scripts/verify_public_data.py` volvió a verificar 13 hashes públicos,
ocho recomputaciones de transferencia y 12 campos de resultados TTC archivados.
Es una comprobación de evidencia conservada, no una evaluación nueva.

El split candidato 24/8/8, semilla 7, se reprodujo desde el manifest local: solo
cambia `code_commit` respecto al candidato recibido. Su estado sigue siendo
`CANDIDATE_METADATA_ONLY`; no se han validado grupos de adquisición ni se ha
eliminado la exposición previa de PHASE a TRAIN40.

Desde el worktree local:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$auditPython = '.venv-audit/Scripts/python.exe'

& $auditPython -X utf8 -m pytest -q tests/unit/test_eap_detection_audit.py `
  tests/unit/test_detection_protocol.py tests/unit/test_detection_environment.py `
  tests/unit/test_eap_garl_alignment.py

& $auditPython -X utf8 scripts/audit_eap_detection.py audit `
  --metadata-root E:/eAP_dataset --dataset-root E:/eAP_dataset `
  --sample-submission "$env:USERPROFILE/Downloads/sample_submission.zip" `
  --output output/local_handoff_intake/g0_recheck.json
# Código 2 esperado: G0 bloqueado.

& $auditPython -X utf8 scripts/audit_eap_garl_alignment.py `
  --eap-root E:/eAP_dataset --garl-root E:/GarlTTC_dataset --per-sequence 3 `
  --output output/local_handoff_intake/garl_recheck.json
# Código 0 significa diagnóstico completado sin errores, no aprobación de G0.

& $auditPython -X utf8 paper/scripts/verify_public_data.py
```

## Continuación necesaria

1. Resolver identidad y convenciones original/HF/submission con evidencia que
   explique las discrepancias observadas. Validar reproyección en imágenes train
   reales, rectificación, reloj de disponibilidad y velocidad; completar la
   comprobación de medios necesaria para el protocolo.
2. Preparar un entorno exclusivo del detector con versiones compatibles de
   Torch, toolkit y compilador; portar y compilar el operador, ejecutar paridad
   forward/gradientes y después el smoke completo de ocho clases. Conservar
   logs, hashes, pérdidas sin saneamiento, actualización, decode y memoria.
3. Tras superar G0 y G1, admitir el loader y los adaptadores, el overfit funcional
   de 16 frames y el piloto limitado descritos en el plan. Mantener la separación
   entre score de detección, incertidumbre, asociación y salidas de fase.

Las nuevas pruebas acotan los problemas de integración. No autorizan a declarar
una convención geométrica definitiva, un baseline entrenado ni robustez del
sistema conjunto.
