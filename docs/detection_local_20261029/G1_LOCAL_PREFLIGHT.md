# G1 local: GPU disponible; detector todavía bloqueado

Fecha: 2026-10-29. Este informe añade evidencia del equipo local al
[informe remoto](../detection_g0_g2_20261029/G1_DETECTOR_FEASIBILITY.md).
Los recibos remotos se conservan con su contexto original.

**Resultado: una operación CUDA con backward pasa; G1 permanece BLOCKED.**
La GPU y Torch están disponibles en un entorno existente. Ninguno de los dos
operadores deformables está instalado en ese entorno. No se compiló una extensión,
no se ejecutó el detector y no se inició entrenamiento.

## Evidencia local

| Comprobación | Resultado medido | Alcance |
|---|---|---|
| GPU/driver | NVIDIA GeForce RTX 5070 Ti Laptop GPU; 591.86; capability 12.0 | Inventario real |
| Memoria GPU | 12.820.480.000 bytes; `nvidia-smi`: 12.227 MiB | Memoria total, no presupuesto garantizado para el detector |
| Python/Torch/torchvision | 3.11.15 / 2.11.0+cu128 / 0.26.0+cu128 | Entorno PHASE existente reutilizado para auditoría |
| Arquitecturas del wheel Torch | Incluye `sm_120` | No certifica extensiones externas |
| Primitiva CUDA | Matriz 128×128, producto, pérdida cuadrática y backward: PASS, pérdida y gradientes finitos | No ejecuta el detector |
| Pico de la primitiva | 17.302.528 bytes asignados; 23.068.672 reservados | No es una medición de memoria del modelo |
| MonoDGP y MonoDETR | `ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'` | Fallo de importación previo a paridad numérica |
| Forward/loss/backward/update/decode del detector | NOT_RUN | G1 no puede aprobarse |

Recibos: [entorno y primitiva](evidence/g1_local_environment.json),
[toolchains](evidence/g1_toolchain.json),
[operador MonoDGP](evidence/g1_monodgp_operator.json) y
[operador MonoDETR](evidence/g1_monodetr_operator.json).
Los comprobadores de operadores terminaron con código 2 y `dtype_results=[]`:
no hay resultado de paridad que interpretar como aprobado o fallido numéricamente.

Se utilizó `../phase-ttc-main-integration/.venv/Scripts/python.exe` sin instalar ni
cambiar paquetes. El campo `isolated_environment=true` del probe solo comprueba
que Python pertenece a un virtualenv: **no acredita un entorno nuevo dedicado al
detector**. La configuración propuesta en el bundle pide Python 3.12.14; esta
auditoría con Python 3.11.15 no valida esa configuración ni completa su lock.

La `.venv` de `e-jepa-ttc-v18-joint-measurement` falla durante el arranque de
`site` por `UnicodeDecodeError` al leer un archivo `.pth`; además falta
`Lib/site-packages/torch/__init__.py`. No se reparó ni utilizó para estas pruebas.
Un metadato `torch-*.dist-info` presente no demuestra una instalación funcional.

## Fuentes fijadas

Se clonaron únicamente los repositorios de código, sin pesos ni datasets, en
`output/detector_sources` del worktree local:

| Fuente | Revisión comprobada |
|---|---|
| MonoDGP | `aa059a18214aebf644510e7f0793971b403f9d14` |
| MonoDETR | `6994b9f512400b258c6edb75f77423beb9c126f2` |

Los diez hashes de archivos fijados para cada fuente coinciden con
`configs/detection/detector_sources_v1.json`. No hay modificaciones de archivos
versionados. La importación del wrapper generó archivos `__pycache__` no
versionados, registrados en el recibo de toolchain. Las fuentes clonadas son
artefactos locales bajo `output/`, no nuevos componentes publicados del proyecto.

El `ops/setup.py` fijado de ambos candidatos conserva `-arch=sm_60` y flags
sm60/61/70/75. Ambos comparten hashes de los archivos centrales del operador
auditados. Por ello, pasar a MonoDETR no elimina este trabajo de port.

## Toolchains: hechos y límites

| Toolkit instalado | `compute_120` en `nvcc --list-gpu-arch` | Precheck real de Torch 2.11.0+cu128 |
|---|---|---|
| 12.4, V12.4.131; seleccionado por PATH y `CUDA_HOME` | No | Código 0 con advertencia por diferencia menor 12.4/12.8 |
| 13.1, V13.1.115 | Sí | Código 1: excepción por diferencia mayor 13.1/12.8 |
| 13.2, V13.2.51 | Sí | Código 1: excepción por diferencia mayor 13.2/12.8 |

El precheck ejecutado fue `torch.utils.cpp_extension._check_cuda_version`, con
`CUDA_HOME` cambiado solo dentro del subproceso. **No fue un intento de build.**
El rechazo de 13.x está demostrado para esa comprobación del Torch instalado;
no demuestra que otras combinaciones de versiones sean inviables. La ausencia
de `compute_120` en 12.4 limita la ruta nativa propuesta; no se probó fallback PTX
ni se afirma que todas las estrategias alternativas sean imposibles.

MSVC está instalado en Visual Studio Community 18: el ejecutable medido informa
versión 19.50.35729 para x64, bajo el directorio de herramientas 14.50.35717.
`cl.exe` no está en PATH en esta sesión. Activación del entorno del compilador y
compatibilidad de ese MSVC con el toolkit que finalmente se elija siguen sin
validarse. No basta con encontrar el ejecutable para admitir un build.

WSL enumera Ubuntu-24.04 y docker-desktop, ambos detenidos. No se arrancó ninguna
distribución ni se comprobó allí CUDA o el detector.

La documentación oficial confirma soporte Blackwell y wheels CUDA 12.8 desde
PyTorch 2.7; su alcance es el framework, no el operador externo de estos
repositorios. [Anuncio oficial de PyTorch](https://pytorch.org/blog/pytorch-2-7/).
La evidencia decisiva de esta laptop es la ejecución local registrada arriba.

## Reproducción desde este worktree

Los comandos siguientes reutilizan el entorno de auditoría existente. No son
comandos de instalación ni de entrenamiento; se espera bloqueo explícito de G1.

```powershell
$auditPython = '../phase-ttc-main-integration/.venv/Scripts/python.exe'

& $auditPython -X utf8 scripts/probe_detection_environment.py `
  --monodgp-root output/detector_sources/MonoDGP `
  --monodetr-root output/detector_sources/MonoDETR `
  --run-cuda-primitives `
  --output docs/detection_local_20261029/evidence/g1_local_environment.json
$LASTEXITCODE # 2 por G1 BLOCKED

& $auditPython -B -X utf8 scripts/check_deformable_operator.py `
  --source-root output/detector_sources/MonoDGP --family monodgp `
  --output docs/detection_local_20261029/evidence/g1_monodgp_operator.json
$LASTEXITCODE # 2 por extensión ausente

& $auditPython -B -X utf8 scripts/check_deformable_operator.py `
  --source-root output/detector_sources/MonoDETR --family monodetr `
  --output docs/detection_local_20261029/evidence/g1_monodetr_operator.json
$LASTEXITCODE # 2 por extensión ausente
```

El recibo `g1_toolchain.json` conserva los comandos y salidas individuales de
`nvcc`, MSVC, inventario WSL y prechecks de Torch. Estos últimos pueden
reproducirse en el mismo entorno así, sustituyendo la versión de toolkit:

```powershell
& $auditPython -I -X utf8 -c 'import torch.utils.cpp_extension as e; from torch.torch_version import TorchVersion; e.CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"; e._check_cuda_version("cl",TorchVersion("19.50.35729"))'
```

## Continuación concreta de NEXT-04 y NEXT-05

1. Crear un entorno propio del detector con dependencias resueltas y hashes.
   La ruta candidata del plan es Torch cu128 con toolkit 12.8 y un compilador host
   admitido; requiere preparar y validar esa combinación. Una ruta alternativa
   debe fijar conjuntamente Torch/toolkit/compilador. No se ha instalado ninguna
   en esta auditoría y no debe reutilizarse PHASE como entorno de desarrollo.
2. En una copia de fuentes con patch versionado, adaptar arquitecturas y APIs
   históricas del operador. Guardar build/logs/binario y hashes. No omitir
   prechecks de versiones para aparentar compatibilidad.
3. Ejecutar el comprobador de paridad de forward y gradientes FP32/FP64. La
   importación satisfactoria por sí sola no supera este requisito.
4. Implementar y ejecutar el smoke completo de NEXT-05: train/eval, objetivos
   sintéticos de ocho clases, pérdidas finitas antes de cualquier saneamiento,
   backward, actualización de parámetros y decode. Desactivar descargas
   implícitas de pesos. Registrar memoria y configuración con batch 1, inicialmente
   sin PHASE residente en GPU.
5. Admitir G1 solo con esos artefactos reales. G2 sigue condicionado también a
   los subgates pendientes de G0. No hay aún baseline eAP entrenado ni evidencia
   de funcionamiento conjunto detector–PHASE.

El cambio respecto al informe remoto es concreto: **ya existe evidencia local de
Torch ejecutando CUDA y backward en la GPU disponible**. Persisten el port del
operador, su paridad y la ejecución del detector completo; ninguno puede
sustituirse por la prueba matricial.
