# Preparación de migración a Runpod

Estado: **preparación local y comandos implementados; Linux/CUDA en Runpod aún no verificados**.
No se ha desplegado ningún Pod, enviado ningún archivo, ejecutado nuevos optimizer updates,
modificado la cola local ni cambiado los 246 archivos de autoridad científica.

## Decisión y tamaño

El piloto usa una RTX 4090, Python 3.11.15 y PyTorch 2.11.0+cu128. La plantilla
PyTorch 2.9.1 / CUDA 12.8.1 / Ubuntu 24.04 es solamente la base del contenedor.
El bootstrap instala el entorno en `/workspace/ejepa_env`, sin sustituir los paquetes
globales de la plantilla. La combinación PyTorch 2.11.0 / torchvision 0.26.0 / CUDA 12.8
está publicada en https://pytorch.org/get-started/previous-versions/.

El inventario regenerable está en `artifacts/runpod_migration_20261026/INVENTORY.json`.
Las cifras exactas son las del JSON; pueden cambiar unos KiB al actualizar este adaptador.

| Perfil | Entradas existentes | Persistente recomendado | Alcance |
|---|---:|---:|---|
| `rgb_pilot` | aproximadamente 9,83 GiB | 50 GB | Productores RGB con cachés lossless |
| `all_producers` | aproximadamente 50,81 GiB | 150 GB | Cuatro productores por rotación |
| `rotations` | aproximadamente 587,38 GiB | 800 GB | Productores, PAIR, CTX y generación OOF |

Separar estos volúmenes de los **50 GB de container disk**. Las capacidades persistentes
suponen transferencia por streaming sin conservar el tar remoto: entradas + 10 GiB de
salidas piloto o 50 GiB para trabajo completo + reserva máxima entre 20 GiB y 15% de entradas.
Son reservas de planificación, no tamaños de outputs ya medidos en Linux.

Los hashes de grandes shards se toman inicialmente de sus manifests históricos. El empaquetado
recalcula el hash de cada archivo y verifica también los bytes leídos desde el tar. No se afirma
que los cientos de GiB de crudos se hayan vuelto a hashear. El perfil completo tiene 136 tar RGB
sin digest previo en este inventario; `pack` debe calcularlos antes de publicarlo.
La raíz eAP local completa (698,7 GiB) incluye datos que estos perfiles no necesitan.

## Aislamiento científico y operativo

`operational/runpod_migration/` contiene exclusivamente nuevos archivos. El adaptador de rutas
traduce referencias Windows al abrir archivos, conservando los JSON, datos, código e identidades
históricas byte a byte. `runtime.load_config` vuelve a verificar la autoridad original, sus 246
fuentes, sus 15 entradas y el plan histórico. El adaptador y el manifest de transferencia quedan
registrados en `MIGRATION_BINDING.json`; los nuevos productores añaden ese hash a su identidad.

El guard Linux usa `memory.current`, `memory.max`, límites de ancestros, `MemAvailable` y espacio
libre del filesystem de salida. Conserva 2 GiB de margen, reserva de checkpoint y rechazo ante
lecturas desconocidas. No llama "commit de Windows" a la memoria Linux. Solo consulta compromiso
de memoria del host cuando Linux utiliza overcommit estricto. cgroup v1 no está admitido todavía.
Referencia del controlador: https://docs.kernel.org/admin-guide/cgroup-v2.html.

Una GPU visible por proceso, un run separado por rotación y un lock de proceso por run.
Semilla 7, BF16, batch, pérdidas, sampler, exposición y checkpoints cada 100 updates permanecen.
El prefetch histórico no está activado en este primer runtime Linux: primero se mide la ruta
original, y solo se puede incorporar después con paridad local Linux y nueva identidad runtime.
No se afirma igualdad bit a bit entre RTX 5070 Ti/Windows y RTX 4090/Linux.

El 26-10-2026 a las 11:23 locales, ROTATION_1 ya estaba completa y ROTATION_2/E_A5 había comenzado.
**No asignar ROTATION_2 a la nube por defecto.** El piloto propuesto utiliza ROTATION_4/R_A5.
Antes de cualquier entrenamiento científico se debe excluir realmente esa tarea de la cola local,
documentarlo en `LOCAL_QUEUE_EXCLUSION.json` y vincular su SHA al assignment. Un texto de intención
no constituye exclusión. Esta preparación no toca la cola para fabricarlo.

## Preparación local (sin gasto)

Desde la raíz del repositorio, PowerShell:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$python='../e-jepa-ttc/.venv/Scripts/python.exe'
& $python -m operational.runpod_migration.inventory --run artifacts/final_refit_20261019_scientific_oof --output artifacts/runpod_migration_20261026/INVENTORY.json
& $python -m operational.runpod_migration.bundle pack --inventory artifacts/runpod_migration_20261026/INVENTORY.json --profile rgb_pilot --output artifacts/runpod_migration_20261026/RGB_PILOT.tar
```

Las seis referencias `RGB_INPUT_REFERENCE_R{2,3,4}.json` y `EVENT_INPUT_REFERENCE_R{2,3,4}.json`
se han generado con CPU, cuatro muestras homogéneas por referencia y cero llamadas al modelo.
Sirven para detectar cambios de decodificación/orden tras la transferencia, no para medir precisión.

El tar piloto requiere unos 10 GB adicionales locales. **No intentar empaquetar el perfil de
587 GiB en el disco local actual:** no hay espacio. Ese perfil precisa transferencia fragmentada
o descarga de los medios públicos en destino, con verificación contra el mismo inventario.
Esa transferencia masiva no se ha ejecutado ni automatizado en esta entrega.

## Después de aprobar el piloto y desplegar un Pod

Configurar almacenamiento persistente antes de desplegar. Un volume disk conserva `/workspace`
al parar/arrancar, pero se elimina al terminar el Pod; un network volume es independiente.
Referencia: https://docs.runpod.io/pods/storage/types. Conservar copia externa de checkpoints.

1. En Runpod, `Connect` proporciona host y puerto SSH. Añadir la **clave pública** de acceso.
2. Verificar la identidad del host SSH por un canal confiable. No desactivar comprobación de host.
3. Sustituir los marcadores por la conexión real. Este comando sin `--execute` solo comprueba el tar:

```powershell
& $python -m operational.runpod_migration.transfer --archive artifacts/runpod_migration_20261026/RGB_PILOT.tar --host HOST --port PUERTO --identity-file RUTA_CLAVE_PRIVADA
```

Añadir `--execute` únicamente cuando la transferencia esté autorizada. La clave no se imprime.
El destino `/workspace/ejepa_bundle` debe ser nuevo; un fallo parcial exige inspección explícita,
no borrar automáticamente. La transferencia es binaria y evita la conversión de PowerShell.

En el Pod, una vez transferido el tar por streaming:

```bash
repo=/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement
bash "$repo/scripts/runpod_bootstrap.sh" /workspace/ejepa_bundle
export PYTHONPATH="$repo/src:$repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
/workspace/ejepa_env/bin/python -m operational.runpod_migration.runner preflight --bundle /workspace/ejepa_bundle --run /workspace/cloud_rotation4 --rotation 4 --fit R_A5
```

El preflight no entrena: comprueba hashes, entorno, BF16, recursos y paridad de los inputs con la
referencia CPU. Si falla, conservar el error; no saltar controles para arrancar.

## Prueba de reanudación y ejecución

`ASSIGNMENT.example.json` es una plantilla **deshabilitada**. Antes de activarla deben concretarse
presupuesto, deadline, binding de migración y exclusión local. El deadline detiene updates de forma
durable, **no detiene la facturación del Pod**: el cierre del recurso debe gestionarse en Runpod.

```bash
/workspace/ejepa_env/bin/python -m operational.runpod_migration.runner resume-probe --bundle /workspace/ejepa_bundle --run /workspace/cloud_rotation4 --rotation 4 --fit R_A5 --assignment /workspace/control/ASSIGNMENT.json
/workspace/ejepa_env/bin/python -m operational.runpod_migration.runner producer --bundle /workspace/ejepa_bundle --run /workspace/cloud_rotation4 --rotation 4 --fit R_A5 --additional-updates 100 --assignment /workspace/control/ASSIGNMENT.json
```

El probe consume ocho updates **técnicos** en directorios propios: 4 continuos frente a 2+restore+2.
Compara pesos, AdamW, scheduler, RNG, sampler y cursor. Nunca produce un endpoint científico.
Comprueba reconstrucción y restauración dentro del proceso; no simula corte eléctrico real.
El productor exige PASS del probe y exclusión local. Para continuar al endpoint se omite
`--additional-updates`; el checkpoint y el assignment deben mantener la misma identidad.
`touch /workspace/cloud_rotation4/STOP_REQUEST` solicita pausa durable. Retirar ese marcador
solo cuando se haya autorizado reanudar, y volver a ejecutar el mismo comando.

Para `all_producers`, repetir por E_A5/E_C2F/R_A5/R_C2F con una tarea autorizada y probe por fit.
Cada rotación debe tener su propio directorio. La programación multigpu automática no está
habilitada: separar `CUDA_VISIBLE_DEVICES` y asignaciones exclusivas explícitas.

Con el perfil `rotations` completo y cuatro endpoints admitidos:

```bash
/workspace/ejepa_env/bin/python -m operational.runpod_migration.runner finish-rotation --bundle /workspace/ejepa_bundle --run /workspace/cloud_rotation4 --rotation 4 --fit R_A5 --assignment /workspace/control/ASSIGNMENT.json
```

Requiere además la tarea `FINISH_ROTATION`. Reutiliza productores completos y ejecuta PAIR,
E_CTX/R_CTX y el fragmento PHASE20 mediante la implementación congelada y su corrección RGB
ya documentada. No permite sustituir padres silenciosamente.

## Retorno y límites de esta entrega

Antes de detener o terminar un Pod: descargar el run completo, `MIGRATION_BINDING.json`,
assignment/exclusión, environment lock, preflights, probes, receipts, logs y checkpoints.
Verificar SHA-256 origen/destino y los endpoints en el runtime Linux antes de importar.
**No copiar encima de la cola Windows activa:** los receipts nuevos contienen rutas Linux.
Existe un verificador de retorno aislado, todavía sin resultados cloud reales que verificar:

```powershell
& $python -m operational.runpod_migration.return_verify --downloaded-run RUTA_DESCARGA_SEPARADA --rotation 4 --output artifacts/runpod_migration_20261026/RETURN_VERIFICATION.json
```

Traduce las rutas solo durante lectura y verifica endpoints/ancestros conservando bytes y hashes.
Falta implementar y probar la admisión al coordinador local y la transición de propiedad de tareas.
La preparación no autoriza marcar etapas locales completas.

Tampoco migra todavía el ajuste final TRAIN40, exportación o generación test12. Esos componentes
siguen en la cadena local; su consumo de resultados cloud necesita la integración anterior.
Por ello el estado no es “migración de toda la campaña verificada”. Se dispone de un piloto
concreto y código de rotación, pendiente de validación Linux/CUDA y coordinación de resultados.

Pruebas locales: 13 unitarias PASS, Ruff PASS y Pyright PASS con el entorno real del proyecto.
Verificación adicional read-only: autoridad original PASS, 246 fuentes y 15 entradas intactas.
No se ha verificado todavía bootstrap Linux, throughput, consumo, probe GPU ni entrenamiento cloud.
