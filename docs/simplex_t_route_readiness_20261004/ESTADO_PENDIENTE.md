El entrenamiento autorizado está terminado: 24 endpoints y 60.000 updates científicos guardados. El análisis, los contrastes, el perfilado de las nueve cabezas con entradas preparadas y su bundle reproducible ya están entregados. H8 conserva su identidad histórica.

La comprobación local del 4 de octubre de 2026 resuelve las fuentes del trabajo restante: las 64 consultas elegidas por hash pertenecen a productores TRAIN de fold0; sus nueve checkpoints, que suman 11.218.581 bytes, se han vuelto a verificar contra los hashes del manifiesto histórico. Existen las 23 rutas de eventos TRAIN referenciadas. Esta comprobación no abre eventos ni valida de nuevo sus contenidos crudos. Los dos índices se verificaron contra sus hashes y cada familia contra la genealogía reconocida.

P3 sigue bloqueado. `nvidia-smi` identifica el proceso Python 26768 y la inspección de su comando confirma un entrenamiento de `sonar-representation-lab` (`src/train_detector.py`, finetune). La observación inicial registró 6.950 MiB de memoria GPU total usada; Windows/WDDM no permite atribuir ese número íntegro a un proceso. El ACK compartido sigue sin un turno exclusivo de GPU y lectura pesada. La comprobación no detuvo ni cambió ningún proceso.

Faltan la integración y paridad de los adaptadores reales completos/reducidos, el protocolo prospectivo de repeticiones de ruta caliente/fría de aplicación y sus medidas, y la actualización del informe/bundle con esos resultados. Las interfaces probadas con fixtures no acreditan paridad real. Una GPU compartida no satisface el contrato de perfilado exclusivo, aunque pueda quedar VRAM disponible.

La dependencia externa necesaria es un turno exclusivo de GPU y lectura pesada coordinado con sus propietarios, después de la liberación de los trabajos ajenos. Debe especificar alcance y vigencia; la mera ausencia de un PID antiguo no lo concede. Con ese turno se verifican recursos y fuentes crudas autorizadas, se valida la ruta canónica y se ejecuta únicamente inferencia. No quedan fits por entrenar ni se autoriza otro update.

El nuevo comprobador es deliberadamente de inspección y no inicia un worker:

```powershell
$env:PYTHONUTF8='1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_route_readiness.inspect --output artifacts/simplex_t/route_readiness_20261004/READINESS_RECHECK.json
```

El archivo de salida debe ser nuevo. Un código de salida 2 registra que la ejecución todavía no está admitida. Este comando comprueba dependencias; no es un comando de ejecución del perfilado integral.

Se ejecutaron 14 pruebas focalizadas sin modelos ni optimizador; Ruff y Pyright pasan. Los nuevos recibos y esta nota forman un suplemento independiente. El ZIP anterior, sus entradas y sus hashes se conservan. No se afirma latencia integral, reacción AEB, tracking persistente ni incertidumbre calibrada.
