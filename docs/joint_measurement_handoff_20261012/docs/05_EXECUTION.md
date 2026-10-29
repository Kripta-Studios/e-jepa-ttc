# Recursos, autonomía y recuperación

## Permisos al enviar el prompt

Nuevo techo: 1.000.000 updates físicos y64.800s dentro de scopes GPU propios, incluyendo carga/guardado que ya se cuenten ahí. Es una nueva ventana de trabajo, NO saldo V17. La planificación no promete completar todas las familias en ese tiempo ni obliga a consumirlo.

Reserva técnica hasta1.000 updates sintéticos; recuperación hasta5.000; el resto sólo para fits científicos registrados del programa. Los pasos sobre ejemplos reales P (incluido overfit smoke) se contabilizan como trabajo científico de diagnóstico, no como datos sintéticos. Máximo un preflight de ajuste P de200updates, separado de pesos científicos finales.

Techo de proyecto23GB RSS, reserva host2GiB/reentrada3GiB, mínimo10GB libres tras reservas; medir además Windows commit y disco previsto antes de crear cachés. No instalar cómputo cloud, modificar drivers/paging/Control Center ni recurrir a Runpod. Si el usuario cambia límites, registrar la modificación sin reescribir historial.

V13/V14/V15/V16/V17 y Stage70 se leen sólo en roots declarados, preservando propietarios. No asumir que un PID ausente es una reserva eterna ni esperar un OWNER antiguo sin revisar proceso, hora y cadena real. Un único escritor por campaña; un único entrenamiento GPU pesado del programa. Revisión/CPU ligera se permite dentro del límite agregado.

## Ejecución sostenible

No generar otra framework de orquestación si el repo ya aporta checkpoint, ledger, batch efectivo y atomic IO. Reutilizar componentes, no sus rutas globales hardcoded ni checkpointers que escribirían en campañas ajenas.

Verificar manifest/source/parent al admitir y en fronteras durables. No volver a hashear cientos de GB ni ejecutar git por cada ejemplo/gradient. Mantener verificación final y detección de cambio en inputs.

Guardar cada100 updates y en pausa: modelo, buffers, optimizer, scheduler, precisión, RNG CPU/CUDA/sampler, cursor, draws, configuración, identidad de datos y contador. Distinguir trabajo ejecutado de durable y recuperación. Evitar reseñalizar el mismo checkpoint si su estado no cambió.

Analizar por fragmentos durables. Contar progreso en updates, shards, cohortes y comparaciones; un heartbeat no es un resultado. No relanzar cada30s el mismo proceso pesado si falta memoria/disco. Ante un bloqueo independiente, continuar otra tarea viable. Si no queda ninguna, entregar checkpoint y dependencia exacta, no prometer trabajo invisible.

## Autonomía científica

`EXPERIMENT_REGISTRY.jsonl` debe registrar cada decisión ANTES de los updates, con enlace a una observación y a un contraste. Máximo4 experimentos adaptativos emparejados más las ramas/replicas descritas. El agente puede elegir dentro de ese espacio sin preguntarle al usuario tras cada negativo.

No emitir sólo planes o código si hay un fit admitido que puede ejecutarse. No detener toda la investigación porque el primer brazo cierre NO_CANDIDATE. No modificar un endpoint, población, métrica o gate viejo para fabricar un éxito. No escoger un checkpoint por mirar repetidamente V ni abrir labels privados.

No push, merge, submissions, mensajes/email o publicación automática en este contrato. Commits nuevos locales para freeze son permitidos, conservando cambios ajenos. No reset/clean masivo. Se permite leer documentación/repos públicos pertinentes y descargar pequeñas dependencias ya admitidas; grandes datasets o pagos requieren ámbito explícito adicional.

## Qué entregar

RESEARCH_STATUS.md, EXPERIMENT_REGISTRY.jsonl, candidatos y baselines, ledger, cuentas GPU, resultados completos por grupo/banda, negativos, curvas, pruebas, muestras de inferencia sin labels, rutas/sha de checkpoints grandes, métricas y reproducción portable. Export local de test sólo si es compatible. Un ZIP esencial verificable y un comando REAL probado de resume.

Separar estado de investigación, candidato de desarrollo, evidencia de generalización y score oficial. El estado `SOTA_CONFIRMED` nunca se establece sin la comparación externa que lo sostenga.

## Reproducir esta revisión (opcional; no requisito que bloquee la nueva campaña)

Con V17 extraído en una carpeta que contiene `source/` y `campaign/`:

```powershell
python scripts/replay_v17_numerically.py --bundle-root /ruta/v17_extraido --output /ruta/nueva/revision_numerica
python scripts/probe_native_interface.py --source /ruta/v17_extraido/source --output /ruta/nueva/native_probe.json
python -m pytest tests/test_reference.py -q -p no:cacheprovider
```

El script numérico requiere NumPy, Pandas y PyTorch compatibles. Publica diferencias, no declara paridad exacta entre entornos. Las tolerancias descriptivas de la revisión no amplían las tolerancias históricas de producción. El probe nativo utiliza pesos aleatorios y cero updates; no confundirlo con replay de un checkpoint entrenado. No repetir estas verificaciones a cada update o para consumir tiempo sobrante.
