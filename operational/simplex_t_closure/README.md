# Cierre operativo SIMPLEX-T, 2026-10-02

Esta carpeta conserva el código científico congelado y adapta sólo admisión,
reanudación, persistencia, serialización y entrega. Nunca construye un optimizador
ni invoca los entrypoints de entrenamiento. Las inferencias posteriores usan
exclusivamente cabezas endpoint2500, contextos cacheados autorizados y CPU FP32.

## Identidades y límites

- Scientific code: `0c1a7285b6b5af869b0bf5a13a9629f08997e7a2`.
- Scientific freeze: `2eb3fff9ea24145c4af16c0c8645eb65893144580e41880790e169239fa8f3a7`.
- Candidato: `TPR-D1-H8-C160`; la reparación no selecciona otro brazo.
- `resource_policy.json` recoge las dos enmiendas explícitas del usuario: RAM
  disponible mínima 2 GiB, emergencia de almacenamiento 10 GB decimales;
  se mantienen techo RSS del árbol 4 GiB y reserva propia de T6 1.070.596.096 bytes.
  Stage70–76 mantiene su propia política sin cambios.
- Un lease compartido conserva un único escritor T6; nunca se roba un PID vivo.

## Reparaciones concretas

1. Admisión científica completa inicial y final; instantánea de entradas con
   detección de cambios en límites confirmados. No hay validación Git por draw.
2. Replay de los 8.195 intentos T2 sellados, incluidos tres rechazos, en 33
   fragmentos de hasta 256 intentos. Cada recibo liga offsets, IDs aceptados y
   rechazados, bytes, población, publicación, freeze y receta. No queda generación
   RNG pendiente; el log T2 entero es la autoridad. Un fragmento sin recibo no se
   considera confirmado. Un hash diferente falla sin sustituir su autoridad.
3. Los checkpoints completos y parciales siguen en su ruta. La publicación usa
   temporales y reemplazo atómico; `fsync` abre el Parquet con `r+b` en Windows.
4. El grafo se compara en su representación JSON: la tupla TRAIN original se
   serializa como lista, conservando números y gates. No se redondea.
5. La entrega reutiliza un `POSTPROCESSING.json` sellado sólo a través del
   verificador real de inventario, procedencia, contabilidad y grafo. Los datos
   inmutables se enlazan físicamente en el destino canónico para evitar duplicados.
6. Se añaden recibos históricos T2/T4 a la declaración incompleta T3/T5. El recibo
   T2 registra una invocación fallida: no se reinterpreta como entrenamiento
   completo ni como garantía de picos/tiempos exhaustivos.
7. El supervisor distingue Windows1455 de hashes corruptos, registra memoria
   comprometida y no modifica el pagefile ni relanza workers en un bucle.
8. El transporte reutiliza el soporte TRAIN calculado por la función original
   para el mismo objeto de fuentes admitidas. Cada frontera comprueba la
   instantánea; la admisión científica y el hash completo se repiten al final.
   Esto evita recargar fuentes TRAIN para cada comprobación del inventario ZIP.
9. Las baselines registran la identidad de caché original; el freeze registra
   esa identidad envuelta por `ArmBinding` D0/H8/NONE. Se verifica el envoltorio
   con el hash original y después la caché real, sin renombrar fuentes o recibos.

Las transformaciones de funciones congeladas sustituyen patrones explícitos y
fallan si el código original cambia. `src/` y `scripts/` científicos permanecen
íntegros. `saved_analysis.py` fue una vía de aritmética de bajo consumo previa a
la admisión canónica; su recibo dice explícitamente que no certifica el cierre.

## Continuación recuperable

Desde el worktree, con el entorno ya instalado:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONUTF8 = '1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/supervisor.py
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/supervisor.py --verify-only
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/extras.py --mode arithmetic
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/extras.py --mode sources
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/package.py --stage prepare
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/verify_extracted.py --stage pre
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/package.py --stage final
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -u operational/simplex_t_closure/verify_extracted.py --stage final
```

No iniciar otro análisis pesado mientras el propietario T6 esté vivo.
La verificación canónica requiere el HEAD de su launch; se ejecuta antes de
registrar un nuevo commit operativo. Los paquetes usan directorios propios,
CRC, hashes internos y una extracción física independiente para cada prueba.
El último estado está en `artifacts/simplex_t/closure_20261002/STATUS.json`,
los 13 checkpoints en `T6/CHECKPOINTED_WORK/control/STATE.json`, y cada fragmento
en el `.resume` de su grupo. Los hashes de las publicaciones son el oráculo.

## Verificación

`test_runtime.py` interrumpe antes del recibo de fragmento y después de publicar
el log; exige igualdad bit a bit de draws, IDs, rechazos y pérdidas. Comprueba
paridad numérica contra la implementación original y T2 publicado, raíz ausente,
Windows1455, journal cortado, corrupción completa y reemplazo de Parquet.
Los tests no construyen modelos. `regenerate.py` sí reconstruye las cabezas
compactas en inferencia, sin optimización ni forwards de expertos.

El bundle incluye código histórico, predicciones, normalizadores y entradas
normalizadas compactas. Su alcance de regeneración es explícito; los 36 expertos
históricos quedan como referencias verificadas, sin sus pesos. No equivale a
reconstrucción autónoma desde eventos crudos.

En una carpeta extraída del bundle, con el entorno y versión Torch registrados:

```powershell
python -B -u operational/regenerate.py --bundle-root . --output ../regeneration.json
```

La comprobación usa sólo los hashes, código científico y entradas incluidos.
Reproduce TTC, q10/q90, residuales, costes relativos, decisiones y agregados;
no abre datasets ni checkpoints externos de expertos. Los campos FP32 usan
tolerancia absoluta 1e-7 fijada antes de la inferencia; TTC usa 1e-10.
