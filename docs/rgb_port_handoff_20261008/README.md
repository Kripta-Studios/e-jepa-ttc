# RGB-PORT: trasladar la familia A5/C2F/PAIR/contexto a RGB

Plan propuesto el 8 de octubre de 2026. Base inspeccionada: `b44443ae331b5e84b94e940801468449b2858de4`.

## Decisión

Priorizar un traslado controlado de la arquitectura compacta que ya existe, antes de convertir la propuesta anterior ResNet50/FPN/CORR en una dependencia. `CausalScaleTTCConfig` ya admite `modality="rgb"`; `_sensor_support` tiene una rama RGB. Eso acredita soporte de código, **no** una ejecución RGB equivalente a los productores TRAIN40 actuales.

El primer resultado buscado es una comparación de información: evento, RGB y fusión tardía, con lectura temporal real y con encoders/cabezas correctamente separados de los grupos evaluados. No se garantiza SOTA.

## Qué hace este paquete

- Analiza los CSV, el ZIP y los módulos revisados.
- Especifica la adaptación y una campaña inicial, acotada y entrenable.
- Aporta contratos de referencia con 33 tests de CPU: tiempos enteros, frames disponibles y distintos, RGB [0,1], esquema RGB_PHASE17, roles de datos y anclaje geométrico.
- Incluye preparación de worktree y prompts, no un trainer de producción terminado.
- Conserva `evidence/SOTA_ESSENTIAL.zip` byte a byte. No añade pesos de modelos ni datos privados.

## Lectura

1. `docs/01_RESULTADOS.md`
2. `docs/02_ARQUITECTURA_RGB.md`
3. `docs/03_DATOS_Y_COMPARACION.md`
4. `docs/04_CAMPANA.md`
5. `docs/05_IMPLEMENTACION.md`
6. `docs/06_RECURSOS_Y_ENTREGA.md`
7. `execution_policy.json`
8. `PROMPT_CODEX_RGB_PORT.md`

## Autoridad

La creación/descarga del paquete no ejecuta nada. Enviar expresamente el prompt al agente abre la nueva autorización descrita. La entrega SOTA precedente sigue siendo de cero updates. No se suma esta cola a otra V13 idéntica ni se cancelan entrenamientos activos por suposiciones: registrar primero el estado.

Esta campaña tiene como máximo 228.408 updates científicos nuevos, 500 técnicos y 11.092 de recuperación, hasta 240.000 físicos. Los máximos no son una obligación de consumo ni son intercambiables entre brazos. Ver el cálculo y prerrequisitos en `docs/04_CAMPANA.md`.

## Pruebas del handoff

Desde la carpeta del paquete:

```powershell
python -B scripts/verify_package.py --root .
python -B -m unittest discover -s tests -v
```

Los tests no acreditan inferencia RGB de A5 en el repositorio, entrenamiento eAP, uso de GPU, calibración FCWD ni ejecución de PowerShell en Windows. Esas pruebas deben realizarse localmente con el entorno del proyecto.
