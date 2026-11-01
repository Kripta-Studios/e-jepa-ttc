# Trabajo realizado sobre el release PHASE-TTC

Fecha de cierre experimental: 1 de noviembre de 2026. Este documento describe
la campaña iniciada a partir del dossier de trece propuestas para mejorar el
release ganador de GarlTTC. Su documento compañero recoge los
[resultados, conclusiones y límites](RESULTADOS_Y_LIMITES.md).

La campaña ejecutó los diagnósticos y validaciones admitidos con los datos
disponibles. Encontró una señal favorable de cadencia en desarrollo y completó
su comparación nativa y sus mediciones de latencia. No produjo un nuevo release
oficial ni entrenó todos los brazos condicionales del dossier.

## 1. Alcance y punto de partida

Se conservó el [plan original](../release_research_plan_20261031/README.md),
su orden de prioridades y sus condiciones de continuación: cadencia física,
observabilidad, reloj causal, formación de eventos, superficies y visibilidad,
y JEPA solamente si los diagnósticos justificaban investigar el productor.
El nombre público del proyecto es PHASE-TTC; los identificadores históricos y
el espacio de importación `e_jepa_ttc` se mantienen.

El objeto de estudio fue R0 TRAIN40: cuatro productores, dos bloques PAIR,
dos contextos y la cabeza `PHASE_ONLY_EVENT_S7_TRAIN40`, junto con los dos
normalizadores, la receta y su sello. Se comprobó su identidad local. El sello
registrado es
`cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.

El usuario confirmó que no existen adquisiciones reservadas ni calibración
fotométrica adicional y autorizó continuar con los datos existentes. Por eso
las evaluaciones siguen siendo de desarrollo expuesto. No se leyeron etiquetas
privadas Test12, no se seleccionó mediante el leaderboard y no se modificaron
pesos, normalizadores ni predicciones oficiales.

## 2. Infraestructura implementada

Se creó el paquete aislado
[`operational/release_research`](../../operational/release_research/), con
ejecución desde [`scripts/release_research.py`](../../scripts/release_research.py)
y runners específicos para cada contraste. El runtime oficial no importa
estos módulos.

| Componente | Trabajo realizado |
|---|---|
| Protocolo | Presupuestos, disjunción de adquisiciones, admisión condicional y controles de identidad; directorios nuevos, fuentes ejecutadas, hashes y recibos `COMPLETE`/`FAILED` |
| Física | Expansión firmada antes de saturación, tiempo real separado del nominal, transporte causal, observadores y condiciones de identificabilidad por SVD |
| Observabilidad | Extracción RAW de indicadores, análisis con adquisición excluida, controles de soporte y permutación, auditoría de fallos y lectura robusta |
| Sensor | Banco de identificabilidad fotométrica, inventario de calibración, proxy sobre datos reales, referencia RGB y lecturas radial, traslacional y afín |
| Superficies | Correspondencias RGB, coherencia de unidades y recortes, cobertura y fallback explícitos |
| Auxiliares | Atenuación con identidades comprobadas, auxiliar sensorial diferenciable y predictor condicionado por tiempo con stop-gradient y anclaje observable |
| Evaluación | Comparaciones pareadas, denominadores explícitos, bootstrap por adquisición, signos, cobertura, colas y tablas generadas desde resultados |
| Ejecución nativa | Captura de productores, replay de PAIR/contextos/cabeza, diagnóstico numérico, recuperación verificada y dos fronteras de latencia |

Los auxiliares son implementaciones y puntos de integración probados con
fixtures. No equivalen a productores entrenados ni a una integración completa
de JEPA con EMA y trainer. Los ajustes ridge auxiliares de observabilidad sí
se ejecutaron; no son una nueva cabeza de producción.

## 3. Estudios ejecutados

### Cadencia física y localización de la intervención

Se separó la cadencia real RGB del intervalo nominal de eventos y se corrigió
PAIR en una ruta experimental. Se contrastaron por separado eventos, RGB y
ambos, conservando el soporte causal máximo de 650 ms. La identidad del caso
nominal de eventos se comprobó explícitamente.

El screen de 149 consultas se amplió a una población completa de 2608:
2531 de carretera y 77 de Slider. La ampliación añadió 2459 consultas de
carretera y reutilizó las 149 del screen. La extracción no leyó targets ni
actualizó parámetros; la evaluación posterior consumió artefactos completos.
Se calcularon métricas pareadas y un bootstrap de carretera por adquisición
con 10 000 remuestreos y semilla 7.

### Observabilidad y calidad de la medida

Se extrajeron seis indicadores físicos de eventos RAW en las ROI del release.
El análisis comprendió 149 consultas, 148 pares RAW y 25 adquisiciones. Se
compararon cuatro brazos —constante, soporte, físico y permutado— mediante
100 ajustes ridge con adquisición excluida. El escalado y la imputación se
ajustaron dentro de cada partición de entrenamiento.

Después se auditó por qué el brazo físico fallaba: cobertura de q, grados de
libertad, signo y dominio canónico. Se ejecutó además una lectura robusta de
flujo normal y se evaluó su TTC con el mismo fallback. Estos controles no
convierten los padres expuestos en una genealogía OOF nueva.

### Reloj causal

Se compararon persistencia, EWMA, transporte y observador de q con
`EVENT_CONTEXT`, primero con PAIR original y después con PAIR canónico.
Todos respetan soporte causal y fallback. Un banco separado comprobó
transporte a velocidad constante, invariancia ante cambios futuros y el
compromiso entre reducción de ruido y retraso al frenar.

### Formación de eventos con los datos existentes

El inventario cubrió 25 adquisiciones. Había geometría y exposición, pero no
una calibración identificada de umbral de contraste, respuesta fotométrica o
refractariedad que permitiera tratar el proxy como sensor calibrado.

Se ejecutó un banco S0 de ambigüedad y un ensayo S1 real sobre 72 consultas
de carretera de 24 adquisiciones. Las 77 de Slider se conservaron en el
inventario sin admitir una alineación fotométrica no demostrada. Se utilizaron
ventanas causales, ROI del release y controles de actividad, textura y
permutación. La extracción fue CPU, con un hilo y menos de 1 GiB de RAM pico.

Se añadió un control basado en cambio RGB observado y después se intentó
recuperar q mediante modelos radial, con traslación y afín, además de un
control espacial permutado. La escala se ajustó excluyendo cada adquisición;
la lectura de q exigió identificabilidad y suficientes celdas. Se evaluó TTC,
además del residuo sensorial, para no admitir un auxiliar por resolver una
tarea que no mejoraba el objetivo final.

### Superficies y visibilidad

Se ejecutaron correspondencias RGB y medición geométrica sobre 149 consultas,
con 133 mediciones y 16 fallbacks. Se auditaron 146 historias de recorte y
redimensionado. Se comparó TTC con R0 conservando las ausencias originales.
Este ensayo no resuelve por sí mismo segmentación, oclusión o paralaje.

Los resultados de estos contrastes están en el
[informe ampliado de mecanismos](../release_research_implementation_20261031/followup_results_v2/RESULTS.md)
y en el [informe de continuación](../release_research_implementation_20261031/continued_results_v1/RESULTS.md).

## 4. Comparación nativa completa y diagnóstico de plataforma

El perfil inicial mostró repetibilidad local exacta, pero diferencias frente
a intermediarios históricos. Se probaron cuatro políticas TF32 en el original;
ninguna recuperó la paridad histórica bajo el criterio registrado. Se conservó
ese fallo sin relajar tolerancias.

Se registró entonces una comparación distinta: original frente al candidato
en el mismo equipo. Se ejecutaron las 5805 consultas físicas, equivalentes a
6762 aliases lógicos que no se cuentan como muestras independientes. Antes
de comparar el candidato se exigió replay exacto de PAIR, contextos y cabeza
del original local. También se midió la deriva histórica del original en toda
la población y se desglosó por magnitud de predicción.

El servidor histórico y el equipo local difieren: Linux/RTX 4090 frente a
Windows/RTX 5070 Ti Laptop, además de sus controladores. La política TF32
efectiva del proceso histórico no quedó registrada. El diagnóstico no permite
atribuir toda la deriva a una única causa.

## 5. Interrupción, recuperación y control del progreso real

La primera ejecución nativa perdió sus procesos sin recibo final. Quedaron
5805 capturas de eventos y 1474 RGB completas, más un fragmento RGB con bytes
nulos. Se conservó la carpeta interrumpida; no se declaró terminada.

La recuperación vinculó 7279 fragmentos válidos por inventario y hashes,
excluyó el fragmento incompleto y recalculó 34 controles de productores desde
los medios originales. Todos fueron exactos antes de admitir la reutilización.
Se corrigieron dos fallos de preflight —identificación de secuencia en metadata
RGB y lectura del fragmento incompleto—, conservando sus errores.

La inferencia RGB pendiente se reanudó a las 00:33:08 del 1 de noviembre y
la captura completa terminó a las 01:15:17, en hora local. El coordinador se
ejecutó desacoplado de la sesión de comandos, con ventana oculta. Los registros
de progreso distinguen fragmentos copiados de inferencias nuevas; durante la
interrupción no se atribuyó avance a los diez minutos sin nuevas observaciones.

Después se completaron, de forma secuencial en la GPU, replay, informes,
latencia NPZ y latencia RAW. Las carpetas parciales antiguas permanecen para
auditoría. El cierre experimental registra que no quedan procesos activos de
esta campaña. Véase el
[registro de recuperación](../release_research_implementation_20261031/NATIVE_INTERRUPTION_RECOVERY.md).

## 6. Latencia medida en dos fronteras

Se fijaron 120 consultas: las diez primeras consultas físicas de cada una de
doce secuencias, con sus historias admitidas. Se alternó el orden de original
y candidato, con pesos calientes y cachés independientes por brazo y secuencia.
La pasada principal midió tiempo de pared con sincronización CUDA; una segunda
pasada instrumentada separó etapas.

La primera frontera comienza en NPZ de eventos ya preparado y RGB. La segunda
comienza en HDF5 RAW y RGB e incluye lectura, recorte, voxelización,
transferencia, productores, contextos y fusión. Se verificó igualdad bit a bit
de las 120 reconstrucciones RAW frente a los tensores preparados antes de medir.
Todas las salidas cronometradas reproducen exactamente el replay local.

Los doce HDF5 locales suman aproximadamente 117,21 GiB. Se verificaron sus hashes
fuera del tramo cronometrado. Se utilizaron lectores acotados y se reiniciaron
cachés por secuencia. Las cajas son suministradas: detección de objetos,
adquisición física y transporte del sensor quedan fuera de esta medición.

## 7. Correcciones y conservación de evidencia negativa

Se detectó un intercambio de etiquetas RGB/eventos al interpretar el vector
nativo, cuyo orden correcto es **F0 / RGB / eventos**. El informe inicial
`results/` quedó marcado como sustituido y se conservaron los intentos
anteriores. Las decisiones usan `results_v2/` y los informes posteriores.

También se conservaron el fallo inicial de comparación exacta CPU/CUDA, el
intento S0 sustituido por el control de ordenación, los fallos de paridad
histórica y las ejecuciones interrumpidas. Las fuentes archivadas describen
el código realmente ejecutado, aunque después cambiase el código de trabajo.

Los resultados se publican con JSON, tablas, fuentes y recibos; los datos
crudos, checkpoints y cachés voluminosas siguen fuera de Git. Las reglas
`.gitattributes` de esta publicación preservan los bytes de documentos,
recibos y fuentes vinculados por SHA-256.

## 8. Validación y reproducción

La campaña acumula 126 pruebas focalizadas distintas de unidad e integración,
incluidas cadencia, admisión, causalidad, identificabilidad, recuperación y
latencia. Ruff y Pyright del alcance de investigación pasaron. Estas pruebas
verifican implementación y contratos; no prueban generalización del modelo.

El [índice técnico](../release_research_implementation_20261031/README.md)
contiene comandos, inventario de runs y enlaces a los recibos de validación.
En este equipo el entorno válido es `.venv-transfer`; el Pyright específico
declara ese entorno. Los runners ofrecen `--help` y requieren destinos nuevos.
No debe sobrescribirse un run sellado ni admitirse una carpeta parcial como
entrada completa.

Los cierres portables principales son:

- [Campaña completa: cadencia, comparación nativa y NPZ](../release_research_implementation_20261031/campaign_results_v1/RESULTS.md).
- [Suplemento de latencia RAW](../release_research_implementation_20261031/raw_latency_results_v1/RESULTS.md).
- [Distribución de deriva histórica](../release_research_implementation_20261031/native_parity_distribution_v1/RESULTS.md).
- [Diez contrastes de mecanismos y datos existentes](../release_research_implementation_20261031/followup_results_v2/RESULTS.md).
- [Recibo de cierre](../release_research_implementation_20261031/evidence/FINAL_CLOSEOUT.json).

El inventario local global permanece en
`output/release_research_20261031/CURRENT_STATE.json`. Su condición de cierre es
`AVAILABLE_DATA_STUDIES_COMPLETE`; la condición científica sigue siendo
«candidato de desarrollo de cadencia; sin promoción oficial».

## 9. Preparación de la publicación en Git

Se redactaron estos dos documentos de síntesis y se repitieron las 126 pruebas
focalizadas: todas pasaron. Al extender Pyright también a los tests nuevos se
añadieron cuatro aserciones de tipo/presencia en dos archivos de pruebas; no
se cambió la lógica experimental ni se reescribieron resultados sellados.
Ruff comprobó 57 archivos de código mantenido y Pyright terminó sin errores.

Se verificaron 24 recibos `COMPLETE` y 494 referencias a artefactos. De ellas,
272 se resolvieron en los runs locales declarados: los subinformes de mecanismos
copian métricas y recibos, pero sus recibos originales también describen trazas
y fuentes que permanecen en `output/`. El recibo del informe contenedor autentica
las copias publicadas. No se presenta el repositorio como una copia íntegra de
todos los datos y runs locales. Los 24 enlaces locales de estos dos documentos
se comprobaron sin ausencias.

La revisión recibida y las copias de fuentes ejecutadas se excluyen del
reformateado automático como evidencia histórica. La implementación mantenida
y sus tests siguen sujetos a Ruff y Pyright. Se separa la publicación en commits
de contexto/plan, implementación y resultados/documentación, conforme a la
política de commits del repositorio.

La comprobación global `ruff check .` también pasa. `ruff format --check .`
señala dos archivos ya versionados y sin cambios en esta publicación:
`scripts/report_phase_producer_timewarp.py` y
`tests/unit/test_phase_canonical_pair.py`. El formato del alcance nuevo pasa;
no se afirma que toda la comprobación global de formato o la CI remota esté
en verde por ese resultado parcial.
