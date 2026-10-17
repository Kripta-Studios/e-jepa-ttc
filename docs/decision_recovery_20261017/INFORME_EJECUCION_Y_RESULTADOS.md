# V19: trabajo realizado, pruebas ejecutadas y resultados

Fecha de corte: 17 de octubre de 2026. Horas operativas en Europe/Madrid (UTC+2).

**Se completaron dos entrenamientos controlados, sus cuatro evaluaciones P/H, tres pares de selectores R0, una extensión R1, diagnósticos de signo/magnitud y gradientes, una lectura lineal de signo y la medición técnica de latencia. Hay mejoras reales en varias métricas y en ejecución, pero ninguna configuración nueva satisface conjuntamente los criterios registrados de promoción. No se ha demostrado SOTA.**

Este documento consolida trabajo efectivamente ejecutado. Las hipótesis y siguientes pasos figuran aparte de los resultados. Los informes especializados enlazados contienen el detalle; [EVIDENCE_MANIFEST.md](EVIDENCE_MANIFEST.md) identifica por SHA-256 los artefactos locales utilizados.

## 1. Alcance y punto de partida

La rama es `scientific-recovery-v19-decision-recovery`, en el repositorio `Kripta-Studios/e-jepa-ttc`. Se trabajó en un worktree separado. V18, sus checkpoints y su cierre histórico `NO_CANDIDATE` se conservaron.

El problema inicial era concreto: A1-FULL había mejorado frente a A5-P2 el agregado H un 29,7% y el crítico un 39,1%, pero empeoraba el negativo un 32,9%. Una tasa global de signo algo mejor no demostraba que el error negativo fuera exclusivamente de magnitud. Había que medir las dos cosas.

Se admitieron predicciones existentes de A5, NATIVE, CORR, C1, B1, FULL y LAST-BLOCK. Compartir consultas permite comparar errores de forma emparejada; no convierte siete predicciones de una consulta en siete ejemplos independientes.

| Población | Consultas | Grupos | Uso en este bloque |
|---|---:|---:|---|
| P | 53.890 | 24 | Entrenamiento de productores y probe; diagnósticos y oráculos |
| H | 18.422 | 8 | Desarrollo ya utilizado; evaluación completa y LOGO de routers |

Se comprobaron identidades, orden, targets, cobertura, manifiestos, relojes y procedencia de los productores. No se descartaron filas difíciles para construir una intersección favorable. A5/NATIVE pueden guardar fase reconstruida desde TTC; no se trató ese campo como equivalente a la fase raw original de FULL/CORR.

H no es un test independiente: ya había influido en decisiones de investigación. Excluir un grupo al ajustar el router evita entrenar esa cabeza con dicho grupo, pero no borra la exposición histórica de la arquitectura. Los intervalos por grupos son descriptivos. V y etiquetas privadas no se abrieron para estos experimentos.

### Qué significan las métricas

Para el contrato utilizado, la fase de un TTC válido es `phi(T) = -log(1 - 0,1/T)`. El MiD por consulta es `10.000 × abs(phi(predicción) - phi(target))`. Menor es mejor.

Se calcula la media por grupo dentro de cada banda y después el agregado con pesos 50/30/10/10%: crucial `(0,1;3]`, pequeña `(3;6]`, larga `(6;10]` y negativa `(-10;0)`. Este agregado interno no equivale automáticamente a una media pooled oficial. El MAE mide segundos; el error de signo, dirección incorrecta.

MiD y MAE pueden moverse en sentidos distintos. Una predicción positiva incorrecta que se aleja hacia +60 s puede acercar su fase a cero y reducir MiD, aunque empeore el error en segundos. La conversión original conserva el límite finito ±60 s y asigna fase cero a +60 s. No se cambió el clipping para mejorar resultados.

## 2. Inventario de lo implementado y ejecutado

| Trabajo | Implementación principal | Ejecución y estado |
|---|---|---|
| Registro y admisión | `common.py`, `registration.py`, `campaign.py` | Panel completo admitido; fuentes y entradas vinculadas |
| Diagnóstico y oráculos | `diagnostics.py` | P/H descriptivos; oráculos únicamente P; cero nuevas inferencias |
| Selector básico R0 | `router.py` | Tres pares completos con LOGO H y control permutado |
| Runtime y entrenamiento | `sensor_runtime.py`, `training.py`, `pipeline.py` | QA admitida; L0 y LW completos, 12.000 updates cada uno |
| Evaluación y export | `evaluation.py` | L0-P, L0-H, LW-P, LW-H completos; exports conservados |
| Descomposición negativa | `loss_negative_analysis.py` | Todos los negativos P/H, por signo, rango y grupo |
| Oráculo del contraste L0/LW | `loss_pair_oracle.py` | Selección ideal P con un único vector de predicciones |
| Gradientes | `gradient_diagnostics.py` | Panel P de 384 consultas; backward diagnóstico sin update |
| Descriptores congelados | `sign_probe.py` | 72.312 extracciones P/H; paridad exacta de raw y TTC |
| Lectura de signo | `sign_probe_continuation.py` | Dos ajustes CPU convergidos, S1 y SHUFFLED; rechazo interpretativo |
| Extensión R1 | `feature_router.py` | 202 entradas, LOGO H completo; rechazo registrado |
| Coste técnico | `cost_measurement_v5.py` y diagnósticos previos | 128 consultas P, dos endpoints, batch4/batch1; completo |
| Observación horaria | `hourly_monitor.ps1` | Registro local cada hora; no entrena ni manda mensajes al chat |

Los módulos anteriores están bajo `operational/decision_recovery/`. Se utilizó trabajo delegado para implementación, auditoría de runtime, gradientes, lectura de signo, revisión de resultados y documentación. Las revisiones independientes se hicieron antes de las ejecuciones pertinentes; su aprobación técnica no constituye validación científica.

No se implementó un MoE disperso dentro del encoder ni una arquitectura de varias cabezas. Los routers seleccionan un output completo entre expertos congelados: no mezclan fases opuestas y no ahorran ejecutar ambos expertos. La extensión R1 probada usa descriptores espaciales; **no es** la propuesta pendiente de entropía de correlación, consistencia afín o curvatura de segundo orden.

## 3. Entrenamiento controlado L0 frente a LW

Ambos brazos parten del mismo checkpoint B1-CORRESPONDENCE-STAGE1-S7, SHA-256 `c2eaa9606de06812c659e318b8f53e2b1fc7b95f0e28de977e600e67501db5b5`. Se mantiene adaptación FULL, semilla 7, arquitectura, optimizador, precisión FP32 sin TF32, sampler uniforme, batch efectivo 32 y microbatch 4. Se conserva la pérdida de fase emitida y raw.

| Brazo | Crucial | 3–6 s | 6–10 s | Negativo | Updates |
|---|---:|---:|---:|---:|---:|
| L0, control nuevo | 50% | 30% | 10% | 10% | 12.000 |
| LW, mayor peso negativo | 44,44% | 26,67% | 8,89% | 20% | 12.000 |

Solo cambia la masa del objetivo. No se cambió simultáneamente el sampler, ni se seleccionó un checkpoint a partir de H. Las evaluaciones completas terminaron a las 03:33. L0 reproduce exactamente los valores numéricos de las métricas P/H del A1-FULL histórico; eso no significa que los archivos checkpoint tengan que ser idénticos.

### Resultado H

| Modelo | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Error signo |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5-P2 | 112,285 | 157,267 | 68,257 | 66,231 | 65,520 | 5,637 | 5,694% |
| L0 / FULL | 78,920 | 95,846 | 53,485 | 62,426 | 87,087 | 5,446 | 5,607% |
| LW | 79,867 | 100,418 | 54,545 | 62,170 | 70,775 | 4,613 | 4,055% |

Frente a L0, LW mejora MiDn **18,73%**, MAE **15,29%**, error de signo **1,55 puntos porcentuales** y ligeramente la banda larga. Empeora el agregado **1,20%**, el crítico **4,77%** y la banda pequeña **1,98%**. Ha cambiado el compromiso entre objetivos; no mejora todas las métricas.

Los IC95 descriptivos de LW menos L0 son `[+0,017; +2,055]` en agregado, `[+1,306; +7,888]` en crítico y `[−23,859; −9,869]` en negativo. Son comparaciones de una semilla sobre H reutilizado, no réplicas independientes de entrenamiento.

### Resultado P y decisión

En P, L0/LW obtienen agregado 59,652/60,028, crítico 59,786/63,688, negativo 93,547/71,782, MAE 5,926/4,785 s y signo 6,413/4,363%. El patrón también aparece en entrenamiento; no se explica únicamente por generalización a H.

Hay tres condiciones distintas:

1. La protección negativa de desarrollo frente a A5 admite hasta `1,10 × 65,520 = 72,072` en H. LW, con 70,775, la satisface.
2. La protección original más estricta del sistema es `referencia + max(2; 5% × referencia)`: en este caso 68,796. LW la incumple.
3. La réplica requería al menos 3% de mejora agregada frente a L0 y otras salvaguardas. Al empeorar el agregado, queda `REPLICATION_NOT_ENABLED`.

No se ejecutaron automáticamente semillas 13/23 ni se cambiaron los umbrales después de ver el resultado. Detalle: [LOSS_SCREEN_RESULTS.md](LOSS_SCREEN_RESULTS.md).

## 4. Qué explica la mejora negativa

Se cruzaron los mismos targets negativos según el signo de L0 y LW. En H hay 7.626 negativos:

| Celda | Consultas | MAE L0 → LW | Interpretación |
|---|---:|---:|---|
| Ambos signos correctos | 6.686 | 6,603 → 3,589 s | Mejora de magnitud conservando dirección correcta |
| Solo L0 correcto | 15 | 50,816 → 64,185 s | Nuevos errores de dirección |
| Solo LW correcto | 355 | 50,311 → 33,697 s | Rescates de signo, con magnitud aún imperfecta |
| Ambos incorrectos | 570 | 23,744 → 33,763 s | Fallos compartidos; aumenta el error en segundos |

El **64,2% de la reducción neta de MiDn H** procede de la celda donde ambos aciertan el signo; en P es el 69,6%. Por tanto, reparar el signo no describe por sí solo el mecanismo de mejora. En el total negativo H, los errores de signo bajan de 925 a 585, las saturaciones de 456 a 287 y el MAE de 10,006 a 7,365 s.

Los 570 errores de signo compartidos empeoran MAE mientras mejoran MiD pooled: es un ejemplo real de la diferencia entre las métricas. No se oculta esa regresión dentro de la media favorable.

Un perfil descriptivo P adicional sitúa 1.466 de los 1.964 fallos de signo compartidos en targets `(-10;−6]`, el 74,6%, con casos repartidos por 23 de 24 grupos. En 1.710 casos la fase emitida se acerca a cero, pero sigue teniendo signo incorrecto. Esto localiza un patrón; no prueba una causa sensorial ni justifica etiquetarlo como «TTC cercano a cero». Detalle: [LOSS_NEGATIVE_ANALYSIS.md](LOSS_NEGATIVE_ANALYSIS.md).

## 5. Complementariedad ideal frente a selección aprendida

### Oráculos: cuánto permiten aprovechar los outputs existentes

Los oráculos usan el target; no pueden desplegarse. Los oráculos de selección minimizan MiD por consulta; el oráculo de signo corrige únicamente la dirección y conserva la magnitud. Cada uno calcula todas sus métricas sobre un único vector de predicciones. Se ejecutaron en P, no como evaluación H independiente.

| Oráculo P | Agregado MiD | MiDn | MAE s | Error signo |
|---|---:|---:|---:|---:|
| A5/FULL | 43,743 | 44,146 | 2,771 | 1,423% |
| NATIVE/FULL | 41,647 | 38,558 | 2,279 | 1,284% |
| NATIVE/CORR | 43,899 | 41,051 | 2,254 | 1,434% |
| NATIVE/B1 | 43,875 | 38,108 | 2,325 | 1,295% |
| FULL con signo ideal y magnitud conservada | 57,310 | 74,251 | 5,072 | 0% |

Incluso el signo perfecto deja en FULL un MiDn P de 74,251 frente a 55,522 de A5: queda un problema de magnitud. Es un diagnóstico contrafactual, no un modelo entrenado.

El oráculo adicional L0/LW consigue agregado P 49,746, crítico 51,160, negativo 68,027 y MAE 4,328 s. Elige LW en 32.023 consultas y L0 en 21.867. Su negativo sigue por encima de los límites P de A5: aproximadamente 61,074 en desarrollo y 58,298 bajo la protección original. Con estos mismos outputs y pesos positivos, ningún selector duro L0/LW puede mejorar ese mínimo negativo. El límite no se aplica a nuevos outputs, interpolaciones ni otras poblaciones. Detalle: [LOSS_PAIR_ORACLE.md](LOSS_PAIR_ORACLE.md).

### R0: aprender el coste relativo sin conocer el target al inferir

Se ajustó una ridge fija de 0,01 para predecir `(coste_especialista − coste_base)/100`. Usa ocho variables derivadas únicamente de los outputs: fases, diferencias, magnitudes, discrepancia de signos y diferencia logarítmica de magnitudes. Umbral cero y empate al experto base.

Cada uno de los ocho grupos H se evalúa con una cabeza que no recibió ese grupo. Los pesos y el normalizador se calculan dentro del fold de entrenamiento. Se conservan los controles de coste permutado. El target enseña el coste durante el ajuste; no entra como feature ni como regla de selección durante inferencia.

| Pareja R0 en H | Agregado | Crítico | Negativo | MAE s | Error signo |
|---|---:|---:|---:|---:|---:|
| NATIVE/FULL, principal | 77,629 | 93,793 | 79,071 | 3,935 | 5,618% |
| A5/FULL | 78,078 | 94,870 | 80,129 | 3,929 | 5,477% |
| NATIVE/CORR, histórica | 93,028 | 104,274 | 107,195 | 3,447 | 7,366% |

NATIVE/FULL mejora agregado, crítico y MAE frente a FULL, pero el negativo 79,071 supera el límite de desarrollo 72,072; tampoco conserva todas las salvaguardas. Estado: `NO_PRIMARY_ROUTER_CANDIDATE`. La mejora ideal del oráculo no se trasladó suficientemente al selector aprendido. Detalle y controles: [ROUTER_RESULTS.md](ROUTER_RESULTS.md).

### R1: la única extensión de variables ejecutada

Se añadieron 194 descriptores FULL congelados: medias y máximos de 96 canales, delta temporal y log(delta). Son 202 entradas contando las ocho de R0. La elección se congeló antes de ver resultados H del probe de signo. R1 no utiliza probabilidades ni salidas aprendidas de ese probe.

R1 conserva familia, regularización, umbral, folds y controles. Obtiene agregado **78,680**, crítico **96,389**, negativo **79,091**, MAE **3,909 s** y signo **5,548%**. Frente a R0 mejora ligeramente MAE y signo, pero empeora agregado y crítico; el IC95 descriptivo del cambio agregado es `[+0,584; +1,590]`.

Queda `further_evaluation_enabled=false`. El protocolo no abre otra extensión ni un barrido de regularización o umbrales. Los modelos ALL_H son exports de desarrollo in-sample; no sustituyen las predicciones LOGO. Detalle: [FEATURE_ROUTER_RESULTS.md](FEATURE_ROUTER_RESULTS.md).

## 6. Diagnóstico de representación y gradientes

### Lectura lineal de signo

Se extrajeron los 194 descriptores de FULL sobre P/H completos, con batch 4 y pesos congelados. Las 72.312 filas reprodujeron exactamente fase raw y TTC históricos, con máximo error absoluto cero y cero cambios de signo.

El clasificador se entrenó en P con StandardScaler, regresión logística C=1, LBFGS, tolerancia 1e−10 y umbral 0,5. El primer intento alcanzó 2.000 iteraciones sin converger. Se conservó el fallo y se registró una continuación CPU que solo amplió el máximo a 10.000: S1 convergió en 2.743 iteraciones y el control SHUFFLED en 2.958. Se calcularon scores H después de converger ambos.

| Lectura H | Error global | Error negativo | Error positivo | Error balanceado |
|---|---:|---:|---:|---:|
| Signo emitido por FULL | 5,607% | 12,130% | 1,000% | 6,565% |
| S1 lineal | 17,381% | 20,378% | 15,265% | 17,821% |
| Control permutado | 41,412% | 99,921% | 0,083% | 50,002% |

S1 tiene AUC ROC 0,910 y Brier 0,119749: hay señal predictiva frente al control. Sin embargo, empeora mucho el signo de FULL y falla tres de cinco condiciones interpretativas. El IC95 del cambio de error balanceado es `[+7,446; +16,123]` puntos porcentuales. No justifica sustituir el signo actual por esta lectura ni añadir automáticamente una cabeza de signo. Tampoco demuestra ausencia universal de información en la representación. Detalle: [SIGN_PROBE_RESULTS.md](SIGN_PROBE_RESULTS.md).

### Gradientes por banda y bloque

Se hizo backward diagnóstico sobre 384 consultas P fijas: cuatro por banda y grupo. Se separaron encoder, torre espacial y cabeza; se emparejaron estados RNG/dropout y se conservaron los pesos.

Los cosenos crucial/negativa fueron **−0,684 en encoder**, **−0,551 en torre** y **+0,118 en cabeza**. La oposición no está localizada únicamente en la cabeza. Los negativos sí aportan señal finita y relevante; no se confirmó la explicación simple de que el optimizador los ignore.

También se reconstruyó analíticamente el paso AdamW a partir de los momentos guardados, sin `optimizer.step`. Para negativos, `g · delta` fue favorable en total (`−2,213e−5`) pero adverso en encoder (`+5,380e−6`), compensado por torre y cabeza. Un coseno negativo por sí solo no determina el efecto del optimizador.

Es un panel, un checkpoint, una semilla y una realización de dropout. No prueba conflicto persistente durante todo el entrenamiento ni su causalidad sobre H; no se ejecutaron PCGrad o GradNorm. Detalle: [GRADIENT_RESULTS.md](GRADIENT_RESULTS.md).

## 7. Optimización de ejecución y velocidad realmente medida

Se reutilizó el runtime V18 y se integraron:

- Sesión CUDA persistente con capturas del modelo y la pérdida.
- Transferencia del batch32 completo antes de crear ocho vistas microbatch4, conservando el orden de reducción.
- Transferencia conjunta de masas FP32, memoria pinned y precarga de un lote.
- Ocho hilos de decodificación y caché LRU de 4 GiB durante entrenamiento; sin caché de filas en evaluación secuencial completa.
- Checkpoints completos de modelo, Adam, scheduler, sampler y RNG; locks y guardas para evitar propietarios GPU concurrentes.

Los ZIP sensoriales FP32 históricos permanecieron inmutables. No se cambió compresión, precisión, datos o pérdida para obtener velocidad. No se multiplicaron procesos y copias de caché sin medir una ventaja.

La QA GPU tuvo **21 comprobaciones aprobadas** y **24 updates técnicos**. Comparó pérdidas, gradientes, datos, RNG y reanudación: cuatro updates continuos frente a 2+2, tanto en sesión como restaurando un checkpoint, con coincidencia exacta.

| Tramo medido | Resultado |
|---|---|
| Transferencia preparada + forward/backward L0 | 9,02% menos tiempo; ratio nuevo/anterior 0,909781 |
| Mismo tramo LW | 11,43% menos tiempo; ratio 0,885666 |
| Lector CPU sobre un pack P, dos observaciones | 3,01× y 4,55× frente a serie; caché del SO no controlada |

El tramo de entrenamiento excluye lectura sensorial y optimizador: no se atribuyen esos porcentajes a toda la campaña. Las observaciones del lector tampoco son aceleraciones end-to-end. Detalle: [RUNTIME_ADMISSION.md](RUNTIME_ADMISSION.md).

### Inferencia medida al final

La medición V5 usa 128 consultas P fijas, dos endpoints, 20 calentamientos y timings sincronizados. Incluye decodificar P2 guardado, colación, H2D y forward hasta TTC.

| Ruta | L0 mediana / p95 | LW mediana / p95 |
|---|---:|---:|
| Batch1 descriptivo | 20,066 / 26,773 ms | 20,076 / 27,163 ms |
| Batch4, lote completo | 36,847 / 47,533 ms | 37,120 / 46,519 ms |
| Batch4, amortizado por consulta | 9,212 / 11,883 ms | 9,280 / 11,630 ms |

Batch4 mantiene paridad exacta raw/TTC. Batch1 conserva signo y saturación y presenta deriva máxima TTC de 0,000030518 s en L0 y 0,000415802 s en LW. La ruta batch1 no se declara bit a bit idéntica. La fase derivada se compara en el mismo backend CUDA que la referencia.

No incluye formar P2 desde eventos crudos, esperar para agrupar consultas ni ejecutar ambos expertos con un router. No demuestra tiempo real del sistema completo ni permite comparar directamente con los segundos de la medición sensorial histórica V18. Detalle: [CACHED_RUNTIME_RESULTS.md](CACHED_RUNTIME_RESULTS.md).

## 8. Contabilidad, incidencias y estado operativo

| Concepto | Cantidad registrada |
|---|---:|
| Updates científicos nuevos | 24.000 |
| Updates técnicos de QA | 24 |
| Cota de updates de recuperación | 87 |
| Cota física total | 24.111 |
| Updates pendientes en el registro | 0 |

Las iteraciones LBFGS del probe son ajustes CPU de un clasificador; no se suman como updates del encoder. Los diagnósticos de gradientes tienen backward, pero cero updates. Los oráculos reutilizan predicciones y no son inferencias nuevas.

El presupuesto local registra 9.995,310 s liquidados y una reserva histórica conservadora de 900 s del primer proceso: total debitado/reservado 10.895,310 s frente al tope 60.000 s. Es contabilidad de scopes, con preparación y esperas; no son horas de kernels GPU. La reserva `RESERVED` no indica un entrenamiento vivo ni se presenta como tiempo medido.

Incidencias conservadas y resueltas:

1. El guard inicial confundía MLflow MCP y aplicaciones WDDM con entrenamiento. Se corrigió antes de la ejecución científica, preservando el intento.
2. Se interpretó incorrectamente el plan externo pegado como orden de parada. El primer entrenamiento se interrumpió después de 787 updates completados, con checkpoint durable en 700. Tras la corrección del usuario se reanudó y se contabilizaron 87 updates de recuperación. No se oculta esta intervención ni se presenta como réplica.
3. La primera extracción de signo comparaba una conversión CPU/CUDA de forma inadecuada. La revisión V2 conservó el contrato numérico y utilizó el backend correspondiente.
4. Tras extraer P completo, el ensamblaje NPZ falló porque los identificadores eran arrays object. Se conservaron los 211 chunks; tras validar hashes se reconstruyeron solo los identificadores desde el registro y se escribieron cachés derivados Unicode sin habilitar pickle. No se reextrajo P ni se alteraron sus matrices numéricas.
5. El límite LBFGS 2.000 fue insuficiente. La continuación separada mantuvo objetivo y tolerancias y convergió, sin seleccionar parámetros mediante scores H.
6. La primera ruta de coste intentó mover a CUDA un export trazado con contrato CPU. Las revisiones posteriores usan el modelo eager CUDA. Se aislaron además la deriva batch1/batch4 y un ULP de diferencia de fase derivada CPU/CUDA; se preservaron filas, logs y recibos fallidos antes de la medición V5 completa.
7. Dos entradas a GPU se aplazaron por procesos de otra comparación local. Las guardas actuaron antes de GPU; no se terminaron esos procesos para forzar la medición.

La extracción P/H finalizó a las 11:59:48; el probe convergido quedó completado a las 12:06:33; el bloque de diagnósticos y coste quedó documentado como completo a las 12:09. La observación de las 12:00 se registró. El monitor horario solo escribe estado local: no reinicia entrenamientos, no descubre candidatos y no envía notificaciones automáticas al chat.

Los archivos `RUNNER_STATE.json` y `PIPELINE_STATE.json` conservan snapshots de etapas anteriores. Por ejemplo, el primero todavía contiene `PENDING_COMPLETE_P_H`; no prevalece sobre los cuatro recibos de evaluación completos. La continuación se consulta en `CONTINUATION_STATE.json` y en los recibos de cada tarea. La latencia desde P2 completa no convierte `cost_complete=false` del sistema/router completo en true.

## 9. Pruebas de software y reparación de GitHub Actions

Las pruebas unitarias sintéticas comprueban contratos y código; no sustituyen las evaluaciones de modelos. Entre las comprobaciones dirigidas se ejecutaron paridad/reanudación del runtime, los 17 casos del probe y recuperación de tokens, los 24 casos R1/R0, los 23 casos de probe/continuación y las pruebas de medición/replay y backend. Esas suites se solapan: no deben sumarse para fabricar un número de tests independientes.

El primer CI del push detectó un único archivo con formato pendiente: `campaign.py`. Se corrigió en `f96a4be`, verificando AST idéntico. La siguiente ejecución pasó los chequeos iniciales y detectó una prueba que leía accidentalmente un registro local. Se aisló con un archivo temporal en `593ed6c`, sin cambiar el comportamiento científico.

La ejecución [39024446650](https://github.com/Kripta-Studios/e-jepa-ttc/actions/runs/39024446650), sobre `593ed6c25ca528c063754e6c6b13024538729dbe`, terminó con **success** el 17 de octubre a las 12:25:43:

| Comprobación CI | Resultado |
|---|---|
| Ruff lint | Aprobada |
| Ruff format | 2.366 archivos con formato correcto |
| Pyright 1.1.411 | 0 errores, 0 warnings |
| Sintaxis PowerShell | Aprobada |
| Unitarias | 4.234 aprobadas, 9 omitidas |
| Integración | 158 aprobadas, 12 omitidas |
| Regresión | 11 aprobadas, 2 omitidas |
| Científicas | 12 aprobadas |
| Total pytest de esa ejecución | **4.415 aprobadas, 23 omitidas** |

Las omitidas no se cuentan como aprobadas. Hubo warnings no fatales, principalmente de APIs/exportación y trazado; no se declara una ejecución libre de warnings. El job completo tardó aproximadamente 6 min 22 s. No descargó los checkpoints ni reprodujo los entrenamientos científicos locales.

## 10. Qué queda demostrado y qué permanece abierto

| Pregunta | Respuesta sustentada por este bloque |
|---|---|
| ¿Aumentar peso negativo ayuda? | Sí, en negativos, MAE y signo; sacrifica crítico y agregado |
| ¿El problema es solo el signo? | No; hay error de magnitud incluso con ambos signos correctos y en el oráculo de signo perfecto |
| ¿Hay complementariedad? | Sí en los outputs existentes; los oráculos P la cuantifican |
| ¿R0/R1 la aprovechan lo suficiente? | No bajo los controles registrados |
| ¿Una lectura lineal de signo resuelve el problema? | No esta lectura; empeora FULL pese a superar el control permutado |
| ¿Existe conflicto de gradientes? | Se observa oposición local en encoder/torre; persistencia y causalidad siguen abiertas |
| ¿Se aceleró el runtime con equivalencia? | Sí en el tramo y QA medidos; no hay porcentaje end-to-end demostrado |
| ¿Hay nuevo candidato promocionado o SOTA? | No |

No se ejecutaron la pérdida adicional LS de signo, varias cabezas, MoE interno, PCGrad/GradNorm, nuevos datasets, un refit general de TRAIN, nuevas semillas de la pantalla L0/LW ni un envío oficial. No se atribuyen a V19 entrenamientos históricos V18 ni comparaciones realizadas por otro hilo.

La siguiente hipótesis debe registrarse como otro contraste, no reutilizar H para barrer umbrales hasta obtener aprobación. La evidencia favorece localizar mejor el conflicto de adaptación y el error de magnitud; antes de modificar gradientes habría que comprobar su persistencia con paneles/checkpoints P fijados. R1 y S1 no justifican escalar automáticamente a un MoE grande. Una futura evaluación oficial requiere su propio protocolo y las reglas de modalidad, historia y cobertura correspondientes.

## 11. Código, evidencia y reproducción

El cierre ejecutado se conserva en `44f64831e93f9a09d351cc80149fd223b13aa805`. El CI de código corregido está en `593ed6c25ca528c063754e6c6b13024538729dbe`. Las revisiones de formato/tests cambian hashes de fuentes, aunque no cambien el resultado científico: **no se reescribieron los registros históricos para hacerlos coincidir con HEAD**.

Para verificar un recibo estricto hay que usar la revisión de fuentes que declara —o el cierre ejecutado compatible— y disponer de los artefactos, caches y padres originales. Las rutas y hashes del [manifiesto de evidencia](EVIDENCE_MANIFEST.md) permiten identificar ese material. Los artefactos grandes, datos y checkpoints permanecen locales y no están incluidos en este push de documentación. Un clon de GitHub permite leer el código y el informe y ejecutar las pruebas sintéticas, pero no reproduce por sí solo los resultados científicos sin esos datos.

Reproducir no significa volver a lanzar a ciegas `pipeline.py`: primero se admiten identidad, fuentes y recibos. Los replays de tareas completas se comprobaron antes de las revisiones CI y no volvieron a ajustar ni inferir. La redacción de este informe solo leyó evidencias existentes y registros de CI; no realizó nuevos optimizer updates ni evaluaciones de modelos.

Documentos de detalle: [índice V19](README.md), [pérdidas](LOSS_SCREEN_RESULTS.md), [negativos](LOSS_NEGATIVE_ANALYSIS.md), [oráculo L0/LW](LOSS_PAIR_ORACLE.md), [R0](ROUTER_RESULTS.md), [R1](FEATURE_ROUTER_RESULTS.md), [signo](SIGN_PROBE_RESULTS.md), [gradientes](GRADIENT_RESULTS.md), [runtime](RUNTIME_ADMISSION.md) y [latencia](CACHED_RUNTIME_RESULTS.md).
