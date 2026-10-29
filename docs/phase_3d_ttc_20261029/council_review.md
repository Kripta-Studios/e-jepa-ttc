# Council: decisiones para detector 3D + PHASE-TTC

Fecha: **29 de octubre de 2026**. Repositorio revisado: `main`,
`e58d0106e1f114f00f796987f70cd622cc6d186e`.

Documento de evidencia: [investigación del swarm](swarm_research.md).
Estado de esta edición: **revisión cerrada; los tres miembros consideran los
documentos publicables internamente tras comprobar las correcciones**. Este consejo
evalúa una propuesta, no certifica un detector ya entrenado ni autoriza una
publicación de resultados.

## 1. Composición y forma de revisión

El usuario solicitó un swarm y un council separados. Cinco agentes investigaron
literatura, repositorio y datos. Otros tres agentes recibieron los hallazgos y
revisaron su ámbito de forma independiente, con acceso de lectura al código y a
fuentes primarias. El agente principal sintetizó, contrastó y redactó los dos MD.

| Miembro del council | Pregunta principal | Dictamen de primera ronda |
|---|---|---|
| `council_architecture` | ¿Qué piezas faltan y cómo preservar geometría, historia e interfaces? | Piloto modular condicionado a geometría; posponer backbone compartido |
| `council_science` | ¿Qué resultados serían válidos sin fuga ni selección engañosa? | Dos carriles de evidencia; no atribuir robustez conjunta ni calibración sin medir |
| `council_delivery` | ¿Qué se puede implementar y verificar con el entorno y hardware actuales? | Etapas con prueba de operadores/memoria; no comprometer FPS o tiempo de entrenamiento |

Son revisores de IA de esta sesión. No representan a los autores de eAP, una
comisión humana, una revisión por pares ni al organizador del benchmark. Ninguno
ejecutó un entrenamiento para respaldar estos dictámenes.

## 2. Decisión conjunta propuesta

**Avanzar hacia un piloto modular, empezando por contrato de datos y viabilidad de
un detector RGB 3D.** Conectar PHASE-TTC después mediante asociación causal y
preparación de regiones compatibles. Medir eventos contra ese baseline y dejar
backbone compartido/entrenamiento multitarea para una fase posterior.

```text
                    detector 3D
RGB/eventos  ----------+----------------> todas las detecciones -> JSON eAP
                      |
                      v
            asociación causal / IDs propios
                      |
                      v
       ROIs + historia en coordenadas compatibles
                      |
                      v
           servicio PHASE-TTC multiobjeto
                      |
                      v
       TTC + límites de fase + estados de validez
```

La salida de detección permanece válida aunque un objeto todavía no tenga TTC.
El exportador eAP no debe filtrar cajas para ocultar fallos de historia, asociación
o capacidad del servicio TTC. El artefacto interno conserva ambos resultados y
sus causas de indisponibilidad.

Esta decisión establece una secuencia de investigación; el usuario solo encargó
los documentos en este turno. No se ha implementado la nueva arquitectura.

## 3. Correcciones que el council introdujo

### 3.1 Ya existe servicio multiobjeto

La primera descripción del swarm enfatizaba el componente `NativeStream` por
track. El council encontró y verificó
[`MultiObjectTrain40Service`](../../operational/train40_runtime/multiobject.py#L214),
que comparte sensores y exige regiones e IDs estables externos. La extensión debe
reutilizarlo cuando proceda. Faltan el detector y el origen causal de las identidades;
no hace falta reconstruir toda la infraestructura multiobjeto.

Su límite actual es de **uno a veinte objetos**. No es un límite del benchmark.
Un frame sin detecciones exporta `[]` sin llamar al servicio; con más de veinte se
conservan todas las detecciones y se evalúa cómo procesar TTC por chunks o expresar
indisponibilidad. El throughput de esa política sigue sin medirse.

### 3.2 Los límites de fase existen, pero no en todas las rutas

El [export de cabeza](../../operational/final_refit/export.py#L358) ya incluye TTC,
fase y límites q10/q90. El wrapper escalar y el
[contexto batched](../../operational/train40_runtime/batched_context.py#L92) no
transportan todos esos campos. La implementación deberá ampliar la ruta elegida y
comprobar paridad. No corresponde afirmar que el proyecto carece de cuantiles ni
que cualquier runtime ya los entrega.

### 3.3 «Congelado» no significa «no expuesto»

Los pesos finales vieron TRAIN40. Es lícito usarlos para probar interfaces, costes
y errores en población conocida, declarando esa exposición. Eso evita reentrenar
todo antes de descubrir un fallo de integración.

Para una afirmación de generalización conjunta por secuencia, el detector y los
productores TTC deben excluir los grupos evaluados. El council científico añade
un matiz: la AP del detector puede seguir siendo disjunta si PHASE permanece
downstream y no interviene en cajas, scores, filtrado, entrenamiento ni selección.
La validez de esa AP no convierte el TTC de TRAIN40 en evaluación independiente.

### 3.4 La geometría se debe resolver por formato

La diferencia entre `size`, `bbox_3d_ego`, bundle original y helper local no prueba
por sí sola un bug. Puede existir una conversión en el release HF. El council
exige un fixture externo o conversor documentado que establezca el significado
de cada campo antes de entrenar o interpretar AP. No basta con un round-trip
construido con la misma suposición en ambas direcciones.

## 4. Registro de decisiones y alternativas

| ID | Decisión propuesta | Razón | Qué permitiría revisarla |
|---|---|---|---|
| C01 | Modular primero | Aísla detección, asociación, recorte y TTC; conserva referencia existente | Evidencia de que duplicar cómputo domina el coste y compartir features aporta una mejora controlada |
| C02 | MonoDGP candidato inicial; MonoDETR control | Código/pesos documentados, familia próxima y cámara única | Fallo de operadores/memoria, coste de port desproporcionado o baseline alternativo más reproducible |
| C03 | ImVoxelNet como alternativa acotada | Cercanía a la familia geométrica de eAP | Perfil de volumen/memoria y disponibilidad del entorno deciden; no portar todos los candidatos |
| C04 | MAGneT-3D como referencia temporal posterior | Trabajo 2026 monocular con implementación real | Ego-motion causal disponible/estimable, ocho clases y port Blackwell verificados |
| C05 | Eventos mediante ablation sobre el mismo detector | Separa su contribución de cambiar arquitectura/presupuesto | Resultado medido que justifique routing o backbone evento más complejo |
| C06 | No importar LiDAR/estéreo de otros métodos | El input público inspeccionado es RGB+una cámara evento | Cambio explícito del contrato público de modalidades |
| C07 | TTC no determina scores ni filtra cajas en el MVP | Mantiene interpretación AP e independencia de la etapa TTC | Experimento separado, con entrenamiento y selección sin exposición a validación |
| C08 | Conservar fase y estados, no prometer intervalo TTC calibrado | Inversa singular/saturada y calibración no demostrada | Definición probabilística y cobertura verificadas en población reservada |
| C09 | Mantener namespace y artefactos históricos | Preserva trazabilidad científica y comparaciones | ADR posterior explícito; no renombrar por introducir una propuesta |

Las fuentes de la selección son [MonoDGP](https://github.com/PuFanqi23/MonoDGP),
[MonoDETR](https://github.com/ZrrSkywalker/MonoDETR),
[ImVoxelNet](https://arxiv.org/abs/2106.01178) y
[MAGneT-3D](https://arxiv.org/abs/2608.14282). La selección es de **candidatos para
medir**, no una clasificación SOTA de eAP.

## 5. Problemas, soluciones propuestas y límites residuales

| Problema comprobado o riesgo concreto | Solución que se propone | Límite que permanece |
|---|---|---|
| 40 etiquetas train y nueve TAR test ausentes | Descarga selectiva fijada a revisión; hashes y referencias | Integridad no equivale a geometría correcta; test completo no es requisito para piloto train |
| Órdenes de dimensiones/yaw entre formatos | Fixtures externos, reproyección y conversión versionada | Scorer/conversor oficial aún no verificado |
| Rectificación RGB/evento y velocidad poco documentadas | Verificar release y ejemplos; declarar marco y punto geométrico | No asumir poses ego temporales a partir de extrínseca fija |
| ROIs predichas introducen jitter | Comparar ROI 2D y proyección 3D; controles de ruido y recorte común | Filtrar puede borrar expansión o añadir retardo |
| Asociación mezcla identidades | Estado por track, gates de asociación y reset explícito | Una métrica global de tracking puede ocultar contaminación en 650 ms |
| Objeto nuevo sin historia | Entregar detección con TTC ausente y razón | Menor cobertura TTC al inicio, que debe publicarse |
| Más de veinte detecciones | Exportar todas; perfil de chunks/política TTC | No hay medición de latencia para esa carga |
| Fase cruza cero | Conservar límites de fase y marcar signo ambiguo | No hay un intervalo continuo de TTC obtenido ordenando extremos |
| Cambio de clase/forma/movimiento | Evaluación por clase y visibilidad; adaptación con exclusiones | Ocho clases de detección no prueban ocho clases TTC |
| Extensiones CUDA antiguas | Entorno candidato aislado y prueba forward/backward/decode | Soporte PyTorch Blackwell no garantiza operaciones de terceros |
| Subconjunto detectado parece más fácil | Publicar soporte común y cobertura total elegible | FP no tienen TTC verdadero; contabilizarlos aparte |
| Scorer CodaBench no accesible en revisión | Métrica local declarada provisional, pruebas sintéticas | No afirmar paridad oficial ni leaderboard |

Las etiquetas de test de otra tarea no resuelven estos problemas: ROIs GarlTTC
no deben alimentar como localización oracle el detector eAP.

## 6. Diseño científico mínimo aceptado por el consejo

### 6.1 Separar dos carriles

**Operativo:** detector independiente y PHASE TRAIN40 congelado. Medir integración,
paridad, memoria, disponibilidad y coste; etiquetar TTC interno como expuesto a
desarrollo. No presentar una nueva partición de esas secuencias como test nuevo.

**Generalización:** excluir grupos completos de todos los módulos que influyan en
la predicción evaluada. Generar predicciones OOF para entrenar etapas posteriores;
registrar pretraining, teachers, calibración y selección. Un refit final a TRAIN40
no preserva automáticamente la distribución de calibración de productores por fold.

### 6.2 Separar causas del error

| Brazo | Inputs | Interpretación |
|---|---|---|
| O | ROI e identidad anotadas | Referencia TTC condicionada, solo diagnóstico |
| R | ROI predicha; asociación GT solo para análisis | Degradación espacial/recorte |
| T | ROI predicha y asociación causal real | Sistema que se pretende usar |

Comparar TTC sobre soporte común O/R/T y publicar además, sobre todas las consultas
con TTC verificado, detección, asociación, historia disponible, emisión y abstención.
No usar todas las cajas 3D como denominador TTC si no tienen verdad válida. Informar
misses y FP aunque no entren en el MAE condicionado.

Separar métricas de detección, tracking, TTC e incertidumbre. No construir tras ver
los resultados un índice agregado que cambie el ganador. Las mejoras principales
requieren tres semillas y diferencias pareadas con bootstrap por secuencia. Fijar
antes del experimento márgenes relevantes y no inferioridad, con justificación.

### 6.3 Fase, velocidad y calibración

La fase actual acepta TTC negativo o positivo >0,1 s, limita la inversa a magnitud
60 s y convierte fase cero a +60 s. Mantener esa semántica como contrato histórico
o versionar cualquier cambio. Los nombres q10/q90 no acreditan cobertura del 80 %.

Una velocidad planar en marco ego no equivale automáticamente a derivada óptica
relativa. La consistencia `TTC = profundidad / velocidad` necesita definir signo,
movimiento ego y centro/superficie de referencia. El council rechaza introducir
esa pérdida antes de verificar esas condiciones.

Calibrar con secuencias reservadas, publicar cobertura/anchura y cruces de cero.
No prometer garantías de métodos conformales sobre ventanas correlacionadas sin
verificar sus supuestos, ni recalibración online con etiquetas de test privadas.
Referencia de alcance: [Angelopoulos y Bates](https://arxiv.org/abs/2107.07511).

## 7. Viabilidad de implementación

El proyecto usa Python >=3.11,<3.14 y Torch >=2.5,<2.12. La GPU local tiene
12.227 MiB totales. [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) documenta
soporte Blackwell/CUDA 12.8, pero eso no valida MMCV antiguo, atención deformable,
NMS 3D ni compilación en Windows. No rebajar el entorno actual para satisfacer
un repositorio externo; aislar el experimento y registrar la compatibilidad real.

Leer cada frame una vez y compartirlo entre detector/ROIs. Compartir la ventana de
eventos cuando sea posible, sin suponer que detector y PHASE usan el mismo binning
o normalización. Medir modelos residentes, activaciones, buffers, estados por track,
disco frío/caliente, colas y cambios de ROI que invaliden caches.

La medición previa de p95 29,14 ms corresponde a un objeto, cinco segundos y RAW
precargado, con disco excluido. No es presupuesto de una escena nueva. Tampoco se
obtiene la latencia p95 sumando percentiles de módulos. El consejo no estima días,
coste de nube ni FPS sin el primer perfil del candidato.

Una lista `[]` es una salida válida cuando el detector no encuentra objetos; no es
una reparación silenciosa de un RGB ausente. Los fallos de datos deben registrarse
y resolverse antes de declarar inferencia test completa.

## 8. Criterios de avance y de detención

Se utilizan los mismos gates G0–G7 que en el informe del swarm:

- **G0, datos/geometría:** completar lo necesario para train y demostrar convenciones.
  Bloquea entrenamiento geométrico si no se resuelve; permite seguir inspeccionando.
- **G1, entorno:** ejecutar forward, backward y decode reales con pérdida finita y
  memoria registrada. Si falla, cambiar configuración o candidato, no el entorno
  histórico de PHASE.
- **G2, baseline:** overfit pequeño como prueba funcional y validación disjunta como
  prueba de generalización. No confundirlas.
- **G3, eventos:** control RGB equivalente y coste declarado. Un resultado negativo
  mantiene el baseline RGB y se documenta.
- **G4, puente:** paridad con ROIs controladas, luego O/R/T, identidad y latencia de
  escena. TTC ausente no borra detecciones.
- **G5, adaptación/calibración:** exclusiones verificables y cobertura medida. Si no
  se cumple, conservar límites nativos sin afirmación probabilística.
- **G6, benchmark:** contrato/scorer/reglas verificados, tokens exactos, datos completos
  y modelo congelado. Sin acceso, mantener resultados locales como tales.
- **G7, compartición:** comparar contra modular con presupuesto y población explícitos.
  Si no mejora un compromiso aceptado, conservar la arquitectura separada.

Las pruebas de salida y export deben incluir cero objetos, más de veinte, valores
no finitos, dimensiones erróneas, clases inválidas y todos los tokens de test.
La prueba temporal debe incluir nacimiento, oclusión, cruce, cambio de identidad,
rollback, falta de modalidad y observación que llega tarde. El test de intervalos
debe incluir fases de ambos signos y saturación.

## 9. Disensos y asuntos que el council no resuelve

Hay acuerdo en la ruta modular, pero no evidencia para escoger un ganador final
entre los candidatos. La prioridad MonoDGP se apoya en reproducibilidad y encaje,
no en una comparación eAP propia.

El council científico rechaza exigir reentrenamiento completo antes de un piloto
operativo; sí lo exige, o exclusiones equivalentes verificables, para afirmar
generalización conjunta sin exposición. Arquitectura e ingeniería aceptan esa
distinción y mantienen paridad como objetivo inicial.

Los tres miembros rechazan «solo falta una cabeza», «TTC ya es robusto para cualquier
ROI/clase» y «los cuantiles ya son intervalos calibrados». Son afirmaciones que
las pruebas actuales no respaldan.

Permanecen abiertos: geometría HF exacta, rectificación, semántica de velocidad,
scorer y reglas actuales, soporte TTC por clase, compatibilidad de operaciones,
memoria con productores residentes y beneficio de eventos/backbone compartido.
Resolverlos requiere implementación o documentación adicional; el consejo no los
convierte en hechos por consenso.

## 10. Revisión de los documentos y alcance de la entrega

Primera ronda: los tres dictámenes anteriores se incorporaron al diseño y al
inventario. En la segunda ronda, los tres miembros leyeron ambos borradores.
Arquitectura y ciencia solicitaron corregir dos detalles antes de publicar;
ingeniería los consideró publicables internamente y añadió una precisión de perfil.

| Hallazgo de segunda ronda | Corrección aplicada |
|---|---|
| Se habían contado siete campos por detección | Son seis; enumerados en el contrato de salida |
| Se equiparaba tocar fase cero con cruzar ambos signos | Se distinguen cruce estricto, borde positivo, singleton cero y extremo cero desde fase negativa |
| El texto mezclaba memoria de entrenamiento e inferencia conjunta | Forward/backward del detector se perfila aparte de inferencia con productores TTC residentes |

La edición añade una tabla de supervisión/pérdidas para concretar la cabeza
aprendida, manteniéndola como propuesta. Los tres miembros hicieron una
comprobación focalizada de esos cambios y emitieron el cierre:

| Miembro | Dictamen de cierre |
|---|---|
| Arquitectura | Publicable internamente; sin fallos materiales restantes en su ámbito |
| Validez científica | Publicable internamente; sin objeciones materiales restantes |
| Ingeniería | Publicable internamente; gates de implementación todavía pendientes |

El agente principal comprobó la existencia de los archivos locales enlazados,
referencias Markdown definidas, bloques de código cerrados y alcance de los cambios.
La entrega solo añade estos dos documentos. No se ejecutó la suite de modelos:
ningún cambio altera código, configuración ejecutable ni resultados científicos.

Esta entrega contiene dos MD de investigación. Conserva resultados, pesos, código
de inferencia, artículo y evidencias históricas. No incluye un detector entrenado,
un score 3D, una descarga completa de datos ni una submission enviada.
