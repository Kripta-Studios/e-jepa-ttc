# Fuentes, razonamiento y alcance de las propuestas

Consulta documental y bibliográfica: 31 de octubre de 2026. Las fuentes externas se contrastaron en páginas primarias. La revisión no reproduce sus experimentos. Las recomendaciones de este plan son inferencias y diseños de pruebas para PHASE-TTC, no resultados ya obtenidos.

## Evidencia interna que determina el orden

| Evidencia | Implicación para el plan | Límite |
|---|---|---|
| [Release y arquitectura](../../paper/sections/final_model.tex) | Intervenir PAIR, contexto o corrección final tiene una frontera concreta | La cabeza pequeña no representa el coste de toda la cadena |
| [Score oficial](../../paper/data/official_benchmark.json) | R0 es la referencia de producto y benchmark | Clasificación fechada, no score de nuevas variantes |
| [Recálculo de transferencia](../physics_review_20261031/RESULTS_VERIFIED.md) | Contrastar siempre contra el RGB del propio sistema | Cohorte positiva expuesta; no implica daño general de eventos |
| [screen149](../research_log_20261031_canonical_screen149/README.md) | La cadencia ya merece ampliación sobre F0 | Router posterior falla; screening no prueba promoción |
| [Historial OOF y sellado](../engineering_history_20261027/02_TRAIN40_OOF_AND_FINAL_TRAINING.md) | Recalcular ancestros/descendientes compatibles | Congelar TRAIN40 no restaura independencia |
| [Observabilidad histórica](../physics_review_20261031/07_OBSERVABILIDAD_Y_FUSION.md) | Revisar rechazo y estimabilidad específica | Stage68 no es una ablation del release |
| [Cierre JEPA](../jepa_closure_20261031/README.md) | Exigir un motivo nuevo antes de abrir esa vía | Un piloto negativo no demuestra imposibilidad universal |

## Fuentes primarias externas

| Fuente | Uso en el diseño | Lo que no permite concluir |
|---|---|---|
| [Horn, Fang y Masaki: Time to Contact Relative to a Planar Surface](https://people.csail.mit.edu/bkph/articles/Time_To_Contact.pdf) | Relación proyectiva y magnitudes relativas | Que toda expansión observada sea TTC de un objeto con giro |
| [Gallego et al.: Event-based Vision, v3](https://arxiv.org/abs/1904.08405v3) | Modelo y limitaciones del sensor de eventos | Que recuento identifique velocidad |
| [Guo et al.: Unsupervised Joint Learning of Optical Flow and Intensity, v2](https://arxiv.org/abs/2503.17262v2) | Auxiliar conjunto de movimiento/apariencia | Que mejore este release sin un contraste de TTC |
| [Low y Lee: Robust e-NeRF](https://arxiv.org/abs/2309.08596) | Umbrales asimétricos, refractario y sensor no ideal | Que haga falta entrenar un NeRF completo |
| [Yang y Ramanan: Optical Expansion](https://gengshan-y.github.io/expansion/expansion.pdf) | Expansión local y correspondencias | Que el determinante sea una contribución nueva |
| [Shiba et al.: Geometric Regularizer for Event Collapse](https://arxiv.org/abs/2212.07350) | Controlar soluciones degeneradas de alineación | Que deba prohibirse cambio de área real |
| [eAP, v1](https://arxiv.org/html/2603.16303v1) | Definición de target, adquisición y calibración | Que TTC instantáneo sea choque futuro con frenada |
| [Semigroup-JEPA, v1](https://arxiv.org/html/2609.10464v1) | Diseño condicional de rollout latente | Recuperación espontánea de física o mejora de TTC |
| [What Can Latent World Models Know?, v4](https://arxiv.org/html/2607.27017v4) | Separar información legible de información usada | Explicación causal del fallo local del piloto |

## Razonamiento que conduce a la prioridad

El release ya cuenta con mediciones y contexto. C-TIME tiene una anomalía observada y una intervención de coste reducido. C-OBS aprovecha la estructura de corrección anclada sin exigir rehacer los productores. C-CLOCK contrasta si falta una dinámica explícita, con un control sencillo que puede resolver el problema sin nueva representación. Esas tres vías producen evidencia sobre qué limita el sistema.

C-SENSOR aborda una fuente más profunda de desajuste, pero requiere RAW, referencias y fotometría suficientes. C-VIS es pertinente si el diagnóstico localiza error en oclusión, fondo o coordenadas. JEPA vuelve a ser razonable solo si esos análisis señalan pérdida de información dinámica dentro del productor y no un problema de lectura o fusión.

No se asignan probabilidades numéricas de éxito ni mejoras esperadas: los antecedentes no las identifican. La prioridad refleja evidencia local, coste de refutación y facilidad de atribuir cambios, no una garantía de resultado.

## Ideas que se descartan como restricciones generales

No usar brillo proporcional a 1/Z² para una superficie extendida resuelta; no conservar masa de eventos como partículas; no conservar área cuando se quiere medir expansión; no imponer qdot=q² en todas las maniobras; no tratar toda ausencia de eventos como inmovilidad; no usar una regularización numérica como prueba de observabilidad.

Estas exclusiones no impiden modelos locales con hipótesis verificables. Obligan a registrar el dominio donde la medición tiene sentido y a contrastar sus fallos contra R0 con cobertura completa.

## Trazabilidad de esta entrega

El archivo `evidence_snapshot.json` registra hashes de las fuentes locales principales y el HEAD observado al cerrar. `document_validation.json` registra comprobaciones del dossier. Esos recibos acreditan qué se consultó y que los enlaces locales resuelven; no acreditan ejecución de las pruebas propuestas ni revalidación de los checkpoints oficiales.
