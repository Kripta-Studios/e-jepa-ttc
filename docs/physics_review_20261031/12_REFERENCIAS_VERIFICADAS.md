# Referencias primarias contrastadas

Consulta realizada el 31 de octubre de 2026. Se verificaron identidad, versión o contenido pertinente en fuentes primarias. Algunos enlaces PDF de CVF devolvieron 403 y se contrastaron mediante arXiv o resultados indexados del documento oficial. No se afirma haber reproducido sus experimentos.

| Fuente | Qué respalda | Límite para PHASE |
|---|---|---|
| [Horn, Fang y Masaki: Time to Contact Relative to a Planar Surface](https://people.csail.mit.edu/bkph/articles/Time_To_Contact.pdf) | Cociente de distancia/velocidad y medición proyectiva directa | Hipótesis geométricas; no evidencia de entrenamiento de PHASE |
| [Gallego et al.: Event-based Vision: A Survey](https://arxiv.org/abs/1904.08405) | Modelo y propiedades del sensor | Event rate no identifica por sí solo velocidad |
| [Guo et al.: flujo e intensidad, ICCV 2025](https://arxiv.org/abs/2503.17262) | Objetivos conjuntos basados en generación de eventos | Tarea de flujo/intensidad, no prueba de TTC |
| [Klenk et al.: E-NeRF](https://cvai.cit.tum.de/_media/spezial/bib/klenk2023nerf.pdf) | Cruces de umbral e intervalos sin eventos | NeRF y condiciones de observación distintas |
| [Low et al.: Robust e-NeRF](https://arxiv.org/abs/2309.08596) | Sensor no ideal, umbrales/refractario | No exige adoptar reconstrucción 3D completa |
| [Shah et al.: CodedEvents](https://arxiv.org/abs/2406.09409) | Información/Cramér–Rao para diseño óptico de tracking | No demuestra un gate TTC óptimo |
| [Bakarji et al.: Dimensionally consistent learning with Buckingham Pi](https://www.nature.com/articles/s43588-022-00355-5) | Consistencia dimensional como estructura de aprendizaje | No sustituye validación sensorial ni anclaje |
| [Yang y Ramanan: Optical Expansion](https://gengshan-y.github.io/expansion/expansion.pdf) | Cambio de área/profundidad y ambigüedad de escala | Giro/deformación requieren hipótesis adicionales |
| [Hu et al.: From Contrast to Consistency](https://arxiv.org/abs/2605.25570) | Estructura y continuidad en flujo de eventos | Arquitectura bidireccional; adaptar frontera causal |
| [Shiba et al.: regularizador de event collapse](https://arxiv.org/abs/2212.07350) | Degeneraciones de maximización de contraste | No justifica prohibir expansión física |
| [Semigroup-JEPA, v1](https://arxiv.org/html/2609.10464v1) | Rollout conjunto condicionado por física conocida | Composición discreta por iteración no es identificación de q |
| [What Can Latent World Models Know?, v1](https://arxiv.org/abs/2607.27017v1) | Versión y título que aparecen en la propuesta | La interpretación debe indicar versión |
| [What Can Latent World Models Know?, v4](https://arxiv.org/html/2607.27017v4) | Diferencia entre información leíble y usada | No identifica el mecanismo del fallo local JEPA |
| [eAP, v1](https://arxiv.org/html/2603.16303v1) | Definición de TTC y contexto de adquisición | Targets públicos reconstruidos y privados no se equiparan |
| [Durand: Light and physics](https://people.csail.mit.edu/fredo/comp-photo-book/02-fundamentals-01-light-and-physics.html) | Distinción radiancia/irradiancia | No usar brillo inverso al cuadrado como telémetro general |

## Cambios de lectura frente al texto recibido

La propuesta usa el título de julio de *What Can Latent World Models Know?*. La página principal consultada muestra v4, con otro subtítulo y énfasis en separar lectura y uso. Se contrastaron ambas. No se debe citar el enlace sin versión y atribuirle todas las conclusiones fuertes de v1 sin revisar la revisión posterior.

Semigroup-JEPA no demuestra que la ley qdot=q² aparezca espontáneamente. Su interés aquí es un diseño de aprendizaje temporal y su alcance declarado. Usarlo como antecedente no reduce la necesidad de los controles propios.

La relación de TTC, el criterio c ortogonal al núcleo, la regla de la cadena para Jacobianos y los cambios de unidades se analizan como derivaciones matemáticas. La decisión de combinarlos en PHASE es una hipótesis de esta revisión. Ninguna fuente externa certifica que esa combinación mejore sus resultados.

Los enlaces de los informes internos remiten a evidencia local concreta; `PHASE_TTC_Research_Report` en el texto enviado no es por sí solo una cita resoluble dentro del árbol. El dossier identifica las secciones y los recibos que respaldan cada afirmación usada.
