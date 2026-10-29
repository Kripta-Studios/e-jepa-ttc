# Revisión independiente: fuentes, estrategia SOTA y temporalidad

Fecha: 2026-10-29. Revisor: agente independiente de investigación de fuentes; no autor del plan. Alcance: exactitud de las fuentes y claims externos, selección de métodos, comparabilidad eAP y estrategia temporal/eventos. No se editó el plan ni se reprodujeron resultados del piloto, entrenamientos o kernels GPU.

Documento revisado: `deliverables/sota_continuation_20261029/SOTA_DEVELOPMENT_PLAN.md`.

SHA-256 de la versión inicialmente revisada: `6bf8a68f5af979b3e43180fdf3be1b624be3eac23afd2f0b25b8bcfd4c7d712f`.

SHA-256 de la versión revisada tras las correcciones: `4cd86dbeab35a86ab95bfda85921da064f36ae3006578e84d2bcbeb57b003aaf`.

**SHA-256 final de cierre:** `d63065f341594613554aecb48219b4bbc967aed2fd5628d3c6bd599cce7013b7`.

## Dictamen

**Sin hallazgos materiales ni observaciones menores abiertas dentro de este alcance.** El plan diferencia evidencia publicada, decisiones propuestas y bloqueos; no promete SOTA, no certifica compatibilidad de stacks externos y no atribuye precisión TTC a la demo. Las dos precisiones menores de la primera revisión fueron incorporadas y se confirmó su cierre mediante relectura de los párrafos afectados.

La relectura de la versión final confirmó las precisiones añadidas sobre GT v2 tanto en validación interna como externa, rol `audit` explícito y protegido, comandos de comparación que preservan tablas históricas, bloqueo del flip con semántica de velocidad no resuelta y posibilidad de avanzar de S1 a S3 conservando el baseline si S2 resulta negativo. Son coherentes con el alcance científico y con la estrategia previamente revisados; no generan hallazgos nuevos. F1 y F2 permanecen cerrados.

## Hallazgos

### F1 — Baja: hacer inequívoco el alcance de «SOTA» en la lista final

- **Estado: cerrado.** En la versión corregida, el punto 2 exige scorer oficial/versionado para el claim oficial y limita la reproducción local a «mejor entre estos métodos reproducidos bajo el protocolo X», excluyendo liderazgo oficial o general. Texto releído y cierre confirmado.
- Localización: sección 11, «Condiciones acumulativas para decir SOTA», punto 2.
- El texto admite «scorer y versión verificables, o reproducción local claramente delimitada de los comparadores». El resto del documento distingue correctamente resultados locales y oficiales, pero esta disyunción bajo ese título podría interpretarse como autorización para declarar SOTA general tras superar solamente los comparadores reproducidos.
- Sugerencia: añadir que esa segunda vía solo permite afirmar «mejor resultado entre los métodos reproducidos bajo el protocolo X»; un claim de liderazgo oficial continúa condicionado al protocolo/ranking oficial verificado. Mantener la distinción ya presente en la sección 1.
- Severidad baja: no invalida la estrategia ni las conclusiones actuales; es una precisión de alcance.

### F2 — Baja: distinguir poses de cámara y trayectoria del ego en ImVoxelNet

- **Estado: cerrado.** La fila corregida distingue datos/calibración/protocolo del requisito adicional de poses de ego-movimiento para fusión BEV temporal, y afirma que esas poses no son requisito del control de un frame. Texto releído y cierre confirmado.
- Localización: sección 3, fila «ImVoxelNet / baseline eAP», condición «si datos/poses/protocolo y coste permiten».
- Una referencia monocular de un frame necesita intrínsecas/extrínsecas verificadas; la trayectoria de ego-movimiento es una dependencia adicional para el warping temporal del baseline eAP. La redacción conjunta puede hacer parecer que la falta de esa trayectoria impide también probar un comparador monocular sin historia.
- Sugerencia: precisar «calibración para un frame; poses de ego-movimiento adicionales para temporalidad BEV». La sección 8 ya trata bien esta diferencia.
- Severidad baja: no bloquea la ruta portable elegida ni exige cambiar la arquitectura.

## Comprobaciones satisfactorias

| Área | Resultado de la revisión |
|---|---|
| eAP Tabla IV | Cifras Car Driving/HDR RGB .510/.403, fusión un frame .515/.460 y tres frames .531/.558 fieles a la fuente. Se evita equipararlas a APdev de varias clases. |
| Release y clasificación | Diferencia 46 train del paper/proyecto frente a 40 del release HF señalada como pendiente. Acceso no verificable a CodaBench no se convierte en afirmación de ausencia del benchmark/scorer. |
| SPAN | Descripción de alineación espacial/proyección y pérdidas progresivas correcta. Código y pesos existen. Se propone adaptación controlada, no copia de un stack sin validar. |
| RARE | Hay código oficial; no se describe incorrectamente como paper sin implementación ni se atribuyen checkpoints no comprobados. La adaptación de calidad al criterio eAP está marcada como propuesta. |
| MonoCoP | Código/modelos/logs publicados y uso como challenger posterior coherentes con las fuentes. No se presenta como mejor modelo actual ni como compatible por defecto con 12 GB. |
| MAGneT-3D | Workshop y carácter monocular temporal bien identificados. El plan no mezcla sus cifras de proyecto y repositorio ni supone que su entorno antiguo funcione en Blackwell. |
| MonoIA | Prioridad condicionada a transferencia/cambio de intrínsecas razonable; no se añade como requisito para una cámara calibrada. |
| Temporalidad | RGB temporal precede a atribuciones sobre eventos; controles de historia repetida/permutada, llegada desconocida, cutoff y memoria acotada son pertinentes. No se usa la posición del objeto como pose del vehículo. |
| Eventos | Factorial RGB/historia/eventos y control de capacidad permiten separar contribuciones. Se reconocen perturbaciones fuera de distribución y se fija la causalidad; gating se presenta como hipótesis, sin garantía de novedad. |
| PHASE | Se conserva separado, con productores/ABI y exposición histórica explícitos. Mejorar detección no se convierte en precisión TTC ni en independencia de TRAIN40. |
| Compatibilidad y recursos | Ninguna mejora externa depende solo de `torch.cuda.is_available`; se exigen admisión, finitud, decode, reload y recursos medidos. Presupuestos futuros no se confunden con resultados. |

## Fuentes contrastadas

- [eAP, artículo y Tabla IV](https://arxiv.org/html/2603.16303v1), [proyecto](https://nail-hnu.github.io/eAP_dataset/), [repositorio oficial](https://github.com/NAIL-HNU/eAP_dataset), [README HF](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/raw/main/README.md).
- [SPAN, artículo](https://arxiv.org/html/2511.06702v2) y [código](https://github.com/WYFDUT/SPAN).
- [RARE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.html) y [código](https://github.com/HyeonjeongPark37/RARE).
- [MonoCoP, artículo](https://arxiv.org/abs/2505.04594) y [código](https://github.com/alanzhangcs/MonoCoP).
- [ImVoxelNet, artículo WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Rukhovich_ImVoxelNet_Image_to_Voxels_Projection_for_Monocular_and_Multi-View_General-Purpose_WACV_2022_paper.html).
- [MAGneT-3D](https://arxiv.org/abs/2608.14282), [entorno publicado](https://github.com/mo-sameh/MAGneT-3D/blob/main/docs/setup.md) y [entrenamiento](https://github.com/mo-sameh/MAGneT-3D/blob/main/docs/training_inference.md).
- [MonoIA, artículo](https://arxiv.org/abs/2603.27059) y [código](https://github.com/alanzhangcs/MonoIA).

La revisión usa la investigación primaria realizada en esta misma sesión y la lectura completa de las secciones pertinentes del plan. No afirma que esta selección sea exhaustiva ni que certifique el líder vigente de ningún benchmark.
