# Fuentes primarias y estrategia de continuación de PHASE-TTC

Fecha de consulta: 2026-10-29. Nota de investigación independiente, fuera del repositorio. Se verificaron publicaciones, documentación y disponibilidad de código; no se ejecutaron estos métodos ni se comprobó su rendimiento o compatibilidad en la GPU local.

## 1. eAP: separar comparadores publicados y clasificación oficial

**Hechos verificados.** El artículo eAP es [2603.16303](https://arxiv.org/html/2603.16303v1), no 2604.24033; este último corresponde a un benchmark de SLAM. La tabla IV del artículo eAP evalúa **Car** y separa Driving/HDR. Para el modelo con cabeza de anclas informa:

| Entrada | Frames | AP Driving | AP HDR |
|---|---:|---:|---:|
| RGB | 1 | 0.510 | 0.403 |
| RGB + eventos | 1 | 0.515 | 0.460 |
| RGB + eventos | 3 | 0.531 | 0.558 |

Son resultados publicados bajo su protocolo, no una verificación del líder actual. No son comparables directamente con AP de desarrollo multiclase del piloto. El artículo adapta métricas nuScenes basadas en distancia entre centros y utiliza poses para fusionar historia BEV. El protocolo exacto debe contrastarse con el scorer.

El [proyecto oficial](https://nail-hnu.github.io/eAP_dataset/) describe 46 secuencias train y 12 test. El [README público de Hugging Face](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/raw/main/README.md), obtenido por HTTP, declara 118.247 muestras de 40 secuencias train y 35.940 de 12 test. **Pendiente:** reconciliar versiones y población; no asumir seis secuencias accesibles que permitan completar el protocolo del artículo.

El [repositorio oficial](https://github.com/NAIL-HNU/eAP_dataset) enlaza [CodaBench 16717](https://www.codabench.org/competitions/16717/#/pages-tab). En su árbol inspeccionado aparecen README, assets y visualizadores; no se localizaron el scorer, entrenamiento ni checkpoints del baseline. El README HF tampoco contiene esos enlaces y especifica que el test no publica etiquetas.

**Acceso no verificado:** frontend CodaBench y GET públicos a `/api/competitions/16717/` y `/api/competitions/16717/phases/` devolvieron HTML ajeno al benchmark, incluso con HTTP 200; no se obtuvo JSON ni tabla de clasificación válidos. Esto no demuestra ausencia del scorer ni permite identificar un líder.

**Causalidad:** el README del visualizador eAP declara una ventana de eventos de 50 ms centrada en el punto medio de exposición RGB. No trasladar ese default a producción sin verificar `decision_time` y disponibilidad de todos los inputs.

## 2. Direcciones prioritarias y límites de la evidencia

### MonoCoP: challenger RGB, después de una referencia estable

- [Artículo, revisión v7 del 29-03-2026](https://arxiv.org/abs/2505.04594), [proyecto](https://alanzhangcs.github.io/monocop-page/), [código oficial](https://github.com/alanzhangcs/MonoCoP).
- CVPR 2026; predicción condicionada entre atributos y selección mediante incertidumbre. Código, modelos y registros están enlazados públicamente.
- La receta publicada usa PyTorch 2.4.1/CUDA 12.1 y atención deformable compilada. La configuración KITTI inspeccionada emplea batch 16; anunciar una GPU no prueba que quepa en 12 GB ni que sus operaciones funcionen en Blackwell.
- **Propuesta:** conservar el detector portable y MonoDGP como controles; considerar MonoCoP como challenger bajo idénticos datos, clases y evaluación. No convertir un port externo en dependencia de la siguiente iteración. Sus resultados KITTI no establecen superioridad en eAP.

### SPAN: modificación geométrica aislable

- [Artículo CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_SPAN_Spatial-Projection_Alignment_for_Monocular_3D_Object_Detection_CVPR_2026_paper.html), [texto completo](https://arxiv.org/html/2511.06702v2), [código](https://github.com/WYFDUT/SPAN), [pesos](https://huggingface.co/yifanwang08/SPAN_Validation/tree/main).
- Pérdidas de alineación espacial y proyección, incorporadas progresivamente, sin módulos adicionales de inferencia. El entorno original basado en MonoDGP es antiguo y no está probado aquí.
- **Propuesta:** estudiar una regularización geométrica pequeña dentro del detector existente. Verificar antes la correspondencia entre cajas 2D visibles, truncadas y proyección de la caja 3D completa. Medir mejoras en centro/profundidad, orientación y clases minoritarias; no asumir que una mejora de IoU implica una mejora del AP por distancia.

### RARE: ranking de calidad si el diagnóstico lo justifica

- [Artículo CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.html), [código oficial](https://github.com/HyeonjeongPark37/RARE).
- **Existe código**, con publicación inicial anunciada el 28-05-2026 y directorios `lib/`, `tools/`, `utils/`. Documentación escasa, dependiente de recetas MonoLSS/MonoDETR. No se verificaron checkpoints oficiales descargables.
- **Propuesta:** diagnosticar primero la distancia entre ranking actual y ranking por calidad geométrica. Si el score explica una fracción importante de los errores, evaluar una cabeza/pérdida de calidad compatible con el criterio de acierto eAP. Reservar múltiples hipótesis para errores de profundidad que lo justifiquen. Mantener score de detección y confianza TTC separados.

## 3. Referencias temporales y de transferencia: condicionales

### MAGneT-3D

- [Artículo de agosto de 2026](https://arxiv.org/abs/2608.14282), [proyecto](https://mo-sameh.github.io/MAGneT-3D-Project-Page/), [código](https://github.com/mo-sameh/MAGneT-3D), [entorno](https://github.com/mo-sameh/MAGneT-3D/blob/main/docs/setup.md), [entrenamiento](https://github.com/mo-sameh/MAGneT-3D/blob/main/docs/training_inference.md).
- ECCV 2026 DriveX Workshop. Monocular temporal, propuestas dependientes de imagen/intrínsecas y refinamiento de historia por identidad. Código y enlaces a checkpoints ya publicados; snippets antiguos de «próximamente» están desactualizados.
- El entorno publicado está probado hasta Ampere: PyTorch 1.9/CUDA 11.1/MMCV 1.6/MMDetection3D 1.0.0rc6. Configuraciones de entrenamiento: dos GPUs por ocho muestras. No está validado en la GPU del usuario.
- El README actual informa NDS sin velocidad y resultados reentrenados distintos de los de la página del proyecto. No mezclar tablas ni presentar esas cifras como métrica eAP.
- **Propuesta:** referencia conceptual para memoria causal pequeña sobre el detector portable, no sustitución inmediata del stack. Warping por ego-movimiento requiere poses verificadas; la posición ego de un objeto no es una pose de ego.

### MonoIA

- [Artículo CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Towards_Intrinsic-Aware_Monocular_3D_Object_Detection_CVPR_2026_paper.html), [arXiv](https://arxiv.org/abs/2603.27059), [código](https://github.com/alanzhangcs/MonoIA), [modelos](https://huggingface.co/zhihao406/MonoIA).
- Adaptación a intrínsecas para transferencia entre cámaras/datasets; código y modelos publicados.
- **Propuesta:** reservarlo para un error de transferencia demostrado o entrenamiento externo con varias cámaras. No priorizar su complejidad sobre cobertura de clases, geometría, datos y representación visual.

## 4. Hipótesis de investigación recomendada

Estabilizar RGB y su evaluación; probar historia causal RGB; después comparar RGB, RGB temporal, RGB+eventos y RGB temporal+eventos. Evaluar gating de eventos solo tras identificar cuándo aportan información o degradan el resultado.

Controlar datos, capacidad, resolución, actualizaciones y disponibilidad temporal. Incorporar eventos anulados, permutados y desplazados en tiempo; desglosar por HDR/iluminación, movimiento, distancia y clase. Mantener PHASE separado hasta evaluar las ROIs predichas y la calidad de la historia de identidades.

Esto es una estrategia experimental propuesta, no una afirmación de novedad o garantía de SOTA. Distinguir tres objetivos: superar referencias reproducidas, alcanzar una clasificación oficial verificada y demostrar mejora conjunta de detección–TTC. Ninguno sustituye a los demás.
