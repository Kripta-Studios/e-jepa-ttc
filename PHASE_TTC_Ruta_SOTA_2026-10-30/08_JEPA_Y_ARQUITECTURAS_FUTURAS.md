# 08 — JEPA y arquitecturas futuras: investigar movimiento, no solo embeddings

**Objetivo propuesto:** determinar si aprender una representación temporal nueva mejora TTC más allá de la preparación correcta, la geometría y una fusión pequeña. Esta rama corresponde a `PHASE-MOTION`, nombre propuesto. No modifica el candidato congelado ni hereda automáticamente sus resultados.

## 1. Condición para abrir esta rama

Abrirla después de completar el diagnóstico y el piloto de [07 — Fusión y fine-tuning](07_FUSION_Y_FINE_TUNING.md), si queda evidencia de que los productores no conservan información dinámica útil. La evidencia puede ser un probe incapaz de recuperar expansión/signo con datos correctos, errores de movimiento que persisten con fusión adecuada o una ventaja reproducible de otro encoder congelado.

No abrirla para resolver un offset de timestamps, una cámara mal emparejada o ausencia de desarrollo Slider independiente. Esas causas siguen teniendo experimentos más específicos.

## 2. Diseño mínimo y causal

**Propuesta, pendiente de implementación:** encoder compacto de ROI RGB y/o eventos, agregación temporal pequeña y cabeza TTC/expansión firmada. Mantener por separado tres señales: apariencia, cambio temporal y calidad de observación. No imponer que una representación de acercamiento sea igual a otra con diferente velocidad.

Entrada de cada consulta: datos disponibles hasta \(t\), tiempos físicos, geometría de recorte, duración efectiva y soporte. Empezar con el mismo horizonte del candidato comparable; cualquier cambio de horizonte será una ablación separada. La cabeza debe conservar TTC negativo y la política de cobertura del evaluador.

En preentrenamiento se pueden usar observaciones futuras **solo como targets de ejemplos de entrenamiento**. El encoder de contexto y la salida TTC a tiempo \(t\) no pueden consumirlas. Los clips de validación/test nunca se usan como targets de preentrenamiento, aunque se ignoren sus etiquetas.

El teacher recibe el target futuro de TRAIN, produce un embedding con `stop-gradient` y se actualiza por la receta fijada —por ejemplo EMA—, idéntica en J1 y J2. El predictor contextual recibe únicamente pasado, máscara y horizonte. Un teacher congelado externo también debe tener procedencia documentada. Pseudolabels de TTC de un teacher supervisado son **destilación**, no prueba de entrenamiento sin supervisión.

## 3. Contrastes J0, J1 y J2

| ID propuesto | Entrenamiento | Pregunta que responde |
|---|---|---|
| **J0 — supervisado desde cero** | Mismo encoder aleatorio, entrenado con TTC elegible; presupuesto total de actualizaciones equiparado | ¿La complejidad del preentrenamiento aporta ventaja práctica frente a aprendizaje directo? |
| **J1 — JEPA base** | Predicción de embeddings + mecanismo anti-colapso elegido; después misma cabeza/ajuste TTC | ¿Ayuda el preentrenamiento predictivo bajo este presupuesto? |
| **J2 — J1 + objetivo dinámico** | Exactamente J1, añadiendo un término que preserve cambios temporales | ¿El objetivo dinámico mejora sobre el JEPA base? |

**El contraste causal principal es J2 frente a J1:** mismo encoder, inicialización emparejada por semilla, datos, lotes, teacher, actualizaciones, máscaras, optimizador, cabeza y selección. Solo cambia el término dinámico. J0 es un control práctico fuerte; sus diferencias en exposición a etiquetas deben declararse, no ocultarse bajo «mismo presupuesto».

Para la primera comparación usar un subconjunto TRAIN con etiquetas elegibles disponibles para todas las variantes; J1/J2 ignoran esas etiquetas durante preentrenamiento. J0 usa el mismo presupuesto total en entrenamiento supervisado. Todas las variantes comparten la misma evaluación y la fase final de lectura/ajuste. Reportar por separado actualizaciones totales, exposiciones a imágenes y exposiciones supervisadas. Si después se añaden datos sin etiquetas a J1/J2, esa es otra condición experimental; J0 ya no tiene idéntica información.

Controles adicionales baratos: encoder aleatorio congelado con la misma cabeza, y productor actual congelado. No comparar una cabeza débil del baseline con una cabeza temporal mayor del candidato.

## 4. Objetivo dinámico concreto

[MotionJEPA](https://arxiv.org/abs/2609.23881) y su [código](https://github.com/mkarmann/motion-jepa) motivan predecir embeddings de cambios temporales: evitar que el encoder represente el fondo y omita la parte rápida. La transferencia a TTC será una hipótesis propia, no un resultado publicado de MotionJEPA.

Para J2, implementar primero un objetivo auxiliar sencillo: a partir de embeddings de dos instantes TRAIN, predecir una representación del cambio temporal calculada por una rama objetivo separada. Con RGB puede partir de diferencias de intensidad normalizadas; con eventos, de estadísticas espaciotemporales/expansión con validez explícita. Usar el mismo horizonte físico y corregir diferencias de exposición. No confundir cambios de brillo con movimiento.

Una formulación propuesta es:

\[
L_{J2}=L_{J1}+\lambda_{dyn}L_{dyn},
\]

con \(\lambda_{dyn}\) elegido únicamente en desarrollo. Registrar escala y gradientes de ambos términos para detectar que uno domina. No cambiar simultáneamente encoder, augmentations y pérdida. La primera búsqueda tendrá cero y dos valores positivos predefinidos: **J1 es el baseline con cero y las dos configuraciones J2 son los dos candidatos**, conforme al presupuesto de [09](09_MATRIZ_EXPERIMENTAL_Y_DECISIONES.md). J0 es el control supervisado prescrito, no otra búsqueda de hiperparámetros. Ampliar solo ante un fallo concreto de escala.

Si se añade una pérdida geométrica, hacerlo en una ablación posterior. La consistencia \(\tau(t+\Delta)=\tau(t)-\Delta\) solo corresponde a velocidad relativa constante; usar una incertidumbre de proceso o máscara de validez y comprobar aceleración/frenada. No forzar una trayectoria suavizada a costa de retrasar una colisión real.

## 5. Probes y controles que pueden refutar la hipótesis

Ejecutar antes del entrenamiento completo y después con el encoder congelado:

1. **Expansión y signo:** recuperar escala relativa y acercamiento/alejamiento con una cabeza pequeña, fijada para todas las variantes.
2. **Orden temporal:** permutar o invertir el orden en fixtures físicamente interpretables. La invariancia completa puede indicar que el encoder ignora dinámica; no implica por sí sola que cualquier sensibilidad sea útil.
3. **Cambio de velocidad:** si \(t'=k t\), el target cinemático cambia como \(\tau'=k\tau\). Comprobar equivarianza en datos sintéticos válidos y transformaciones coherentes.
4. **Apariencia constante, movimiento diferente:** mismo fondo/objeto con tasas de expansión distintas. Descartar un modelo que predice TTC a partir de identidad o tamaño aparente sin distinguir dinámica.
5. **Movimiento constante, apariencia diferente:** variar iluminación/textura sin alterar geometría; medir robustez y evitar premiar invariancia a cambios físicos reales.
6. **Causalidad:** modificar todos los frames/eventos posteriores a \(t\); la predicción a \(t\) debe permanecer igual dentro de la tolerancia numérica fijada.
7. **Colapso parcial:** medir varianza y rango efectivo, pero también probes. La varianza alta global no garantiza que se conserve expansión.

La mejora de pérdida latente no basta. La promoción requiere mejorar TTC por grupos y colas relevantes, sin deterioro material de negativos, cobertura y retención eAP. Usar las reglas de [09](09_MATRIZ_EXPERIMENTAL_Y_DECISIONES.md) y [10](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md), no añadir umbrales después de ver el reservado.

## 6. V-JEPA 2.1 como referencia congelada

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) incorpora objetivo denso, supervisión intermedia y features temporales. El [repositorio oficial](https://github.com/facebookresearch/vjepa2) ofrece ViT-B de unos 80M parámetros. Compararlo como encoder congelado con idéntica cabeza, clips y presupuesto de entrenamiento de la cabeza. Cachear features solo de particiones autorizadas, con hash de checkpoint, preprocessing y clips.

Un encoder bidireccional dentro de un clip enteramente anterior a \(t\) puede ser causal respecto a la consulta. Un clip centrado en \(t\) con frames posteriores no lo es. Registrar el timestamp máximo real de cada tensor, incluidas interpolación y duplicación de frames.

La GPU de 12 GB obliga a medir resolución y longitud de clip antes de lanzar nada. Empezar con batch pequeño, encoder congelado y precisión adecuada al hardware verificado. No prometer memoria exacta ni entrenar un foundation model desde cero. Solo descongelar un bloque si el encoder congelado demuestra una ventaja que la justifique.

## 7. SSM, SNN y probabilidad: ramas secundarias

[REACT](https://arxiv.org/abs/2609.19204) inspira estado temporal con intervalos físicos. Antes de una SNN, comparar una GRU pequeña, un SSM convencional y el filtro causal con las mismas entradas e historia. La ventaja debe sobrevivir a medir latencia de streaming, arranque, actualización y consumo; el bajo coste teórico de un spike no demuestra menor tiempo en una GPU.

[LeWorldModel](https://arxiv.org/abs/2603.19312) puede servir como referencia compacta de preentrenamiento; [Fast-LeWM](https://arxiv.org/abs/2606.26217) se centra en planificación con acciones y no es una sustitución inmediata de PHASE. [Variational JEPA](https://arxiv.org/abs/2601.14354) y [MDN-VJEPA](https://arxiv.org/abs/2610.13956) justifican estudiar incertidumbre multimodal solo si una distribución simple y un ensemble no representan la ambigüedad observada. El [predictor JEPA portable](https://arxiv.org/abs/2607.16274v3) es posterior a demostrar que la oclusión explica errores importantes.

## 8. Implementación y cierre

Archivos **propuestos**, pendientes de contrastar con la estructura final del repositorio:

| Archivo | Contrato |
|---|---|
| `configs/research/motion_jepa_j0.yaml` | Presupuesto supervisado, particiones, encoder y semilla |
| `configs/research/motion_jepa_j1.yaml` | Igual base, teacher/predictor y pérdida JEPA |
| `configs/research/motion_jepa_j2.yaml` | Diferencia mínima: objetivo dinámico y coeficiente |
| `tools/research/probe_motion_representation.py` | Probes, causalidad y controles físicos |
| `docs/research/motion_jepa_result.md` | Resultados emparejados, recursos, fallos y decisión |

Terminar el piloto con una decisión: **promover**, **reformular una hipótesis concreta** o **cerrar la línea**. Si J2 solo mejora el entrenamiento o una secuencia inspeccionada, no promover. Si gana en desarrollo pero el reservado no confirma, publicar la discrepancia. La novedad defendible sería conservar dinámica relevante para TTC y mejorar transferencia con igual capacidad/datos, especialmente en arranque, dominio nuevo y colas de error; añadir la palabra JEPA al modelo no constituye esa contribución.
