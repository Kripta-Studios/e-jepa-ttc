# Qué tendría que demostrar una afirmación de SOTA

Fuentes primarias consultadas el 12-10-2026. Esta revisión no reproduce inferencia ajena ni establece un ranking exhaustivo.

| Referencia | Población/modalidad | Implicación para nuestra comparación |
|---|---|---|
| [GarlTTC, repositorio oficial](https://github.com/NAIL-HNU/Garl-TTC) | RGB+eventos; test12 publicado como inputs sin TTC GT | El release anuncia 6.762 consultas y MiD por bandas 53,1/37,6/40,6/31,3. Esos números no se comparan directamente con nuestras 16.432 consultas V. El test oficial requiere la evaluación del organizador. |
| [Anotaciones oficiales GarlTTC](https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset) | Train público; test sin etiquetas | Mantener P/H/V derivados de train no convierte V en test independiente. No adquirir etiquetas privadas. |
| [eAP/Garl-TTC, artículo](https://arxiv.org/html/2603.16303v1) | eAP, FCWD y EvTTC; evaluación por cuatro bandas | MiD usa log de motion-in-depth, con eta=1−delta/T. Hay que conservar reloj, rangos, negativos y definición al comparar. |
| [REACT, preprint septiembre2026](https://arxiv.org/html/2609.19204v1) | Eventos full-field sin bbox; validación EvTTC; TTC positivo | Reporta RTE9,59±0,74% y latencia4,6ms. Su MiD ponderado usa log inverse-TTC y omite negativos; no es nuestra métrica de cuatro bandas. La página indica que el código se publicará tras la publicación. Es una referencia relevante, no una reproducción local. |

El acceso a CodaBench no devolvió aquí una tabla de clasificación interpretable. No se afirma haber comprobado el líder actual. No se realizaron submissions, descargas de nuevos datasets ni cambios en V16, cuya comparación multimodal tiene propietario distinto.

La prioridad local es obtener una ganancia reproducida con controles y medir el coste de la ruta completa. Aun si ocurre, V seguirá siendo desarrollo expuesto. Una futura comparación externa debe fijar población, inputs disponibles, sincronización, cobertura, hardware, precisión y scope antes de ejecutarse; publicar una cifra menor en otra población no acredita SOTA.
