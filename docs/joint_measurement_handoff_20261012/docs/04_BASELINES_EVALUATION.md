# Qué significa ganar y cómo comparar

## Referencias comprobadas en web, 12-10-2026

- Garl oficial: https://github.com/NAIL-HNU/Garl-TTC . Release de RGB+eventos, visual-only y event-only; test12=6.762 inputs sin TTC GT público. Fila full publicada MiDc/s/l/n=53.1/37.6/40.6/31.3 (overall aritmético45.02). Son test12, no V/P/H.
- Artículo eAP: https://arxiv.org/html/2603.16303v1 . No mezclar MiD, RTE y MAE ni modalidades.
- REACT: https://arxiv.org/abs/2609.19204 . Preprint2026, full-field eventos, sin bbox, reporta9.59% RTE/4.6ms. No sustituye la evaluación de objetos eAP ni acredita por sí solo un líder comparable.
- Benchmark: https://www.codabench.org/competitions/17289/ . En este entorno no se recuperó un leaderboard legible; no se afirma haber verificado el líder actual.
- RAFT: https://arxiv.org/abs/2003.12039 . Correlación espacial es un antecedente de estimación de movimiento, no prueba que el nuevo TTC gane.

## Matriz de comparadores

1. A5 matched congelado P2 y nativo original; NATIVE-P2-FT; DIRECT/CORR/FROZEN con input especificado.
2. E_CTX matched y F_TRUE/R_CTX vigentes como sistemas completos, sobre V con sus padres correctos. No comparar productor H con cabeza V por el número bruto.
3. Checkpoints públicos Garl event/visual/full en transferencia, con su preprocesado nativo y disponibilidad. Procedencia de entrenamiento no certificada se declara; no se usan como init de una evaluación P/H/V que supuestamente excluyera sus grupos.
4. Test12 oficial cuando haya candidatos congelados y autorización de envío. No adquirir etiquetas privadas ni iterar hiperparámetros a partir del leaderboard.

El trabajo puede derivar de Garl si eso produce el mejor modelo. No existe una obligación de preservar A5 por identidad del proyecto. Un baseline Garl fuerte no debe degradarse con ventanas/capas que no le pertenecen o checkpoints parciales.

## Métricas

Conservar scorer pin, dominio, dt de la métrica, población y bandas c/s/l/n. Grupo-macro ponderado no es automáticamente el overall pooled del servidor. Si faltan bandas, no fabricar overall ni renormalizarlo para evaluación oficial.

Reportar además MAE/mediana/RTE, signos, MiDc, subrangos descriptivos, p95 de sobreestimación crucial, cobertura y fallos. No borrar salidas no finitas. Presentar clipping nativo y sensibilidad común aparte; ninguna sensibilidad sustituye el score principal.

Bootstrap por grupos de adquisición/secuencia, no por ventanas independientes. Seeds de heads no crean nuevas escenas. Los gates V17 siguen rechazados; los nuevos umbrales de EXPLORACIÓN no promueven esos modelos retroactivamente. La promoción final del sistema debe cumplir todas las salvaguardas registradas y la comparación con el baseline completo.

No definir SOTA como dominar cada muestra o cada métrica concebible. Definir modalidades, benchmark y recursos ANTES: event-only y RGB+eventos llevan tablas distintas. Mejoras de desarrollo pasan a una comprobación independiente; no se garantiza una puntuación.

## Transferencia y confirmación

FCWD/Dev32 ya han sido examinados. Se pueden reevaluar candidatos congelados como transferencia DESCRIPTIVA, nunca seleccionar su seed, pérdida o checkpoint con esos retornos y seguir llamándolos test ciego. La intersección RGB disponible debe ir acompañada de cobertura total/fallback. Registro local no es calibración extrínseca.

Coordinar con V16 la comparación Garl full FCWD; reutilizar resultados admitidos si ya existen. No bajar NCC según el error TTC. Si falta geometría, bloquear ese comparador, no toda la investigación event.

Tras refit congelado, autorizar sólo la preparación local de predicciones test12 y un ZIP root submission.json si el contrato de inputs lo admite. No hacer upload/submission en esta autorización. Entregar candidatos, hashes y comparación prevista; el envío al servidor requiere autorización expresa. La ausencia de score impide declarar SOTA, no entrenar una hipótesis mejor.

## Coste

Primero forward batch1 con tensores preparados; luego replay real con preparación y todos los encoders/heads, timestamps y ROI. Medir Garl nativo Y optimizado en el mismo hardware. Frío/warm, p50/p95/p99, memoria, throughput separados. Evitar medir bajo otro job sin declararlo. No culpar a la MLP de las copias/commit de caché. No denominar fase/GRU a un AEB certificado.
