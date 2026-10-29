# NEXT_DECISION al corte del 8 de octubre de 2026

**Decisión actual:** conservar los modelos TRAIN40 y ambas comparaciones Garl congeladas, cerrar R1 con el protocolo admitido y evitar nuevas afirmaciones de SOTA hasta disponer de una comparación oficial o independiente suficientemente equivalente.

## Trabajo cerrado

- E0 control analítico, con resultado desfavorable documentado; WIDE, con nueve fits y réplicas completas.
- Entrenamiento A5, C2F, PAIR y tres cabezas H8: endpoints completos y checkpoints conservados.
- Extracción de 88.744 features/predicciones TRAIN40 y análisis descriptivos de ajuste y coste.
- Comparación congelada EvTTC event-only y extensión RGB+eventos, con cobertura, métricas, predicciones, curvas en bundles y backups verificados.
- Revisión de contratos de entrada, sincronización, paridad y contabilidad. Ningún resultado autoriza reinterpretar TRAIN como test.

## Trabajo pendiente

1. **R1 eficiencia causal:** cerrar los 768 pares previstos. La instantánea conserva 486; faltan 282. El supervisor consta RUNNING al corte de las 19:00 de Madrid, tras reanudación automática. La paridad CPU completó 256/256 exactos, pero eso no reemplaza el cierre GPU. Sólo el informe final podrá comparar rutas incluyendo ingesta, memoria y preparación. Los datos de esa ruta proceden del protocolo histórico congelado; no son un benchmark nuevo de velocidad del sistema TRAIN40.
2. **Protocolo de precisión publicable:** fijar antes de ejecutar una nueva evaluación su población, métrica, geometría, selección de pesos y tratamiento de fallos. La comparación actual es transferencia local muestreada sobre desarrollo histórico, no reproducción exacta de la tabla oficial.
3. **Genealogía del checkpoint Garl:** obtener información verificable de entrenamiento y selección si se necesita afirmar independencia respecto a holdouts concretos. No sustituir esa dependencia por la simple existencia de TRAIN40 en Hugging Face.
4. **Replicación del sistema completo:** si se pretende estimar variación por entrenamiento de encoder, tres semillas de cabeza no bastan. No se inicia ese coste con esta entrega ni se presume autorizado por el saldo físico.

La rama E3 original permanece pausada por cambio de estrategia. Su ausencia no bloquea el resultado de las comparaciones públicas ya ejecutadas. No se plantea reentrenar todo Garl para rehacer una comparación que dispone de pesos finales.

## ETA y criterios de cierre

Los entrenamientos y comparaciones tienen duración restante cero. R1 conserva pausas y coste de reconstrucción de historia. No se da una fecha final usando un pico antiguo ni una ventana que estaba en espera. Se actualizará su ETA únicamente con pares nuevos y un intervalo sostenido posterior a la recuperación.

R1 se considerará cerrado cuando estén todos los fragmentos, paridad y contratos verificados, análisis por ruta y bundle esencial con SHA-256. Si una dependencia impide terminar, se registrará exactamente cuál y el comando de reanudación. Una espera o dependencia no se convertirá en resultado experimental negativo.

La siguiente conclusión de investigación debe distinguir precisión media, colas de error, mediana y coste. En esta población H8 presenta menores estimaciones puntuales de MAE/RMSE, Garl full presenta menor mediana y hay incertidumbre entre secuencias. Esa es la conclusión que se conserva en el informe y en GitHub.
