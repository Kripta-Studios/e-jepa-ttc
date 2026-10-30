# Decisión de piloto

**Decisión: ninguna intervención aprendida justificada todavía.**

La auditoría reproduce la identidad y permite diagnóstico expuesto. Quedan sin
resolver las relaciones de adquisición de las 24 candidatas y la equivalencia
geométrica/temporal de los targets locales con el scorer oficial. El inventario
tampoco certifica todos los ancestros y decisiones históricas de selección.
Estos límites impiden presentar F0/F1 como una comparación de desarrollo elegible.

## Hipótesis y evidencia

| Hipótesis | Estado | Evidencia y siguiente observación |
|---|---|---|
| Los datos de competición comparten grabaciones con EvTTC-32 | supported | Ocho matches únicos y 365 frames; excluir grabaciones y relaciones |
| Un traslado global del reloj cambia la predicción | refuted en fixture | La misma historia relativa mantiene la salida; prueba con contextos y cabeza reales |
| Se confunden los extremos anterior/actual en la fórmula analítica | discriminada en fixture | Para delta 0,1 y crecimiento 1,1: TTC actual 1, anterior 1,1; falta certificar la semántica aprendida y del target oficial |
| Omitir traslación puede producir error dependiente de profundidad | supported físicamente | Fixture de cámara: desplazamiento omitido de 10 px a 1 m y 1 px a 10 m; no atribuye el fallo real de Slider |
| Un único problema de arranque explica ambos Slider | unresolved; patrón distinto | Las consultas predeclaradas conservan negativos/saturación en Slider-750 y salidas positivas sin ese rechazo en Slider-1000 |
| Cambiar la fusión mejora desarrollo independiente | unresolved | La tabla expuesta no sustituye validación agrupada |
| JEPA resolvería el límite de representación | unresolved | No se ha demostrado pérdida de señal dinámica en productores con datos elegibles |

## Próxima acción concreta

1. Revisar sesiones/relaciones de las 24 grabaciones con metadatos de adquisición.
   Asignar roles completos sin consultar scores del candidato.
2. Confirmar con los organizadores el endpoint, geometría, reloj y reglas del scorer.
   La ficha revisable está en [ORGANIZER_QUESTIONS.md](ORGANIZER_QUESTIONS.md).
3. Documentar la relación de Slider-500 con las dos adquisiciones conocidas.
   Implementar una política de lector separada para sus topics solo después de
   fijar el contrato, conservando el adaptador congelado.
4. Repetir las pruebas discriminantes en desarrollo elegible. Si conservan señal
   útil y sitúan el sesgo en la fusión, abrir F0/F1 con capacidad fija y semilla 7.

El presupuesto posterior sigue el ZIP: hasta dos candidatos motivados, selección
por grupos, confirmación 7/13/21 tras elegir receta. Margen práctico, protección de
TTC corto/negativos y presupuesto de latencia se fijan antes de selección con
datos y requisitos reales. No se han inventado valores para abrir el gate.

No se han ejecutado F1/F2/Fcap, P0, J0/J1/J2, confirmación externa ni nueva
submission. El capítulo 09 del paquete condiciona esas ramas; ejecutarlas todas
ahora contradiría su orden y sus criterios de promoción.
