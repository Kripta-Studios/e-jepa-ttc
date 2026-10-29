# V17: información anterior a PHASE17

Autorización nueva del usuario: «otras 18h de GPU y 1M de updates o lo que necesites». Se adoptan como techos duros 64.800 segundos de ocupación GPU de procesos propios y 1.000.000 updates físicos, incluyendo técnicos y recuperación. No se reutiliza presupuesto V15. El límite no obliga a agotarlo. V13/R1/V16 y los endpoints V14/V15 quedan en lectura solamente; no push, submission, merge ni cambios globales.

## Evidencia y pregunta

El diagnóstico P/H de endpoints V15 muestra que en H≤1s C20 tiene los mismos tres errores de signo que C05, pero MAE de signo correcto 3,106 frente a 1,654 s. Cinco consultas acumulan el 63,0% del error absoluto C20 y seis de siete grupos empeoran MAE. No basta atribuir el fallo al límite o al clipping.

Pregunta prospectiva: ¿las señales geométricas y de transporte de A5, antes del residual y de la compresión a PHASE17, permiten a H8 mejorar sin sacrificar otras bandas?

## Cola cerrada A

Dos brazos Z (control con bypass desactivado) y G (bypass activo), cada uno con seeds 7,13,23, 2.500 updates. Se congelan los seis antes de predecir V. No se elige epoch ni semilla. Total científico de esta cola: 15.000 updates. Son seis readouts de encoders compartidos, no seis representaciones independientes.

Ambos usan GRU H8 nativa de dos capas/160, mismos pesos iniciales de la ruta nativa por seed, AdamW3e-4, wd1e-3, batch128, FP32, TF32 off, clipping1, mismo sampler uniforme y masas M0/L0 globales, scheduler nativo2500, normalizador PHASE17 nativo H, reloj, PAIR y decoder. Entrenamiento GPU en ambos; no se presenta como reproducción bit a bit del fit CPU V14.

El bypass es MLP(896,160)-LayerNorm-SiLU-Linear(160,160), con última capa cero. Se suma al embedding de entrada de la última observación antes de GRU. Entrada: tres geometry_tokens128 y dos pares de forward_transport128/reverse_transport128 de A5 original matched; normalización media/std ajustada sólo H. No RGB, target, identificadores, ni features del A5 C20. Z multiplica por cero la salida del mismo módulo; controla arquitectura e inicialización. La ruta G reutiliza señales del mismo encoder y no agrega otro encoder. Esta prueba usa vectores geométricos/transportados actuales; no es una prueba de tokens espaciales densos ni de toda su historia.

## Decisión y evaluación

V es desarrollo expuesto y nunca se llama test nuevo. Misma población16432; no FCWD/Dev32 ni etiquetas privadas. Métricas y gates V14 sin modificaciones. Comparar G vs Z por seed y G seed7 vs histórico V14M0; además promedio aritmético de métricas por seed y CI pareado por grupos. No ensemble de predicciones. Candidato sólo si los tres contrastes pareados y el histórico cumplen todos los gates, incluidas cobertura y coste. El coste se mide head-only emparejado y se distingue de E2E; ninguna promoción del sistema antes de coste E2E comparable. Si falla, registrar negativo, sin rescate de umbrales, seed o coeficientes.

Las fuentes nuevas, entradas, normalizador, padres, protocolo, entorno y RNG se ligan por SHA antes del primer update. Pruebas: control=nativo, bypass inicial cero, targets prohibidos en inferencia, normalizador H, alineación por sample_token, integridad de shards, y resume continuo10 vs5+restore5 sintético. Contabilidad durable antes/después de cada optimizer.step; reservas conservadoras para tiempo GPU y recuperación.

Esta cola no autoriza agotar el millón en variantes arbitrarias. Cualquier siguiente cola requiere hipótesis y configuración registradas antes de sus fits. Los análisis descriptivos no cambian los gates y todos los resultados permanecen.
