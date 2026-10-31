# Plantilla — Reporte de resultados y decisión

**Estado:** pendiente de cumplimentar. Las tablas numéricas se generan desde JSON/CSV; no se introducen resultados supuestos.

## 1. Identidad y alcance

| Campo | Valor |
|---|---|
| Experimento / hipótesis | PENDIENTE |
| Variante y baseline | PENDIENTE |
| Commit / hashes de pesos y normalizadores | PENDIENTE |
| Configuración / scorer / manifest | PENDIENTE |
| Semillas y regla de checkpoint | PENDIENTE |
| Tipo de evidencia | Oficial recibido / reproducción equivalente / diagnóstico parcial / desarrollo / confirmación |
| Qué test o scores se conocían al diseñar el cambio | PENDIENTE |
| Uso autorizado de datos y modalidades | PENDIENTE |

**Pregunta prefijada:** PENDIENTE.  
**Modificación realizada:** PENDIENTE.  
**Criterio de selección y margen práctico fijado previamente:** PENDIENTE.  
**Desviaciones respecto al protocolo y cuándo ocurrieron:** PENDIENTE.

## 2. Población y cobertura

| Modelo | Consultas requeridas | Inputs disponibles | Salidas finitas | GT elegible | Consultas compartidas | Fallos / razones |
|---|---:|---:|---:|---:|---:|---|
| Baseline | — | — | — | — | — | PENDIENTE |
| Candidato | — | — | — | — | — | PENDIENTE |

Una salida finita no acredita soporte fiable ni ausencia de fallo según el benchmark. No quitar consultas para mejorar una media. Separar GT ausente de fallo del modelo.

## 3. Métricas generadas

| Dominio / población | Métrica y unidad | Baseline | Candidato | Diferencia pareada | Grupos independientes | Incertidumbre y método |
|---|---|---:|---:|---:|---:|---|
| PENDIENTE | PENDIENTE | — | — | — | — | PENDIENTE |

- Publicar macro, micro y MAE cuando corresponda, en columnas distintas.
- Separar RTE EvTTC, MiD GarlTTC y pérdidas de entrenamiento.
- Mostrar salida bruta y postprocesamiento oficial por separado.
- Adjuntar errores por secuencia/grupo, rango TTC y signo.
- Marcar derivados de tablas redondeadas y diagnósticos parciales.

## 4. Réplicas aprendidas y runtime

| Semilla | Resultado primario | Cobertura | Estado | Artefacto |
|---:|---:|---:|---|---|
| 7 | — | — | PENDIENTE | PENDIENTE |
| 13 | — | — | PENDIENTE | PENDIENTE |
| 21 | — | — | PENDIENTE | PENDIENTE |

Para métodos deterministas, sustituir esta tabla por grupos y perturbaciones prefijadas; no simular replicación mediante semillas irrelevantes.

| Hardware / ejecución | Frontera del cronómetro | Mediana / p95 | Memoria | Arranque | Cobertura |
|---|---|---|---|---|---|
| PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

No equiparar kernels, adaptador offline, replay y adquisición-a-respuesta. Indicar reutilización de features, sincronización y costes excluidos.

## 5. Controles y resultados negativos

- [ ] Identidad de grupos y componentes comprobada.
- [ ] Pruebas de causalidad, referencia temporal y cobertura superadas.
- [ ] Controles de rama, capacidad o filtro simple pertinentes incluidos.
- [ ] Todos los intentos de la campaña registrados, incluidos descartados y fallos.
- [ ] Regresiones por dominio/seguridad visibles.
- [ ] No se eligió semilla ni parámetro por la reserva.

**Qué hipótesis refuta o apoya el resultado:** PENDIENTE.  
**Qué explicaciones alternativas siguen abiertas:** PENDIENTE.

## 6. Decisión y afirmación permitida

**Decisión:** promover / rechazar / inconcluso / fallo técnico.  
**Motivo cuantitativo y práctico:** PENDIENTE.  
**Estado de la reserva después de este análisis:** PENDIENTE.  
**Frase exacta que permite afirmar la evidencia:** PENDIENTE.  
**Limitaciones y siguiente experimento:** PENDIENTE.  
**Recibo de revisión y enlaces a artefactos:** PENDIENTE.

Guías: [04 — Evaluador](../04_EVALUADOR_Y_BASELINES.md) y [10 — Confirmación](../10_CONFIRMACION_ESTADISTICA_Y_SOTA.md).
