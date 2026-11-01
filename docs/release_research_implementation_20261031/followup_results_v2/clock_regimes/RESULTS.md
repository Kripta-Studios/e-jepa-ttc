# Reloj causal bajo cambios de régimen

Banco sintético prefijado; no es evidencia de mejora frente a R0.
El transporte sin ruido y velocidad constante es exacto (error <=1e-12).
Los futuros distintos conservan predicciones idénticas hasta su divergencia.

| Régimen/ruido | Método | MAE q | MAE q tras cambio | Errores de signo | Demora inversión (s) |
|---|---|---:|---:|---:|---:|
| constant_noise_0.0 | persistence | 0.020882 | 0.019832 | 0 | — |
| constant_noise_0.0 | ewma | 0.037108 | 0.030893 | 0 | — |
| constant_noise_0.0 | transport | 0.000000 | 0.000000 | 0 | — |
| constant_noise_0.0 | observer | 0.000000 | 0.000000 | 0 | — |
| braking_noise_0.0 | persistence | 0.043225 | 0.057038 | 2 | 0.030 |
| braking_noise_0.0 | ewma | 0.081876 | 0.074897 | 20 | 0.210 |
| braking_noise_0.0 | transport | 0.038267 | 0.063605 | 2 | 0.030 |
| braking_noise_0.0 | observer | 0.042334 | 0.068347 | 6 | 0.070 |
| accelerating_noise_0.0 | persistence | 0.123272 | 0.065678 | 0 | — |
| accelerating_noise_0.0 | ewma | 0.202158 | 0.094873 | 0 | — |
| accelerating_noise_0.0 | transport | 0.043566 | 0.043452 | 0 | — |
| accelerating_noise_0.0 | observer | 0.048688 | 0.046737 | 0 | — |
| receding_noise_0.0 | persistence | 0.005709 | 0.005623 | 0 | — |
| receding_noise_0.0 | ewma | 0.010634 | 0.009281 | 0 | — |
| receding_noise_0.0 | transport | 0.000000 | 0.000000 | 0 | — |
| receding_noise_0.0 | observer | 0.000000 | 0.000000 | 0 | — |
| constant_noise_0.05 | persistence | 0.034319 | 0.036156 | 0 | — |
| constant_noise_0.05 | ewma | 0.043036 | 0.040469 | 0 | — |
| constant_noise_0.05 | transport | 0.032483 | 0.037363 | 0 | — |
| constant_noise_0.05 | observer | 0.018977 | 0.016184 | 0 | — |
| braking_noise_0.05 | persistence | 0.046479 | 0.039797 | 2 | 0.030 |
| braking_noise_0.05 | ewma | 0.079345 | 0.059719 | 20 | 0.210 |
| braking_noise_0.05 | transport | 0.054289 | 0.057258 | 2 | 0.030 |
| braking_noise_0.05 | observer | 0.082614 | 0.082492 | 20 | 0.210 |
| accelerating_noise_0.05 | persistence | 0.134601 | 0.075504 | 0 | — |
| accelerating_noise_0.05 | ewma | 0.208086 | 0.104449 | 0 | — |
| accelerating_noise_0.05 | transport | 0.069993 | 0.070092 | 0 | — |
| accelerating_noise_0.05 | observer | 0.100556 | 0.072717 | 0 | — |
| receding_noise_0.05 | persistence | 0.028498 | 0.033255 | 0 | — |
| receding_noise_0.05 | ewma | 0.020585 | 0.020004 | 0 | — |
| receding_noise_0.05 | transport | 0.027712 | 0.031018 | 0 | — |
| receding_noise_0.05 | observer | 0.014944 | 0.013158 | 0 | — |

El ruido de medida es conocido solo en este banco; las ventanas reales pueden compartir evidencia.
Los tiempos de régimen son etiquetas de análisis, nunca entradas del estimador.
La ausencia de aviso o inversión queda registrada como null; no se transforma en demora cero.
El banco no admite D4 y no modifica parámetros tras observar sus resultados.
