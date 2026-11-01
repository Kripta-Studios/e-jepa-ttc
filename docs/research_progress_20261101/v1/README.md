# Avance sobre los tres requisitos científicos

El objetivo sigue sin estar alcanzado. Se conserva el entregable anterior. Este informe no es un cierre SOTA ni sustituye evaluaciones ausentes.

## 1. eAP

Se reconstruyó la cadena local de 5805 consultas físicas / 6762 aliases. El original local se repite exactamente y el contexto de eventos conserva identidad. El router congelado se auditó por separado.

El original conserva 5805 fallos de paridad histórica; 504 afectan al control de TTC. No se sustituyen por la repetibilidad del nuevo equipo.

| Ruta | Nuevas ausencias | Cambios de signo | Máximo cambio absoluto (s) |
|---|---:|---:|---:|
| F0 | 0 | 0 | 0.006496429443359375 |
| Router congelado | 0 | 0 | 0.0017948150634765625 |

La precisión Test12 sigue sin medirse. El paquete oficial no distribuye targets; requiere evaluación en CodaBench. Su plantilla actual coincide exactamente con nuestros 6762 tokens. No se ha enviado una nueva entrega.

## 2. Nuevo reajuste emparejado

Se realizaron 48 ajustes: dos brazos, ocho familias excluidas y tres semillas. Arquitectura, pérdida, aumentos y presupuesto se conservaron. Cada brazo usa sus estados originales o canónicos correspondientes. Son resultados OOF de desarrollo con exposición histórica; no confirmación independiente.

| Vista | Dominio | MAE original → canónico (s) | RAE original → canónico (%) |
|---|---|---:|---:|
| population2608 | road | 0.5396081119384536 → 0.537801226636036 | 17.418661215067214 → 17.291983263087527 |
| population2608 | slider | 0.36498710009958835 → 0.35651482157892994 | 26.877530251373184 → 21.468543661694326 |
| screen149 | road | 0.48475203787287074 → 0.5118206669059064 | 16.72085968514506 → 17.431738167291815 |
| screen149 | slider | 0.36498710009958835 → 0.35651482157892994 | 26.877530251373184 → 21.468543661694326 |

No supera el filtro de desarrollo; no se realiza ajuste final ni exportación. El fallo previo del router congelado permanece intacto.

## 3. Comparabilidad y superioridad

Slider-750 sigue sin equivalencia establecida entre el evaluador local y el resultado recibido. La auditoría anterior se conserva; no se ajustan offsets ni targets para forzar coincidencia. Tampoco se acredita una reproducción completa equivalente de los baselines SOTA.

Se mantiene la distinción entre las diez secuencias de la competición, las tres filas de transferencia Garl-TTC y la validación in-domain de [REACT](https://arxiv.org/html/2609.19204v1). El contrato histórico copiado documenta estas poblaciones distintas; sus antiguos flags de descarga no describen el estado operativo actual. No se mezclan sus cifras para construir una clasificación nueva.

La validación local no elimina estas dependencias externas. La consulta a los organizadores permanece pendiente de autorización explícita; no se presupone el envío por haber preparado un borrador.

## Reproducción y evidencia

STATUS.json enlaza por hash los informes exactos copiados en evidence. Los runs completos, cachés y pesos permanecen fuera de Git. Los protocolos y runners conservan los controles previos y las genealogías de recuperación.
