# PHASE-TTC: resultados conseguidos y requisitos pendientes

Estado documentado el 1 de noviembre de 2026, con evidencia publicada hasta
`189b6e27`. El [documento 01](01_TRABAJO_REALIZADO.md) describe la ejecución.
Hemos completado el screen, la población de desarrollo, la sensibilidad local
de eAP y un reajuste emparejado del router. El modelo entregable sigue siendo
el anterior: no hay candidato nuevo aprobado ni demostración de SOTA.

## 1. Qué hemos conseguido

| Avance | Evidencia | Límite de la conclusión |
|---|---|---|
| Aplicar PAIR canónico a la cadena completa | PAIR133, raw17, contextos y cabeza recalculados; originales reproducidos antes de intervenir. | Una intervención ejecutada no garantiza menor error. |
| Medir F0 en 149 y 2.608 consultas | Predicciones selladas antes de unir targets; métricas macro y cobertura explícitas. | Son datos de desarrollo con exposición histórica. |
| Completar eAP en alcance local | 5.805 consultas físicas y 6.762 aliases; F0 y router auditados por separado. | Faltan precisión Test12 y paridad histórica completa. |
| Contrastar una hipótesis de reajuste | 48 ajustes, ocho folds, tres semillas y dos brazos emparejados. | El nuevo router falla el criterio fijado. |
| Verificar la identidad de la entrega pública | Coincidencia exacta de los 6.762 tokens con la plantilla oficial fijada por revisión. | No equivale a haber enviado o evaluado la entrega. |
| Conservar evidencia reproducible | Runners, protocolos, hashes y 13 JSON del último informe publicados. | Los artefactos voluminosos siguen siendo locales. |

## 2. F0 canónico y router congelado en EvTTC de desarrollo

El [screen149](../research_log_20261031_canonical_screen149/README.md) incluye
72 consultas de carretera y 77 de Slider-500. Las métricas son medias macro
por grabación. Las cifras de esta tabla están redondeadas a seis decimales;
los informes enlazados conservan la precisión de origen.

| Ruta | Dominio | MAE original → canónico (s) | Error relativo original → canónico (%) | Decisión |
|---|---|---:|---:|---|
| F0 | Carretera | 0,748853 → 0,748672 | 31,370683 → 31,358111 | Pasa. |
| F0 | Slider-500 | 1,178156 → 1,120809 | 83,885894 → 82,442047 | Pasa. |
| Router congelado | Carretera | 0,338041 → 0,338051 | 10,295071 → 10,295517 | Falla. |
| Router congelado | Slider-500 | 0,205066 → 0,198997 | 13,578564 → 13,426316 | Pasa este dominio; no el criterio conjunto. |

No aparecieron nuevas ausencias ni cambios de signo. F0 conservó 76 de las
77 salidas de Slider-500; su ausencia previa siguió contando en el informe.
El router dispone de geometría y produjo las 77. No relajamos el criterio de
carretera por ser pequeña la regresión del router.

En la [población de 2.608 consultas](../research_log_canonical_population2608/README.md),
F0 obtuvo:

| Dominio | Pares / población | MAE original → canónico (s) | Error relativo original → canónico (%) |
|---|---:|---:|---:|
| Carretera | 2.531 / 2.531 | 0,810620018 → 0,810489612 | 30,600389999 → 30,591045605 |
| Slider-500 | 76 / 77 | 1,178155898 → 1,120808950 | 83,885893960 → 82,442047433 |

F0 cumple los criterios numéricos de desarrollo. En carretera, el cambio medio
de MAE es −0,000130406 s y su intervalo descriptivo del 95% es
[−0,000504700; 0,000093150] s. El intervalo incluye cero. El del cambio de error
relativo también incluye cero. La mejora agregada es modesta y no demuestra
superioridad estadística independiente.

El router congelado mejora los agregados de la población completa, pero su
fallo previo del screen sigue vigente. Los resultados del router en esta
población son diagnósticos. Conservamos también la regresión temporal del
contexto de eventos del piloto anterior.

## 3. eAP: cobertura completa local, precisión todavía sin evaluar

La recuperación v2 terminó y permitió comparar original y canónico en la misma
plataforma. El original local se repite exactamente; el contexto de eventos
permanece idéntico entre brazos. Después ejecutamos el router congelado sobre
esos estados con historia y geometría causales.

| Control | F0 | Router entregable congelado |
|---|---:|---:|
| Consultas físicas emparejadas | 5.805 | 5.805 |
| Nuevas ausencias | 0 | 0 |
| Cambios de signo | 0 | 0 |
| Predicciones idénticas entre original y canónico | 86 | 72 |
| Mediana del cambio absoluto (s) | 0,0000143051 | 0,0000133514 |
| Percentil 95 del cambio absoluto (s) | 0,0000767708 | 0,0000610352 |
| Percentil 99 del cambio absoluto (s) | 0,0001210403 | 0,0000920296 |
| Máximo cambio absoluto (s) | 0,0064964291 | 0,0017948151 |

El cambio absoluto medio del router es 0,0000204651 s. La vista de entrega
contiene 6.762 aliases de esas 5.805 consultas; no son observaciones independientes
adicionales. Las tablas por secuencia y las cifras completas están en
[native_result.json](v1/evidence/native_result.json) y
[router_result.json](v1/evidence/router_result.json).

Estas diferencias miden sensibilidad de salida. No demuestran conservación de
MAE, error relativo ni ranking oficial porque no usamos targets Test12.

### Repetibilidad local y paridad histórica

El replay previo desde diagnósticos archivados pasó el control de contextos y
cabeza en 5.805 consultas. La recaptura completa de productores exige otro
control: el piloto RGB corregido tuvo 24 fallos en 24 observaciones. En la
cadena nativa posterior hubo 5.805 consultas que fallaron algún control histórico,
y 504 fallaron el control específico de TTC.

La repetibilidad exacta del nuevo original local no borra esos fallos. Tampoco
los valores de raw17 se pueden interpretar como segundos de TTC. Conservamos
los informes separados para no atribuir a PAIR canónico diferencias que ya
existían antes de la intervención.

### Targets y acceso oficial

La [revisión pública verificada](https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset/blob/53fbaade17f3b76aa56dc7df2b527ce279062f59/README.md)
mantiene privados los targets Test12 y remite a
[CodaBench](https://www.codabench.org/competitions/17289/). La plantilla coincide
con nuestros tokens, según [external.json](v1/evidence/external.json).
No descargamos targets privados ni enviamos una nueva entrega. Sigue pendiente
el acceso autorizado a la evaluación oficial solicitado al usuario.

## 4. Nuevo candidato: reajuste emparejado rechazado

Entrenamos la misma receta con estados originales recapturados y estados
canónicos. Las siguientes cifras corresponden al ensemble OOF de tres semillas,
no al router entregable ajustado sobre todos sus datos. No deben compararse
entre sí como si ambos experimentos tuvieran el mismo entrenamiento.

| Vista | Dominio | MAE original → canónico (s) | Error relativo original → canónico (%) |
|---|---|---:|---:|
| Población 2.608 | Carretera | 0,539608 → 0,537801 | 17,418661 → 17,291983 |
| Población 2.608 | Slider-500 | 0,364987 → 0,356515 | 26,877530 → 21,468544 |
| Screen149 | Carretera | 0,484752 → 0,511821 | 16,720860 → 17,431738 |
| Screen149 | Slider-500 | 0,364987 → 0,356515 | 26,877530 → 21,468544 |

El ensemble mejora los agregados de la población completa, pero empeora ambas
métricas de carretera en el screen fijo. Por ello,
`admit_further_validation=false`. No hicimos ajuste final ni exportación.

Conservamos además los resultados de cada semilla:

| Semilla | Pasa población 2.608 | Pasa screen149 |
|---|---|---|
| 7 | No | Sí |
| 13 | Sí | No |
| 21 | No | No |

Ninguna pasa ambas vistas. El [informe completo](v1/evidence/refit_result.json)
incluye métricas por grabación, cobertura y signos. Los 48 ajustes contrastan
una hipótesis fijada; no son una repetición de la búsqueda histórica de 312
variantes ni una demostración de que ninguna arquitectura futura pueda mejorar.

## 5. Comparación equivalente y SOTA

La auditoría del evaluador mantiene nueve de diez filas coincidentes al redondear.
Slider-750 sigue discrepando: 51 consultas, error relativo local
283,931058% frente a 132% en el resultado recibido. Conservamos los bytes de la
[auditoría](v1/evidence/evaluator_audit.json) y no ajustamos offsets, targets o
métricas para obtener coincidencia.

Falta aclarar el vector de ground truth alineado, la versión del evaluador y la
identidad de la entrega anterior. El
[borrador para los organizadores](../research_log_20261031_sign_safe/organizer_evaluator_question.txt)
formula esas preguntas. No lo enviamos porque el usuario pidió autorización
previa para contactar; la consulta de permiso sigue pendiente.

También falta una comparación completa y reproducida con baselines bajo el
mismo protocolo. Distinguimos estas poblaciones en el
[contrato de comparación](v1/evidence/historical_comparison_contract.json):

| Fuente | Población / régimen | Restricción |
|---|---|---|
| Competición EvTTC | Diez secuencias: ocho de carretera y dos Slider. | El evaluador aún tiene la discrepancia de Slider-750. |
| Tabla de transferencia Garl-TTC | Tres secuencias de carretera, transferencia desde eAP. | No representa por sí sola las diez secuencias. |
| REACT | Validación in-domain y protocolo propio. | No equivale a transferencia ni al screen local. |

No unimos cifras de esas poblaciones para construir un ranking. El contrato
histórico documenta sus diferencias; sus flags antiguos de descarga no describen
el estado operativo actual.

## 6. Estado del modelo y de la evidencia

Verificamos la identidad y los componentes de `output/phase_submission/FINAL_MODEL`.
El SHA-256 de su `MODEL.json` sigue siendo:

```text
fe1460c407ccd08fc670c1955435b0fe9b4719737e22e3dec7534a57e66bb005
```

El [STATUS publicado](v1/STATUS.json) mantiene:

| Estado | Valor |
|---|---|
| Sensibilidad local eAP completa | `true` |
| Precisión Test12 validada | `false` |
| Nuevo router supera el filtro de desarrollo | `false` |
| Modelo nuevo aprobado | `false` |
| Superioridad SOTA comparable demostrada | `false` |
| Objetivo alcanzado | `false` |

Las 17 pruebas nuevas del reajuste y de la auditoría eAP pasaron, junto con Ruff
y Pyright en ese alcance. Eso verifica software y controles; los resultados
negativos de los experimentos siguen siendo negativos.

Publicamos doce JSON fuente y el STATUS. Verificamos sus hashes y la igualdad
de los 13 archivos con los bytes publicados en Git en `189b6e27`. El
[README de reproducción](README.md) contiene el comando para regenerar el
informe en una salida nueva. Los runs sellados, pesos y cachés siguen locales.

## 7. Qué falta para cumplir los tres requisitos

| Requisito | Trabajo pendiente | Condición para declararlo cumplido |
|---|---|---|
| Validar eAP completo | Obtener evaluación autorizada con targets Test12; tratar la falta de paridad histórica según el protocolo. | Evidencia de precisión y conservación bajo condiciones comparables, además de cobertura. |
| Conseguir candidato aprobado | Una hipótesis nueva y justificada, con entrenamiento y selección en desarrollo, seguida de todos los controles. | Pasar los criterios previos, conservación de eAP y validación equivalente antes de sustituir el entregable. |
| Demostrar superioridad | Resolver Slider-750 y reproducir baselines con idénticos datos, consultas, métricas y régimen de entrenamiento. | Comparación reproducible y estadística apropiada que sustente la afirmación. |

Al retomar, comprobar primero si llegaron acceso a CodaBench o respuesta sobre
el correo. No reiniciar runs completos, repetir los 312 ajustes o elegir una
semilla porque favorezca el resultado del screen. El cierre documental y el
push no convierten estos requisitos científicos pendientes en tareas cumplidas.
