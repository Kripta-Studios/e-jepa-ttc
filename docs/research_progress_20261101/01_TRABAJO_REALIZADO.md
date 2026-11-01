# PHASE-TTC: trabajo realizado hasta el 1 de noviembre de 2026

Este documento reúne la continuación desde el cierre `f1f9c8ed` hasta la
publicación de resultados `189b6e27`. El [documento 02](02_RESULTADOS_Y_PENDIENTES.md)
explica qué conseguimos, qué falló y qué falta. Los informes enlazados conservan
los resultados detallados y los recibos de cada experimento.

Repositorio: `Kripta-Studios/phase-ttc`. Rama: `research/evttc-transfer-v2`.
El objetivo científico sigue siendo demostrar superioridad reproducible en
EvTTC bajo un protocolo comparable, preservando eAP. No lo hemos alcanzado.

## 1. Punto de partida y alcance

Retomamos el [handoff del 31 de octubre](../handoff_20261031_pause/README.md),
sus cuatro documentos y las instrucciones del repositorio. El trabajo anterior
ya había implementado la conversión temporal explícita de PAIR y la había
probado en dos consultas. También había auditado las observaciones actuales
de eAP: 88.744 salidas PAIR de eventos idénticas y un cambio medio absoluto RGB
aproximado de 0,000092 s. Aquello dejaba pendientes los contextos y la cabeza.

Conservamos ese trabajo y sus resultados negativos, incluida la regresión del
contexto de eventos de carretera al ralentizar el tiempo. No repetimos los 312
ajustes históricos de fusión ni los pilotos temporales terminados.

Distinguimos tres rutas durante la continuación:

| Ruta | Uso |
|---|---|
| F0 | Predicción nativa de la cadena PHASE, controlada antes del router adaptado. |
| Router entregable congelado | Ensemble existente de `FINAL_MODEL`, evaluado sin reajustarlo. |
| Nuevo reajuste emparejado | Experimento posterior con estados originales y canónicos, evaluado mediante predicciones fuera de fold. |

Comprobamos el estado de Git, los procesos y los recibos antes de reanudar.
Los archivos de trabajo concurrente permanecieron separados de nuestros commits.
No presentamos el worktree completo como limpio.

## 2. Screen canónico de 149 consultas

Ejecutamos el [screen fijo](../research_log_20261031_canonical_screen149/README.md)
de 72 consultas de carretera y 77 de Slider-500. Reprodujimos primero los arrays
originales disponibles de productores, contextos y cabeza. Capturamos PAIR133,
reconstruimos raw17 completo, incluidos los desacuerdos, y recalculamos contextos
y cabeza para la intervención canónica.

Conservamos pesos, normalizadores, lotes y relojes. No hubo entrenamiento en
este experimento. Sellamos predicciones y ausencias antes de abrir targets en
otro proceso; evaluamos cobertura, MAE y error relativo por separado para F0
y para el router congelado. El proceso tardó 999,372 s, incluida extracción y
replays; ese tiempo no representa latencia de servicio.

Comprobamos también que el replay compacto reprodujera los artefactos canónicos
del piloto anterior en las dos consultas comunes. El lote rechazado en
Slider-500 ya fallaba en el original antes de PAIR. Conservamos la ausencia
nativa de Slider-500:76 por fin de cobertura de eventos.

F0 superó el criterio de desarrollo. El router congelado falló el de carretera,
por lo que sus mediciones posteriores siguieron siendo diagnósticas.

## 3. Ampliación a 2.608 consultas

Completamos la [población fijada](../research_log_canonical_population2608/README.md):
2.531 consultas de carretera en 24 grabaciones y 77 de Slider-500. Reutilizamos
las 149 consultas selladas y extrajimos 2.459 nuevas. Cada consulta nueva
reprodujo su original en PAIR, raw17, contextos y cabeza antes de la intervención.

La ejecución tardó 13.243,949 s. No hubo actualizaciones del optimizador.
Sellamos las salidas antes de unir targets y calculamos métricas macro por
grabación, cobertura, signos y bootstrap pareado de grabaciones de carretera
con 10.000 remuestreos y semilla 7. No tratamos las ventanas como observaciones
independientes ni calculamos un intervalo entre grabaciones para el único Slider.

Usamos como control el original recapturado en el mismo entorno. No asumimos
identidad con las características nativas de la antigua caché de entrenamiento.
La mejora agregada del router en esta población no revocó su fallo del screen.

## 4. Reproducción de eAP y diagnóstico de paridad

Ejecutamos el [replay original de contextos y cabeza](../research_log_20261031_canonical_screen149/EAP_ORIGINAL_CONTEXT_REPLAY.md)
en las 5.805 consultas físicas Test12 desde los diagnósticos sellados. Recalculamos
los contextos, sus bloques de 170 coordenadas, los logits RGB y la cabeza.
Comparamos 22 campos por consulta con tolerancias fijadas antes de ejecutar.
El replay pasó en 117,910 s; ese resultado corresponde al original y no incluye
una recaptura completa de los productores de sensores.

Preparamos después la recaptura original RGB. El primer intento falló al omitir
el adaptador ABI histórico. Corregimos el runner y conservamos ambos intentos.
El [piloto corregido](../research_log_canonical_population2608/EAP_RGB_CAPTURE_RESULT.md)
capturó 24 observaciones y encontró 24 fallos de paridad histórica. No ampliamos
esa ruta como si hubiera pasado ni relajamos las tolerancias.

Verificamos en lectura el diagnóstico concurrente de cuatro políticas TF32:
ninguna recuperó la paridad histórica en sus dos consultas. No repetimos ese
trabajo ni atribuimos una causa definitiva a las diferencias de plataforma.

## 5. Recuperación y cadena local completa de eAP

Al revisar la captura nativa interrumpida, comprobamos la ausencia de sus
procesos y que no tenía recibo COMPLETE. Había 7.280 carpetas con los dos nombres
de archivo esperados; el inventario detallado admitía 7.279 fragmentos completos.
Documentamos esa diferencia para no confundir presencia de archivos con validez.

Encontramos ya activa la recuperación concurrente v2. No lanzamos otra captura.
Retiramos un borrador alternativo que no llegó a ejecutarse y conservamos el
intento v1. La recuperación concurrente inventarió los fragmentos, recapturó
centinelas y faltantes y completó los replays original y emparejado.
La [nota de recuperación](../research_log_canonical_population2608/NATIVE_CAPTURE_RECOVERY.md)
describe la procedencia y los controles.

Esperamos a que terminaran también sus mediciones de latencia NPZ/RGB y RAW/RGB
antes de iniciar nuestra auditoría, para no añadir carga a sus mediciones.
La recuperación y esas mediciones pertenecen a la línea concurrente: consumimos
sus resultados completos, sin atribuirnos una segunda ejecución.

Implementamos y ejecutamos después
`scripts/audit_phase_eap_router_sensitivity.py` sobre el router entregable
congelado. El runner:

- Verifica COMPLETE, ausencia de FAILED, genealogía y hashes de los tres padres.
- Exige las 5.805 identidades físicas y sus 6.762 aliases de entrega.
- Recalcula la conversión nativa con CUDA y exige coincidencia exacta de F0
  con las predicciones selladas.
- Reconstruye historia y geometría causales y ejecuta el ensemble congelado
  de tres semillas en CPU, con batch uno.
- Comprueba el retorno exacto a F0 cuando la historia no es admisible.
- Publica cobertura, signos y diferencias globales y por secuencia, sin targets.

Optimizamos la verificación de padres con ocho workers de lectura y una raíz
resuelta por padre. Conservamos la comprobación de cada hash, la contención de
rutas y la propagación de errores. No paralelizamos la inferencia ni cambiamos
fuentes de un experimento que estuviera ejecutándose. La auditoría terminó con
código cero; inferencia y sellado tardaron 62,357 s, aparte de verificar padres.

## 6. Un reajuste nuevo y emparejado del router

Fijamos y publicamos el [protocolo](../research_log_canonical_population2608/CANONICAL_ROUTER_REFIT_PROTOCOL.md)
antes del entrenamiento. La hipótesis era que ajustar la misma receta del router
con los estados canónicos podía corregir el desajuste de sus entradas.

Implementamos `scripts/train_phase_canonical_router.py` y ejecutamos 48 ajustes:
dos brazos, ocho folds que excluyen familias completas y tres semillas
(7, 13 y 21). Mantuvimos arquitectura, pérdida, aumentos, 300 pasos de AdamW,
learning rate 0,01 y weight decay 0,01. Cada brazo usó sus estados correspondientes;
geometría, calidad y población permanecieron iguales.

Las ocho grabaciones del benchmark y Slider-750/1000 quedaron excluidas.
Calculamos normalización y pesos de familias dentro de cada conjunto de ajuste.
El entrenamiento sí usó targets de desarrollo de sus familias de ajuste; no leyó
targets Test12. Conservamos checkpoints y predicciones por semilla y fold, y
sellamos las predicciones antes del cálculo de métricas fuera de fold (OOF).

El criterio exigía pasar tanto la población completa como el screen149 fijo.
El nuevo router falló ese criterio. Conservamos las tres semillas y no hicimos
ajuste final, exportación ni selección de una semilla favorable.

## 7. Dependencias oficiales y comparación

Verificamos mediante Hugging Face CLI la revisión pública
`53fbaade17f3b76aa56dc7df2b527ce279062f59` de `NAIL-HNU/GarlTTC-dataset`.
Descargamos el README y la plantilla de entrega de esa revisión, y ejecutamos
`scripts/audit_phase_external_requirements.py`. La plantilla contiene los mismos
6.762 tokens lógicos que nuestro inventario confiable.

Confirmamos en esa documentación que Test12 mantiene privados los targets y
requiere evaluación oficial en CodaBench. No encontramos targets Test12 en la
búsqueda local realizada ni mensajes relevantes en la búsqueda de correo.
Tampoco había una sesión de navegador conectada desde la que acceder a CodaBench.
Solicitamos acceso al usuario y permiso para enviar el borrador a los organizadores.
No enviamos correo ni una nueva entrega.

Revisamos la auditoría publicada de Slider-750 y distinguimos las poblaciones
de la competición EvTTC, la transferencia de Garl-TTC y la validación in-domain
de REACT. No ajustamos offsets ni targets para forzar la coincidencia del evaluador.

## 8. Publicación, pruebas y conservación

Implementamos `scripts/report_phase_three_requirements.py` y generamos el
[informe v1](v1/README.md). El generador comprueba recibos, genealogía, auditoría
publicada e identidad y componentes del modelo entregable. Copia doce JSON
fuente y añade `STATUS.json`, con `goal_achieved: false`.

La validación nueva incluyó ocho pruebas del reajuste y nueve del router eAP,
además de Ruff y Pyright sobre ese alcance. Las 17 pruebas pasaron. Los informes
anteriores conservan sus verificaciones propias: 49 pruebas del screen y once
del runner de recaptura RGB. Estos conteos pertenecen a alcances distintos y no
constituyen una nueva ejecución de toda la suite del repositorio.

Publicamos código, protocolos y evidencia pequeña en commits sucesivos:

| Commit | Contenido principal |
|---|---|
| `30410e60` | Evaluación sellada de la población de 2.608 consultas. |
| `85410f4d` | Fallo de paridad histórica del piloto RGB conservado. |
| `2e83053a` | Preparación de la auditoría del router nativo. |
| `6276725b` | Protocolo e implementación del reajuste emparejado. |
| `a84122ce` | Verificación de la plantilla Test12 y generador de informe. |
| `80e3441c` | Verificación paralela de padres sin omitir controles. |
| `e918ca0f` | Vinculación del informe al evaluador y modelo entregable. |
| `189b6e27` | Resultados eAP completos en alcance local y reajuste rechazado. |

Tras el último push verificamos que el remoto apuntara a `189b6e27` y que los
13 JSON del informe coincidieran byte a byte con Git. Los datos, NPZ, cachés y
checkpoints voluminosos permanecen fuera de Git. El generador de cierre acotado
preparado durante la conversación no se ejecutó; el usuario pidió resolver los
requisitos científicos y no dimos el objetivo por completado.
