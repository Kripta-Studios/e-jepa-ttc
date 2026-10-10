# Datos, cronología real y roles

## Un timeline RGB no es un stream event de 20 Hz

eAP registra imágenes a 10 Hz. Obtener contexto de ocho puntos de 50 ms no crea ocho imágenes diferentes. Construir un índice desde frames reales y su disponibilidad, independiente de filtros de TTC. Duplicar una imagen no aporta movimiento y no permite inventar timestamps intermedios.

Primera comparación: misma frontera de información retrospectiva máxima que el sistema event, 650 ms, y como máximo ocho observaciones expertas. R_CTX forma observaciones de tres frames consecutivos distintos dentro de esa frontera. Con adquisición regular a 10 Hz puede haber cinco tripletas, no ocho. Guardar `distinct_frames`, `distinct_observations`, `actual_span_us`, `frame_ids`, deltas y máscaras; no llamar a eso ocho observaciones independientes.

R_H1 usa la tripleta más reciente; R_CTX usa las tripletas distintas admisibles. Los tamaños de historia no se seleccionan con Dev32/FCWD. Todas las consultas originales se conservan. Si hay sólo dos frames, el productor admite T=2: registrar ese modo y ensayarlo antes de la campaña, sin fabricar un tercero. Con menos de dos se declara RGB no disponible; la fusión usa fallback event. No borrar consultas difíciles para que mejore la cobertura.

La inferencia es al cutoff actual. Ninguna imagen disponible después del cutoff puede entrar. El tiempo de label/consulta y el anchor del último frame se guardan por separado. Al entrenar el refiner, su target es el del query, y su entrada incluye el retraso RGB: no comparar el TTC del último frame con otra etiqueta fingiendo sincronía. Para puntuar el productor geométrico puro se requiere un contrato explícito: primero pruebas en anchors RGB coincidentes con la consulta; en consultas asíncronas, emitir el anchor propio y evaluar sólo mediante una adaptación temporal predeclarada, o marcar la comparación de ese productor como no disponible. No retargetear el GT ni afinar el offset con error observado.

Las imágenes de los distintos endpoints deben compartir un recorte/escala geométrica dentro de la observación. Redimensionar cada bbox por separado a 128×128 puede borrar la expansión. Usar la ROI actual y sus datos disponibles de forma consistente; no una trayectoria de cajas futura ni toda la track. Esta es una interfaz detection-assisted, no un tracker acreditado.

## Calibración por dataset

En eAP, separar RGB y cámara event; no confundir sus intrínsecos, desplazamientos y centros de pixel. Para fusionar tarde no hace falta forzar alineamiento píxel a píxel, pero sí identificar el objeto, los crops y tiempos correctos.

FCWD RGB continúa NO DISPONIBLE hasta exportar/verificar calibración. Un factor 1920/1280 no demuestra correspondencia entre cámaras; el MAT MCOS requiere exportación documentada o datos portables del proveedor. La falta de FCWD RGB no bloquea entrenar RGB en eAP ni evaluarlo bajo el contrato EvTTC ya admitido.

No importar a FCWD el ajuste x+5 de eAP. No usar GT depth o navegación para alinear inputs de test. Un JSON de calibración debe incluir origen, unidades, camera IDs, distorsión, intrínsecos, extrínsecos, resolución y prueba de reproyección sin optimizar TTC.

## Desarrollo con encoders y cabezas separados

Fijar por grupo de adquisición antes de los fits tres roles aproximadamente 60/20/20 dentro de TRAIN40:

- P: aprender encoders A5/C2F event y RGB y las PAIR correspondientes.
- H: producir features con encoders congelados que NO vieron H; aprender refinadores y fusión.
- V: evaluación de desarrollo de todas las rutas nuevas; no aparece en P, H ni en teachers de tarea.

Con 40 grupos independientes serían aproximadamente 24/8/8. No asumir que 40 nombres equivalen a 40 adquisiciones. Usar agrupación real disponible; si sólo hay secuencias verificadas, declarar `sequence-held-out provisional`, no `fresh independent holdout`. Un split debe conservar soporte suficiente de buckets en TRAIN; no mover grupos después de conocer los scores V.

El diseño P/H/V reduce disponibilidad de datos para cada etapa. Es un experimento controlado inicial, no el entrenamiento final con todos los datos. Como todos los nuevos E/R/fusión comparten roles, permite estudiar la intervención. Replicación/cross-fitting completa y refit TRAIN40 se proponen después, sin gastarlos dentro de este primer cribado.

Reutilizar encoders/cachés existentes únicamente si se demuestra exclusión de H y V y equivalencia del pool/receta. Los productores TRAIN40 completos NO valen como control limpio en V. Si no existen controles event admisibles, entrenar E_A5/E_C2F sobre P dentro de la reserva autorizada.

Teachers DINO genéricos congelados pueden consumirse sobre P. Teachers TTC o PAIR con train/selection que incluya H/V no son admisibles. Los normalizadores de cabezas se ajustan sobre observaciones únicas H, nunca V. Las PAIR se entrenan en P sobre su encoder P congelado, antes de crear las features H/V.

## Comparación con Garl

No reanudar los productores Garl parciales supersedidos. Los pesos públicos sirven como referencias externas y para preparar comparación oficial; no son controles limpios en V si ya entrenaron con esos grupos. No usarlos para escoger nuestras recetas en V.

Publicar una tabla explícita de privilegios: número de frames, intervalos de eventos, alcance retrospectivo, ROI/oracle, timestamp de decisión, teacher/pretraining y soporte de salida. Garl pairwise y un sistema con 650 ms de pasado pueden compararse como sistemas, pero esa mejora no es automáticamente de arquitectura con información idéntica. Un control de dos endpoints permitido puede ejecutarse sobre los productores nuevos que aceptan T=2, etiquetándolo como ablation de inferencia sin seleccionar pesos con sus resultados; no cambiar el horizonte de la cabeza entrenada para fingir equivalencia.

Dev32 y FCWD ya vistos son transferencia exploratoria; ampliar timestamps no crea escenas independientes. Mantener pesos, thresholds, split y endpoints congelados antes de su evaluación. La Tabla VI y eAP MiD son contratos diferentes de este RTE local. Pedir los datos de contrato faltantes sin bloquear el entrenamiento.

El primer plan no autoriza submissions, lectura de labels privados ni Stage76. Preparar una propuesta concreta de confirmación tras seleccionar candidatos, con roles/modalidad/horizonte/salidas y fallos iguales. Entrenar, validar y calibrar no es lo mismo que consultar etiquetas finales.
