# Continuidad autónoma sin falsificar un éxito

## Principio

«No parar hasta SOTA» se implementa como un objetivo persistente y una cola de investigación con evidencia, no como una promesa de victoria ni permiso de variar el test. Un brazo fallido cierra SU hipótesis, no la misión. Un techo de recursos, una acción externa no autorizada o ninguna rama viable sí exige checkpoint y handoff veraz.

V17 fue una auditoría válida de representación congelada. No se relajan sus gates ni se promociona el bypass retrospectivamente. Para V18 se distinguen: señal mecanística, decisión de gastar más cómputo, promoción de un pipeline completo y confirmación oficial. No exigir a cada componente aislado la dominancia final en todas las bandas antes de poder entrenar su readout coherente.

## Tras la matriz primaria

El agente puede ejecutar hasta CUATRO experimentos adicionales EMPAREJADOS de las siguientes familias, hasta 24.000 updates por experimento (2×12k por defecto). Registra antes de cada primer update hipótesis, origen, población, único factor, baseline, parámetros, endpoint, métricas y criterios. No pedir otra autorización entre estas ramas ya incluidas, pero no salir de ellas. Nunca buscar un nuevo coeficiente sobre V17 o FCWD y presentarlo como preregistrado.

### Familia A: medir la causa de una representación insuficiente

Si CORR funciona sólo con encoder entrenable, explorar profundidad de adaptación (últimos bloques frente a encoder completo) o un control de descriptor congelado actualizado. No necesita otra arquitectura de cabeza. Las dos rutas difieren explícitamente en qué parámetros se entrenan; encoder congelado debe estar eval.

Si todos ajustan P pero transfieren mal a H, NO aumentar capacidad como primer reflejo. Comparar regularización/augmentación geométricamente consistente mediante una única receta fija y su control. Aplicar el MISMO warp espacial a ambos instantes, máscaras y targets geométricos TRAIN cuando existan; preservar TTC para transformaciones del plano de imagen que no cambian el movimiento. No inventar TTC para streams event simulados sin verificar física ni alterar el reloj de la etiqueta.

### Familia B: enseñar correspondencias sin TTC sintético

Pretraining corto sobre texturas event de P y transformaciones conocidas; comparación con igual número de updates adicionales de supervisión TTC sin pretraining. Etiquetar como correspondencia sintética, NO simulación física de cámara ni ground-truth de colisión. Ningún RGB/teacher/imagen de H/V en pretraining. Reutilizar una implementación previa equivalente de Stage70 sólo con prueba de configuración/ejecución; una propuesta antigua no cuenta como experimento negativo.

### Familia C: arquitectura verdaderamente geométrica relativa

Si la correlación contiene desplazamientos coherentes pero la torre no generaliza, contrastar un readout explícito de escala+traslación sobre el MISMO volumen frente al readout libre, con gradiente al encoder. La hipótesis temporal debe anclar TTC al instante correcto; bajo cierre constante T(t+dt)=T(t)−dt, NO expansión constante. No exigir isotropía universal, foreground perfecto o≥32 planos válidos antes de emitir: conservar máscaras y un fallback definido para TODAS las consultas.

Esta familia debe registrarse numéricamente por completo antes de entrenar. No puede convertirse en un sweep de grids/loss/thresholds. No presentar la capa geométrica como una garantía de verdad física en rotación/oclusión.

### Familia D: capacidad o datos, distinguiendo sus efectos

Sólo cuando el ajuste P esté también limitado tras verificar gradientes/loader y el control de correspondencia, probar anchura×2 con el mismo input/objetivo/presupuesto. No sustituir enseguida por un gran Transformer no medido.

Inventariar datos TRAIN adicionales es permitido; su uso exige cobertura, licencia, tarea, exclusiones y presupuesto de disco explícitos ANTES de consultar su evaluación. No llamar nuevos datos a ventanas más densas de las 40 secuencias o a V renombrado. No efectuar descargas masivas/pagos fuera del límite local. Una adquisición que no pueda acreditarse deja esa rama pendiente, no impide otras.

## RGB y fusión: rama complementaria, no esperar indefinidamente

V13 R_CTX y F_TRUE ya demostraron utilidad exploratoria. Conservarlos como comparadores. Leer el estado real V16 y sus resultados Garl/registro; no duplicar su trabajo.

La receta de par más informativa (CORR por defecto, DIRECT si el contraste la favorece) puede entrenarse sobre RGB P con el encoder RGB matched inicial. Contrastar readout congelado frente a encoder RGB entrenable si hace falta atribución. Son pesos, normalización y relojes RGB propios; no repetir canales event ni inventar frames de50ms. Máximo de esta rama: dos fits12k y dos fusiones/control2.5k (29k updates), definidos antes de sus resultados.

Para fusión, preservar la salida R_CTX o la rama RGB elegida como anchor y entrenar un complemento event con control ZERO de igual capacidad; no promediar TTC de signos opuestos. Conservar E sólo cuando falte RGB con fallback exacto admitido. Padres P excluyen H/V, normalización H. Si se usan heads entrenados previamente en H como features/initialization, declarar ajuste secuencial supervisado; no llamarlo OOF. No utilizar scores H in-sample para proclamar generalización.

## Réplicas y refit

Máximo una familia de encoders/readouts y sus controles se replica con seeds13/23 tras señal práctica admisible. Repetir TODOS los módulos nuevos entrenables del camino, no sólo una última MLP. Si se comparte un encoder de inicialización previamente entrenado, declararlo: son réplicas de la etapa de adaptación, no pretraining independiente. Para confirmar variabilidad completa, rehacer la inicialización/pipeline necesario bajo el protocolo final.

Después de seleccionar la receta sin test: un refit de los candidatos event y/o multimodal sobre el pool TRAIN público completo admisible. H/V que pasan al refit dejan de ser evaluación de esos pesos. Preservar receta y presupuesto antes del refit; no volver a evaluar V y llamarlo confirmación. Para heads sobre expertos se necesitan inputs de entrenamiento compatibles/crossfit según el contrato, no se copian cachés de padres anteriores. Para el candidato standalone end-to-end no existe esa segunda capa de crossfit.

## Criterio de fin

Completar toda rama habilitada, o suspender con checkpoint por recurso/límite externo; no emitir un resultado SOTA porque se alcanzó una hora. Registrar `RESEARCH_ONGOING`, `CANDIDATE_FOR_CONFIRMATION`, `EXTERNALLY_BLOCKED` o `CONFIRMED_BENCHMARK_RESULT` según evidencia. NO_CANDIDATE por experimento no se transforma en un resultado favorable; la investigación continúa en la siguiente hipótesis disponible dentro del contrato.
