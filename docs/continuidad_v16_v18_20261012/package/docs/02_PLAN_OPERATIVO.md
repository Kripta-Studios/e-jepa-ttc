# Plan operativo: conservar, medir, aprender, confirmar

## 0. Un único mapa vivo de las campañas

V13: productores/modos E/R terminados. V14/V15/V17: preservar conclusiones y
pesos, sin rescatar variantes rechazadas. V16: candidato PHASE_ONLY y evaluación
multimodal cerrados. V18: consultar su estado real; el informe sólo acredita
una observación pasada de su supervisor. No inferir que sigue esperando o que
ya entrenó por existir una carpeta o un checkpoint parcial.

No abrir por defecto otra rama V19. Usar la sesión V16 para evaluación y entrega,
y la sesión V18 para la nueva representación. Si hay que abrir un chat nuevo,
conservar el mismo worktree/identidades y verificar que no existe otro escritor.

## 1. Evaluación oficial del candidato ya preparado

Producto inmediato: carpeta de envío de PHASE_ONLY_EVENT_S7 con el ZIP original,
freeze, model card y manifest. Verificar hash, 6.762 tokens, finitud, esquema y
compatibilidad del historial declarado con las reglas efectivas del benchmark.
No reentrenar, escoger otra seed, retocar signo o mezclar outputs para este envío.

Un envío requiere autorización expresa de la acción externa. El texto opcional
se proporciona por separado. Registrar recibo, fecha, versión de scorer y tabla
official exacta si se obtiene. No hay score conocido hasta recibirlo; el parser
local no estima precisión. Tampoco la fila del paper acredita el líder actual.

El agente de entrenamiento puede continuar sin recibir ese score. La primera
puntuación no debe alimentar una búsqueda de hiperparámetros sobre test12.
Una eventual nueva familia requiere una decisión científica anterior a sus
resultados; futuras puntuaciones ya no son todas pruebas ciegas independientes.

Si las reglas/modalidad/historial no se pueden confirmar, registrar esa carencia.
No falsificar independencia, detener ciencia viable ni afirmar que el test fue
superado. No usar acceso inexistente a CodaBench ni enviar credenciales en logs.

## 2. Preservar un frente de candidatos

PHASE_ONLY: candidato seleccionado por los gates internos y submission congelada.
F_TRUE: referencia multimodal con mejor FCWD mean_MiD/MAE que PHASE_ONLY.
R_CTX: referencia RGB fuerte y necesaria para medir aporte incremental event.
E_CTX y H8 V12: referencias event-only, con TRAIN/contextos distintos declarados.
Garl event/full/visual: checkpoints nativos y políticas de fallback explícitas.

No elegir el modelo por etiqueta verdadera ni por el nombre FCWD/Dev32 durante
inferencia. No construir un selector oráculo por banda. Una política aprendida
sería un modelo nuevo y no está justificada aún por el mero cruce de curvas.

## 3. Investigación principal V18: entrenar mediciones nuevas

Mantener la matriz primaria ya definida: NATIVE-P2-FT, PAIR-DIRECT, PAIR-CORR,
PAIR-CORR-FROZEN. No cancelar CORR entrenable por el fracaso de congelado.
No cambiar recetas a mitad de un fit a raíz de este documento. Si ya comenzó,
los nuevos diagnósticos son descriptivos y su manifest debe decir que se
solicitaron después del inicio. No se cambian gates retroactivamente.

Registrar para cada endpoint:
- Métrica admitida por bandas y grupos, pooled diagnóstica y cobertura.
- Signo de verdad/predicción, errores de signo separados de magnitud.
- Raw phase, emitted phase, cambios de signo, cap y regiones de gradiente plano.
- Sesgo y sobre/infraestimación en P/H: 0,1–1, 1–2, 2–3, 3–6, 6–10 y negativos.
- Diferencias por grupo/episodio y presupuesto de cómputo real.

FCWD/Dev32 motivan la pregunta, pero no se usan sus etiquetas para ajustar
recetas o checkpoints. El hecho de que no entrenen pesos no borra su exposición.
Los experimentos H sirven como desarrollo adaptativo; V tampoco es test fresco.

### Decisiones al terminar la matriz

DIRECT gana a CORR: conservar DIRECT; más estructura física no es una obligación.
CORR entrenable gana a FROZEN: la adaptación de representación está aportando;
probar réplicas/control previstos antes de otra arquitectura.
Todos ajustan P pero fallan H: priorizar diversidad/regularización y robustez de
inputs, no extender pasos indefinidamente sobre el mismo pool.
P tampoco ajusta: comprobar unidades, gradientes, capacidad e inicialización
con ejemplos P y pruebas de sobreajuste acotadas previstas por V18.
Signo concentra fallos tras mejorar la medición: registrar un ensayo específico
signo/magnitud antes de ejecutarlo. No transformar esas salidas por abs().

No lanzar diez nuevos calibradores PHASE_ONLY: las 32 cabezas V16 ya respondieron
muchas variantes de ese nivel. Tampoco congelar toda investigación porque una
prueba pequeña no domine simultáneamente cada tabla. Separar señal de mecanismo,
candidato exploratorio y promoción de sistema.

## 4. Geometría y cobertura: trabajo independiente sin tuning de TTC

Ya existen 12 paneles y 72 filas de correspondencias por completar. Anotarlas
sin mostrar TTC ni errores, distinguiendo puntos ambiguos/invisibles de puntos
válidos. Separar ejemplos de ajuste y comprobación geométrica por escena.
Si no hay personas/anotaciones independientes, marcarlo pendiente; no inventarlas.

20/40/80 ms: repetir la tabla de aceptación no es validación. Evaluar error de
reproyección, falsos matches, cobertura y coste con el criterio fijado antes.
Un prior previo sólo puede usar registro pasado aceptado y su edad; nunca futuro.
Conservar una transformación coherente a lo largo de la historia. Ningún warp
puede recuperar eventos que un ROI anterior descartó.

Promoción de geometría: sólo tras validación geométrica, reejecutar TODOS los
comparadores afectados con el mismo registro y políticas de fallback. No recalcular
sólo las 125 filas donde nuestro modelo pierde. Registrar datos destino empleados.

## 5. Ingeniería: optimizar la ruta, no sólo la cabeza

Ya está probada parte de la reutilización exacta event. Próximos candidatos son
la decodificación/crop RGB y la reutilización de mapas de endpoints idénticos,
previos al suavizado temporal. La clave incluye fuente/frame, ROI, transformación,
normalización, checkpoint, dtype/dispositivo y ajustes relevantes. Identidad
nominal de timestamp no basta si cambia el recorte o tensor.

Perfil de aceptación: mismas consultas completas, orden cronológico y baseline
Garl nativo más optimizado; p50/p95/p99, frío/caliente, resets, ROI variable,
memoria y cobertura. Declarar detector/tracker fuera del scope cuando no existen.
No usar la suma de medianas como tiempo total. Cambios de lote o precisión pasan
pruebas previas; no ampliar tolerancias después de ver un fallo.

Si se reutiliza exactamente un encoder por frame, puede compartir cómputo entre
varias parejas. No reutilizar todo el estado temporal de otra query como si fuera
el mismo forward. La cabezita de 1.955 parámetros sigue necesitando sus padres.

## 6. Cierre y entrega

Entregar DECISION_ACTUALIZADA.json con propietario/estado por campaña,
REGISTRO_CANDIDATOS.csv con contratos, EVALUACION_OFICIAL.md con score o carencia,
DIAGNOSTICO_SIGNO_RANGO.csv, GEOMETRY_STATUS.json y comparación de latencia.
Referenciar outputs V18 sin recontarlos como nuevos resultados V16.
Nunca detenerse únicamente tras documentación si queda trabajo ejecutable
admitido. Tampoco simular actividad cuando falta una dependencia externa real.
No se amplía en este documento el presupuesto de V18 ni se autoriza push o envío.
