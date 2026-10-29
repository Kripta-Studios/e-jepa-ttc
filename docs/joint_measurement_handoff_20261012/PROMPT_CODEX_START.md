/goal Desarrollar y validar un predictor TTC mejor que los actuales, con aspiración de SOTA verificable, mediante el programa V18 JOINT-MEASUREMENT. No limitarse a auditar o escribir comandos: integrar, probar, entrenar, evaluar y continuar las hipótesis habilitadas tras los negativos. Nunca declarar victoria sin la comparación que la pruebe.

Esta es una autorización NUEVA. Base revisada `396abe922537d9dac19d6db3451e59d283636e20`. Lee README.md, execution_policy.json y docs/01–06 de `docs/joint_measurement_handoff_20261012/`. V17 permanece cerrado; no reabrir sus gates/recetas ni reutilizar su saldo.

Trabaja en la rama/worktree V18 nuevo. Conserva V13/R1/V14/V15/V16/Stage70, sus recursos y archivos ajenos. Antes de inferir ejecución, comprueba endpoints reales. Reutiliza código/resultados previos equivalentes sólo con identidad verificada; no hagas otra auditoría histórica interminable.

Primera cola obligatoria: NATIVE-P2-FT, PAIR-DIRECT, PAIR-CORR y PAIR-CORR-FROZEN, seed7, 12.000 updates cada uno según protocolo. En DIRECT/CORR el gradiente debe llegar a encoder.features y cambiar realmente sus pesos. No entrenes sobre vectores congelados y lo llames nueva representación. El resultado negativo del control congelado NO bloquea los modelos entrenables.

Registra recetas/inputs/controles antes de entrenar y congela los cuatro endpoints antes de evaluar H. P entrena esos modelos; H selecciona; V sigue siendo desarrollo expuesto. No inicializar una supuesta evaluación held-out con padres TRAIN40 que ya la vieron. Conservar reloj, ROI, dominios, negativos, cobertura y scorer. Comparaciones entre roles/modalidades distintas deben etiquetarse.

Si la primera cola no produce candidato, continúa en las familias adaptativas del documento03, hasta cuatro experimentos adicionales emparejados, registro prospectivo y mismos controles. Puedes tomar decisiones dentro de ese alcance sin preguntar después de cada fallo. No improvises un nuevo threshold, dataset, loss o encoder fuera del espacio autorizado; no rescates una seed tras mirar V.

Objetivo persistente: separar señal mecanística, continuidad de investigación, promoción completa y confirmación. NO_CANDIDATE cierra una hipótesis, no la misión. Sigue hasta completar la cola viable o agotar un límite externo/recursos; conserva siempre la verdad del resultado. No trabajes artificialmente para consumir un presupuesto ni prometas SOTA por agotarlo.

Presupuesto nuevo: máximo1.000.000 updates físicos (incluye≤1.000 sintéticos y≤5.000 recuperación) y64.800segundos dentro de scopes GPU propios. Recursos: 23GB proyecto, reserva host2GiB/reentrada3GiB y10GB libres tras reservas. Cuenta también I/O dentro del scope; no reportarlo como kernel puro. No cómputo cloud ni cambios globales de equipo.

Usa productores/datasets reales admitidos, optimiza sus loaders existentes, checkpoints completos y análisis fragmentado. No rehacer hashes enormes por update ni crear supervisores duplicados. Ante falta de una dependencia, termina las tareas independientes y pasa a la siguiente rama viable. Si no queda ninguna, entrega causa exacta y comando real de resume, no un estado de éxito.

Mantén comparadores E_CTX/R_CTX/F_TRUE y Garl event/full. Completa el trabajo externo con el propietario V16. El sistema más fuerte puede derivar de Garl; la identidad A5 no es un requisito de éxito. Entrena RGB/fusión, réplicas y refit sólo conforme a las fases del protocolo.

No FCWD/Dev32 para ajustar o escoger hiperparámetros. No etiquetas privadas, submissions, push, merge, emails ni apertura Stage76. Puedes preparar un export local test12 tras congelación si el contrato lo admite; su score oficial requiere un envío autorizado separado.

Entrega al final de cada fase endpoints, predicciones, curvas, métricas por grupo, costes con scope y estado durable. La entrega final lleva RESEARCH_STATUS.md, NEXT_DECISION, registro de experimentos, contabilidad, pesos y bundle reproducible con SHA256. No regreses únicamente con implementación cuando exista trabajo científico admitido ejecutable.
