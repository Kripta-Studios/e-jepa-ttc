# Encargo para Codex — RGB-PORT

Autorizo una campaña NUEVA, separada, para comprobar la transferencia de A5/C2F/PAIR y del refinador contextual a RGB, y su fusión con eventos. La lectura/envío explícito de este encargo por el usuario es la autorización; la existencia de un archivo en el repo no lo es.

Base inspeccionada: b44443ae331b5e84b94e940801468449b2858de4.
Lee el README de este handoff, execution_policy.json y los documentos de docs. Sus contratos científicos gobernarán sólo esta nueva campaña, sin reescribir los anteriores.

La entrega SOTA_ESSENTIAL añadió cero optimizer updates. No repitas TRAIN40, T6, WIDE/H16 o las evaluaciones ya cerradas. No reinicies los productores Garl pausados. La propuesta DUAL-TTC anterior no es evidencia de entrenamiento ni obligación de lanzar ResNet50: este traslado compacto es la prioridad nueva para trabajo no iniciado. Si ya existe otra campaña en ejecución, registra su estado y coordina sin modificarla ni matarla.

Tu primer hallazgo a verificar localmente: CausalScaleTTCConfig ya admite modality='rgb'; _sensor_support ya distingue RGB/event. Reutiliza esa arquitectura, no reimplementes otro modelo como sustituto oculto. C2F es el experto llamado informalmente C2H en la conversación.

Ejecuta P0→P4. Antes de cada fase con ajustes, completa QA, fija configs/inputs/roles y congela su entrenamiento. No regreses sólo con un plan, código sin ejecutar o comandos sugeridos cuando las dependencias permitan ejecución.

Parte científica:
- fija roles P/H/V por grupos dentro de TRAIN40, aproximadamente60/20/20;
- aprende R_A5/R_C2F en P con RGB real y deltas reales;
- reutiliza sólo controles event excluyentes de H/V o entrena E_A5/E_C2F en P;
- entrena PAIR de cada modalidad en P;
- congela productores, crea features H/V;
- aprende E_H1/E_CTX/R_H1/R_CTX/F_TRUE/F_ZERO en H;
- congela seis heads antes de puntuar V;
- después evalúa transferencia sólo como desarrollo ya expuesto, manteniendo el contrato congelado.

No fabriques imágenes de50ms ni conteos event para RGB. R_CTX usa hasta8 tripletas reales dentro650ms, que pueden ser menos de8 a10Hz. Usa máscara, disponibilidad y espacios de cámara correctos. R_H1 ya tiene varios frames; no lo llames single-image. Inputs RGB [0,1] para soporte; no introducir ahí normalización ImageNet. No reutilices pesos, PAIR, normalizadores o PHASE17 event como RGB por igualdad de dimensiones.

No inicialices un modelo evaluado en V con TRAIN40 full o un teacher de tarea que vio V. Si sólo hay grouping por secuencia, declara la limitación; no lo llames confirmación fresca. Nunca uses GT para decidir las filas históricas disponibles o una transformación de entrada.

El soporte RGB/FCWD no resuelto bloquea sólo esa evaluación, no eAP RGB ni el entrenamiento propio. No extrapoles boxes de1280x720 a1920x1200 multiplicando tamaños; no consumas depth GT. Continúa tareas independientes listas.

Límite nuevo: científicos228408, técnicos500, recuperación11092, máximo físico240000. No es el remanente previo ni permiso para rescates. Productores18 épocas/endpoint fijo con recipe de base auditada; PAIR6840 máximo; seis heads2500. No nuevas seeds/arquitecturas/KD dentro de esta cuota.

Conserva R1: no duplicar supervisor ni competir por GPU. Slot para un trainer pesado; CPU ligera puede avanzar dentro de23GB agregados, reserva dura2GiB y recuperación3GiB, margen disco10GB. Usa checkpoints cada100, análisis fragmentado y guardas baratas. No cambies drivers/paging/Control Center ni uses cómputo externo pagado.

Conserva métricas nativas y presenta separadamente el clipping común ±60 ya calculado. Reporta también 0–3s, RTE, mediana, signos, colas y costes. Los nuevos splits/thresholds no usan Dev32/FCWD para elegir recetas o rescatar brazos. Un MAE mejor no basta para proclamar mejor seguridad o SOTA.

No abras Stage76, labels privados, test sellado o CodaBench; no submissions, push, merge ni emails. Prepara el siguiente protocolo de confirmación en la entrega, sin ejecutarlo.

Termina con report, NEXT_DECISION, configs, pesos nuevos, predicciones, normalizadores, curvas, pruebas, procedencia, accounting, coste y ZIP verificable. Si falta una dependencia real, preserva lo viable y entrega el bloqueo exacto. No detenerse sólo porque los autores aún no hayan contestado el protocolo oficial.
