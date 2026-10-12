# Encargo para una sesión nueva: cierre comparativo multimodal con pesos congelados

Analiza y ejecuta este handoff sobre el resultado publicado V13. El objetivo es
obtener las comparaciones y trazas que faltan para decidir el siguiente cambio,
no repetir una campaña de entrenamiento ya completa.

Base revisada: `62cf4dbe914ea077c68e42983f869cd5422ab71c`.
Lee README.md, execution_policy.json y docs/01_REVIEW.md–03_NEXT_TRAINING.md
antes de implementar. Usa una rama/worktree propios, por ejemplo
`scientific-recovery-v16-multimodal-validation`; V13/V14 quedan intactos.

Autorización nueva de este encargo: CERO optimizer updates, científicos,
técnicos y de recuperación. Se permiten implementación, pruebas sin optimizer,
inferencia con modelos congelados, registro geométrico sin TTC, análisis y
perfilado del alcance definido. Las propuestas de entrenamiento del documento
03 requieren otra autorización y no se ejecutan con el saldo de V14/V15.

1. Comprueba el estado real y hashes. V13 ya terminó sus doce fits: no reanudar
   TRAIN ni H_R_RAW, mediana determinista, known o scorers que ya se repararon.
   V14 NO_CANDIDATE se conserva. Verifica si V15 se inició; no lo dupliques ni
   canceles. Mantén sus propietarios, recetas y outputs.
2. Reproduce/reutiliza Q0 y conserva la reconciliación de cohortes. Dev32 actual
   es 1024/946, no 3947/3640. No unas esos CSV sólo por query_id: se renumeraron.
3. Completa Q1: Garl visual-only y full con la geometría FCWD registrada y
   sellada, configuraciones/pesos nativos y auditoría de disponibilidad temporal.
   No hace falta fingir calibración oficial para una comparación exploratoria.
   No supongas que toda tripleta admisible V13 es una pareja admisible Garl.
   Conserva filas rechazadas e intersección de soportes. Publica además una
   política explícita GarlFull_registered_fallbackEvent para comparar sistemas
   completos. No entrenes Garl, no cambies NCC según TTC, no uses depth GT.
4. Obtén trazas multimodales Q2 con paridad, distinguiendo productores, medianas,
   residual de contexto y residual de fusión. No inventes pesos de router H8.
   Completa F_TRUE–R_CTX en V donde estén sus predicciones locales.
5. Ejecuta Q3 de cobertura/registro sin etiquetas TTC. Conserva rechazos y
   reporta las referencias geométricas que falten. No promociones una mejora
   de registro sólo por NCC, ni uses un frame futuro como prior. Las alternativas
   se evalúan geométricamente antes de alterar el sistema candidato.
6. Mide Q4 con el mismo perímetro, comparadores Garl optimizados, recursos
   aislados cuando corresponda, arranque, resets y latencias explícitas. No
   atribuyas a la cabeza sola lectura, registro, productor o commit de estado.
7. Entrega todos los outputs mínimos, casos negativos e impedimentos concretos.
   No regreses sólo con documentación o comandos cuando existen tareas viables.
   Un bloqueo de Garl full no cancela trazas/coste/V15. Guarda fragmentos para
   reanudar, no lances workers repetidos para el mismo fallo no diagnosticado.

Recursos: conserva límites vigentes, presupuesto agregado de 23 GB, reserva
física de 2 GiB/reentrada3 GiB, margen de commit Windows de 1 GiB y disco libre
10 GB tras reservas, salvo una orden local posterior explícita. La nueva cola
CPU debe acotarse a 4 GiB y pocos hilos cuando comparta host. No cambios globales
de drivers, paging, energía, entorno ni prioridades. No más de un propietario
del slot GPU pesado. No modificación de cambios ajenos ni reformateo legacy.

No subas datasets o checkpoints privados. No push/merge/emails/submissions por
este encargo. No abras Stage76, labels privados ni scores sellados. FCWD/Dev32
ya expuestos sirven de desarrollo exploratorio, nunca de test ciego nuevo.

El informe final debe separar: hecho verificado, recálculo descriptivo,
inferencia causal no demostrada y experimento propuesto. No ocultes que F_TRUE
usa RGB donde Garl histórico es sólo eventos. No confundas calibración métrica
con registro local event-assisted. Si Garl full gana, publica esa conclusión.
