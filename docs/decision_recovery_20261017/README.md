# V19: recuperación del equilibrio de TTC

Continuación autorizada el 17 de octubre de 2026, desde V18 `fb41681`, en una rama y un directorio de artefactos propios. El código CPU quedó registrado en `e768765` antes de ajustar los routers. V18 y sus resultados permanecen intactos.

## Preguntas y orden

1. Diagnosticar signo y magnitud sobre consultas idénticas de P/H; los oráculos usan etiquetas y no son candidatos.
2. Medir selección aprendida entre expertos congelados: NATIVE/FULL como pareja principal, A5/FULL y NATIVE/CORR como comparaciones registradas. Una regresión ridge fija predice el coste relativo. Cada grupo H se evalúa con un router que excluyó ese grupo de su ajuste y normalización. H sigue siendo desarrollo reutilizado.
3. Comparar FULL-L0 y FULL-LW desde el mismo padre B1 de correspondencias, con 12.000 updates por brazo y semilla 7. Solo cambia la masa negativa del objetivo del 10 al 20%; las positivas mantienen su proporción 5:3:1. Se crea un control nuevo.
4. Evaluar ambos endpoints completos P/H, sin selección de checkpoints por H. Replicar únicamente si se satisface el disparador registrado; las demás familias necesitan un contraste separado justificado por estos resultados.

El protocolo ejecutable está en `configs/decision_recovery/protocol.json`. Los registros y recibos se guardan bajo `artifacts/decision_recovery_20261017`. El presupuesto inicial incluye un máximo de 100.000 updates físicos y 60.000 segundos de scopes GPU, con límites adicionales para QA y recuperación. No autoriza gasto en nube.

## Resultado CPU ejecutado

Se admitieron las mismas 53.890 consultas P y 18.422 H para siete expertos, verificando identidades, targets, cobertura y hashes. Las poblaciones de entrenamiento P y desarrollo H tienen grupos distintos; la exposición histórica de H y el preentrenamiento genérico se declaran en el registro.

La regresión negativa de FULL combina dos problemas. En H introduce 576 errores de signo donde A5 acertaba y rescata 217 donde A5 fallaba. Sobre los 6.484 negativos donde **ambos aciertan el signo**, el MAE de FULL es 6,290 s frente a 4,108 s de A5. Por tanto, una cabeza de signo por sí sola no arreglaría la magnitud. Frente a NATIVE, sobre 6.510 negativos con ambos signos correctos, los MAE son 6,384 y 3,193 s.

El selector NATIVE/FULL reduce el agregado H de FULL de 78,920 a 77,629, el crítico de 95,846 a 93,793 y el MAE de 5,446 a 3,935 s. El negativo baja de 87,087 a 79,071, pero la protección frente a A5 exige aproximadamente 72,072. Su error global de signo es 5,618%, ligeramente peor que el 5,607% de FULL. No es una mejora en todas las métricas ni un candidato admitido. Los otros dos pares también se conservan y rechazan bajo sus controles registrados.

Resultados completos: [ROUTER_RESULTS.md](ROUTER_RESULTS.md). Los intervalos de los JSON remuestrean grupos sobre las predicciones OOF ya ajustadas; no son repeticiones independientes del aprendizaje. El modelo ALL_H exportado es de desarrollo y sus predicciones sobre H no sustituyen la evidencia OOF.

La única extensión registrada R1, con 194 descriptores espaciales congelados adicionales, también ha terminado: agregado H 78,680 frente a 77,629 de R0 y negativo 79,091. Mejora ligeramente MAE y signo, pero no supera R0 ni la protección negativa. El contraste queda rechazado, sin habilitar otra extensión dentro de este protocolo. Véase [FEATURE_ROUTER_RESULTS.md](FEATURE_ROUTER_RESULTS.md).

La lectura lineal de signo sobre esos descriptores terminó tras una continuación CPU registrada del límite de convergencia. Su error H es 17,381% frente a 5,607% del signo emitido por FULL: no justifica sustituir la lectura actual por este clasificador. La AUC 0,910 muestra señal accesible, pero no una mejora sobre FULL. Esto no prueba ausencia de información ni descarta todas las lecturas no lineales. Se conservaron íntegros los fallos técnicos y las 72.312 extracciones P/H con paridad exacta; véase [SIGN_PROBE_RESULTS.md](SIGN_PROBE_RESULTS.md).

## Rendimiento y reproducibilidad

El contraste L0/LW y las cuatro evaluaciones completas ya terminaron. LW mejora negativos, MAE y signo, pero empeora el agregado un 1,20% frente a L0: no habilita la replicación registrada. Véase [LOSS_SCREEN_RESULTS.md](LOSS_SCREEN_RESULTS.md). V19 continúa con diagnósticos de gradientes y lectura, y medición de coste; no se reanudan los endpoints ya completados.

Se reutilizan las optimizaciones de V18: modelo y pérdida capturados con CUDA Graphs, sesión persistente, muestreo reproducible, precarga del siguiente lote, memoria pinned, decodificación paralela y caché acotada. Las optimizaciones adicionales de transferencias y workers se someten a paridad de datos, gradientes, RNG y reanudación antes de su uso científico. La compresión sensorial histórica permanece sin cambios. El tiempo de captura, calentamiento y QA se contabiliza separadamente del tiempo estable por update.

Los dos brazos usan el mismo runtime admitido. Los checkpoints conservan modelo, Adam, scheduler, sampler y RNG; las escrituras son propias de V19 y los locks coordinan la GPU con V18. Las pruebas de velocidad no permiten modificar pérdidas, cohortes, precisión numérica ni criterios científicos.

## Qué permitiría hablar de SOTA

La mejora interna y la comparación pública deben informarse separadas. El [artículo Garl-TTC](https://arxiv.org/html/2603.16303v1) publica 66,2 para event-only con LHR y 45,0 para RGB+event en su evaluación; no son umbrales directamente comparables con el agregado interno por grupos de H. El [repositorio oficial](https://github.com/NAIL-HNU/Garl-TTC) contiene el contrato de submission y el scorer. Una afirmación SOTA requiere población, modalidad, métrica, reglas y evaluación oficial comparables; esta fase local no presenta una submission ni acredita un ranking actual.

Los resultados de EvTTC, incluido [REACT](https://arxiv.org/html/2609.19204v1), corresponden a otro protocolo y no sustituyen esa comparación. No se reetiquetará H reutilizado como test independiente ni se eliminarán los brazos negativos del informe.
