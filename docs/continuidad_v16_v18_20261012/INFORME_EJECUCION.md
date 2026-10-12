# Continuidad V16–V18 ejecutada

Se conserva **PHASE_ONLY_EVENT_S7** y **F_TRUE** como referencia de transferencia.
El candidato sigue sin puntuación oficial. Por tu última instrucción, el envío
en CodaBench lo haces tú: [pasos y archivo exacto](EVALUACION_OFICIAL.md).

La continuación no ha entrenado modelos, modificado V18, ampliado presupuestos,
hecho push/merge ni creado otra rama. Todas las nuevas operaciones de modelo
han sido inferencia CPU para reproducir cabezas y medir coste. V16 sigue limpio
en el commit 71ec9d900189a120b06c3956029496e6fd704f6f.

## Comprobaciones terminadas

- ZIP de continuidad: CRC y 25 payloads del manifiesto verificados; extracción segura.
- Nueve pruebas del paquete, seis del adaptador RGB y dos del lector diagnóstico:
  **17 pruebas pasadas**, repetidas desde la extracción independiente al verificar
  la entrega. Las dos últimas usan fixtures sintéticos, no resultados V18.
  Los 50 tests mencionados en el informe original son históricos, no 50 nuevos tests.
- Reproducción publicada: **46.778 escalares**, 13 tablas internas, 34 externas
  y hashes/recibos de 32 fits verificados; no se repitieron esos entrenamientos.
- Seis cabezas EVENT/ZERO, semillas 7/13/23: **98.592 predicciones V** desde
  las 16.432 entradas cacheadas. Diferencia máxima TTC 3.81469726562e-06 s,
  agregado interno 2.56338495319e-07 MiD. No hay igualdad bit a bit con la publicación;
  las tres EVENT siguen en 89,223/89,199/89,220. No se reconstruyeron los encoders V.
- Submission: copia byte a byte, 6.762 tokens exactos, TTC finitos e idénticos al
  CSV congelado; 5.805 observaciones físicas y 957 aliases preservados; **46**
  fuentes/padres/normalizadores/readouts ligados a la congelación comprobados.
- BANDS, GAP_DECOMPOSITION y SIGN_ERROR_SHARE reproducidos: diferencias menores
  que 1e-13; nuevo desglose descriptivo por subrango, modalidad disponible y fallback.
- Perfil RGB: 64 consultas, 128 pares, 256 mediciones de ruta completa con paridad
  exacta; Garl: 128 rutas completas y controles residentes. [Resultados](COSTES.md).

El paquete completo histórico de 47.842 archivos y el ZIP exterior de informes
no se reextrajeron de nuevo aquí. Sus recibos previos se leyeron; este informe
distingue esa evidencia histórica de las verificaciones ejecutadas ahora.

## Qué concluyen los resultados

PHASE_ONLY pasa los criterios originales del 2 %; el 4 % no aporta predicciones
nuevas. En FCWD, la mejora crucial convive con infraestimación en 3–6 s y peor
agregado/MAE que F_TRUE. Esa banda aporta +6,586 puntos y explica más que el déficit
neto +5,575, compensado parcialmente por la mejora crucial. Las 13 inversiones de
signo en Dev32 concentran 20,09 % del AE y 8,80 % de MiD; las cinco de FCWD están
en el fallback event. Son diagnósticos de cohortes ya expuestas, no criterios
nuevos para ajustar o seleccionar V18. No se construyó un selector por target.

La interfaz PHASE_ONLY mejora la combinación de mediciones congeladas; no prueba
que los mapas espaciales sobren ni que se hayan replicado encoders independientes.
La siguiente investigación sigue siendo V18, con los comparadores y negativos
conservados en [REGISTRO_CANDIDATOS.csv](REGISTRO_CANDIDATOS.csv). No hay SOTA
demostrado ni un refit TRAIN40 nuevo.

## Estado V18 observado

Lectura de 2026-10-12T21:10:30.556313+00:00; es una instantánea no atómica de una campaña viva.

| Fit | Estado del recibo | Updates durables | Journal observado |
|---|---|---:|---:|
| NATIVE-P2-FT-S7 | PAUSED | 3108 | 3108 |
| PAIR-DIRECT-S7 | PAUSED | 2008 | 2070 |
| PAIR-CORR-S7 | NOT_STARTED | 0 | 0 |
| PAIR-CORR-FROZEN-S7 | NOT_STARTED | 0 | 0 |

El lanzador y su hijo Python son un solo supervisor, no dos propietarios.
La matriz conserva cuatro endpoints de 12.000 updates y el límite existente de
64.800 segundos GPU contabilizados; el orden operativo registrado prioriza los
tres pares antes de reanudar NATIVE. No hay resultados P/H/V sellados para analizar
todavía. El mensaje está en E_JEPA_SHARED_COORDINATION; su lectura no está confirmada.

Se dejó un lector externo de predicciones selladas para bandas, grupos/secuencias,
signo, raw/emitted y cap. Para NATIVE, el evaluador actual reconstruye raw_phase
desde el TTC: no es la fase interna predecoder. Esa instrumentación y la
identificación de episodios distintos de sequence_id quedan al propietario V18,
sin cambiar fits activos. [Reanudación y comandos](COMANDOS_REANUDACION.md).

## Dependencias pendientes

1. **Evaluación oficial manual**: archivo y ficha listos. Tu cuenta necesita
   participación aprobada y falta aclarar el contrato de historial de hasta
   650 ms. El agente no enviará nada ni solicita permiso para hacerlo.
2. **Geometría independiente**: 12 paneles conservados, 72 filas sin anotación
   válida. Hay paquete ciego, reparto por escena y protocolo. No se inventaron
   coordenadas ni se promovió una transformación por aceptación NCC.
3. **Resultados V18**: dependen de su entrenamiento autorizado y posterior
   congelación. La falta de score oficial no bloquea esa campaña.

Los resultados de coste CPU no se han instalado en el candidato ni acreditan
paridad GPU. El intento que falló conserva sus resultados negativos. Los costes
de la cabeza aislada y la reutilización event ya completada no se presentan
como el coste del sistema ni como nuevas aportaciones.

Los manifiestos contienen hashes de todos los archivos entregados. El ZIP de
continuidad ejecutada es para archivo/revisión; **el único ZIP para CodaBench es
PHASE_ONLY_S7_GarlTTC_submission.zip**, también copiado directamente a Descargas.
