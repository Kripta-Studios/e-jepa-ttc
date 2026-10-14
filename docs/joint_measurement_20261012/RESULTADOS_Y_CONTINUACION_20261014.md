# V18: resultados del 12–14 de octubre de 2026 y razón del entrenamiento C1

**Conclusión: todavía no hay un sustituto validado del sistema anterior ni una
demostración SOTA.** Los cuatro entrenamientos primarios y los dos RGB han
terminado. Hay mejoras parciales relevantes, pero las variantes nuevas PAIR
empeoran el signo y otras bandas. El entrenamiento activo C1 comprueba si una
lectura explícita de expansión y traslación aprovecha mejor las correspondencias.

Esta entrega es un corte de investigación en curso. No es el cierre de V18.
Los valores de precisión proceden de endpoints completos; el checkpoint C1
incluido es parcial y no tiene una evaluación final. La hora y el update exactos
del corte constan en `SNAPSHOT.json` del ZIP.

Para leer en GitHub sin descargar el paquete, se incluyen copias de la evidencia
resumida: [matriz primaria](evidence_20261014/PRIMARY_RESULTS.json),
[comparación del sistema](evidence_20261014/SYSTEM_COMPARISON.json),
[métricas RGB](evidence_20261014/RGB_RESULTS.json),
[decisión RGB](evidence_20261014/RGB_SELECTION.json),
[motivación de C1](evidence_20261014/C1_MECHANISM_DECISION.json) y
[transición de ejecución C1](evidence_20261014/C1_RUNTIME_TRANSITION.json).
Las predicciones y pesos completos están en el ZIP, no en estas tablas resumidas.

## 1. Qué cambió respecto a V17

V17 quedó cerrado como NO_CANDIDATE. Sus variantes investigaron principalmente
lecturas y correcciones sobre representaciones visuales congeladas. Esto no
equivalía a demostrar que otro encoder entrenado no pudiera mejorar TTC.

V18 entrena realmente el encoder en los brazos DIRECT/CORR y en RGB-FT, junto
con una nueva medición espacial entre dos observaciones. Los controles congelados
permiten contrastar esa adaptación. Se verificaron gradientes y cambios de pesos;
no se presenta una caché de features detached como aprendizaje del encoder.

El resultado actual actualiza la hipótesis: **descongelar el encoder y conservar
correlaciones no ha sido suficiente bajo estas arquitecturas, objetivos y datos**.
No demuestra que toda adaptación de representación sea inútil. Tampoco permite
seguir atribuyendo los fallos exclusivamente al congelamiento de V17.

## 2. Protocolo y alcance de las cifras

- P: entrenamiento, 53.890 consultas. H: selección de recetas, 18.422 consultas
  de ocho grupos. V: desarrollo interno ya expuesto, 16.432 consultas.
- Endpoints primarios fijados antes de evaluar H; NATIVE seleccionado mediante H
  antes de comparar su resultado V con E_CTX. No se rescataron otros brazos con V.
- Cada fit principal tiene 12.000 updates, seed7. Son seis fits y 72.000 updates
  principales; esta cifra excluye diagnósticos, pruebas técnicas y recuperación.
- El agregado MiD y MiDc de estas tablas son métricas internas; menor es mejor.
  No son puntuaciones intercambiables con el benchmark oficial.
- Los intervalos se obtienen mediante bootstrap por grupos de adquisición.
  Una sola semilla no acredita estabilidad entre semillas.
- NATIVE conserva su objetivo nativo; PAIR utiliza el objetivo de fase nuevo.
  Su comparación no aísla exclusivamente la arquitectura. P2 tiene menos contexto
  temporal que los sistemas históricos con historial y expertos.

## 3. Matriz primaria de eventos

Resultados H, copiados de la tabla regenerable `primary_analysis/RESULTS.json`:

| Modelo | Agregado | MiDc | MiD negativo | Signo incorrecto | MAE (s) |
|---|---:|---:|---:|---:|---:|
| A5-P2-FROZEN | 112,285 | 157,267 | 65,520 | 5,694 % | 5,637 |
| NATIVE-P2-FT | 104,053 | 145,117 | 61,259 | 5,434 % | 4,646 |
| PAIR-DIRECT | 103,870 | 117,980 | 148,167 | 13,533 % | 4,999 |
| PAIR-CORR | 103,510 | 115,556 | 156,002 | 13,804 % | 5,496 |
| PAIR-CORR-FROZEN | 106,722 | 123,608 | 146,990 | 12,257 % | 5,398 |

NATIVE mejora el agregado H un 7,332 % y MiDc un 7,725 % frente a A5-P2
congelado, con las protecciones satisfechas: fue candidato de desarrollo.
Los tres PAIR mejoran MiDc, pero fallan las protecciones de signo y bandas.
La pequeña diferencia CORR/DIRECT no establece superioridad general del volumen
de correspondencias.

El fallo también aparece en P. El MiD negativo del control es 55,522, frente a
168,892 / 175,753 / 163,661 de DIRECT / CORR / FROZEN. Sus errores de signo
son 16,762 % / 17,575 % / 15,890 %, frente a 3,333 % del control. Por tanto,
no es defendible atribuir todo el problema a transferencia hacia grupos nuevos.
Esto no aísla si domina la pérdida, el readout, la representación o su interacción.

### Comparación del candidato NATIVE contra el sistema completo

| Modelo en V | Agregado | MiDc | MAE (s) | Signo incorrecto | E2E p50 (ms) |
|---|---:|---:|---:|---:|---:|
| E_CTX_MATCHED | 106,082 | 156,557 | 2,497 | 4,704 % | 991,200 |
| NATIVE-P2-FT | 126,106 | 178,140 | 5,445 | 7,540 % | 189,140 |

La mayor velocidad no compensa el incumplimiento de los gates de precisión.
Decisión: **NO_SYSTEM_PROMOTION**. El coste E2E procede de 64 consultas
emparejadas, incluye eventos crudos y preprocesado, con pesos residentes y caché
del sistema operativo no controlada. E_CTX conserva su contexto y expertos;
no se atribuye toda la diferencia a la arquitectura.

## 4. Rama RGB completada el 14 de octubre

Dos fits de 12.000 updates: encoder entrenable y encoder congelado. Utilizan
frames reales con los relojes propios de RGB, sin sustituirlos por tiempos event.
Los endpoints se verificaron por separado: encoder cambiado en FT e idéntico
en FROZEN, con estados del optimizador y trayectoria de muestras completos.

| Modelo en H | Agregado | MiDc | MiD negativo | MAE (s) | Signo incorrecto |
|---|---:|---:|---:|---:|---:|
| R_A5-P2-FROZEN | 113,910 | 167,602 | 52,967 | 3,819 | 3,995 % |
| RGB-PAIR-CORR-FT | 90,910 | 115,951 | 105,633 | 5,437 | 7,665 % |
| RGB-PAIR-CORR-FROZEN | 97,852 | 126,816 | 102,215 | 5,331 | 6,840 % |

FT mejora el agregado un 20,19 % y MiDc un 30,82 %. Sin embargo, casi duplica
el MiD negativo y aumenta MAE y errores de signo. FROZEN muestra el mismo tipo
de compromiso. En P también empeoran signo y banda negativa: no es sólo una
regresión held-out. Ambos fallan bandas/signo y no se admiten para V, réplica
o refit bajo esos criterios. Decisión: **NO_CANDIDATE_RGB_PAIR**.

Se conservaron seis tablas P/H, recálculo NumPy, bootstrap por grupos y tres
exports portables. El informe regenerable es `RGB_RESULTS.md`, con sus datos
en `rgb/evidence/REPORT_DATA.json`. R_CTX y F_TRUE siguen siendo comparadores;
no se han sustituido por estos endpoints rechazados.

## 5. Qué evidencia motivó C1

Se separaron dos diagnósticos, ambos sobre P y sin usar targets TTC:

1. Traslaciones conocidas de texturas event: 24 grupos, desplazamientos de ocho
   píxeles. CORR y CORR-FROZEN recuperaron 24.576/24.576 correspondencias interiores;
   los controles permutados, aproximadamente 3,9 %. Esto valida una capacidad de
   correspondencia sobre transformaciones conocidas, no movimiento físico real.
2. Pares temporales reales: 96 pares de 24 grupos. En la región débil delimitada
   por cajas, CORR obtuvo 23.665/32.867 matches recíprocos (72,0 %), frente a
   7.829/32.867 (23,8 %) con posiciones permutadas. FROZEN obtuvo aproximadamente
   80,1 % frente a 24,1 %. Los 24 grupos superaron el contraste registrado.

La reciprocidad no es ground truth de flujo ni prueba de expansión correcta.
Las cajas sólo delimitan el diagnóstico; no entran en el forward del modelo.
El hecho de que FROZEN también muestre correspondencias fuertes impide atribuir
la señal exclusivamente a haber entrenado el encoder.

La hipótesis habilitada es concreta: existe coherencia espacial detectable que
la cabeza de fase libre no convirtió en un TTC aceptable. Probar una lectura
geométrica del mismo volumen puede distinguir una limitación del readout.

## 6. Qué está entrenando ahora

**PAIR-CORR-GEOMETRY-S7, experimento adaptativo C1**, registrado antes de updates:

```text
Dos ventanas event en ROI común + separación temporal real
  -> encoder CNN compartido y entrenable
  -> correspondencias bidireccionales en dos resoluciones
  -> ajuste afín ponderado de escala y traslación
  -> mezcla aprendida de ejes/resoluciones
  -> fase y emisión TTC firmada
```

La relación temporal usa `T_actual = dt / expm1(log_ratio)`. El modelo conserva
la conversión firmada y la extensión de fase declarada para el gradiente; no
recibe cajas o TTC GT en inferencia ni añade una corrección TTC libre.

Se entrena sobre P, seed7, batch efectivo32/micro4, FP32 sin TF32, 12.000 updates.
El objetivo y el control PAIR-CORR-S7 se conservan. El control ya completado se
reutiliza sin repetir sus updates o cargar dos veces su coste. La capacidad pasa
de 578.337 a 578.854 parámetros: es un contraste de readout, no paridad exacta.

Al terminar: sellado del endpoint, verificación de estados, predicciones P/H
completas, coste emparejado, recálculo independiente, bootstrap y export. Sólo
después se interpreta. La cola no abre V automáticamente. Una curva de pérdida
parcial no se presenta como evidencia de precisión o de promoción.

## 7. Rendimiento y optimizaciones: qué está demostrado

El perfil anterior a la optimización mostró 21.375 lanzamientos CUDA en el paso
preparado: `cudaLaunchKernel` consumía 1,825 s de CPU, el 43,65 % del tiempo CPU
perfilado. La GPU observada estaba entre 4 y 15 % de utilización. Esto apuntaba
al coste de lanzar muchas operaciones pequeñas y sincronizar, no a falta de VRAM.

La caché sensorial, archivos ZIP persistentes y prefetch ya estaban activos. En
el primer scope observado, la LRU tuvo 251 aciertos y 7.685 fallos (3,16 %).
Las entradas P ocupan 20,11 GB comprimidas y 86,76 GB densas; la máquina tiene
32 GB RAM. No es posible mantener todo ese conjunto denso en RAM. Una prueba
previa de caché mapeada/descomprimida no pasó sus criterios de velocidad.

La ruta admitida ahora captura forward/backward con CUDA Graphs, mantiene
buffers de gradiente independientes al acumular microbatches y usa vistas
contiguas para conservar memoria pinned. Los intentos fallidos se conservaron:
una aparente aceleración de aproximadamente 9,3 veces fue rechazada porque
alteraba gradientes. No es una mejora válida ni la versión en producción.

La versión corregida obtuvo pérdidas y gradientes bit a bit iguales en el probe,
con RNG y pesos preservados. Medianas preparadas: eager 0,316952 s y captura
0,093844 s, aproximadamente 3,377 veces más rápido. Después pasó la prueba de
optimización/reanudación continua frente a interrumpida. La transición real
restauró 472 updates completos con once comprobaciones y cero pérdida adicional.

Observación de las 17:46: 876 updates ejecutados; checkpoint durable800. Ritmo
desde el arranque optimizado: aproximadamente 2.190 updates/h; tramo reciente
entre checkpoints700 y800: aproximadamente 2.670/h. Es una muestra provisional,
no una garantía de toda la ejecución. La ETA entonces era 21:55–22:55 para el fit,
sin incluir evaluación. No confundir esta aceleración con precisión científica.

Se mantiene FP32 y compresión sin pérdida. No se activó precisión mixta,
aproximaciones numéricas ni una caché congelada del encoder bajo el nombre de
optimización equivalente. El perfil Chrome no se pudo escribir por un problema
de ruta Unicode; sí se conservan tablas CPU/CUDA y timings. No se afirma disponer
de un trace que falló.

## 8. Coste, recuperación y límites

La autoridad aplicada del 14 de octubre permite 172.800 segundos acumulados
dentro de scopes GPU propios y 1.000.000 updates físicos. Amplía el alcance
autorizado; no reinicia la contabilidad. Los scopes incluyen carga/guardado y no
son exclusivamente tiempo de kernels. `campaign/GPU_BUDGET.json` y
`campaign/ACCOUNTING.json` del ZIP conservan el corte; un scope vivo puede no
estar liquidado todavía y los archivos no forman una transacción simultánea.

Los 72.000 updates principales no incluyen los 200 diagnósticos reales P,
pruebas técnicas, C1 en curso ni recuperación. Se conserva la recuperación
histórica: por ejemplo, el reinicio del equipo durante RGB dejó 91 updates
no durables que se contabilizaron al reanudar desde9.900. No se afirma que toda
la campaña carezca de repetición; la transición optimizada de C1 sí preservó
su origen472 completo.

## 9. Conclusión y decisiones pendientes

La evidencia sostiene tres afirmaciones: existen mejoras parciales de TTC;
existe correspondencia espacial detectable; ninguna variante nueva evaluada
justifica reemplazar el sistema completo. No sostiene una causa única del fallo,
superioridad general del encoder entrenable, ni SOTA.

C1 responde a una observación concreta. Después corresponde interpretar sus
resultados completos y elegir sólo continuaciones habilitadas dentro del programa.
La fusión con anchor RGB es una prueba distinta que exige registrar padres,
objetivo, control ZERO y fallback event. El uso de una cabeza histórica entrenada
en H se declarará como supervisión secuencial, no evaluación OOF independiente.
Réplica y refit requieren candidatos admisibles; no se harán para rescatar
retrospectivamente una variante rechazada. La evaluación oficial, si llega a
prepararse, sigue necesitando autorización separada para enviarla.

## 10. Contenido del ZIP y reproducción

- `source/`: código y documentación del commit indicado en el manifiesto.
- `campaign/`: resultados cerrados event/RGB, predicciones, métricas, curvas,
  checkpoints, exports y evidencia seleccionada de diagnósticos/QA/runtime.
- `live_C1_checkpoint/`: una versión durable parcial, con hash y contador,
  separada expresamente de los endpoints terminados.
- `SNAPSHOT.json`: intervalo de captura, exclusiones y trabajo pendiente.
- `BUNDLE_MANIFEST.json`: tamaños y SHA-256 de cada miembro.

No contiene sensores crudos ni padres históricos externos. Se omiten grandes
tensores de diagnóstico/QA, locks y logs vivos; algunos recibos históricos
referencian esos payloads omitidos. No es un paquete completo para reentrenar
sin dependencias. Los informes históricos preservados pueden describir fases
anteriores; este documento y SNAPSHOT.json delimitan el estado de esta entrega.

Desde una copia extraída con el entorno del proyecto:

```powershell
$env:PYTHONPATH = 'src;.'
python -B -m operational.joint_measurement.bundle verify 'RUTA_AL_ZIP'
```

La verificación recorre miembros, tamaños, SHA-256 y CRC. No constituye un
reentrenamiento ni una reevaluación oficial. El paquete lleva además los recibos
históricos de métricas/exports. No se deben ejecutar los drivers de entrenamiento
desde esta copia ni introducir sus journals en el árbol de contabilidad vivo.

Fuentes principales dentro del paquete: `primary_analysis/RESULTS.json`,
`system_compare/RESULT.json`, `rgb/evidence/REPORT_DATA.json`,
`rgb/SELECTION_FINAL.json`, `NEXT_DECISION_MECHANISM_20261014.json`,
`adaptive/C1_geometry/REGISTRATION.json`,
`runtime/geometry_captured_training_v1/TRANSITION.json` y
`runtime/C1_bottleneck_20261014/AUDIT.json`, bajo `campaign/`.
