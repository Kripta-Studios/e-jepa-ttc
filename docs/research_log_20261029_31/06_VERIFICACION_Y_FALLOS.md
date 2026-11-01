# 06. Verificaciones, fallos conservados y revisiones

## Qué cuenta como evidencia de funcionamiento

Se comprobaron propiedades diferentes: ausencia de fuga entre grupos, integridad
de los archivos, causalidad de entradas, equivalencia numérica de rutas, formato
de submission, reproducción de pesos y regresión del repositorio. Ninguna prueba
de ingeniería sustituye targets correctos o una adquisición independiente.

La evidencia se conserva en JSON, XML, logs y manifiestos. Los informes citan la
última ejecución pertinente y mantienen las fallidas. Una repetición que pasa
no borra que la primera falló ni transforma una prueba corregida tras observar
su error en un test científico ciego.

## Entorno Windows: dos fallos distintos

El entorno anterior tenía problemas de `.pth` y una instalación incompleta de
Torch. Python 3.11 intentaba decodificar ciertos archivos editable con la página
de códigos local al atravesar una ruta con «Álvaro». Se creó `.venv-transfer`,
se hizo explícito UTF-8 y se usó instalación no editable para reproducción.

En la prueba final de instalación separada volvió a aparecer un Torch de caché
sin `__init__.py`: el import producía un namespace sin `__version__`. Un refresh
de metadatos no lo arregló. Se reinstaló la rueda oficial fijada usando
`--reinstall-package torch --no-cache`; después se verificaron import, operación
tensorial, evaluación y entrenamiento. Los logs distinguen este problema del
entorno intacto que había entrenado y generado las predicciones originales.

El comando publicado usa `uv sync --group dev --no-editable`, ya que `dev` es un
grupo de dependencias, no un extra. Los comandos posteriores usan `uv run
--no-sync` para no reintroducir un editable de forma implícita. En el entorno
local de tests, `UV_PROJECT_ENVIRONMENT` apunta al entorno corregido para que
los subprocesos no vuelvan al `.venv` antiguo.

## Resultados históricos de las baterías

| Iteración | Resultado registrado | Interpretación |
|---|---|---|
| Auditoría inicial | 7.254 passed, 46 skipped, un timeout | La batería completa no estaba verde; el archivo repetido sin carga pasó |
| Campaña F1 corregida | 7.297 passed, 46 skipped, 17 subtests; 709,93 s | Primera corrida tuvo siete fallos de codificación; se conservan ambos registros |
| Gate robusto | 7.309 passed, 46 skipped, 17 subtests | Correcciones finales cubiertas además por 14 tests focalizados |
| Router de arranque | 7.320 passed, 46 skipped, 17 subtests; 952,74 s | 11 pruebas focalizadas finales; alcance por etapa en FINAL_CHECKS |
| Nuevo envío final | 7.329 passed, 46 skipped, 17 subtests; 896,729 s | Cero failures/errors; incluye las pruebas añadidas después de baterías previas |

La suma JUnit incluye subtests de manera distinta al número de elementos testcase.
En el cierre final, `tests=7392`, `skipped=46`, `failures=0`, `errors=0`; hay
7.375 elementos testcase y 17 subtests adicionales. Por eso se informa
7.329 tests aprobados más 17 subtests, en lugar de etiquetar todo como tests nuevos.

El crecimiento del contador no demuestra más precisión del modelo. Las pruebas
verifican contratos. Las omitidas conservan sus condiciones de disponibilidad;
no se cuentan como aprobadas. Las advertencias de tracing de código histórico
también permanecen en los logs.

## Paridad y causalidad comprobadas

- La instrumentación de auditoría mantuvo 40/40 salidas idénticas a la ruta sin
  traza. Se verificaron código y dependencias de candidato antes/después del run.
- La aceleración de extracción comprobó 36 consultas en 32 grabaciones, además
  de la paridad inicial de los cuatro CUDA Graphs.
- Los routers conservan F0 bit a bit en 88.744 ejemplos eAP sin historia de cajas.
  Esa prueba no evalúa eAP con geometría ni garantiza ausencia de regresión en
  cualquier nueva entrada del dominio fuente.
- Los exportados TorchScript se compararon con eager y recarga. En el nuevo envío
  hay 2.608 filas de fit y 434 filas de replay verificadas.
- En el replay final, la diferencia máxima entre cálculo por consulta y GEMM por
  batch es `2,86102295e-6 s`; se acepta la tolerancia FP32 declarada. Eager y
  export por batch coinciden exactamente. No se llama bitwise a las tres rutas.
- En 24 grabaciones, desplazar el reloj completo y añadir una observación futura
  mantiene el resultado. El filtrado temporal evita consumir la observación futura.
- Un refit del fold seed 7/CPLA del router previo reprodujo sus pesos. En el envío
  final se repitieron las tres semillas desde otra instalación y coincidieron
  los archivos de checkpoints completos y las 2.608 predicciones.

## Diferencias numéricas que no se ocultaron

En la entrega de arranque, un replay nativo nuevo difirió en unos `9,54e-7 s`.
Ocho consultas de Slider difirieron como máximo en `0,003544450 s` porque la CLI
desactivaba TF32 y la caché anterior había heredado otro contrato de precisión.
Se registró esa diferencia; no se afirmó paridad bit a bit de sensores.

El replay final fija FP32 y desactiva TF32. La evaluación desde el ZIP extraído
y desde la instalación separada reproduce el mismo resultado, dentro del contrato
de paridad ya comprobado. El ZIP conserva los tiempos medidos originales; no se
reemplazan por tiempos de una ejecución posterior más rápida.

## Pruebas de integridad de submission

El verificador independiente compara el inventario exacto de miembros del ZIP,
cabecera, cantidad de filas, índices, texto de timestamps, TTC y costes contra
el ledger. Rechaza identidades duplicadas, filas extra/ausentes, no finitos,
costes negativos o una cifra modificada. Los TTC negativos finitos son válidos
según el contrato comunicado por el organizador.

Las pruebas de interpolación usan timestamps no uniformes y negativos
consecutivos, confirman que la entrada no se modifica y rechazan extremos sin
vecinos o relojes no crecientes. La prueba de población cambia las etiquetas de
Slider-750/1000 y comprueba que no cambian las filas elegibles del fit final.

## Revisión independiente del nuevo envío

Un revisor con contexto nuevo encontró cero problemas Critical, cuatro Important
y dos Minor. Los cuatro importantes se corrigieron antes del ZIP:

1. Faltaba el README raíz que requiere la metadata de instalación.
2. El comando de instalación trataba `dev` como un extra inexistente.
3. El paquete podía combinar un modelo internamente válido con replay o inputs
   pertenecientes a otra ejecución; ahora se comprueban las identidades cruzadas.
4. El freeze de código no incluía todas las dependencias ejecutables neuronales.

Las pruebas de sustitución e inventario fallaron antes de implementar las
correcciones y después pasaron. Se ensayan cambios de modelo, target, población,
dependencia de código y commit. La batería focalizada final tiene siete tests.

Para el cuarto punto se conservó el runner ya sellado y se añadió una auditoría
del checkout retenido contra el commit anterior. Coinciden 2.054 archivos Python,
con 17 diferencias únicamente CRLF/LF. El suplemento enlaza los hashes exactos
actuales en el empaquetado. Es una comprobación posterior: no demuestra que
hubiera monitorización continua ni un freeze completo de dependencias al empezar.
La documentación limita la garantía de reanudación a los archivos inventariados.

Los dos detalles menores quedaron visibles: la guía copiada a la raíz del ZIP
usa una referencia al informe que debe resolverse en la carpeta documental del
repositorio, y el fit final original carece de sus propios timestamps exactos
de inicio/fin y entorno completo. Se conservan duración, semillas, hashes y
checkpoints; no se inventan horas históricas para completar una plantilla.

## Validación de la entrega y publicación de esta memoria

El ZIP final contiene 2.588 miembros del manifiesto, más el propio manifiesto.
Se releyeron y verificaron todos sus hashes, se extrajo en otra carpeta y se
ejecutó allí el evaluador con los pesos extraídos. Coinciden métricas, identidades,
negativos, ausencias y el sello de las predicciones. El resultado se conserva en
[FINAL_DELIVERY_VERIFICATION.json](evidence/FINAL_DELIVERY_VERIFICATION.json).

Para la memoria y publicación actual no se reentrena ni se repite una suite de
quince minutos: no cambia el código de inferencia. Se verifican enlaces, cifras,
inventario, ausencia de sensores crudos en los ZIP, commit remoto y assets. La
evidencia previa sigue identificando el commit del modelo, no el commit documental
posterior. Un push correcto tampoco equivale a aceptación por EvTTC.
