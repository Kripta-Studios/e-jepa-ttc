# Datos, resultados, métricas y exposición

**Actualización posterior al corte base:** el screen149 terminó durante esta revisión; F0 pasa y el router no. La ampliación de F0 está registrada en curso y la conservación de historias eAP sigue pendiente. Los pasajes que describen el screen como pendiente corresponden al corte inicial. Véase [actualización de cierre](13_ACTUALIZACION_CONCURRENTE.md).

El principal riesgo al analizar este proyecto es comparar números que tienen el mismo nombre pero distinta población, referencia temporal o genealogía. La tabla siguiente fija los denominadores antes de discutir mejoras.

| Población | Tamaño registrado | Papel y límite |
|---|---:|---|
| OLD_DEV / recovery | 8.192 consultas, nueve secuencias | Desarrollo reutilizado; folds y parents varían por campaña |
| P/H/V anterior | 53.890 / 18.422 / 16.432 | Roles 24/8/8 secuencias; V ya expuesto |
| TRAIN40 | 88.744 consultas, cuarenta secuencias | Refit; P/H/V anteriores dejan de ser validación del modelo final |
| Test12 oficial | 6.762 tokens, 5.805 consultas físicas, 957 aliases | Score privado archivado; aliases no son inferencias independientes |
| EvTTC inicial/expandido H8 | 946 / 3.640 targets | Historial de transferencia distinto de Extended-new |
| Extended-new del paper | 468 targets, 32 secuencias | Positivos, ROI anotada, desarrollo expuesto |
| Historical46 | 46 targets | Consultas disjuntas, pero tres secuencias compartidas con Extended-new |
| Competición EvTTC | 434 consultas, diez secuencias | 365 carretera y 69 Slider; targets no verificados en el paper |
| Auditoría local posterior | 423/434 targets | Once ausentes y tres discrepancias frente al PDF del organizador |
| Desarrollo del adaptador final | 2.608, 25 grabaciones | 2.531 carretera/24 + 77 Slider-500; exposición histórica |
| Piloto temporal PAIR | Dos consultas | Sin targets, solo respuesta al warp; no precision poblacional |
| Screen canónico registrado | 149 consultas | 72 carretera + 77 Slider-500; resultado completo no localizado en este corte |
| Piloto JEPA cerrado | 512 P + 256 H, 24+8 secuencias | Nueve entrenamientos; probe de log-altura, no score final TTC |

Fuentes: [protocolo del paper](../../paper/sections/protocol.tex), [transferencia](../../paper/sections/transfer.tex), [datos 29–31](../research_log_20261029_31/02_DATOS_EVALUADOR_Y_FISICA.md), [cierre JEPA](../jepa_closure_20261031/README.md).

## Resultados cuya interpretación importa para las propuestas

El resultado oficial es 35,801451580824 Overall MiD; la comparación 45,03 corresponde a Garl publicado. No son segundos. La reducción usa pesos de bandas 0,5/0,3/0,1/0,1, no frecuencia empírica de todas las consultas. La ausencia de una banda impide reconstruir el Overall completo; no se deben redistribuir pesos para obtenerlo.

En Extended-new, el recálculo desde CSV da:

| Modelo | MAE s | RTE % | p95 s | Máximo s | Error >10 s |
|---|---:|---:|---:|---:|---:|
| PHASE | 1,022677 | 41,226716 | 3,243010 | 32,505471 | 5 |
| RGB | 1,015377 | 38,555541 | 3,798781 | 16,644880 | 2 |
| Eventos | 1,451422 | 65,695499 | 4,397003 | 35,954098 | 4 |
| Garl, adaptador anterior | 3,257596 | 140,068866 | 5,252376 | 624,072595 | 13 |

Las cifras completas y su script están en [RESULTS_VERIFIED.md](RESULTS_VERIFIED.md). La diferencia media PHASE–RGB es +0,007299808 s. El intervalo archivado es [−0,110305331; +0,128971192]; el de p95 también cruza cero. El modelo de fusión no domina RGB y no hay prueba de equivalencia. Comparar con el Garl anterior oculta un control más exigente: el propio RGB de PHASE.

La [entrega adaptada del 31](../evttc_submission_20261031/REPORT.md) registra, tras aplicar a ambas versiones la interpolación de negativos indicada por el organizador, RAE local 58,9781→25,8452% y MAE 1,057825→0,561545 s sobre 423 targets. Esta mejora local es sustancial, pero no equivale a un score oficial: faltan once targets, tres secuencias no reproducen el PDF original y el historial de selección está expuesto. No se deben insertar esos números en la tabla de 468 filas.

## Genealogía y réplicas

Congelar un encoder impide nuevos updates, pero mantiene toda su exposición. Si vio TRAIN40, no se puede volver a llamar independientes a P/H/V. OOF debe separar también teacher caches adaptadas, normalizadores, parents y reglas de selección. Un profesor genérico externo puede aportar supervisión privilegiada sin fuga local; debe declararse su procedencia y acceso.

Tres semillas de cabeza sobre el mismo productor estiman variabilidad condicional. En V18 C1 se repiten adaptación y cabeza desde A5/RGB compartidos. En el piloto JEPA reciente sí hay tres estudiantes entrenados, pero el profesor es común y el conjunto H ya había estado expuesto en el proyecto. Ninguno de esos casos constituye tres adquisiciones independientes.

La independencia se debe establecer por grabación/adquisición y sus aliases/cámaras. Agrupar por familia de escenario no certifica que dos vídeos sean adquisiciones distintas. El dossier reciente identificó ocho grabaciones de competición dentro de las 32 nativas mediante hashes de frames. El fit final de 2.608 excluye esas ocho y Slider-750/1000, pero esa exclusión no deshace las decisiones tomadas tras ver sus errores.

## Contratos de sensor y target

El entrenamiento nativo de carretera y el replay de competición utilizan preparaciones y cámaras distintas. La competición usa RGB/eventos derechos; Slider usa otro conjunto de cámaras y una rotación provisional. Una loss fotométrica necesita revisar esa geometría y la respuesta del ISP; no basta que ambos tensores tengan igual tamaño.

El paper eAP define TTC a partir de la profundidad mínima y la velocidad relativa axial. Por ello, seguir una región material concreta puede dar un cociente distinto del punto más cercano, sobre todo si cambia orientación o el punto que realiza el mínimo. Es una discrepancia de definición, no necesariamente un error del estimador. La salida oficial debe mantener su target y timestamp. [eAP, definición y adquisición](https://arxiv.org/html/2603.16303v1).

La label pública reconstruida y la privada suavizada no son idénticas por contrato. El TTC firmado del benchmark tampoco es la solución de choque entre volúmenes bajo frenada futura. Una cabeza de aceleración puede describir dinámica auxiliar; no debe sustituir el target de comparación.

## Datos que faltan para decidir con más fuerza

Se necesita una población nueva con procedencia de adquisición resuelta, recesión, arranque, cambios de velocidad y varios regímenes de iluminación. También un mapeo autoritativo de los targets/tiempos de competición y calibración intersensor suficiente para objetivos fotométricos. Los cientos o miles de nuevas ventanas de las mismas secuencias no sustituyen ese material.

Se puede hacer diagnóstico sobre lo disponible, etiquetándolo como desarrollo. La confirmación independiente debe reservarse antes de entrenar. El benchmark privado no debe convertirse en un bucle de selección para reloj, sensor y observabilidad.
