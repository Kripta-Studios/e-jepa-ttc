# Datos, exposición y protocolo común

## Poblaciones que no deben mezclarse

| Población | Uso permitido en este plan | Límite |
|---|---|---|
| TRAIN40, 88.744 consultas/40 secuencias | Replay nativo, entrenamiento y diagnóstico | Todos los productores R0 vieron sus antiguos P/H/V |
| Test12, 6.762 tokens/5.805 consultas físicas | Verificación contractual y evaluación oficial final del candidato fijado | Targets privados; no usar leaderboard para elegir brazos |
| Extended-new, 468 consultas/32 secuencias | Comparación exploratoria con las rutas R0/RGB/eventos | Exposición histórica, ROI anotada y TTC positivo |
| screen149 | Resultado existente de diagnóstico temporal | No nueva confirmación independiente |
| Población 2.608/25 grabaciones | Ampliación de desarrollo de C-TIME | Subset relacionado con adaptación posterior; no test nuevo |
| Competición EvTTC 434, 423 targets locales | Diagnóstico separado del adaptador A0 | Targets ausentes y discrepancias aún documentadas |
| Nuevas adquisiciones reservadas | Confirmación de transferencia de candidatos fijados | Requiere comprobar independencia y calidad de targets |

Fuente de tamaños y exposición: [auditoría previa](../physics_review_20261031/03_DATOS_RESULTADOS_Y_EXPOSICION.md). Ningún número de esta tabla convierte miles de ventanas correlacionadas en miles de adquisiciones independientes.

## Tres niveles de evidencia

**Mecanismo:** trayectorias controladas con tiempo, geometría y sensor conocidos. Permite comprobar una ley y descartar atajos. No demuestra transferencia real.

**Desarrollo:** datos ya vistos por el proyecto, incluidos los ensayos actuales. Permite decidir qué candidato merece una confirmación, declarando la exposición. No se borrará esa exposición al congelar pesos.

**Confirmación:** grupos nuevos y reservados antes de elegir parámetros. Separar adquisiciones completas y cámaras sincronizadas de la misma adquisición. Sin esa población, entregar resultados de desarrollo y una candidatura; no titularlos mejora independiente del release.

## Cómo entrenar una modificación sin falsear el OOF

Para módulos posteriores a productores R0 congelados, un split nuevo de TRAIN40 no crea validación independiente de la cadena. Sirve como desarrollo in-sample respecto a sus padres, con ese nombre. Una primera comparación barata puede usarlo, pero no justificar promoción científica.

Para entradas de entrenamiento fuera de muestra, reutilizar únicamente cachés auxiliares cuyo linaje excluya el grupo correspondiente. En el diseño histórico, cada rotación separa 24 grupos para productores/PAIR, ocho para contextos y ocho para emisión OOF. Si cambia un productor, reconstruir su familia de descendientes dentro de cada rotación afectada. [Contrato histórico](../engineering_history_20261027/02_TRAIN40_OOF_AND_FINAL_TRAINING.md).

Los contextos finales entrenados sobre todas las filas OOF también han usado los targets de todas esas filas. Aplicarlos después sobre las mismas filas no produce automáticamente PHASE20 OOF para una nueva cabeza. Mantener la generación superior con contextos auxiliares excluyentes. Un nombre de archivo `OOF` no basta: comprobar parents, grupos y columnas.

Para comparar receta antigua y nueva con evaluación interna, reservar un fold externo completo y construir los ancestros dentro de los grupos restantes. Eso evalúa réplicas de la receta, no los pesos oficiales R0. El contraste con R0 sobre datos externos no usados por ninguno resuelve la pregunta del producto.

No mezclar el `E_CTX_MATCHED/R_CTX` de entrenamiento OOF con PAIR133 de productores de despliegue para afirmar paridad de eAP. [Inventario de genealogía disponible](../research_log_20261031_canonical_screen149/evidence/eap_history_inventory.json).

## Banco controlado propuesto

Separar un banco geométrico y otro fotométrico. El primero basta para tiempo, ROI y observabilidad; el segundo hace falta para formación del sensor. Antes de entrenar, definir grupos por semilla de escena, textura y trayectoria, reservando combinaciones completas.

| Factor | Valores iniciales propuestos | Resultado que debe medirse |
|---|---|---|
| Dinámica axial | Aproximación, q≈0, recesión, frenada y aceleración | Signo, q y TTC instantáneo |
| Movimiento accesorio | Traslación lateral y giro | Sesgo de expansión y detección de desajuste |
| Tiempo | Escalas 0,75/1/1,25; extremos 0,5/2 de estrés | Equivarianza y sensibilidad a cadencia |
| Espacio | Escalas 0,75/1/1,25, ROI estable y zoom conocido | Invariancia y pérdidas de soporte |
| Fondo/visibilidad | Sin mezcla, 25/50/75% fondo; oclusión gradual | Error aceptado, rechazo y cobertura |
| Sensor | Umbral fijo, asimetría, ruido y refractario | Separación de movimiento y actividad |
| Luz | Constante, rampa global y cambio localizado | Expansión espuria y alcance del modelo de luz |
| Transporte | Stream completo, dropout y paquetes ausentes etiquetados | Distinguir silencio de pérdida de datos |

Son parámetros de diseño, no rangos ya validados ni una campaña cartesiana obligatoria. Empezar con una intervención cada vez y un bloque combinado de fondo+oclusiones. Medir cruces deliberados: mismo TTC con distinto tamaño y textura; mismo aspecto con distinta tasa; igual recuento con distinta información geométrica.

## Causalidad y acceso a información

Para una consulta t, todo input termina en t. El futuro puede ser target de entrenamiento, nunca parte de la normalización por ejemplo, selección del recorte pasado o gate de aceleración disponible en inferencia. El teacher bidireccional, si se usa, queda explícitamente en entrenamiento y no certifica causalidad del estudiante.

Las historias del sistema pueden estar condicionadas por la ROI de la consulta. No reutilizar una predicción anterior como medida idéntica a la de un snapshot reconstruido con otra ROI sin registrar la diferencia. Identificar track, referencia espacial, límites temporales, timestamps reales y cobertura por modalidad.

## Réplicas y análisis

Usar las mismas semillas, ejemplos, número de updates y selección de checkpoint para pares de brazos. Las réplicas abarcan todos los bloques modificados. Si solo cambia un gate, tres gates sobre los mismos productores son una réplica condicional válida de ese gate, no tres entrenamientos del sistema completo.

Bootstrap por adquisición/secuencia, conservando pares y aliases. Separar variabilidad entre entrenamientos de incertidumbre entre secuencias. Fijar receta y métrica primaria antes de abrir la confirmación; con muchos brazos, elegir uno en desarrollo y confirmar ese candidato una vez. Los diagnósticos por bandas se publican completos, incluso cuando contradicen el agregado.
