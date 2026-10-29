# Correspondencia del autor y alcance de la reconstrucción

Fuente: correo de Hongxin proporcionado íntegramente por el usuario en esta sesión.
No se ha verificado su autenticidad fuera de la conversación ni enviado una respuesta.
La fecha de recepción del correo no consta. Este documento registra su contenido
técnico y lo contrasta con los archivos locales; no atribuye al autor nuestros resultados.

## Qué establece el correo

- Los datos públicos y las anotaciones se alinearon con medios y manifests durante
  la preparación de la entrega. Recomienda los recursos publicados en Hugging Face.
- Test12 conserva TTC GT privado. La evaluación pública oficial pasa por
  [CodaBench](https://www.codabench.org/competitions/17289/); no proporciona etiquetas,
  paquete offline privado ni evaluación individual por correo.
- Garl se entrenó en eAP y se evaluó en EvTTC sin fine-tuning. El checkpoint full
  publicado es `paper_ours_full.pth`; el artículo no identifica explícitamente
  el filename utilizado para la Tabla VI.
- Escenarios CCRs2-medium, CCRs2-high y CCRm-medium; intervalos configurados
  [9.4, 11.6], [15.5, 17.4] y [21.4, 22.2] s. No son una lista exhaustiva de queries.
- RGB y eventos izquierdos; bbox de `leftlabel/*.json`, seguimiento por proximidad
  al centro anterior; cuatro esquinas RGB proyectadas para formar el ROI de eventos.
- Recorte cuadrado con lado igual a la dimensión mayor, factor 1.0; RGB aproximadamente
  cada 0.1 s, tripletas deslizantes y uso de los dos últimos RGB y dos segmentos de eventos.
- TTC = profundidad / velocidad relativa en el eje óptico; `ttc.csv` con asociación
  por timestamp más cercano y tolerancia máxima 0.05 s. Se omiten muestras inválidas,
  anotaciones/GT ausentes, eventos vacíos y errores de preprocesado.
- RTE = 100·|pred−GT|/|GT|; también se publica runtime.

## Qué se ha ejecutado aquí

| Elemento | Implementación de esta comparación | Límite |
|---|---|---|
| Checkpoint Garl | Full publicado, SHA `e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e`; carga estricta y fuentes fijadas | No certifica identidad con el checkpoint de Tabla VI |
| Escenarios | Recorre los tres nombres y todas las variantes locales CCRm medium | El correo no especifica overlap de CCRm |
| Población | Rejilla 100 ms dentro de cada intervalo, RGB real más cercano con tolerancia 1 ms; tripletas completas | No conocemos las query IDs exactas del artículo |
| Cámaras | RGB izquierdo y eventos izquierdos del HDF5 | No se sustituye por FCWD/right camera |
| Objetos | Bbox de anotación; proximidad al centro anterior; primer centro de referencia en centro de imagen; empates por índice | ROI anotado, no detector desplegado |
| Tiempo de bbox | Disponible en su timestamp; consulta usa sólo información hasta su ancla | Difiere del margen conservador de 50 ms del Dev32 histórico |
| RGB→evento | Calibración local y proyección de cuatro esquinas por rotación, rectángulo envolvente | Omite traslación/paralaje sin profundidad; el correo no da la fórmula exacta |
| Garl ROI | Factor 1.0, recorte cuadrado; RGB comparte el cuadrado final según release; eventos por observación | Reconstrucción desde release, no certificación de preprocessing exacto Tabla VI |
| Rutas propias | Conservan ROI común con margen 0.25 y productores/contextos congelados de su adaptación anterior | No se igualan sus recetas a Garl cambiando pesos o preprocessing después de puntuar |
| Historia | Garl usa el par de la tripleta; sistemas propios conservan sus contextos causales más largos | Se compara el sistema completo, no una ablación con idéntico presupuesto de historia |
| TTC | Tabla pública de cinco columnas; timestamp columna 1, TTC columna 4, índices desde cero; vecino más cercano ≤0.05 s | Dev32 histórico interpolaba; no mezclar sus resultados con esta cohorte |
| Conversión | Cada sistema conserva su conversión nativa; sin clipping adicional ni reajuste | Garl no tiene por qué compartir el soporte numérico de las cabezas propias |
| Estadística | Predicciones completas selladas antes de leer GT; métricas emparejadas; bootstrap por secuencia | EvTTC ya observado históricamente y sólo tres grupos efectivos |
| Runtime | CPU FP32 dos hilos; fases registradas; ruta propia comparte cinco modelos | No equivale a latencia individual ni al hardware de Tabla VI |

## Cobertura y exclusiones antes de leer TTC

| Directorio local | Intervalo | Consultas completas |
|---|---|---:|
| CCRs-2/medium-100/overlap-100 | 9.4–11.6 s | 21 |
| CCRs-2/high-100/overlap-100 | 15.5–17.4 s | 18 |
| CCRm/medium-0/overlap-0 | 21.4–22.2 s | 0 |
| CCRm/medium-50/overlap-50 | 21.4–22.2 s | 0 |
| CCRm/medium-100/overlap-100 | 21.4–22.2 s | 7 |

En overlap0 faltan anotaciones en ese intervalo. La grabación overlap50 termina
antes de él. Se conservaron 18 exclusiones de frames y 14 de tripletas; son
32 registros de dos niveles, **no 32 consultas independientes descartadas**.
El manifiesto registra estos motivos sin consultar TTC. No se eligió overlap100
por su resultado. Las 46 consultas con sensores válidos tuvieron después GT válido
y salida finita de los seis sistemas; no se excluyó ninguna por error predictivo.

## eAP y la independencia

La afirmación de alineación del correo no obliga a reinterpretar sin evidencia
la genealogía de cada checkpoint histórico. Se verificaron los ocho grupos V contra
el split público de 46 IDs y todos aparecen. Esto impide tratar esta V como test
independiente para Garl full. El test12 privado continúa fuera de la tarea.

El subconjunto eAP de 64 queries se fijó por IDs, ocho por grupo. La comprobación
de preprocesado llamó al upstream local en ocho consultas y dio igualdad bit a bit.
No hubo selección por calidad de predicción. Los modelos propios usan sus outputs
V históricos, por lo que esta tabla compara precisión, no tiempos emparejados de ejecución.

## Decisiones que esta evidencia no autoriza

No acredita SOTA oficial, no convierte EvTTC/Dev32/FCWD en test ciego, no prueba la
banda negativa de A1, no promueve V18 y no cambia el gate histórico del 10%.
No sustituye la revisión de reglas de modalidad/historia ni un recibo CodaBench.
No hubo descarga de datasets, nuevos optimizer updates, acceso a GT privado,
refit, modificación de V19, submission, push ni merge en esta tarea.
