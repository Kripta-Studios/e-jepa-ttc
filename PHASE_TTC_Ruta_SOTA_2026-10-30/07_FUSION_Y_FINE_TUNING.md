# 07 — Fusión, adaptación limitada y fine-tuning

## Cuándo abrir esta línea

Abrir F0/F1 después de G3 si las entradas pasan sus contratos, el error aparece de forma consistente en desarrollo elegible y hay indicios de que los productores contienen señal útil. Si faltan datos independientes de un dominio, adaptar la fusión no crea esa independencia. El resultado será una variante `PHASE-ADAPT`, con entrenamiento EvTTC declarado.

La prioridad es pequeña porque la cabeza final ya existe, es barata de ajustar sobre features congeladas y permite aislar si el problema está en la combinación. Esta recomendación es una decisión de diseño; no hay evidencia todavía de cuánto mejorará.

## 1. Arquitectura que realmente se adapta

`PhaseOnlyFinal` recibe un estado de 340 dimensiones, pero utiliza solamente 20 coordenadas físicas: diez de RGB y diez de eventos. Su MLP tiene estructura 20→32→32→3, dos LayerNorm y SiLU, y **1.955 parámetros** contando pesos y sesgos. La predicción puntual es la fase del contexto RGB más `0.03 × correction[0]`; las otras dos salidas corrigen anchos de cuantiles. [Implementación exacta](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/training.py).

Las coordenadas contienen fases de contexto/expertos, tiempos y longitud de historia. No incorporan explícitamente soporte, `known_mask` ni las varianzas de productores en la MLP puntual. Los logits de ancho RGB entran en la salida de cuantiles. No describir esta cabeza como una puerta de fiabilidad calibrada. [Construcción del estado](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/features.py).

## 2. Matriz secuencial de cuatro controles

| ID | Variante | Qué permite atribuir |
|---|---|---|
| F0 | Cadena y cabeza originales congeladas | Referencia del nuevo protocolo |
| F1 | Misma cabeza, mismas 20 coordenadas, ajuste en desarrollo | Beneficio de adaptación con capacidad fija |
| F2 | Cabeza pequeña con indicadores de calidad admitidos | Beneficio conjunto de señal de calidad y capacidad |
| Fcap | Capacidad equivalente a F2 usando solo las coordenadas originales | Control del aumento de capacidad |

Primero ejecutar F0/F1. Abrir F2/Fcap únicamente si la traza muestra que calidad/soporte predicen cuándo la corrección debería cambiar. Mantener igual población, entrenamiento, normalización, evaluación y presupuesto para comparar F1/F2/Fcap. No introducir a la vez nuevas ROI, nuevo backbone y nueva pérdida.

Para Fcap documentar parámetros efectivos y coste. Igualar solo parámetros desconectados no es un control útil; ajustar la anchura de una red que use las mismas coordenadas originales y reportar la diferencia residual si no es posible una igualdad exacta.

## 3. Features de calidad admitidas

Propuesta inicial, a reducir según A00: duración real disponible, número de observaciones, fracción de eventos en ROI, event rate, soporte de productores, edad de la estimación, discrepancia entre expertos, anchos de contexto y cantidad de recorte/clipping. Todas deben poder calcularse causalmente durante inferencia.

Quedan fuera TTC/depth/velocity GT, error frente al target, ID de secuencia como feature y etiquetas de escenario usadas para reconocer una grabación. Un `track_id` puede organizar estado y resets; no debe convertirse en un embedding que memorice escenas. La disponibilidad de una modalidad puede entrar como indicador explícito.

`known_mask` y cuantiles son señales a validar: no representan automáticamente probabilidad calibrada de error en otro sensor. Evaluar si ordenan riesgo y si mejoran la predicción fuera de los grupos de ajuste. No eliminar las observaciones difíciles para simular una mejora.

## 4. Construcción del dataset y normalización

1. Sellar el manifest de roles y su hash.
2. Verificar genealogía de todos los componentes usados para producir features.
3. Extraer F0 una vez por identidad de entrada. El cache incluye ROI, timestamps, representación, calibración, política, modelos y normalizadores.
4. Conservar features y predicciones también cuando exista soporte bajo; motivos de ausencia separados del target ausente.
5. Ajustar cualquier normalizador nuevo exclusivamente con ajuste. Mantenerlo fijo en selección y holdout; no adaptarlo por secuencia test ni con su distribución completa.
6. Ponderar o muestrear por grupo independiente para que grabaciones largas no dominen. Registrar conteos y pesos: una fórmula inicial razonable es peso de muestra inverso al número de muestras de su grupo, normalizado globalmente en ajuste.

La normalización agrupada significa que el ajuste de estadísticos respeta los roles/grupos; no significa aprender una media específica del test. La cadena existente ya utiliza normalizadores sellados y masas por grupo en entrenamiento histórico. Esas identidades no certifican automáticamente el nuevo protocolo. [Normalización y entrenamiento](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/training.py).

### Cuándo hace falta OOF

Si los productores y contextos son externos, están congelados y nunca vieron las grabaciones del desarrollo actual, pueden producir directamente sus features de ajuste/selección: no hace falta generar OOF artificial para ellos.

Si se ajustan productores o contextos con etiquetas del nuevo desarrollo y se entrena otra capa sobre sus predicciones, construir features fuera de fold para ese apilamiento. El grupo de la muestra no puede haber entrenado el componente que genera su feature OOF. Reajustar la cadena de despliegue solo con roles admitidos tras cerrar selección. El código `scientific_oof_*` ofrece piezas reutilizables, pero sus manifests históricos no sustituyen el nuevo contrato. [Pipeline OOF existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/scientific_oof_pipeline.py).

## 5. Piloto propuesto y límites de búsqueda

Estos son **valores iniciales propuestos**, no hiperparámetros óptimos ni recursos medidos. Las restricciones más estrictas del proyecto o del protocolo vigente prevalecen; no editar caps históricos para permitir un run.

| Elemento | Propuesta para F1 |
|---|---|
| Inicialización | Pesos de cabeza original; productores/contextos congelados |
| Semilla exploratoria | 7 |
| Optimizador | AdamW, LR `1e-4`, weight decay `1e-4` |
| Batch de features | 128, reducible si el entorno lo exige; sin alterar masas |
| Gradiente | Clip de norma 1,0; registrar activaciones |
| Smoke | Hasta 100 updates para validar pérdida, serialización y finitud |
| Presupuesto principal | Máximo 2.000 updates o 30 épocas, lo que ocurra primero |
| Selección | Evaluación al final de cada época; paciencia 5; métrica fijada en G1/G2 |
| Configuraciones | Una inicial; como máximo una alternativa predeclarada |
| Confirmación | Semillas 7/13/21 después de cerrar diseño y selección |

La alternativa `LR=3e-5` solo se abre ante inestabilidad del entrenamiento documentada según criterio fijado antes; no porque el test no mejore. Un smoke que falla por bugs no consume una hipótesis científica, pero queda registrado y no autoriza ampliar indefinidamente la búsqueda.

Usar cachés de features permite separar coste de extracción y coste de entrenamiento de cabeza. No estimar horas de GPU sin medir ambos. Una GPU histórica de 12 GiB puede servir para empezar; el preflight debe comprobar dispositivo y memoria libres actuales.

## 6. Pérdida, coordenada y selección

Conservar primero la pérdida original al estudiar exclusivamente adaptación de cabeza. Si G3 identifica una discrepancia entre objetivo fuente y error relativo EvTTC, registrar una segunda hipótesis de pérdida bajo capacidad y datos iguales. Una pérdida robusta relativa puede servir para ajuste, pero cualquier denominador estabilizado es una decisión del entrenamiento: no modifica la métrica oficial ni permite eliminar targets cortos.

La fase canónica, inverse TTC `q` y segundos tienen sensibilidades diferentes. Cambiar a `q` requiere ID de variante, dominio, conversión y controles propios. No combinar esa decisión con F2 en el primer piloto. La pérdida no debe usar `log(T)` indiscriminadamente si el protocolo incluye negativos; definir su dominio y estados censurados explícitamente. [Dominio nativo de fase](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/simplex_t/phase.py).

Seleccionar por el criterio predeclarado, acompañado de cobertura, MAE, error relativo, peor grupo, signo y coste. Si el entrenamiento baja la pérdida pero selección empeora, registrar sobreajuste; no avanzar automáticamente a una red mayor.

## 7. P0: fine-tuning parcial condicionado

Abrir P0 solo si el diagnóstico sitúa el problema en un productor/contexto y F1 no puede corregirlo. Descongelar un bloque concreto, documentar parámetros entrenables y empezar con microbatch 1–2, acumulación si procede y LR de ese bloque menor que el de la cabeza. Son elecciones de piloto, a verificar en preflight.

Mantener baseline de arquitectura igual, estudiar retención en fuente con datos elegibles y regenerar features/OOF donde corresponda. No recuperar datos fuente con genealogía desconocida para aumentar volumen. Si faltan medios eAP, registrar la limitación antes de prometer mezcla de dominios.

Cerrar G4 con decisión única: promover candidato a [confirmación](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md), rechazarlo o declarar evidencia insuficiente. No seleccionar la mejor semilla por el holdout; ninguna fila de competición decide LR, pérdida, checkpoint o gate.
