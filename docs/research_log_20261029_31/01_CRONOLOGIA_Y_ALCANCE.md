# 01. Cronología, decisiones y alcance real

## Antecedentes: 28 y 29 de octubre

El envío `PHASE_TTC_EvTTC_results_2026-10-28.zip` precede a esta campaña. Contiene
434 estimaciones de diez secuencias. El organizador confirmó su evaluación,
interpoló los dos TTC negativos y pidió hardware, cámaras, focales, autores y
afiliaciones. La nueva campaña preserva ese ZIP y sus resultados: una nueva
versión del modelo no cambia retroactivamente la identidad del envío original.

El 29 de octubre, el historial registra la publicación del artículo y su
integración en main (`e58d0106`), correcciones de CI y documentación de detección
3D (`f2903f0b`). Los documentos de detección G0/G1 corresponden a una línea de
preparación: registran gates bloqueados y G2 no ejecutado. No constituyen un
detector entrenado ni un experimento TTC adicional.

Los Markdown recibidos en la raíz y sus ZIP se conservan como material histórico.
Una afirmación de falta de GPU en el entorno de aquella auditoría no describe el
portátil utilizado para las ejecuciones CUDA de los días 30 y 31. El dossier
distingue entorno, fecha y contrato antes de comparar hechos.

## 30 de octubre: de plan a herramientas de auditoría

El commit `30b28b5c` contenía el plan de diagnóstico y el anexo de evidencia. El
paquete recibido añadió 21 Markdown: índice, quince capítulos, cuatro plantillas
y manifiesto. Se verificaron CRC e inventario; el manifiesto enumera los veinte
documentos de contenido y no se cuenta a sí mismo como un hash autocontenido.

La primera implementación produjo herramientas para reconstruir la identidad de
datos, la genealogía de componentes, el protocolo de exclusiones, la alineación
de targets y las trazas pasivas. Los commits `0f091880`, `07bd0497` y `d799345d`
separan esas responsabilidades. La revisión `65edf4ee` reforzó la unión entre
evidencia, protocolo y diagnóstico antes de abrir gates.

La auditoría reprodujo ocho correspondencias de grabación y 365 frames de
carretera. El evaluador creó 434 filas diagnósticas, de las cuales 354 tenían
target local utilizable en ese momento. Se preservaron once consultas fuera de
cobertura y las 69 de Slider sin correspondencia entonces validada. La
instrumentación de 40 consultas mantuvo paridad exacta.

La decisión inicial fue no afirmar que existía desarrollo independiente suficiente
para seleccionar una intervención aprendida. Se documentaron adquisiciones
desconocidas, historial de exposición y desacuerdos con el scorer. Cuando el
usuario pidió continuar hasta entrenar, el alcance pasó a adaptación exploratoria
declarada. No se reinterpretaron esos bloqueos como si hubieran desaparecido.

## 30 de octubre: refit F1, recuperación y rendimiento

`0cc59c62` añadió el refit completo de la cabeza final. `bd8805ad` reforzó la
recuperación de cachés interrumpidas y la verificación del grafo adaptado.
`ed9743d5` añadió informes/model card desde resultados. `dfbd20b7` introdujo la
reutilización de productores mediante CUDA Graphs y cachés de sensores, con
comparación de paridad. `50b2bb68` cerró la entrega F1 y sus mediciones.

El entrenamiento utilizó 92.315 estados: 88.744 eAP/TRAIN40 y 3.571 EvTTC. Se
entrenó la cabeza de 1.955 parámetros durante dos épocas completas, 1.444 updates
por semilla, manteniendo el resto de componentes congelados. Es un refit de la
combinación final; no representa un nuevo entrenamiento de todos los encoders.

La petición de «entrenar con todo» se materializó en esta variante de ajuste.
Su evaluación sobre los mismos estados se etiquetó como resubstitution. El
trabajo posterior añadió protocolos más informativos sin borrar ese primer fit.

## 30 de octubre: campaña agrupada y sensores crudos

Entre `0e3c0469` y `73a74f4d` se ejecutaron 21 ajustes agrupados, pruebas temporales,
probes, controles geométricos, Slider-500 y perturbaciones de sensores. Los
commits intermedios conservaron predicciones ausentes, comparaciones sobre
intersecciones y causas de fallo. El resultado fue mixto: F1 mejoraba error
relativo agrupado, pero empeoraba MAE y no confirmaba mejora relativa en Slider-500.

Se rechazó promover F1 como ganador universal. Tampoco se activó una expansión
automática a JEPA o a entrenamiento de todos los productores. La campaña cerró
con pruebas de Windows corregidas, resultados por grupo y límites explícitos.

## Noche del 30 al 31: fusión informada por calidad

`70efabae` añadió la fusión física con un gate de 401 parámetros por semilla.
La comparación incluyó un control de igual capacidad. El modelo combinaba los
expertos neuronales con un observador geométrico causal y calidad de observación.
Se entregaron `evaluated` y `all_data_refit`, con roles de datos distintos.

Esta iteración mejoró la carretera agrupada más que el control, pero dejó un
fallo grande en el arranque de Slider-750. El observador geométrico aislado mostró
que existía señal útil, aunque perdía consultas por insuficiencia de historia.
Su mejor error en un subconjunto no se confundió con cobertura completa.

## 31 de octubre: router de arranque y coordenada inversa

`3eb5cb5e` investigó dos observaciones causales, calidad de historia y mezcla en
segundos frente a tasa inversa. Se ejecutaron 117 ajustes contando controles,
semillas, folds y refits. Se conservaron tanto el ensayo en segundos rechazado
como el inverso, los pesos intermedios y los fallos de validación.

El router inverso mejoró ligeramente el error OOF de carretera y mucho el ajuste
de Slider. Sin embargo, no dominó el MAE de CCRm ni las perturbaciones crudas.
Por ello se entregó como candidato adicional y se mantuvo la fusión robusta previa
como referencia por defecto en aquella entrega. La decisión no afirma que las
dos variantes hayan sido entrenadas con la misma población de desarrollo.

## 31 de octubre: candidato para un nuevo envío

El correo del organizador resolvió dos puntos prácticos: todos los timestamps son
obligatorios y los negativos pueden permanecer en el fichero original. Se
auditó el ZIP del 28 contra su ledger y se preparó la ficha del único autor.

Para el nuevo envío se fijó la receta inversa ya elegida y se entrenaron tres
routers sobre 2.608 consultas: las 24 grabaciones elegibles de carretera y
Slider-500. Se excluyeron explícitamente las ocho grabaciones solapadas y
Slider-750/1000. La receta conserva exposición histórica y se presenta como
adaptación. No se reabrió una búsqueda de hiperparámetros con el nuevo replay.

Después se ejecutaron las 434 consultas desde sensores, se sellaron las salidas y
solo entonces se calcularon scores locales. El commit `e482c220` reúne el código,
la evidencia pequeña y los borradores. El paquete extraído reprodujo la evaluación
y una instalación separada reprodujo los tres archivos de pesos byte por byte.

## Publicación posterior y cambios de autorización

Los handoffs anteriores decían «sin push», «sin publicación» o «mantener local»
porque ese era el alcance de sus turnos. La solicitud actual autoriza add, commit,
push y subida a GitHub. Se publica la rama de investigación y una prerelease con
los ZIP. No se fuerza main, no se elimina historia y no se envía correo a EvTTC.

La tabla de commits completa permite reconstruir el orden sin depender de la
narración del chat. Los horarios son los registrados por Git, normalmente con
offset `+01:00`; los recibos de inferencia usan UTC. No se inventan horas de
entrenamiento que no se registraron en su momento.
