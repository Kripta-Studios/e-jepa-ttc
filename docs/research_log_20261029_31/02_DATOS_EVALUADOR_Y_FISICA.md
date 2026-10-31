# 02. Datos, evaluador, cámaras y pruebas físicas

## Poblaciones que no deben mezclarse

| Población | Tamaño | Uso y límite |
|---|---:|---|
| Features de eAP/TRAIN40 | 88.744 | Referencia fuente y refit F1; no son consultas EvTTC |
| EvTTC nativo, 32 grabaciones | 3.571 targets válidos | Refit exploratorio F1; incluye grabaciones del benchmark |
| Carretera tras ocho exclusiones | 2.531 consultas / 24 grabaciones | Desarrollo expuesto y entrenamiento de gates |
| Desarrollo carretera del gate | 1.747 | Cinco familias; OOF agrupado |
| CCRm reservado en iteraciones de gate | 580 | Cuatro grupos de grabación, una familia; exposición histórica |
| CPNAO para calibración de gate | 204 | Etiquetas separadas del ajuste de la variante evaluated |
| Slider-500 | 77 | Primero externo al ajuste actual; después desarrollo y entrenamiento |
| Slider-750 y Slider-1000 | 51 + 18 | Benchmark; posteriormente expuestos en investigación de arranque |
| Población all-data de gates | 2.677 | 2.531 carretera + 146 Slider; fit descriptivo |
| Fit del envío nuevo | 2.608 | 2.531 carretera + 77 Slider-500 |
| Consultas oficiales | 434 | 365 carretera + 69 Slider; todas deben tener salida |
| Targets locales finales disponibles | 423 | 354 carretera + 69 Slider; once siguen ausentes |

Estas cuentas proceden de manifests y recibos. Que dos poblaciones contengan
«EvTTC» no demuestra que compartan cámara, timestamps, targets, preparación o rol.
La diferencia entre 2.608 y 2.677 es exactamente la retirada de las 69 consultas
Slider del benchmark del fit final; no cambia el historial de selección previo.

## Identidad de grabaciones y exclusiones

La coincidencia se reconstruyó por píxeles/hashes en 365 frames de carretera,
en lugar de fiarse solo de nombres de carpetas. El CLI identificó ocho matches
únicos contra las 32 grabaciones nativas. Las exclusiones completas son:

```text
CCRm-low-100-overlap-100
CCRm-medium-100-overlap-100
CCRs-1-high-100-overlap-100
CCRs-1-low-100-overlap-100
CCRs-1-medium-100-overlap-100
CCRs-2-high-100-overlap-100
CCRs-2-low-100-overlap-100
CCRs-2-medium-100-overlap-100
```

El protocolo propaga exclusiones por grabación, alias y adquisición relacionada.
No permite que una cámara alternativa de la misma grabación se presente como
test independiente. El inventario inicial conservó `acquisition_group: unknown`
cuando faltaban metadatos; una familia de escenario usada para agrupar folds no
certifica por sí sola una adquisición independiente.

La genealogía identificó nueve checkpoints, normalizadores y recibos del candidato
TRAIN40. Donde no se pudo reconstruir una decisión histórica o un teacher ancestral,
se mantuvo `unresolved`. El hecho de congelar pesos evita updates en esa campaña,
pero no prueba ausencia de exposición anterior ni contaminación de los pesos.

## Cámaras y contratos de preparación

La reproducción de competición del 28 y el nuevo replay usan, en las ocho
secuencias de carretera, los grupos `/prophesee/event_cam_right` y
`/blackflys/right`: Prophesee EVK-v4 y FLIR Blackfly-S. La calibración da campos
horizontales de visión próximos a 22° y 23°. La identificación como pareja de
16 mm procede de compararlos con la tabla del artículo EvTTC; no de un campo
nominal en milímetros supuesto dentro del HDF5.

El entrenamiento nativo de carretera utiliza su propia preparación e historia RGB
izquierda. El replay de competición usa la derecha. Se mantuvieron interfaces
separadas para no pasar un manifest nativo por el adaptador de competición o
atribuir sus métricas a una cámara distinta. La transferencia entre contratos
permanece como limitación del candidato final.

Slider usa un testbed con DVXplorer y DAHENG MER2, beam splitter y calibración
propia. Los focales empleados son `fx=1846.739948`, `fy=1845.352379` píxeles para
eventos y `fx=4672.479462`, `fy=4676.997356` para RGB. No se le asignan las etiquetas
8/16 mm de carretera. La focal nominal del testbed en milímetros sigue pendiente.

Slider-500 procedía de otro contrato de topics: `/dvs/events` y `/aps/image_raw`,
frente a `/capture_node/events` y `/image_raw` del adaptador congelado. El lector
inicial lo rechazó y se añadió un perfil separado; no se alteraron silenciosamente
las reglas del envío histórico para conseguir que el fichero entrase.

## Reloj, consulta y causalidad

La consulta se refiere al timestamp de la caja suministrada. La preparación usa
historia causal, intervalos reales y un horizonte máximo de 650 ms. Las ventanas
de eventos se cortan por sus límites temporales; cargar un chunk que contiene
bytes posteriores no permite que esos eventos entren en la ventana del modelo.

El historial geométrico del router de arranque admite dos cajas separadas al menos
10 ms. Si no hay historia o la última observación es más antigua de 150 ms, el
modelo vuelve al estimador original. Las pruebas añaden observaciones futuras y
desplazan todos los relojes por una constante para comprobar invariancia.

La fase canónica de PHASE tiene una referencia de 100 ms, distinta del delta real
entre observaciones. No se cambió a 80 ms por la cadencia de Slider. Esa modificación
habría cambiado la coordenada de salida sin haber demostrado un defecto temporal.

## Prueba de TTC anterior frente a actual

Para aproximación frontal a velocidad constante, con delta de 0,1 s y cociente
`h_actual/h_anterior=1,1`, el TTC actual es `0,1/(1,1-1)=1 s`; el TTC anterior es
`0,1/(1-1/1,1)=1,1 s`. Ambas expresiones pueden ser correctas con referencias
temporales diferentes. El fixture distingue endpoints, pero no prueba por sí solo
que la semántica de una altura aprendida o del target oficial sea incorrecta.

La comparación evita declarar un bug a partir de una fórmula aislada. Se conserva
el recorrido timestamp → observación → productor → fase → TTC para poder ubicar
un desfase demostrado y corregir solo ese contrato.

## Paralaje, traslación y calibración provisional

El adaptador proyecta la ROI con intrínsecas, distorsión y rotación. Omite traslación
porque no dispone de una profundidad permitida en inferencia. Un fixture muestra
el efecto esperado: una traslación omitida puede introducir 10 px a 1 m y 1 px a
10 m bajo la misma geometría sintética. La auditoría posterior ensaya siete
profundidades y rayos más abiertos. Es evidencia del mecanismo físico, no una
atribución causal cuantificada del fallo real en Slider.

La rotación intersensor de Slider procede de un consenso de poses de calibración
y conserva carácter provisional. No se introdujo la distancia real del target
para mejorar el recorte. La validación física requiere extrínsecas autoritativas
o correspondencias medidas en datos independientes.

## Evaluación, negativos y ausencia de targets

El correo aportado por el usuario establece que todas las cajas requieren
predicción y que los negativos pueden permanecer en los ficheros originales.
El organizador sustituyó los dos negativos originales mediante interpolación
lineal entre predicciones válidas circundantes.

La implementación local aplica esa regla a una copia del vector, en tiempo físico.
No modifica el ZIP. Rechaza un negativo en un extremo sin vecinos a ambos lados,
porque el correo no autoriza inventar una extrapolación. Una fila sin target
continúa en la población y en los archivos de inferencia, aunque no entre en el
promedio local de error.

La incorporación de targets públicos de Slider elevó la cobertura local de 354
a 423. El join usa identidad de frame y tolerancia temporal de 2 μs. Permanecen
sin target `CCRs1-medium:32` a `CCRs1-medium:42`. No se extrapolaron para producir
un supuesto score de 434 consultas. La utilidad pública de MATLAB que usa
`interp1(...,'linear','extrap')` no prueba que el scorer privado utilice esa regla.

Seis secuencias de carretera y Slider-1000 reproducen el error original redondeado
del PDF. CCRs1-low, CCRs1-medium y Slider-750 no. La pregunta pendiente al organizador
solicita sus targets alineados o los relojes y fuentes correspondientes. Las
pruebas de software y la bajada de loss no resuelven esa diferencia de evaluación.
