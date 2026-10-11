# 05 — Qué falla y cómo se habilitó RGB en FCWD

## Geometría disponible y problema original

FCWD contiene RGB derecho de 1920×1200 y eventos de 1280×720. Las cajas
publicadas están en coordenadas de eventos. Redimensionar una imagen no corrige
distorsión, principal point ni paralaje entre cámaras. El archivo
`calibration/lidar_to_rgb_extrinsic/right/extrinsic.txt` transforma LiDAR→RGB;
no permite deducir una extrínseca event→RGB sin la relación restante.

Se leyó el objeto MATLAB MCOS `calibParams.mat` con mat-io en dependencias
aisladas sobre E:. Los intrínsecos se identificaron por cámara, transponiendo
la matriz MATLAB y pasando el principal point de índices 1-based a 0-based.
Hay 131 poses de checkerboard RGB y 33 de eventos sin correspondencia de pareja
acreditada; no se usaron como si fueran observaciones estéreo sincronizadas.

Los números extraídos están en
[PUBLISHED_INTRINSICS.json](../fcwd_rgb_registration_20261012/PUBLISHED_INTRINSICS.json):
focal RGB aproximadamente 4647,5/4649,6 y event 3292,3/3296,1 píxeles.
La disparidad vertical observada varía con la proximidad. No se puede ajustar
toda la escena con una traslación fija universal.

## Algoritmo implementado

Código: `operational/fcwd_rgb_registration/geometry.py`, `prepare.py`,
`transfer.py`, `score.py`, `campaign.py`.

1. Proyectar RGB al plano óptico de eventos utilizando intrínsecos y distorsión.
2. Elegir el último frame RGB con timestamp no posterior a la consulta.
3. Construir plantilla log1p de eventos de los 20 ms anteriores a ese frame.
4. Correlacionar localmente la plantilla con gradiente RGB con CLAHE y suavizado
   sigma 1,5. Estimar una traslación local por consulta sin TTC, LiDAR o profundidad.
5. Exigir al menos 100 píxeles con eventos; correlación normalizada ≥0,22 en la
   ventana completa y sus dos mitades; desacuerdo de traslación ≤5 píxeles;
   máximo de correlación fuera del borde de búsqueda.
6. Transformar el borde de la ROI con 33 muestras por lado y añadir margen 25%
   en una caja cuadrada común. Mantener la misma ROI y transformación durante
   la historia, para no normalizar accidentalmente la expansión del objeto.
7. Construir tripletas de frames reales a ~100 ms, tolerancia 1 ms, deltas
   medidos y span completo ≤650 ms. Rechazar historias no causales.
8. Si falla calidad/soporte, registrar causa. R_H1/R_CTX no producen TTC;
   F_TRUE y F_ZERO usan exactamente E_CTX. No eliminar la consulta.

El registro es una aproximación de correspondencia local en imagen, **no una
calibración métrica extrínseca**. Usa entradas del dominio destino. La ROI de la
consulta se aplica retrospectivamente; no equivale a un tracker que hubiera
producido una ROI nativa en cada observación histórica.

## Qué se probó y qué se rechazó

Redimensionado directo, ECC y SIFT no dieron una transformación fiable. El
ajuste afín global sobre bordes falló el gate geométrico fijado en `PROTOCOL.json`.
No se eligió ese ajuste usando TTC. Los ensayos negativos están conservados en
la raíz E: de calibración; la entrega publica sus recibos textuales en
`evidence/registration_pilots/` cuando están disponibles.

La auditoría inicial seleccionó 60 consultas por hash, 20 por secuencia:
47 aceptadas, dx entre −26 y −21 píxeles y máximo desacuerdo de mitades 4,123.
La campaña completa acepta 495/630: FCWD1 164/214, FCWD2 173/218, FCWD3 158/198.
Los 135 rechazos son parte del resultado. La consistencia NCC/temporal no mide
error contra correspondencias geométricas GT y no prueba exactitud estéreo.

Fuente: [VALIDATION60](../fcwd_rgb_registration_20261012/VALIDATION60.json),
[sellado](../fcwd_rgb_registration_20261012/REGISTRATION_SEALED.json),
[historias reales](../fcwd_rgb_registration_20261012/REAL_INPUT_HISTORY_QA.json)
y [inferencia real](../fcwd_rgb_registration_20261012/REAL_INFERENCE_QA.json).

## FCWD: dónde se concentra el error

F_TRUE tiene MAE global 1,430 s, sesgo −1,034 s y MiDc 102,323. Garl event-only
tiene MAE 2,146 s pero mejor MiDc, 95,057. Por tanto la mejora en segundos
no resuelve el régimen crucial. La tabla 04 regenera los valores exactos.

De las 597 filas etiquetadas, 125 necesitan fallback: 20,94% de las filas y
61,28% del error absoluto total F_TRUE. En ese subconjunto E_CTX obtiene
MAE 4,184 s y sesgo −3,769 s. En las otras 472 filas F_TRUE tiene MAE 0,700 s.
El registro y el experto de eventos en condiciones difíciles son prioridades
de investigación, pero esos agregados no prueban que relajar el gate ayude.

FCWD1 obtiene MAE F_TRUE 2,310 s, FCWD2 1,114 y FCWD3 0,844. FCWD1 incluye
cuatro predicciones negativas con GT positivo y una salida en el límite ±60 s.
La cifra global favorable debe leerse junto a esos fallos. F_ZERO es peor que
F_TRUE y E_CTX no mejora a las tres cabezas H8 V12 en MAE: no hay dominancia
general de V13 sobre V12.

## Dev32: dos secuencias y fallos de signo

`CCRs-1-medium-100-overlap-100` aporta 30 consultas etiquetadas, MAE 12,346 s,
cuatro signos incorrectos y tres salidas en ±60 s. `CCRs-1-low-100-overlap-100`
aporta 31, MAE 5,058 s y ocho signos incorrectos. Juntas concentran 40,72% del
error absoluto F_TRUE con 61/946 consultas (6,45%).

El sesgo global de F_TRUE, +0,068 s, oculta cancelación: otras secuencias
sobreestiman; `CCRs-side-medium` tiene sesgo +3,284 s. Corregir todo con un
offset global puede empeorar unos regímenes para mejorar otros. R_CTX tiene
menor MAE que F_TRUE, aunque F_TRUE mejora MiD positivo: la fusión no gana
automáticamente en todas las métricas.

Los nombres anteriores identifican exactamente las secuencias en
`../dev32_transfer_20261012/DIAGNOSTICS.json`; allí están las 32, sin ocultar
las peores. `SUMMARY_COMPARISONS.json` contiene la aritmética de concentración.

## Hipótesis que todavía no son causas demostradas

| Hipótesis | Evidencia actual | Prueba que la discrimina |
|---|---|---|
| Productores sobre/subestiman antes de la cabeza | Fallos finales localizados; no hay traza completa por etapa en esta entrega | Exportar expansión, residual productor, TTC A5/C2F/PAIR y mediana |
| La cabeza amplifica un experto razonable | R_CTX gana MAE a F_TRUE en Dev32 | Comparar fase base y residual final sobre las mismas filas y controles TRAIN |
| ROI/relojes reducen señal geométrica útil | Registro rechaza muchos casos y usa ROI retrospectiva | Correspondencias manuales sin TTC, cobertura ROI, deltas, soporte, oclusión |
| Pesos de bandas favorecen unos regímenes | Recetas V12/V13 distintas | Auditar masa H materializada, exposición por episodio y gradientes |
| Inestabilidad cerca de fase cero produce extremos TTC | Signos y saturación observados | Traza de fase antes/después, normalización, incertidumbre y clipping |
| Hay domain shift fotométrico/cinemático | Transferencia desigual entre secuencias | Probes de textura, densidad, escala y movimiento con grupos independientes |

La siguiente mejora debe resolver una causa medible y validarse fuera de las
filas usadas para formularla. Los análisis anteriores son diagnósticos, no una
demostración de que un nuevo modelo o pérdida vaya a corregirlos.
