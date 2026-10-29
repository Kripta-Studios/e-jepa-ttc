# Diagnóstico local de correspondencias eAP–GarlTTC

Fecha: 2026-10-29. Estado: **DIAGNOSTIC_ONLY_G0_BLOCKED**.

Se inspeccionaron 120 consultas train: tres posiciones equiespaciadas tras ordenar
por timestamp y token dentro de cada una de las 40 secuencias. Sus 240 extremos
se unieron a los medios eAP por `(sequence_id, rgb_member_path)`, comprobando además
la coincidencia del TAR y del HDF5. No se leyó ningún medio ni etiqueta test, ni se
ejecutó entrenamiento. El recibo [garl_alignment.json](evidence/garl_alignment.json)
contiene selección, tokens, valores por extremo, SHA-256 de los tres manifests y
las cuarenta etiquetas eAP utilizadas, y hash del script. La auditoría procesó
geometría solamente para esos frames seleccionados; no recorrió las 804.510 cajas
para calcular geometría.

## Resultados medidos

| Comprobación | Resultado en la muestra |
|---|---:|
| Unión inequívoca de medios y diagnóstico completo | 240/240; cero errores |
| Timestamp de anotación Garl distinto del inicio de exposición eAP | 240/240 |
| Secuencias con offset de reloj no constante entre extremos muestreados | 40/40 |
| Final de ventana de eventos Garl igual al inicio de exposición RGB | 240/240 |
| Reproyección de corners Garl con K_event, clipping a últimos centros de píxel, error <1 px frente a ROI Garl | 240/240 |
| Máximo error de esa reproyección | 0,9996923455 px |
| Mismo ID Garl disponible como instance_id eAP en el frame | 36/240 |
| Vecino de centro más próximo con el mismo ID | 20/240 |
| Coincidencia de corners dentro de 1 mm, hipótesis mismo ID | 0/36 |
| Coincidencia de corners dentro de 1 mm, diagnóstico vecino más próximo | 0/240 |
| box3d_h distinto de extensión Z de corners en ego, extremo objetivo, >1 mm | 120/120 |
| box3d_h igual a longitud de arista 0→1, extremo objetivo, dentro de 1 mm | 120/120 |

El clipping se comparó explícitamente bajo dos hipótesis: límites de extensión
`[0,1280] × [0,720]` y últimos centros `[0,1279] × [0,719]`. La primera deja 43
extremos con residual de exactamente 1 px. La segunda explica todos dentro de
1 px, compatible con cuantización entera. No se ha demostrado aquí el algoritmo
exacto de generación de las ROIs ni el rectificado de los píxeles RGB.

## Imagen coincidente no implica objeto coincidente

Se aplicó al campo HF `bbox_3d_ego` **como hipótesis** la regla del visualizador
original ya inspeccionado: `[x,y,z,l,h,w,yaw]`, ancho en X local, largo en Y local,
altura en Z local, yaw horario. Se compararon conjuntos de corners mediante
distancia Hausdorff simétrica, sin exigir el mismo orden de vértices. Las corners
Garl se llevaron a ego usando la inversa del `T_event_ego` del frame.

La igualdad literal de IDs no basta. Para la consulta
`2cyv0Oedzg_000005_19312900000`, extremo 0, el medio corresponde al token eAP
`2cyv0Oedzg:000000221_000612216`. El objeto eAP `_000005` queda a **61,3878 m**
del centro reconstruido desde las corners Garl. El objeto eAP más próximo es
`_000001`, a **0,09142 m**, pero sus corners todavía difieren **0,34533 m** bajo
la hipótesis de conversión. La búsqueda del vecino es un diagnóstico y no una
asociación validada ni un nuevo ground truth.

En los 36 extremos con el mismo ID, la distancia mediana de centros es 1,66590 m.
Entre vecinos más próximos es 0,57200 m; máximo 12,07165 m. Estos residuos impiden
certificar equivalencia exacta de las anotaciones a partir de esta muestra. No
permiten atribuir la causa a un fallo concreto del dataset: pueden intervenir
identificadores independientes, versiones o procesamiento de las anotaciones.
No se modificó ni se corrigió ninguna etiqueta.

## Relojes, disponibilidad y altura

Los offsets observados entre reloj de anotación y exposición van desde
123.760.621 hasta 19.091.187.784 microsegundos. Varían también dentro de cada
secuencia muestreada. El contrato del puente debe conservar ambos campos y usar
la identidad explícita del medio; no debe asumir un offset universal ni restar
una constante escogida a mano.

Las ventanas de eventos terminan en el inicio de exposición RGB. Esto no hace
que la imagen esté disponible entonces: una decisión que use RGB debe considerar
al menos el final de exposición y posteriormente el tiempo real de llegada.
Este diagnóstico no midió latencia de lectura, transporte ni disponibilidad.

`box3d_h` coincide con la arista 0→1, y no con la extensión Z en ego, en los 120
extremos objetivo. El constructor público local
`garl_ttc_benchmark/build_garlttc_dataset.py`, función `_box3d_h`, selecciona el
índice 5 de `box3d_ego`. SHA-256 del archivo inspeccionado:
`dccc152fda78f7b7a75ddc4b9532f9c60f27f8e958b7355bf86ce6afb0a2f93c`.
Esto exige aclarar la semántica antes de reutilizar el campo como altura física.
No se cambia el pipeline TTC histórico a partir de este diagnóstico.

## Reproducción y límites

Desde el worktree, usando el entorno separado de auditoría:

```powershell
.venv-audit/Scripts/python.exe scripts/audit_eap_garl_alignment.py --eap-root E:/eAP_dataset --garl-root E:/GarlTTC_dataset --per-sequence 3 --output docs/detection_local_20261029/evidence/garl_alignment.json
.venv-audit/Scripts/python.exe -m unittest tests.unit.test_eap_garl_alignment -v
.venv-audit/Scripts/python.exe -m ruff check scripts/audit_eap_garl_alignment.py tests/unit/test_eap_garl_alignment.py
.venv-audit/Scripts/python.exe -m ruff format --check scripts/audit_eap_garl_alignment.py tests/unit/test_eap_garl_alignment.py
```

Los 12 tests protegen uniones ambiguas, separación de secuencias y splits, errores
de tiempos y rutas, clipping, geometría tras transformaciones, IDs incompatibles,
ausencia de candidatos y corners tras cámara. El código de salida cero significa
que se generó el diagnóstico sin errores estructurales; **no que G0 esté aprobado**.

Faltan el mapeo documentado entre objetos y anotaciones de las dos publicaciones,
la equivalencia original↔HF↔submission, la convención del scorer, la rectificación
RGB real, la semántica de velocidad y la auditoría causal sensorial completa.
No se generó un conversor de targets ni se declaró ninguna convención HF oficial.
La siguiente acción es resolver esas correspondencias mediante procedencia del
constructor o evidencia de los autores, conservando estos contraejemplos como
casos de regresión; no reemplazarlas por una asociación de vecino más próximo.
