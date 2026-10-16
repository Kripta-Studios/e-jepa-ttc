# B1: correspondencias conocidas frente a continuación supervisada

Estado al preparar esta receta: D1 sigue entrenando. B1 todavía no tiene
resultados científicos. Sus decisiones proceden del diagnóstico P, sin consultar
el resultado D1 ni nuevas predicciones H/V de B1.

El diagnóstico de texturas con escala conocida encontró señal clara: C1 acierta
aproximadamente el 72% de los desplazamientos no nulos dentro del soporte de
cajas, frente a aproximadamente el 4% con correspondencias permutadas. El CORR
adaptado directamente a TTC obtiene aproximadamente el 69%. Esto no describe un
encoder muerto. Permite una pregunta más precisa: si entrenamos explícitamente
las correspondencias que todavía falla, ¿mejora el TTC posterior más que dedicar
esos mismos updates a TTC?

Es la tercera comparación adaptativa, después de C1 y D1, dentro de la familia B
del handoff. No abre una búsqueda de temperaturas, transformaciones o semillas.

## Dos brazos y dos etapas

Ambos brazos parten del mismo C1 seed7, completo a 12.000 updates y entrenado sólo
en P. Ese coste histórico se reutiliza y se cuenta una sola vez.
El padre ya recibió supervisión TTC. Sólo la nueva etapa de correspondencia
prescinde de esas etiquetas; no es preentrenamiento independiente desde cero.

| Etapa | CORRESPONDENCE | TTC-CONTROL |
|---|---|---|
| 1: 2.000 updates | Sólo encoder; correspondencia de texturas P transformadas | Sólo encoder; TTC real P2 y objetivo C1 original |
| 2: 10.000 updates | Todo C1; TTC real P2 y objetivo original | Todo C1; TTC real P2 y objetivo original |

Son **24.000 updates científicos nuevos** en total. Las cabezas se congelan en
la primera etapa de ambos brazos. Al comenzar la segunda se cargan los pesos de
su propia primera etapa y se reinician Adam, scheduler y RNG a seed7 en ambos.
El reinicio es una frontera científica prospectiva; una reanudación dentro de
cualquier etapa conserva íntegros los estados y los updates.

El control iguala el número de updates, el origen, la capacidad, la secuencia de
IDs P, los grupos entrenables y la segunda etapa. La primera etapa cambia
objetivo e inputs. No iguala FLOPs ni segundos GPU; ambos se documentan con su
alcance. Batch efectivo32, microbatch4, AdamW y LR originales, FP32 sin TF32.

## Supervisión conocida

La rama de correspondencias usa la última textura event de cada consulta P.
Cada microbatch aplica cuatro transformaciones fijas, en orden:

| Escala | Traslación x, px | Traslación y, px |
|---:|---:|---:|
| 0,9 | −2 | 0 |
| 0,9 | +2 | 0 |
| 1,1 | 0 | −2 |
| 1,1 | 0 | +2 |

El warp es bilineal, centrado, con relleno cero y `align_corners=False`. La
etiqueta de correspondencia es la distribución bilineal del desplazamiento
conocido sobre los25 offsets nativos. Se supervisan las dos direcciones y las
resoluciones32 y16, con temperatura0,07, dentro de la mitad central de cada eje.
Las consultas, posiciones, direcciones y resoluciones tienen igual peso.

Antes de cualquier update B1 se corrigió el origen de coordenadas: las dos
convoluciones nativas tienen centros `4*j+0,5` en píxeles de entrada; el pooling
posterior tiene centros `8*j+2,5`. La etiqueta usa esos centros, con una diferencia
de 1,5 píxeles frente a los centros nominales de celdas uniformes. El registro v1,
sus fuentes y su prueba CPU de cero updates permanecen en
`admission_revisions/v1_before_coordinate_correction`. El registro v2 conserva
presupuesto, transformaciones y comparador; la corrección precede a la QA CUDA y
a cualquier resultado de entrenamiento B1.

No se fabrican etiquetas TTC. No es una simulación física de una cámara de
eventos. Las etiquetas P sí existen en el loader compartido y en el control
supervisado; la pérdida de correspondencia no las utiliza. Tampoco utiliza
RGB, imágenes de H/V ni cajas como máscara de entrenamiento.

## Verificaciones y evaluación

Antes de entrenar: pruebas CPU de coordenadas, targets bilineales y gradientes;
comprobación adicional con el padre nativo real, cero pasos de optimizador.
Después, tres pruebas CUDA de régimen: correspondencia con cabeza congelada,
TTC con cabeza congelada y TTC completo. Cada régimen compara eager10,
capturado10 y capturado5+restore5: **90 updates técnicos** como máximo. Se exige
igualdad exacta de reanudación y comparación numérica con eager, incluidos Adam,
scheduler, sampler, RNG y curva. Un fallo queda preservado y bloquea el uso
científico de ese runtime.

El entrenamiento usa CUDA graphs, gradientes con almacenamiento propio, caché
sensorial de8GiB sin pérdida, prefetch y microbatches contiguos en memoria pinned.
No almacena features detached de un encoder que deba aprender. Checkpoint cada
100 updates y en fronteras durables; scopes GPU de hasta900 segundos.

Se congelan las cuatro etapas completas antes de cualquier inferencia H de B1.
Se regeneran todos los IDs muestreados y se verifican pesos, Adam y genealogía.
Se evalúan las cohortes P/H completas, todos los signos y bandas, métricas NumPy
independientes, bootstrap por grupo y coste forward preparado emparejado. Ese
coste excluye preparación sensorial y transferencias; no se llamará E2E.

La señal de mecanismo exige la mejora prospectiva H frente al control TTC:
agregado≥3%, o MiDc≥5% con agregado≤1,02; todas las bandas≤1,10 y aumento de
error de signo≤0,005. La admisión de desarrollo conserva además las reglas
originales frente a A5-P2. Ambas decisiones se separan de la promoción del
sistema. El diagnóstico de escala conocida se repite sobre las mismas24
consultas P para el padre y las cuatro etapas, conservando también su null.

Sólo una admisión H puede habilitar una evaluación V registrada posteriormente.
No se rescata una receta en V, no se replica otra familia y no se afirma SOTA.

## Comandos

Desde el worktree V18 con `PYTHONPATH=src;.` y el entorno existente:

```powershell
python -B -m operational.joint_measurement.correspondence_train --prototype-cpu
python -B -m operational.joint_measurement.correspondence_train --register-only
python -B -m operational.joint_measurement.correspondence_evaluate --register-only
python -u -B -m operational.joint_measurement.correspondence_train
python -u -B -m operational.joint_measurement.correspondence_evaluate
```

Los dos últimos comandos se ejecutan sólo con la GPU propia libre. El mismo
comando de entrenamiento reanuda la primera etapa pendiente; no crea otra copia
ni reinicia las etapas completas. Registro, contabilidad, pruebas y resultados
quedan en `artifacts/joint_measurement_20261012/adaptive/B1_correspondence` y
`artifacts/joint_measurement_20261012/qa/correspondence_B1`.
