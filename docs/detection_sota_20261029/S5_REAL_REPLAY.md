# Replay real S5 detector→tracker→TRAIN40

Fecha: 2026-10-29. Alcance: integración de desarrollo expuesta; no mide
precisión ni generalización TTC.

Se verificó primero el JSON temporal7 del piloto. Su span sensorial real es
599.982 µs, 18 µs menor que el gate de historia de 600 ms, y produjo cero
tracks maduros. Se añadió el frame público inmediatamente anterior y se ejecutó
la inferencia del detector sin etiquetas sobre ocho frames consecutivos. El
span temporal8 es 699.975 µs.

En el frame final, las 50 propuestas guardadas se redujeron de forma
reproducible a 17: 14 quedaron bajo score 0,1, ninguna falló proyección/tamaño
y 19 se suprimieron por NMS por clase. Había siete tracks confirmados,
no ambiguos y con antigüedad suficiente.

El replay real usó el candidato sellado
`D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE`, el servicio histórico
`MultiObjectTrain40Service`, `max_objects=2` y `rgb_object_chunk_size=1`.
Ingerió un único paquete final de 20 ms tras inicializar exactamente 600 ms de
eventos. Emitió dos TTC firmados: `-12.055676460266113 s` y
`-9.403387069702148 s`. Conservó las otras quince detecciones: cinco por
capacidad, ocho por historia de track insuficiente y dos por track tentative.

La actualización integral tardó 708,6669 ms. Incluyendo carga de modelos y
lectura de medios, la ejecución midió 1,5958044 s. En una RTX 5070 Ti Laptop de
12.820.480.000 bytes, Torch registró 3.526.439.936 bytes asignados y
3.590.324.224 reservados de pico. Son medidas de este replay max2, no una
garantía para otra capacidad o plataforma.

Una ruta adicional y opt-in, `BatchedPhaseContextRuntime`, conserva exactamente
los TTC del runtime batched histórico y evalúa el `forward` canónico del mismo
head congelado con los width logits de R_CTX. No modifica el productor
histórico y desactiva explícitamente CUDA graphs en esta primera versión. Si no
hay historia RGB, devuelve límites no disponibles en vez de inventar límites de
eventos.

El replay final con `--phase-bounds` mantuvo exactamente los dos TTC anteriores
y transportó `point_phase` `[-0.0082606347, -0.0105783176]`, q10
`[-0.0288517810, -0.0208993666]` y q90
`[0.0098759895, -0.0019528866]`. El primer par cruza cero y se conserva como
coordenadas de fase: **no es un intervalo TTC continuo ni una nueva afirmación
de calibración**. El hash SHA-256 de ese productor fue
`9e263435d364395325c849a7f97819c72cc42cdea02bd11da47756ed0f4f22f2`.

El pico de memoria fue idéntico al replay anterior y la ejecución incluyendo
carga y medios tardó 1,5532352 s.

La extensión aditiva posterior `train40_ingest_once_chunked_v1` separa la
ingestión única de consultas acotadas sin modificar el productor sellado. Las
pruebas CPU cubren cero, uno, varios, más que el chunk y 25 objetos, además de
fallo cerrado y generaciones distintas. La preparación real
`temporal8_prepare_logical30_v2.json` fija capacidad sellada ocho y lógica
treinta. El replay GPU de este dataset solo dispone de siete tracks maduros, por
lo que no demuestra todavía una ejecución real con más de veinte; no se duplican
objetos para fabricar esa evidencia.

Evidencias locales:

- `output/s5_replay/inference_temporal8_v1.json`
- `output/s5_replay/temporal8_prepare_v2.json`
- `output/s5_replay/temporal8_gpu_max2_chunk1_v2.json`
- `output/s5_replay/temporal8_gpu_max2_chunk1_phase_bounds_v1.json`
- `output/s5_replay/temporal8_prepare_logical30_v2.json`

Los dos primeros intentos reales fallaron antes de emitir TTC: uno incluyó un
triplete RGB fuera del límite causal de 650 ms y el siguiente no desactivó TF32,
requisito del graph congelado. Ambos guards se mantienen cubiertos en el script
final. La ejecución sellada posterior es la evidencia reportada arriba.
