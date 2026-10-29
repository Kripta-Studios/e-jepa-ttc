# Ventanas de rendimiento observadas

Ventanas secuenciales con carga y datos cambiantes. No son ensayos aleatorizados. La fila fast_4 está invalidada por desconexión de disco.

| Ventana | Updates | Rango | Updates/min | Elegible |
| --- | --- | --- | --- | --- |
| pre_pipeline_control | 400 | 20601–21000 | 45.660 | véase análisis |
| process_pipeline_short | 400 | 21201–21600 | 101.593 | véase análisis |
| four_threads_short | 400 | 22401–22800 | 81.707 | véase análisis |
| restored_one_thread_reference | 800 | 23201–24000 | 36.370 | véase análisis |
| fast_collate_primary | 400 | 24201–24600 | 57.152 | véase análisis |
| fast_collate_confirmation | 2000 | 24201–26200 | 67.466 | véase análisis |
| fast_collate_post_aux_secondary | 1000 | 25201–26200 | 82.626 | véase análisis |
| fast_2_short | 400 | 26401–26800 | 72.038 | véase análisis |
| fast_4_interrupted_ineligible | 400 | 27101–27500 | 23.881 | False |
