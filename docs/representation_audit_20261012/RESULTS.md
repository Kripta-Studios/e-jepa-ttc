# V17 — cola A: bypass geométrico

Decisión: **NO_CANDIDATE**. V es desarrollo interno expuesto.

## Métricas regeneradas

| fit | role | MiDc | MiDs | MiDl | MiDn | overall_MiD | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Z-S7 | H | 123.5735 | 40.7242 | 39.6674 | 35.9888 | 81.5696 | 1.6757 |
| Z-S7 | V | 156.5566 | 54.6613 | 52.4499 | 61.6029 | 106.0820 | 2.4974 |
| G-S7 | H | 92.9304 | 35.9939 | 33.7357 | 29.5213 | 63.5890 | 1.5975 |
| G-S7 | V | 148.1797 | 55.9208 | 52.8701 | 62.1078 | 102.3639 | 2.6501 |
| Z-S13 | H | 121.4825 | 42.0419 | 39.3978 | 36.5640 | 80.9500 | 1.7609 |
| Z-S13 | V | 157.3158 | 57.1715 | 52.9690 | 60.8746 | 107.1937 | 2.6931 |
| G-S13 | H | 91.4767 | 35.9195 | 35.3284 | 30.4773 | 63.0948 | 1.6630 |
| G-S13 | V | 149.7113 | 56.0797 | 53.8674 | 60.6004 | 103.1264 | 2.6903 |
| Z-S23 | H | 124.3522 | 41.2856 | 39.3174 | 37.0241 | 82.1959 | 1.7565 |
| Z-S23 | V | 157.3946 | 54.7049 | 51.8404 | 60.6549 | 106.3583 | 2.5484 |
| G-S23 | H | 90.0499 | 37.1825 | 34.6733 | 31.1194 | 62.7590 | 1.6297 |
| G-S23 | V | 147.1744 | 58.3725 | 54.6488 | 62.3175 | 102.7956 | 2.6171 |

## Media aritmética de métricas entre tres semillas

| arm | MiDc | MiDs | MiDl | MiDn | overall_MiD | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| G | 148.3552 | 56.7910 | 53.7954 | 61.6752 | 102.7619 | 2.6525 |
| Z | 157.0890 | 55.5126 | 52.4198 | 61.0441 | 106.5447 | 2.5797 |

## Clipping

| fit | updates | clipped_fraction | last250_loss |
| --- | --- | --- | --- |
| Z-S7 | 2500 | 0.1692 | 0.1656 |
| G-S7 | 2500 | 0.2356 | 0.1362 |
| Z-S13 | 2500 | 0.2080 | 0.1659 |
| G-S13 | 2500 | 0.2324 | 0.1377 |
| Z-S23 | 2500 | 0.1680 | 0.1709 |
| G-S23 | 2500 | 0.2448 | 0.1397 |

## Bootstrap pareado por grupos

| CI95 | delta_MiDc | groups | seed |
| --- | --- | --- | --- |
| [-15.727437046733748, -1.149457129909269] | -8.3769 | 8 | 7 |
| [-15.21231871982345, -0.04504987855996688] | -7.6045 | 8 | 13 |
| [-21.019189124922484, 0.6031152116886812] | -10.2202 | 8 | 23 |

Los resultados H son ajuste/diagnóstico; V es desarrollo expuesto. Los encoders se comparten entre semillas. Los intervalos no son un gate adicional. No se escoge semilla ni se promedian predicciones.

Métricas reconstruidas exactamente de 98.592 filas CSV. El perfil mide sólo cabeza CPU, con inputs preparados y host compartido; no representa E2E. La contabilidad GPU incluye el tiempo de los procesos propios dentro de cada reserva; las reservas no cerradas se contabilizan completas.
