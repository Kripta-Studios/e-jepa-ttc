# Localización del fallo de observabilidad

Auditoría posterior al resultado. No se reajusta la sonda ni se excluyen datos.

q estimable: 63/149; sin grados de libertad residuales: 8.
Signo incorrecto: 26; fuera del dominio canónico: 10.
Cuantiles |q| (0/50/90/100%): [0.04490279623755278, 5.162400999786638, 49.899113671047836, 57632.48384225686].
Cuantiles de error |q−1/TTC| (50/90/100%): [4.674920728740635, 49.35564893200564, 57632.71397474189].

Identificar algebraicamente q no garantiza un valor físicamente correcto.
Con cero grados de libertad, residuo casi nulo no demuestra fiabilidad.
El target 1/TTC tampoco certifica el modelo de expansión local bajo giro o paralaje.

| Adquisición | MSE físico | MSE soporte | Delta |
|---|---:|---:|---:|
| CCRm-low-0-overlap-0 | 0.223922 | 0.242450 | -0.018528 |
| CCRm-low-50-overlap-50 | 1.152699 | 0.104010 | 1.048690 |
| CCRm-medium-0-overlap-0 | 0.807284 | 0.596864 | 0.210421 |
| CCRm-medium-50-overlap-50 | 0.433064 | 0.267943 | 0.165121 |
| CCRs-1-high-0-overlap-0 | 1.504059 | 0.322386 | 1.181672 |
| CCRs-1-high-50-overlap-50 | 0.105229 | 0.051978 | 0.053251 |
| CCRs-1-low-0-overlap-0 | 0.074747 | 0.125877 | -0.051130 |
| CCRs-1-low-50-overlap-50 | 0.017506 | 0.005264 | 0.012243 |
| CCRs-1-medium-0-overlap-0 | 0.176791 | 0.110873 | 0.065918 |
| CCRs-1-medium-50-overlap-50 | 0.020726 | 0.075420 | -0.054695 |
| CCRs-3-low-100-overlap-100 | 0.224379 | 0.072721 | 0.151658 |
| CCRs-3-medium-100-overlap-100 | 0.061242 | 0.058143 | 0.003099 |
| CCRs-side-high | 0.593473 | 0.047171 | 0.546301 |
| CCRs-side-low | 1.484293 | 1.508722 | -0.024429 |
| CCRs-side-medium | 0.755471 | 0.996204 | -0.240733 |
| CPLA-high | 0.184586 | 0.426103 | -0.241516 |
| CPLA-low | 1.258341 | 0.949285 | 0.309056 |
| CPLA-medium | 0.314520 | 0.072626 | 0.241894 |
| CPNA-high | 1.354741 | 0.476025 | 0.878716 |
| CPNA-low | 0.502747 | 0.441193 | 0.061554 |
| CPNA-medium | 0.087453 | 0.098495 | -0.011042 |
| CPNAO-high | 10.815747 | 4.398928 | 6.416819 |
| CPNAO-low | 0.424574 | 0.342284 | 0.082290 |
| CPNAO-medium | 0.202667 | 0.262172 | -0.059505 |
| Slider-500 | 0.201424 | 0.247956 | -0.046532 |

Se guardan las contribuciones de cada campo y la influencia de cada adquisición.
Excluir un grupo solo mide sensibilidad; no sustituye la población prefijada.
O1 y JEPA requieren más evidencia: primero hace falta una medición de q fiable.
