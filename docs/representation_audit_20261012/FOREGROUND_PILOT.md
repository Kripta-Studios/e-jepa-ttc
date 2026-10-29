# Piloto P: sensibilidad del foreground nativo

Se ejecutó el forward nativo A5 en CPU sobre 40 consultas P: ocho sin reemplazo en cada banda, con seed20261012. Se derivó sólo respecto a los parámetros de `encoder.foreground`; se verificó que todos los parámetros y buffers permanecieran idénticos. No hubo updates de optimizador.

| Banda | n | Norma altura, media | Norma NLL, media | Coseno altura/NLL, media | Coseno <0 |
|---|---:|---:|---:|---:|---:|
| 0-1 | 8 | 0.113572 | 30.222172 | -0.022716 | 4/8 |
| 1-3 | 8 | 0.149475 | 5.275473 | -0.047667 | 3/8 |
| 3-6 | 8 | 0.055092 | 7.109884 | 0.056467 | 1/8 |
| 6plus | 8 | 0.027115 | 8.301591 | 0.069345 | 3/8 |
| negative | 8 | 0.057225 | 6.407894 | 0.209528 | 2/8 |

Gradientes finitos y no nulos: height_gradient_norm: 40/40; temporal_gradient_norm: 40/40; nll_gradient_norm: 40/40.

Esto no respalda que la rama esté completamente muerta en estos controles. Tampoco demuestra que una pérdida domine el entrenamiento nativo: la pérdida de altura del piloto es MSE sobre log-altura, la temporal es MSE de su cambio y la NLL es sólo el componente de ratio de la pareja actual. Sus escalas y reducciones difieren; se excluyen top-k, riesgo y otros términos nativos.

Las derivadas recorren el forward nativo completo, con el resto de pesos congelados, incluidas sus dependencias de geometría, tokens e incertidumbre. No son las derivadas de una futura rama que conservara esos tokens por separado.

Una probabilidad de foreground próxima a0 o1 puede corresponder a una máscara bien delimitada; su fracción de saturación no demuestra por sí sola un fallo. El CSV conserva esa fracción y la dispersión espacial, sin reinterpretarlas como una métrica de calidad de segmentación.

Es un piloto descriptivo P con pocos ejemplos por banda. No estima rendimiento poblacional, no añade evidencia H/V ni selecciona un checkpoint. Todos los tokens, targets, pérdidas, normas y cosenos se conservan en `P40.csv`.
