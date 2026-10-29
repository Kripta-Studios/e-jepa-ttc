# Cola H: recorte de gradiente del scorer

Evidencia: en los6000updates de cada uno de los seis readouts U0/U1 de B, la norma anterior al clipping supera1. Esta observación no prueba que el clipping cause los errores. Los gradientes y pérdidas de productor no son los de las cabezas H8.

Contraste único: N0 frente a U0, C20 equal-slope con pendiente0,05, mismos parámetros, última capa reiniciada, misma semilla, P uniforme, batch128, objetivo nativo U0, FP32, AdamW y scheduler6000 de B. N0 mide la norma y exige gradientes finitos, pero no aplica recorte. No cambia ninguna ponderación relativa ni el riesgo esperado. No es otro multiplicador de la pérdida ni un sweep de umbrales. Si aparece una pérdida o norma no finita, abortar y conservar el fallo; no reducir LR ni añadir otro umbral.

Reutilizar U0-S7/S13/S23 congelados de B como controles exactos; no reentrenarlos bajo otro nombre. Registrar y comprobar la igualdad de inicialización, objetivo, sampler, scheduler y optimizador antes de aplicar clipping. La única diferencia durante updates es el recorte.

Primero N0 seed7 de6000updates; congelar candidato y control antes de H. Aplicar todos los gates originales frente a U0-S7 y A5 histórico. Si ambos pasan, entrenar N0 seeds13/23,6000cada uno, y congelar los seis endpoints contando controles reutilizados antes del cribado H de réplicas. Si cualquier contraste falla, cerrar la cola sin otro umbral o semilla.

Sólo si se admite la familia, registrar integración de features coherentes y normalizadores H, entrenar las seis cabezas H8 M0/L0 de2500updates y congelarlas antes de V. Los gates de calidad, cobertura y coste final no se relajan. El ahorro por controles reutilizados no habilita variantes adicionales de H.

Máximo nuevo:18000updates científicos de scorer,15000condicionales de cabezas,40técnicos. Consumo dentro del presupuesto V17 autorizado de1M físicos/18h GPU, sin reutilizar saldo V15. Encoder, foreground, geometría, incertidumbre, transporte y PAIR quedan congelados. Ningún cambio en V13/R1/V16, ningún dato FCWD/Dev32 para ajustar o seleccionar.
