# Recuperación de la verificación final

Los 24 fits están completos. No se debe relanzar entrenamiento.
El ZIP final fue escrito por completo; la extracción se pausó por memoria
comprometida de Windows. `recover_published.py` termina la extracción del ZIP
existente con un padre sin Torch, valida SHA-256 y CRC de todos los miembros,
y admite después los verificadores canónicos en procesos separados.

Primera recuperación de una publicación que aún no creó su cola de verificadores:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_io_recovery.recover_published --archive 'artifacts/simplex_t/nocturnal_20261003/delivery/E_JEPA_TTC_SIMPLEX_T_NOCTURNO_60000_bbabc02921d2.zip'
```

Una vez que la cola esté admitida, una interrupción se recupera mediante:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.simplex_t_io_recovery.finalize_continuation --resume-verification
```

Ambos conservan el ZIP publicado, exigen un propietario único y ejecutan cero
updates. El helper de extracción se incorporó después de publicar el ZIP y está
en Git, separado de las fuentes científicas congeladas incluidas en el bundle.
La comprobación exige exactamente 24 checkpoints y 18 cabezas C0. El archivo
H16 interno conserva su comprobación independiente original de seis cabezas.
