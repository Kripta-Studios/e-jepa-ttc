# Política de datos

Los datasets crudos no se almacenan en Git.

## IDs permitidos

```text
CARLA_DVS_LOOMING_1406
EVTTC32_LABELLED
EAP_HF_TRAIN40
BENCHMARK10_SEALED
```

No se aceptan nuevas raíces sin una revisión explícita del protocolo.

## Contenido versionable

- manifests pequeños;
- splits legibles;
- hashes;
- fixtures sintéticos;
- documentación.

Cada entrada de manifest debe incluir fuente, ruta local, tamaño, metadata de
split y hash cuando esté disponible.

## Rutas operativas

```text
datasets/evttc
datasets/evttc_official_benchmark_sealed
datasets/CARLA_DVS_Looming_Dataset/random_spawn
E:\eAP_dataset\data\train
E:\eAP_dataset\derived
```

La raíz sellada no se inspecciona durante desarrollo. eAP derived está limitado
a 55 GiB y debe mantener al menos 50 GiB libres en E:.

CARLA se lee mediante mmap desde los `events.npy`; no se permite crear una
segunda copia de 71,64 GiB ni un cache voxel completo. Sus splits viven en
`data/splits/carla_dvs_looming_blocked_v1.json` y nunca sustituyen la evaluación
real EvTTC.

## Prohibiciones

- datos crudos en Git;
- eAP test;
- extracción masiva de TAR RGB;
- voxel cache global;
- logits SAM densos;
- hidden states DINO de todas las capas;
- pseudo-TTC presentado como ground truth.
