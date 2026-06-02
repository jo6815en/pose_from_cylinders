# Pose from Cylinders

Det här repot tränar en bildparmodell som använder cylindrar i scenen som geometriska ankare. Modellen tar två bilder som input och predikterar både

- en `vision`-representation för varje bild: `[occupancy, radius, depth]` per horisontellt bin
- en relativ 2D-pose mellan bilderna: `[tx, ty, sin(yaw), cos(yaw)]`

Huvudflödet ligger i `train.ipynb`. Där skapas dataset, modell, losses, träningsloopar, valideringsplottar och visualiseringar av predikterade cylindrar.

## Repoöversikt

- `train.ipynb`: huvudsaklig notebook för träning, validering, inferens och plotting.
- `analyze_dataset.ipynb`: analys/debug av dataset och kameraposer.
- `model.py`: parmodell med patch-embedding, Transformer-backbone, vision-head och pose-head.
- `dataset.py`: datasetklass för syntetiska cylinderscener med labels i `labels.json`.
- `colmap_pair_dataset.py`: datasetklass för bildpar med relativa COLMAP-poser.
- `losses.py`: supervised loss, pose loss, vision loss, Sinkhorn-matchning, radius consistency och reprojection loss.
- `utils.py`: plotting och små hjälpfunktioner för pose/vision-output.
- `debugger.py`: hjälpfunktioner för att inspektera kameror, yaw och datasetposter.
- `ex_train_loop.py`: minimalt exempel på att läsa `ColmapPairDataset`.
- `working_requirements.txt`: beroenden som använts i arbetsmiljön.

## Data

Datamapparna är git-ignorerade och förväntas ligga lokalt i reporoten.

### Syntetisk cylinderdata

`SceneTwoPairsDataset` läser mappar med den här strukturen:

```text
dataset/
  scene_000/
    labels.json
    images/
      pair_000_cam1.png
      pair_000_cam2.png
      ...
```

Samma format används för exempelvis:

```text
dataset/
valdataset/
testdataset/
```

Varje `labels.json` ska innehålla en lista `pairs`, där varje pair har:

- `image1`, `image2`
- `vision1`, `vision2`
- `camera1`, `camera2`, antingen i pair-objektet eller på scen-nivå

`vision` tolkas som en array med shape `(num_bins, 4)` i labels:

```text
[occupancy, radius, depth, cylinder_id]
```

Modellen predikterar motsvarande `(num_bins, 3)`:

```text
[occupancy, radius, depth]
```

### Forest/COLMAP-data

`ColmapPairDataset` läser:

```text
forestdataset/
  images/
    ...
  relative_poses.json
```

Varje post i `relative_poses.json` ska innehålla:

- `from_image`
- `to_image`
- `T_ab`, en relativ 4x4-transform från kamera/frame A till B

## Installation

Skapa och aktivera en virtuell miljö:

```bash
python3 -m venv venv
source venv/bin/activate
```

Installera beroenden:

```bash
pip install -r working_requirements.txt
```

Starta sedan Jupyter och öppna `train.ipynb`:

```bash
jupyter notebook
```

Om `jupyter` inte finns i miljön behöver det installeras separat:

```bash
pip install notebook
```

## Träning

Det rekommenderade flödet är att köra `train.ipynb` uppifrån och ned:

1. Importera beroenden och skapa `device`.
2. Skapa `PairImageCylinderModel`.
3. Skapa `train_loader`, `val_loader` och `forest_loader`.
4. Kör träningscellen.
5. Kör plotcellerna för loss-kurvor och inferens.

Notebooken använder för närvarande bland annat:

```python
PairImageCylinderModel(
    img_size=128,
    patch_size=8,
    embed_dim=384,
    depth=6,
    num_heads=6,
    num_bins=128,
)
```

De viktigaste loss-delarna är:

- supervised vision loss mot syntetiska labels
- supervised relativ pose loss
- matchad radius consistency mellan vyer
- matchad 2D reprojection loss mellan vyer
- extra forest/COLMAP-konsistens efter en start-epoch

## Visualisering

`utils.py` innehåller hjälpfunktionen:

```python
plot_estimated_cylinders_on_images(images, visions, occ_threshold=0.6, flip_x=True)
```

Den ritar predikterade cylindrar som vertikala overlay-band ovanpå bilderna. `flip_x=True` används eftersom vision-binarnas riktning är speglad relativt bildens x-koordinat.

I `train.ipynb` finns celler för att:

- plotta GT vs predikterad `occupancy`, `radius` och `depth`
- plotta predikterade cylindrar på syntetiska testbilder
- plotta predikterade cylindrar på ett sample från `forest_loader`
- plotta relativ pose med `plot_relative_pose`

## Snabba kontroller

Kontrollera att Python-filerna kompilerar:

```bash
venv/bin/python -m py_compile model.py dataset.py colmap_pair_dataset.py losses.py utils.py debugger.py
```

Kontrollera att notebooken är giltig JSON:

```bash
python3 -c "import json; json.load(open('train.ipynb')); print('notebook json ok')"
```

## Anteckningar

- Datamapparna är inte versionshanterade enligt `.gitignore`.
- `train.ipynb` är det levande experimentflödet och kan innehålla körda celloutputs.
- Forest-datan saknar cylinder-GT; där används modellens prediktioner och geometriska konsistens mellan vyer.
