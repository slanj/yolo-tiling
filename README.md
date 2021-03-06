# YOLO Dataset tiling script

## Tile (slice) YOLO Dataset for Small Objects Detection

This script can cut images and corresponding labels from YOLO dataset into tiles of specified size.

## Usage 

`python3 tyle_yolo.py -source ./yolosample/ts/ -target ./yolosliced/ts/ -ext .JPG -size 512`

## Arguments

- **-source**        Source folder with images and labels needed to be tiled. Default: ./yolosample/ts/
- **-target**        Target folder for a new sliced dataset. Default: ./yolosliced/ts/
- **-ext**           Image extension in a dataset. Default: .JPG
- **-falsefolder**   Folder for tiles without bounding boxes
- **-size**          Size of a tile. Default: 416
- **-ratio**         Train/test split ratio. Dafault: 0.8


