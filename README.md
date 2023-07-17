# Microsurgical Aneurysm Clipping Surgery (MACS) Dataset and Code Base

## Introduction

This is a reposiroty for the MACS dataset proposed in *Shifted-Windows Transformers for the Detection of Cerebral Aneurysms in Microsurgery* and the code base. The dataset can be downloaded from [here](https://rdr.ucl.ac.uk/articles/dataset/Microsurgical_Aneurysm_Clipping_Surgery_MACS_Dataset_with_image-level_aneurysm_presence_absence_annotations/23533731).

## MACS Dataset

The dataset has 16 folders `vid_*\` of frames extracted from 16 surgery videos respectively. Under each folder there are two folders `0\` including all Type-X (negative-aneurysm not present) frames and `1\` including all the Type-Y (positive-aneurysm present) frames.

As reported in the paper, we split the video frames to 4 folds to conduct cross validation experiment as:

| fold-0   | fold-1   | fold-2   | fold-3   |
| -------- | -------- | -------- | -------- |
| `vid_00` | `vid_01` | `vid_05` | `vid_13` |
| `vid_04` | `vid_07` | `vid_09` | `vid_15` |
| `vid_06` | `vid_02` | `vid_11` | `vid_17` |
| `vid_19` | `vid_03` | `vid_12` | `vid_08` |


There is also an `img_test\` folder which contains 15 test images use for the comparison between the AI model and humans.

## MACSSwin-T

The implementation of the MACSSwin-T model is based on the code base of [Swin Transformer](https://github.com/microsoft/Swin-Transformer). See [get_started.md](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) for a quick start. 

You can rearange the MACS Dataset to be in the same format as ImageNet:

  ```bash
  $ tree data
  macs
  ├── train
  │   ├── 0
  │   │   ├── frame1.jpeg
  │   │   ├── frame2.jpeg
  │   │   └── ...
  │   └── 1
  │       ├── frame3.jpeg
  │       └── ...
  │   
  └── val
      ├── 0
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      └── 1
          ├── img6.jpeg
          └── ...
 
  ```

You can use the following command to train the MACSSwin-T model:

```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swin/swin_small_patch4_window7_224.yaml --data-path <macs-path>
```

## vidMACSSwin-T

The implementation of the MACSSwin-T model is based on the code base of [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). See [install.md](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/install.md) for a quick start. 

You can use `cas_data\gen_vid_cas.py` to generate the dataset used to train the vidMACSSwin-T model as:

```
python --cas_path <macs-path> --dest_path <desitmation_path>
```

After generating the dataset, you need to modify the configuration `configs/recognition/swin/swin_tiny_patch244_window877_macs.py` to specify the dataset path and the annotation file.

And use the following command to train the vidMACSSwin-T model:
```
python tools/train.py configs/recognition/swin/swin_tiny_patch244_window877_macs.py
```