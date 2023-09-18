# Universal-Photo-Sketching

Link to this paper: (tbc)

Chun-Chun Hui, Wan-Chi Siu, Wen-Xin Zhang and H. Anthony Chan, "Universak Quality Photo Sketch with Residual Heatmap Network", submitting to ACM Transactions on Graphics.

This repo provides codes for training and evaluating.

We propose a universal photo to sketch deep learning network with using residual heatmap network 

# BibTex

BibTex will update after published

```
@INPROCEEDINGS{9978022,
  author={Hui, Chun-Chuen and Siu, Wan-Chi and Zhang, Wen-Xin and Chan, H. Anthony},  
  booktitle={TENCON 2022 - 2022 IEEE Region 10 Conference (TENCON)},   
  title={Quality Photo Sketch with Improved Deep Learning Structure},   
  year={2022}, 
  volume={},  
  number={},  
  pages={1-6},  
  doi={10.1109/TENCON55691.2022.9978022}}
```

# Implementation

## Training

Download MS COCO 2017 dataset by https://cocodataset.org/#download
You will need ***2017 Train images [118K/18GB]***

Download ImageNet Sketch from https://github.com/HaohanWang/ImageNet-Sketch

* Unzip COCO Train images into data/coco/train2017/

* Unzip COCO Segmentation images into data/coco/seg/

* Unzip ImageNet Sketch into data/sketch/

Run to train

```py
python main.py
```

## Evaluation

Copy your test images to folder Test/test_img/ and run

* Note that under exposure images or blurred images will affect final quality
  
```py
python eval_self.py
```
Results are generated at Test/sketch_result/

