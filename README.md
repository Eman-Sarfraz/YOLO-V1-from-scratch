# YOLOv1 from Scratch in PyTorch

This project implements the **YOLOv1 (You Only Look Once)** object detection model from scratch using **PyTorch**, trained on the **Pascal VOC 2007** dataset. It covers the full pipeline: data preprocessing, target encoding, model architecture, training, loss computation, and visualization.

---

## 📌 Features

- ✅ Custom PyTorch `Dataset` for VOC annotations and images  
- ✅ Full YOLOv1 architecture implementation  
- ✅ Custom loss function matching YOLOv1 paper  
- ✅ Visualization of predictions vs ground truth  
- ✅ Loss curve plotting after training  

---

## 📂 Dataset

The model is trained and tested on the **PASCAL VOC 2007** dataset.

### Download Instructions:

1. Go to: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
2. Download the following files:
   - `VOCtrainval_06-Nov-2007.tar`
3. Extract to create a directory structure like this:

```
VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/
│   │   └── Main/
│   ├── JPEGImages/
│   └── ...
```

---

## 🧠 Model Overview

- **Grid Size (S):** 7 × 7
- **Bounding Boxes (B):** 2 per cell
- **Classes (C):** 20 Pascal VOC classes
- **Input Image Size:** 448 × 448
- **Loss:** Combines localization, confidence, and classification losses

---

## ⚙️ Setup

### Dependencies:

```bash
pip install torch torchvision matplotlib opencv-python
```

### Define Constants:

```python
img_size = 448
S = 7
B = 2
C = 20
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
λ_coord = 5
λ_noobj = 0.5
```

---

## 🚀 Training

To train the model:

```bash
python train_yolov1.py
```

Or run:

```python
model, dataset = train()
```

### During Training:

- Each epoch prints the average loss.
- After training, a **loss curve** is plotted to show training dynamics.

---

## 📊 Visualization

Once trained, visualize predictions and ground truth with:

```python
plot_results(model, dataset, num=5)
```

- **Green boxes**: Ground truth bounding boxes  
- **Red boxes**: Predicted bounding boxes with confidence > 0.5  

---

## 📎 Code Structure

| Component       | Description |
|----------------|-------------|
| `VOCDataset`   | Loads images and XML annotations, resizes images, and encodes targets into YOLO format |
| `YOLOv1`        | CNN based on YOLOv1 architecture from the paper |
| `yolo_loss`     | Custom loss including localization, confidence, and classification components |
| `train()`       | Training loop for model |
| `plot_results()`| Visualizes predictions alongside ground truth |

---

## 📖 References

- Original paper: [You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

---






