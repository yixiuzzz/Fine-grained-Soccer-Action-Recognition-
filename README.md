# ⚽ Efficient Fine-grained Soccer Action Recognition 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-Tracked-yellow)](https://wandb.ai/)

## Overview
This is a lightweight and high-efficiency spatiotemporal video analysis system built upon the **PyTorch** framework. 

The core architecture utilizes **[X3D](https://arxiv.org/abs/2004.04730)**, an advanced 3D convolutional neural network optimized to capture critical **motion features** from consecutive video frames. By processing temporal sequences, the system accurately classifies fine-grained, complex player behaviors within highly dynamic soccer broadcast scenarios.



## Technical Highlights
Fine-grained soccer action recognition suffers from **low inter-class variance**. As shown below, **Kicking** and **Moving** share extremely high visual similarity in static frames, making them difficult to distinguish.

<p align="center">
  <img width="711" height="251" alt="截圖 2026-06-11 晚上11 00 04" src="https://github.com/user-attachments/assets/3200de6c-2b76-465d-adde-ac0eaa8bc41e" width="45%" />
  <br>
</p>

To address this, our framework extracts **robust motion features** to significantly enhance the backbone's ability to capture spatiotemporal cues. This approach allows the model to capture subtle motion nuances for accurate fine-grained classification, all while maintaining a **lightweight and highly efficient** footprint.
<p align="center">
  <img width="987" height="470" alt="截圖 2026-06-12 下午5 58 54" src="https://github.com/user-attachments/assets/3d26c49b-f6de-48a5-bab9-3903ecc754c4" width="40%" />
  <br>
</p>



## Demo
The system classifies player behaviors into four distinct categories, utilizing unique bounding box colors to display the actions in real time:
* **Green:**</font> Moving
* **Yellow:**</font> Standing
* **Red :**</font> Kicking
* **Blue :**</font> Falling

<p align="center">
  <img width="800" height="450" alt="9-23-ezgif com-cut" src="https://github.com/user-attachments/assets/60ab13e2-af42-4951-a4b8-c6ca441910fa" />
</p>



## Performance Comparison
Dataset: [SoccerNet-v2](https://silviogiancola.github.io/SoccerNetv2/)
  
### 1. Overall Accuracy & computational efficiency
  
| Architecture | Overall Accuracy (%) | Params (M) | FLOPs (G) |
| :--- | :---: | :---: | :---: | 
| **Baseline** (X3D) | 88.03% | 3.0M | 20.82G | 
| **Ours (with Motion Features)** | **91.80%** | **3.7M** | **21.78G** | 
| *Improvement* | *+3.77%* | *+0.70M* | *+0.96G* |

### 2. Per-class Precision
  
| Architecture | Idle | Kick | Fall | Move |
| :--- | :---: | :---: | :---: | :---: | 
| **Baseline** (X3D) | 88.75% | 75.43% | 87.30% | 90.40% |
| **Ours (with Motion Features)** | **92.44%** | **89.28%** | **98.24%** | **90.44%** |
| *Improvement* | *+3.69%* | *+13.85%* | *+10.94%* | *+0.04%* | 

