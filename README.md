# Pavement Crack Inspection using CNNs & Vision Transformers (ViTs)

This project implements automated pavement crack detection using **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** inside a single executable script: `main.py`.  
The project demonstrates preprocessing, patch encoding, model construction, training pipeline, and evaluation—all within one file.

---

## 🚧 Overview

Road crack detection is essential for infrastructure maintenance. Deep learning provides accurate, scalable, and automated inspection capabilities.

This project compares:

- **CNN-based models** (for local texture extraction)
- **Vision Transformers (ViTs)** (for global crack pattern understanding)

---

## 🧠 CNN Approach

CNNs extract local spatial features such as edges and crack textures.

**Strengths**
- Excellent for small datasets  
- Strong at extracting local features  
- Works well with segmentation & classification tasks  

**Limitations**
- Limited global context  
- Difficult to capture long crack patterns  

---

## ⚡ Vision Transformer (ViT) Approach

ViTs use self-attention to understand global crack structures.

**Strengths**
- Captures long-range dependencies  
- Better at complex or irregular crack patterns  

**Limitations**
- Requires large datasets  
- Can overfit on small datasets  
- Needs heavy augmentation  

---

## Model Architecture(ViT)
- Split image into fixed-size patched
- Encode using the custom PatchEncoder
- Pass through Transformer blocks
- Flatten output
- Predict crack class with a classifier head

## key features
- CNN based crack detection
- Vision transformer classification
- Custom patch Encoder implementation
- Data preprocessing and augmentation
- Unified training pipeline in a single script
- Visulization and evaluation report

## Contributing
Feel free to open issues or pull requests.
