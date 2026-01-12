**🧠 Brain Tumor MRI Classifier**

Deep learning system for classifying brain tumors from MRI scans using EfficientNetB4, fine-tuning, and Grad-CAM explainability.
This project demonstrates an accurate Brain Tumor MRI Classifier:

**Transfer learning from ImageNet**
- Two-stage fine-tuning
- Cosine-decay learning rate
- Confusion-matrix evaluation
- Visual model interpretability with Grad-CAM

**🔬 Overview**

This model takes a brain MRI scan as input and predicts the tumor class using a convolutional neural network.

**The model was trained using:**
- EfficientNetB4 as the backbone
- Custom convolutional head
- Mixed-precision training (FP16)
- Two-stage training (frozen backbone → fine-tuned backbone)

**🧠 Model Architecture**
- Backbone: EfficientNetB4 (ImageNet pretrained)
- Input size: 380 × 380 × 3
**- Custom Head:**
  - 1×1 and 3×3 Conv layers
  - Batch Normalization
  - ReLU activations
  - Global Average Pooling
  - Dropout (0.3)
  - Softmax classifier

**🚀 Training Strategy**

**The model is trained in two stages:**

Stage 1 - Feature Extraction

- Backbone frozen
- Only classification head trained
- Adam optimizer (1e-4)
- ReduceLROnPlateau + EarlyStopping

Stage 2 - Fine Tuning

- Last 100 layers of EfficientNet unfrozen (except BatchNorm)
- Cosine-decay learning rate starting at 1e-5
- Adam optimizer
- EarlyStopping + ModelCheckpoint


**📊 Evaluation**

**The model is evaluated using:**
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion Matrix

**Each cell in the confusion matrix shows:**
- Number of predictions
- Percentage relative to the true class


**🔍 Explainability (Grad-CAM)**

Grad-CAM is used to visualize which parts of the MRI image most influenced the model’s decision.

**This helps verify:**
- The model is focusing on tumor regions
- Predictions are medically plausible

**Example outputs:**
- Heatmaps overlaid on MRI scans
- Highlighted tumor regions

**📁 Dataset Structure**

**Your dataset should be organized as:**

- Datasets/
  - Training/
    - Class_1/
      - img1.jpg
      - img2.jpg
    - Class_2/
      - ...
  - Testing/
    - Class_1/
    - Class_2














































