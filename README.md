# Synthetic vs Natural Materials for Façades (CAP6415 F25)

**Course:** CAP6415 – Computer Vision (Fall 2025, FAU)  
**Author:** Carlos David González Vargas 

## Abstract

This project investigates whether **synthetic, procedurally generated façade material images** can reduce the amount of **real-world (natural) data** needed to train a material classifier.

We focus on **five façade material classes**:

- Brick  
- Glass  
- Concrete  
- Metal panel  
- Vegetation  

We train and compare three convolutional neural network models (based on **ResNet-18**) under different data regimes:

1. **NAT** – trained only on **natural (real)** façade images.  
2. **SYN** – trained only on **synthetic** images generated procedurally.  
3. **MIX** – **pretrained** on synthetic images and **fine-tuned** on a small subset of natural images.

All models are evaluated on a **held-out NAT test set**. We report **accuracy, macro and per-class F1**, confusion matrices, and qualitative **Grad-CAM** visualizations.  
Finally, we deploy the best model in a **real-time pipeline** that sends predictions to **TouchDesigner**, where they drive an **interactive visualization** 3D point-cloud façades), connecting computer vision with architectural/computational design workflows and further 3d modelling from 2D images.

## Results

--------------------------------------------------
Final experiments and results
--------------------------------------------------
- NAT model:
  - Best overall performance on NAT test set.
  - Around ~0.8 test accuracy and macro F1 in the low 0.7x range.
  - Confusion mainly between glass and concrete.
- SYN model:
  - Lower accuracy and macro F1 compared to NAT.
  - Still demonstrates that synthetic images encode meaningful texture/structure signals.
- MIX model:
  - Improves over SYN-only training and narrows the gap with NAT.
  - Particularly helpful for classes where synthetic patterns transfer reasonably to real images.
- Inference:
  - Real-time script runs single-frame inference fast enough to support interactive behavior in TouchDesigner.

--------------------------------------------------
Issues, limitations, and debugging
--------------------------------------------------
- Domain gap:
  - Despite improvements, a visible gap remains between SYN and NAT performance.
  - Current synthetic textures do not capture all complexities of real façades (mixed materials, occlusions, weathering, clutter).
- Mixed materials and ambiguous labels:
  - Images with multiple materials (for exmaple glass, that normally is in context with other materials or is above another texture) are inherently ambiguous.
  - This leads to persistent confusion between certain classes even with NAT data.
- Real-time + TD setup:
  - Time spent debugging:
    - Correct image transforms to match training preprocessing.
    - Synchronization between Python and TouchDesigner OSC/WebSocket messages.
    - Mapping of material labels to visual parameters and smoothing changes to avoid flicker.
  - Needed to carefully manage performance:
    - Ensured that the PyTorch model runs with no_grad and eval mode.
    - Simplified some TD networks so the 3D point-cloud scenes run smoothly.

--------------------------------------------------
Final status and future iterations…
--------------------------------------------------
- The project is functionally complete:
  - All three experiment types (NAT, SYN, MIX) trained and evaluated.
  - Metrics, confusion matrices, and Grad-CAM examples saved under results/.
  - Real-time classifier integrated with a TouchDesigner façade demo in 3D/point-cloud form.
  - Repo structure and README prepared for grading and reproducibility.

- Future iterations and research…
  - Extend to semantic segmentation,
  - Improve synthetic data with more realistic 3D/lighting,
  - Improve natural data with bigger datasets
  - Improve the readability of the point-cloud–based model and the representation of the information in real time.
  - Continue with the analysis part of the 3D models.


## Acknowledgements

The project uses PyTorch and Torchvision as the main deep learning framework.

Additional code and implementation ideas are inspired by official PyTorch examples, course materials, and public computer vision repositories.

All external code and papers used or adapted in this work are properly attributed in the dedicated attribution section in this repository, in accordance with the course requirements.

