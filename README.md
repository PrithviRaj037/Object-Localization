# Object-Localization
Object Localization Pipeline is an end-to-end solution for detecting and drawing a single bounding box around an object in an image. Built in PyTorch and demonstrated interactively in Google Colab, it leverages timm’s EfficientNet-B0 backbone for feature extraction, a lightweight regression head for box prediction, and OpenCV for on-the-fly visualization.

You’ll find scripts for data loading & preprocessing (custom Dataset → JSON annotations → train/val split), model definition (EfficientBBoxNet), training with MSE loss and checkpointing, plus an inference module that reads new images and writes out JPEGs with predicted boxes. A companion Colab notebook ties it all together—run cells to see data samples, watch live training curves, and inspect output images without any local setup.

Key tools & libraries:

PyTorch (modeling & training)

timm (EfficientNet-B0 backbone)

OpenCV (drawing boxes & saving outputs)

nbformat / Colab (interactive demo)

argparse / JSON (configurable scripts & annotation handling)
