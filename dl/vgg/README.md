# VGG16 Study
- In this directory, we try to break down details about the VGG16 architecture
- This way we can get a better understanding of how the architecture is built and how it works
- I tested different versions because I wanted to replicate a part of the work in [FSL-HDNN](https://arxiv.org/abs/2409.10918) where they use VGG16 for the CNN extraction
- The motivation is to move the MLP layer later to an HDC operation

## File Descriptions
- `vgg16-pretrained-cifar10`: Is a simple pre-downloaded VGG16 model. But the input size are rescaled to match that of ImageNet since the pretrained VGG16 supports ImageNet only. Current accuracy here is 11%.
- `vgg16-train-finetune-cifar10`: Upscales the accuracy up to 87% after fine-tuning or retraining.
- `vgg16-train-from-scratch-cifar10`: Tries to retrain the VGG16 from the beginning.


# VGG16 Model Layers
- The model can be drawn elegantly:

TODO: Add image here

- There are 5 major blocks
  - Each block contains 2 convolution layers followed immediatley with the ReLU activation layers.
  - Each convoltion layer increases the channels.
  - The max-pool in the end reduces the size of the image.
- The final layer is a stack of fully-connected (FC) layers
  - Here, it is quite large since the 1st and 2nd FC layers use 4,096 hidden features
  - The output originally is 1,000 for ImageNet but for CIFAR it scales to 10 or 100 depending on the data set