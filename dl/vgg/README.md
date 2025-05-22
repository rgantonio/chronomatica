# VGG16 Study
- In this directory, we try to break down details about the VGG16 architecture
- This way, we can get a better understanding of how the architecture is built and how it works
- I tested different versions because I wanted to replicate a part of the work in [FSL-HDNN](https://arxiv.org/abs/2409.10918), where they use VGG16 for the CNN extraction
- The motivation is to move the MLP layer later to an HDC operation
- *Note:* VGG is an outdated model, but it is only used as a nice test case for characterization of some HW architectures

## File Descriptions
- `vgg16-pretrained-cifar10`: Is a simple pre-downloaded VGG16 model. However, the input size is rescaled to match that of ImageNet since the pretrained VGG16 supports ImageNet only. Current accuracy here is 11%.
- `vgg16-train-finetune-cifar10`: Upscales the accuracy up to 87% after fine-tuning or retraining.
- `vgg16-train-from-scratch-cifar10`: Tries to retrain the VGG16 from the beginning.


# VGG16 Model Layers
- The model can be drawn elegantly:

![image](https://github.com/user-attachments/assets/8369dd0c-6963-4e00-b78c-b5e854b56d7e)

- There are 5 major blocks
  - Each block contains 2 convolution layers followed immediately by the ReLU activation layers.
  - Each convolution layer increases the channels.
  - The max-pool in the end reduces the size of the image.
- The final layer is a stack of fully-connected (FC) layers
  - Here, it is quite large since the 1st and 2nd FC layers use 4,096 hidden features
  - The output originally is 1,000 for ImageNet, but for CIFAR, it scales to 10 or 100, depending on the data set
