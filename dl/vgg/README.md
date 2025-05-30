# VGG16 Study
- In this directory, we try to break down details about the VGG16 architecture
- This way, we can get a better understanding of how the architecture is built and how it works
- I tested different versions because I wanted to replicate a part of the work in [FSL-HDNN](https://arxiv.org/abs/2409.10918), where they use VGG16 for the CNN extraction
- The motivation is to move the MLP layer later to an HDC operation
- *Note:* VGG is an outdated model, but it is only used as a nice test case for characterization of some HW architectures
- *Note:* When using VGG16 for CIFAR dataset, it will never have a better accuracy if you stick to the vanilla implementation. It needs to upscale the input to $(3,224,224)$ which was intended for ImageNet.
- There is a nice experiment on combining VGG16 with HDC done below. The FSL-HDNN technique though, if you read precisely, shows that the integer-based hypervectors result in 50% accuracy only. It is very poor. Going into binary reduces this even further. It needs to be augmented with binary neural network in the end.

## File Descriptions

### Pytorch VGG16 CIFAR 10
- [pytorch-vgg16-pretrained-cifar10](./pytorch-vgg16-pretrained-cifar10.ipynb): Is a simple pre-downloaded VGG16 model. However, the input size is rescaled to match that of ImageNet since the pretrained VGG16 supports ImageNet only. Current accuracy here is 11%.
- [pytorch-vgg16-train-finetune-cifar10](./pytorch-vgg16-train-finetune-cifar10.ipynb): Upscales the accuracy up to 87% after fine-tuning or retraining.
- [pytorch-vgg16-train-from-scratch-cifar10](./pytorch-vgg16-train-from-scratch-cifar10.ipynb): Tries to retrain the VGG16 from the beginning.

### Pytorch VGG16 CIFAR 100
- [pytorch-vgg16-quantized-cifar100](./pytorch-vgg16-quantized-cifar100.ipynb): Trains VGG16 from scratch and makes a quantized model out of it.
- [pytorch-vgg16-quantized-load-cifar100](./pytorch-vgg16-quantized-load-cifar100.ipynb): Loads the quantized VGG16 trained with CIFAR100.

### Pytorch VGG16 with HDC
- [pytorch-vgg1-hdc-trial](./pytorch-vgg-hdc-trial.ipynb): This is an example trial of the VGG16 and HDC trained and combined together. It uses the quantized VGG16 and then the final HDC layer.

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
