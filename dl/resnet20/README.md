# ResNet20 Study
- In this directory, we try to breakdown details about the ResNet 20 architecture
- This way we can get a better understanding on how the architecture is built and how it works
- The `resnet20.ipynb` file is used to investigate some details. You can check this out if you want.
- ResNet20 was originally used to process CIFAR-10. Means there are 10 classes in the end.

# Model Layers
- There are 4 major sections
  - Initial convolution layer
  - Blocks 1, 2, and 3
  - Final layer with average pooling and fully-connected (FC) unit

## Initial Convolution Layer
- The inputs have a shape of $[C,H,W] = (3,32,32)$
- The initial convolution has parameters:
  - kernel size: $3 \times 3$
  - output channel: 16
  - stride: 1
  - pad: 1
- There is a bnorm and ReLU after
- Output shape of the initial layer is $(16,32,32)$ 

TODO: Insert image here

## Block 1 Layers
- Block 1 is repeated $3 \times$
- 1st convolution has:
  - kernel size: $3 \times 3$
  - output channel: 16
  - stride: 1
  - pad: 1
- Goes to a bnorm and ReLU
- 2nd convolution has the same parameters as the first
- Goes to a bnorm
- Does a skip connection from the output of the initial layer then ReLU at the end
- Final output shape is $(16,32,32)$ 

TODO: Insert image here

## Block 2 layers
- Block 2 has similar looking layers except the 1st convolution block increases the output channels and reduces the image size
- 1st convolution has:
  - kernel size: $3 \times 3$
  - output channel: 32
  - stride: 2
  - pad: 1
  - take note of the stide of 2, hence the decrease in image size
  - The output then becomes shape $(32,16,16)$  
- Goes through a bnorm + relu
- Goes to a 2nd convolution but with stride of 1 instead of 2
- Goes to a batchnorm first before adding residual
- Skip connection but the skip connection has to downsample the image
  - This means a point-wise convolution is needed to make the downsampling
  - The parameters are:
    - kernel size: $1 \times 1$ it's a point-wise
    - output channel: 32
    - stride: 2
    - pad: 0
    - Note that stride of 2 and padding 0 allows to reduce the size but keeping it the same for addition
- The downsampled output and the output of the 2nd convolution are adding as skip connections
- The ReLU at the end is applied
- The blocks are repeated again for 2x more, however, the first convolution now just sticks to a stride of 1 and the same output channel of 32
- Final output shape is $(32,16,16)$ 

TODO: Insert image here

## Block 3 layers
- Block 3 is similar to block 2
- The onyl difference is that the 1st conversion to a higher channel output is not to 64, and the shape to be $8 \times 8$
- Such that as the block progresses the shape is now $(64,8,8)$

TODO: Insert image here

## Final Layer
- The final layer uses an average pool of $8 \times 8$ to bring the output shape to $(64,1,1)$
- After the average pool it now goes to an FC layer converting 64 features to 10 features which is the number of classes of CIFAR-10

TODO: Insert image here