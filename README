# Convolutional Neural Network

**This project utilizes a Convolutional Neural Network to classify images of cats and dogs with 84% accuracy.**

The goal was to maximize accuracy of the classification task while using a relitively small number of parameters in the model. 

---

## Table of Contents
1. [Installation & Usage](#installation-and-usage)  
2. [Network Architecture](#network-architecture)
3. [Design Choices](#design-choices)
4. [Convolutonal Neural Network Structure](#structure)
5. [Contact Info](#contact-info)  
6. [Versions Quick Reference](#versions-quick-reference)


---
## Installation and Usage
1. The `CatDogClassifier.py` script creates, trains, and tests a Convolutional Neural Network. 
   
   *Running the script will:*
   
   **a.** Automatically check for libraries required to run this project. Ask if you want to install them. Entering `Y` in the terminal will install them.
   
   **b.** Randomly split the training and testing data.

   **c.** Train a model with the pre-defined hyperparameters.

   **e.** Tests the model on a testing dataset, not used in training. Displays the accuracy.

   **f.** Saves the model weights in `/"Convolutional Neural Network"/` directory.
   
3. Utilizing `tensorflow-gpu` for this project is optional, but because the training dataset is relatively small and the architecture is simple, it will train quickly on the cpu. It may be useful when changing hyperparameters such as increasing `INPUT_SIZE`, `NUM_FILTERS_#`, `EPOCHS`, or decreasing `POOL_SIZE_#`. 

4. It is recommended that you create a Python virtual environment so that any pre-existing versions of these libraries are not updated.
   
---

## Network Architecture

Utilizing "sliding/striding" kernels in Convolutional Neural Networks allows a significant reduction in the number of parameters (weights and biases) required to get comparable accuracy than from a Deep Neural Network. The key metrics are the same, so designing the hyperparameters (user defined) to allow for **quick convergence** while **minimizing the loss** on training data is an effective guide. Also, of course, **high prediction accuracy** on the testing set.
#### GOAL: Accuratly classifly images of cats and dogs.

## Design Choices

##### Guided by balancing the training speed with the final accuracy of the trained network. *Minimizing overfitting is essential in attaining 84% accuracy on new data.*

- **Input Shape** : Images are downscaled to **32x32x3** by the `input_shape` parameter. The depth of 3 reflects RGB color.
- **Kernel Size** : **3 x 3 x 3** for input data, then **3 x 3 x n** for n number of output filters.
  
- **Filter Activation Function** : I chose Rectified Linear Unit (ReLU). ReLU is superior for feature extraction because it only preserves positive positive pre-activation values. Other activation functions such as tanh or sigmoid wouldnt make sense for this process. With ReLU, activation maps always have values >= 0, and "edge detection" is able to emerge as a consequence of the weights being both negative and positive.

- **Convolving an K (depth) channel Image**: The Initial 3 RGB channels only matter as input to the first convolution layer, further convolutions only work with stacks of K depth activation maps. A convolution with a 3x3x3 kernel on the RGB input, followed by an element-wise summation between all 3 channels create 1 activation map per filter. The number of desired activation maps is determined by how many filters are defined in the `NUM_FILTERS_ONE` hyperparameter. Each filter's weights are uniquely initalized.

- **Convolutional Layer 1** : 64 filters `NUM_FILTERS_ONE` varied by their initalized weights preform convolutions over the entire previous layer, which produce 64 activation maps of dimensions 32x32. Dimensions are **32 x 32 x 64.**

  The kernel has the same depth as the number of channels in the layer it is convolving over. At every position, the products of the positional values and the filter weight values are computed, then summed along with the deeper channels. This sum is added to the bias term, then passed through BatchNorm then the ReLU function to determine the activation in the feature map.
  
   Padding is `same`, so input dimensions (32x32) = output dimensions, with no padding pixels.
  
- **Pooling Layer 1** : Max pooling layer with `POOL_SIZE_ONE` = 2 and `stride` = 2. This reduces the total number of pixels by 75%. The dimensions become **16 x 16 x 64**.

- **Convolutional Layer 2** : This convolutional layer outputs 16 activation maps `NUM_FILTERS_TWO`. The dimensions become **16 x 16 x 16**.

- **Pooling Layer 2** : Max pooling layer with `POOL_SIZE_TWO` = 4 and `stride` = 4. The dimensions become **4 x 4 x 16**.

- **One layer Dense Network** : The shape of the flattened vector after the final pooling layer is 256. The size of the input layer of the Dense Neural Network is 128. This means that there are 128*256+128 parameters in this layer and 128+1 including the output channel weights and output bias. Verified by `model.summary()`. The CNN doesnt do the classification task, its moreso just for feature extraction and creating this vector.

  *Note: Data-flow and Gradient-flow is slightly different than just for a regular dense network because of the filters component of CNNs.*

- **Drop Out** : I used a 30% drop out as overfitting was observed. Higher dropout rate resulted in lower testing accuracy. Lower dropout resulted greater difference between training accuracy and testing accuracy.

- **Loss Function :**
  I used the **Binary Crossentropy function** because I wanted to pivot to categorical-crossentropy for multi-class predictions, which uses softmax rather than sigmoid for classification probabilities. Generative pretrained transformers utilize softmax activation in their output layer, and categorical cross-entropy uses that output vector.
  
- **Output Neuron Activation Function :** **The Sigmoid Function** is used with activation threshold 0.5.
   
- **Optimizer** (Weight Update Function) : **Adaptive Movement Estimation (ADAM)** 

### **Convolutional neural network reduces parameter count**

In my implementation the convolutional layers utilize 11,334 parameters and batch normalization layers   use 320 parameters. The classification 1-layer DNN head utilizes 33,025 parameters. The total parameter count for this architecture is 44,379. If I had trained a fully connected dense network to preform this classification task, assuming 1 hidden layer + 1 output node, this would be 32 * 32 * 3 * 128 + 128 add 128+1 (393,479) parameters.

#### *Using a CNN architecture reduces parameters from 393,479 to 44,709!!!*

### Creation of training and testing data
The original dataset of 25,000 images is split into **80% Training Set** and **20% Testing Set**
 
The Training and Testing set is constructed to have a 50/50 split of dogs and cats. 

This ensures balanced training and allows testing to be reflective of network effectiveness and not sample variance.

**Training Set (80%)** : Used in training the network.

**Testing Set (20%)** : Used in the final evaluation of the model. The model has never seen this data prior.

### Filter Intialization

##### I go into depth about the weight initalization equation because it is critical to understanding how "feature detection" takes place. Additionally, this calculation varies between Deep Neural Networks and Convolutional Neural networks as a result of the way the neuron's input and output channels are organized.

**Glorot Uniform Initalization** of filter weights. The formula is mostly the same between Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs) except for the way input and output channels operate.

### a = ± sqrt( 6 / (in_channels + out_channels) )

#### DNN initalization
For DNNs this is straight foward. Pick a layer, the number of nodes are the input channel, the number of nodes in the next layer (in the direction of the output node) are the output channel. The weights in between them must start off at some value. So this equation gives a stable range to initalize them. 

#### CNN initalization
CNN channels must consider the spatial dimensions of the kernel and the number of filters. 

The result of multiplying the height and width of the kernel by the number of filters used is that the sum of in_channel and out_channel is larger for CNNs than DNNs. This sum is used in the denominator, so this larger value yeilds smaller initalization values for CNNs.

*Using the first set of filters between the Input layer and activation maps as an example:*
   
   The input channels are computed by multiplying the kernel 3x3 kernel by the 3 channels for depth to reflect RGB values of the input layer. The output channels are computed by the kernel dimensions by the the number of filters I want to convolve over the input. In my implementation this is 64 channels. Meaning a total 3 * 3 * 3 + 3 * 3 * 64, which is 603. In a fully connected network with 128 input nodes and 128 nodes in the first layer, this is only 256. This difference produces smaller initial weights for CNNs.
  
   *There are other methods for weight initialization exist such as:*
   *HeNormal/HeUniform (forReLu activation) ; RandomNormal/Uniform (custom params)*
   
## Structure 
#### The architecture is composed of repetitive structures and processes organized as follows:

   1) Input Layer
   2) Convolutional Layer
   3) Batch Normalization
   4) Activation maps
   5) Pooling layer
   6) Repeat Structures 2 - 5
   7) Deep Neural Network Classification Head
   8) Output Node
---


1. **Input Layer**
   
   The shape of the input dimensions are defined by the resolution of the image and other information encoded such as RGB pixel values. 
   
   Using piexif, the images are downscale to match the input dimensions. Because the images use 3-channels (RGB), it will reflect in our input channels and initial kernel shape.
   
   After preprocessing, the dataset only contains 32x32x3 images. Metadata and corrupted images are removed with pillow.
   
2. **Convolutional Layers**
   
   Convolutional layers refer to the combination of convolved input, Batch Normalization, and activation layers. Sometimes pooling layers.

   This portion focuses on the convolved inputs. The convolved input layer requires the user to decide how many filters are desired as output. This determines the depth of the kernel. The dot product of the input and weight vector is computed. The sum of the resulting vector is added to the bias term, which produces a single value. This is done at every single position of the input image. The resulting pre-activation map is then processed through the Batch Normalization Layer.

3. **Batch Normalization Layers**
   
   Batch normalization accelerates model convergence through normalizing layer inputs to zero and unit varince during training.

   *It does this in a few ways*:

   - Standardizes the input data to zero mean and unit variance. "Unit variance" is the preffered term because the scale of the variance is a tunable parameter. This
  
   - Uses the variance of the input mini-batch in the Normalization formula. This essentially adds "random" stochastic noise, which helps reduce extreme pre-activation values. As more batches are normalized, the running average of the mean and standard deviation is used to further regularize the data. This further reduces the "peaks" and "valleys" which result in smoothening loss landscape.
  
   - Utilizes learnable parameters such as the Scale γ  and Shift β. These paremeters are adaptivley tuned with the goal of minimizing the loss function.

4. **Activation Maps**
   After the pre-activation data has been normalzied, the entire output is processed through the ReLU function. Which updates negative values to 0 and preserves the scale of positive values. The resulting data is passed to the pooling layer.

5. **Define Pooling Layers**

    Pooling is done to reduce the spatial dimensions of feature maps and extract significant features.

    This is done by selecting the dimensions of the pooling filter, stride, and operation.

   Pooling kernel Size
   The dimensions of the pooling kernel determine the subset of elements the pooling operation takes place on.

   Decided on 2x2 pooling with `stride = 2` on the first set of activation maps and 4x4 pooling with `stride = 4` on the second set of activation maps. Hyperparameter selection was guided by the intention to :

   - Avoid aggressivly downsampling feature data and allow the flow of gradients during back propagation.
   - Sufficiently downsample so that sufficient feature extraction takes place 
   - Prevent overfitting on testing data.
   
   The max pooling operation generalizes the maximum activation value from the filter onto every position traversed by the kernel.
   
   I chose Max Pooling because I wanted to extract stronger features from the activation map.

   *Other Operations such as Minimum Pooling and Average Pooling do exist, but those are for extracting non-prominent features in a noisy dataset and preserving contextual information for different image tasks, respectivly.*

7) **Dense Neural Network Classification Head**

   The values from the previous pooling layer are "Flattened" to a 1-d vector.

   This flattened vector is used as input for the fully conencted neural network. Conisting of 128 nodes in the first layer and a single output layer, this portion has 33,025 trainable parameters. 

   The `dropout` hyperparameter controls the proportion of neurons in this network that remain inactive. I set this to .3 (30%) as overfitting was observed with lower values and reduced accuracy was observed with higher values.
    

8. **Define Output Layer**

   The desired output of the classification is **Dog** or **Cat**. Either can be the negative or positive class, as long as training examples are enumerated properly.

    The dot product of the input vector and weights vector is computed, then summed with the bias term. This preactivation value is then passed through the **Sigmoid Function** to determine the probability of the possible classification. During training, the activation of this output node will yeild a low loss, if the probability is high because it is an accurate prediction. 

    Binary cossentropy (BCE) Loss requires a single predicted probability from 0 to 1, so passing the preactivation value through the sigmoid function ensures the output will be within this range. This sigmoid output is utilized in BCE. Assuming one pass, the BCE loss is computed as **− ( y ⋅ log(p) +(1−y) ⋅ log(1−p) )**. Where the true label **y** is **0 or 1** (predefined when enumerating the training set). If the true label is 1 and the predicted probability **(p sigmoid output)** is high, then the right side cancels out and you are left with - ( 1 - log(.99) ) -> ( -ln(.99) is .01 ), meaning low loss. But, smaller predicted values for the true class are exponentially more penalized with a higher loss value, since negative ln(0) approaches infinity.
   

10. **Training**

    The overarching goal of this process is to minimize the binary cross-entropy loss of the network and update the weights such that new data will fall into the accurate classification threshold.

    **Gradient Flow from output neuron (DNN)**

    The next step is to compute the gradient vector, which is the contribution of each individual weight and bias to the loss of the overall network. 
    ##### This is done by utilizing the chain rule of calculus to solve for the change in cost relative to the preactivation functions of deeper layers.

    *Solve for:*
    1. The derivative of the loss (L) with respect to the activation neuron (a^L)
    2. The derivative of the activation neuron (a^L) with respect to the Activation_Function(pre-activation (z^L))
    3. The derivative of the pre-activation (z^L) with respect to the weight (w^L)
    4. The derivative of the pre-activation (z^L) with respect to the bias (b^L)
      
      *By this step you utilize the chain rule of calculus to compute the gradient of the loss with respect to weight at layer L (w^L) and gradient of the loss with respect to the bias at layer L (b^L) of deeper layers.*
    
    **Gradient Flow Through Convolutional Layers (CNN)**

    Constructing the gradient vector is a similar computation for convolutional layers.
    The key difference is the addition of the "de-convolution" step.

    ### Deconvoluton Operation
   
    The kernel weights are spatially flipped 180 degrees to undo the foward-pass convolutions. This flipped kernel then strides along the activation map and the summed gradient is passed to the weight optimizer "update" function. Rather than striding across the layer backwards or with a flipped kernel, alegraically manipulating the relevant formulas achieves the same result.

    ### Propagation through Pooling layers

    The gradents are computed as a function of the underlying positional values' contribution to the loss. This means values that are not represented in min-pooling or max-pooling operations do not influence the kernel's weight updates. For average pooling, this ascribes equal contribution to the loss between the subset of pixels in that position.
   
    **Repeating this process through previous layers results in the construction of the *gradient vector*, which is used in updating the weights with the goal of minimizing the cost of the network.**

11. **Update The Weights and Biases**

    **Adaptive Movement Estimation (ADAM)** is used for Back-Propagation as it is sufficiently optimized. 

    **[Adaptive Movement Estimation](https://keras.io/api/optimizers/adam/):** Is too lengthy for the scope of this document, but linked is the keras documentation for this optimizer.

11. **Number of Epochs**
   
    The number of times steps 2 - 10 repeats over entire the dataset is one epoch. There are optimizations here with mini-batching because it would take an insane amount of compute to do this with every individual training example. Conceptually, I refrain from speaking of these concepts in conjunction with mini-batching unless it is nessesary (like in batchnorm). 

    **The number of epochs is defined by the user, but you will notice that the loss will converge to a constant-ish value. This is "convergence" and a good indicator that there are diminishing returns for additional epochs.** Setting `verbose=1` will display the loss after every epoch.

12. Final Prediction

    The weights and biases of the neural network are set and no longer updated.
    
    **The classification threshold varies based on the activation function of the output neuron:**
    
    **Sigmoid**: 0 (negative class) if output neuron activation is <.5 and 1 (positive class) if >=.5*

---

## Contact Info
**Email:** [juandev32@gmail.com](mailto:juandev32@gmail.com)  
**LinkedIn:** [Juan Chavira's Profile](https://www.linkedin.com/in/juan-chavira/)

---

## VERSIONS QUICK REFERENCE
- `piexif==1.1.3` For image(.jpg) manipulation
- `tensorflow==2.18.0` Keras API Convolutional Neural Network Architecture
- `pillow==11.0.0` Remove metadata from images