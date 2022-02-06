# W-net: A Deep Model for Fully Unsupervised Image Segmentation

Wnet, by Xia and Kulis (2017), expands and builds upon the fully convolutional U-net to create a completely unsupervised image segmentation algorithm.
The architecture of W-net is illustrated in the figure below:


<img width="613" alt="Screenshot 2022-02-06 at 20 02 16" src="https://user-images.githubusercontent.com/49812606/152694711-c41b5b88-7df2-4bbd-ae0d-0718898e9419.png">

The network ties a U_Enc and U_Dec into a single Autoencoder, hence the (UU) W shape of the network. The Encoder network takes the input image and produces a k-way segmentation. This segmentation is used as input for the UDec network where it reconstructs the original input image. The output is a segmented image and a reconstructed image of the input. The network contains 18 modules, with 9 modules in each the U_Enc and the UDec network. Like U-net, each module contains two 3x3 convolutional layers, followed by a ReLU and batch normalization, with the modules of the contracting path connected by 2x2 max-pooling layers and the modules of the expansive path connected by 2x2 2D transposed convolutional layers. The final layer of the U_Enc is a 1x1 convolutional layer that maps the features to the number of classes K, followed by a softmax layer that ensures that the output of each class is in the range of (0,1) and that the outputs of the K classes sum up to 1. The output of the U_Dec is a 1x1 convolutional layer that maps the feature map to the reconstructed image. Like U-net, skip connections are also used to retrieve spatial information, however, unlike U-net, W-Nets uses depth-wise separable convolutions, except modules 1, 9, 10, and 18. Two loss functions are used: soft normalized cuts, which is backpropagated onto the Encoder network, and the reconstruction loss is backpropagated onto the Decoder and Encoder networks.
Since we no longer use labeled data that informs us of the performance of the network, we need an alternative mechanism that dictates what constitutes a
good segmentation. This is achieved via the use of the Soft Normalized Cuts loss function.

### Soft Normalized Cuts: 
Soft Normalized Cut Loss, Shi Malek (2000), is a novel segmentation approach where the image segmentation problem is treated as a graph G(V,E) partitioning
problem. To create different segments in the graph, we remove the edges E between different sets of pixels. Ideally, this partitioning process is performed
among dissimilar regions.

### Reconstruction Loss: 
The second loss function implemented is the reconstruction loss, which is the squared error of the reconstructed image of the decoder network and the original
image.

### Results: 

For our implementation, the network was applied on nuclei of cell images. However, the network requires a strong computational power, as a result, only 3 images were used for training. Below are the results after training:

![x_0](https://user-images.githubusercontent.com/49812606/152696968-bac41945-7b34-4310-bc47-64be9b46a51e.png)
![enc_0](https://user-images.githubusercontent.com/49812606/152696976-76b7fa50-de73-43ca-8c5d-97a369842910.png)
![dec_0](https://user-images.githubusercontent.com/49812606/152696985-29904462-b32a-4efa-ba64-527be073948b.png)

![x_1](https://user-images.githubusercontent.com/49812606/152696996-dd5ac87b-635f-47b6-a5ab-e40158bbbe01.png)
![enc_1](https://user-images.githubusercontent.com/49812606/152697001-22a44a83-2285-4b02-928c-63a10403af23.png)
![dec_1](https://user-images.githubusercontent.com/49812606/152697009-19ea380c-77e0-4403-9124-480e77587a90.png)

![x_2](https://user-images.githubusercontent.com/49812606/152697035-d85ff118-65ff-49d1-a2ba-f8cc33be3172.png)
![enc_2](https://user-images.githubusercontent.com/49812606/152697036-33a27295-73fd-443d-ba7b-0a1122e9bb19.png)
![dec_2](https://user-images.githubusercontent.com/49812606/152697041-34fe9745-cc73-45ce-9193-12e137cc0101.png)



 The dataset is a public dataset found on kaggle https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells
 
