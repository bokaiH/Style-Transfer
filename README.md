# Style-Transfer
This is the final project of Computer Vision (Peking University, 2024 fall).
## Motivation
***Style Transfer*** is a computer vision technique that allows us to recompose the content of an image in the style of another, which is interesting but challenging. Inspired by the power of Convolutional Neural Networks (CNNs), Gatys et al. first studied how to use a CNN to reproduce famous painting styles on natural images, which is the first work of neural style transfer. They proposed to model the content of a photo as the feature responses from a pre-trained CNN, and further model the style of an artwork as the summary feature statistics. The key idea behind their algorithm is to iteratively optimise an image with the objective of matching desired CNN feature distributions, which involves both the photo’s content information and artwork’s style information. We call this type of method as Gatys-style neural style transfer.

The method mentioned above only compare content and stylised images in the CNN feature space to make the stylised image semantically similar to the content image. But since CNN features inevitably lose some lowlevel information contained in the image, there are usually some unappealing distorted structures and irregular artefacts in the stylised results. To preserve the coherence of fine structures during stylisation, Li et al. propose to  introduce an additional Laplacian loss, which is defined as the squared euclidean distance between the Laplacian filter responses of a content image and stylised result. We call this type of method as Lapstyle neural style transfer.
