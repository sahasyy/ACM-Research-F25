# Image-to-Image Translation with Diffusion Transformers and CLIP-Based Image Conditioning


## Motivation

Image-to-image translation is fundamental in mapping images to a target domain, used in various applications such as medical imaging and virtual reality. Generative Adversarial Networks (GANs) paved the way in translation, but have limitations such as training instability and lack of detail. This paper proposes a new framework, Diffusion Transformers (DiT), which addresses the limitations of traditional GAN-based image-to-image translation methods. It uses a pretrained CLIP encoder for conditioning to represent the image's characteristics and semantic content using a numerical representation. 2 datasets, face2comics and edges2shoes, are compared across various models to demonstrate that diffusion models have considerable comparison with GAN-based models.

## Novelty

- Diffusion models are only recently used for translation tasks compared to GANs
- CLIP encoder for semantic content representation
- DiT allows better high-resolution with global relationships
- More scalability across image types 

## Methods

### Model Evaluation:
- Image translation is visually compared with other model's output
- Loss graph shows semantic content is accurately represented

## Advantages/Disadvantages 

### Advantages
- Consistent accurate semantic output based on decreasing loss and visual quality
- Adaptable model across multiple translation tasks
- Better high-resolution with global relationships
- Increased training stability than GANs

### Limitations
- Significant computational power needed compared to other models
- Dependence of pretrained models (CLIP encoder) limit the encoding of specific image
domains
- For paired data only

