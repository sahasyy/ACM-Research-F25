![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

## Papers Read

1. Compositional Transformers for Scene Generation
2. High-Resolution Image Synthesis with Latent Diffusion Models
3. Conditional Generative Adversarial Nets
4. Medical Imaging Complexity and its Effects on
GAN Performance
5. Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model

## Paper 3 Chosen

**"Conditional Generative Adversarial Nets"**

## Summary of Paper

This paper talks about Conditional GANs (cGANs), which include auxiliary data (such as class labels or picture attributes) into the generator and discriminator. This research expands on the basic GAN architecture. Tasks such as image tagging, using Flickr data, and digit creation conditioned on class labels are used to test the model. Although performance varies according on the job, results demonstrate that cGANs may effectively direct the generating process and enable multi-modal output production.

## Justification for the Approach

The authors argue that while traditional GANs produce high-quality outputs, they lack control over the generated data. By introducing conditioning variables into the architecture, cGANs allow users to direct the generative process, making it more practical for real-world tasks such as image-to-text generation or class-specific synthesis. This enhancement is relatively simple to implement but significantly increases the model's usability and flexibility.

## Evaluation of Strengths and Weaknesses

**Strengths:**
- Introduces a simple yet powerful method for controlled data generation.
- Effective for multi-label prediction and image synthesis.
- Flexible and extensible to different data modalities (e.g., text, image).

**Weaknesses:**
- Performance is still outperformed by some traditional models in certain tasks.
- Requires careful tuning of hyperparameters and architecture.
- The paper lacks thorough evaluation and presents only preliminary results.
- cGANs may still inherit training instability issues of GANs.

## Some Novelties Noticed in the Paper

- **Conditional data generation:**  Incorporates class labels or image features into GANs to control the output.
- **Multi-modal capability:** Demonstrates the use of cGANs for generating textual tags from images.
- **Architectural simplicity:** Achieves enhanced functionality with minimal structural changes to GANs.