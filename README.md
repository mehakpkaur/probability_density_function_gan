# Learning an Unknown Probability Density Function using GAN

## Overview
This project focuses on learning the probability density function (PDF) of a random variable **when its analytical form is unknown**.  
Unlike traditional methods where the PDF equation is assumed or provided, this work uses a **Generative Adversarial Network (GAN)** to implicitly learn the distribution **only from data samples**.

The task is based on NO₂ concentration data and follows a transformation that makes the resulting distribution analytically intractable, thereby motivating the use of advanced machine learning techniques.

---

## Dataset
- **Feature used:** NO₂ concentration  
- **Source:** India Air Quality Dataset  
- **Input variable:** `x` (NO₂ concentration)

The dataset is preprocessed by removing missing values before further analysis.

---

## Data Transformation
To ensure that the resulting distribution has no simple analytical form, each value of `x` is transformed as:


Where:
- `a_r = 0.5 × (r mod 7)`
- `b_r = 0.3 × (r mod 5 + 1)`
- `r` is the university roll number

This nonlinear transformation introduces oscillatory behavior, making the PDF of `z` unknown and analytically intractable.

---

## Motivation for Using GAN
Since:
- The analytical form of the PDF is unknown  
- No parametric assumption (Gaussian, exponential, etc.) is allowed  
- Only samples of the transformed variable `z` are available  

A **Generative Adversarial Network (GAN)** is used to learn the underlying distribution directly from data.

The GAN learns the PDF **implicitly**, without estimating an explicit probability function.

---

## GAN Architecture

### Generator
- Input: Random noise sampled from `N(0,1)`
- Output: Generated samples `z_f`
- Architecture: Fully connected neural network with ReLU activations

### Discriminator
- Input: Real samples `z` and generated samples `z_f`
- Output: Probability that the sample is real
- Architecture: Fully connected neural network with Sigmoid activation at the output

The generator aims to fool the discriminator, while the discriminator learns to distinguish real samples from generated ones.

---

## Training Process
- **Loss Function:** Binary Cross Entropy Loss  
- **Optimizer:** Adam  
- **Training Strategy:** Mini-batch gradient descent  

The GAN is trained until the generator produces samples that are difficult for the discriminator to distinguish from real samples.

Due to the stochastic nature of GAN training, the learned distribution may vary slightly across different runs.

---

## PDF Estimation from GAN Samples
After training the GAN:
1. A large number of samples are generated from the trained generator
2. The probability density function is approximated using:
   - **Histogram Density Estimation** (coarse approximation)
   - **Kernel Density Estimation (KDE)** for smoother visualization

These methods provide a numerical and visual approximation of the learned probability density.

---

## Observations
- The GAN successfully captures the overall structure of the underlying distribution
- No explicit parametric PDF is assumed at any stage
- Minor variations in the learned PDF are expected due to random initialization and stochastic training
- KDE provides a smoother and more interpretable estimate compared to histogram-based estimation

---

## Conclusion
This project demonstrates the effectiveness of Generative Adversarial Networks in learning an unknown probability density function using data alone.  
The approach avoids analytical assumptions and highlights the capability of GANs to model complex, non-linear distributions.

---

## Tools & Libraries Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- PyTorch  
- SciPy  

---

## How to Run
1. Install the required Python libraries
2. Ensure the dataset file is present in the same directory as the notebook
3. Run the notebook `gan_pdf_estimation.ipynb` cell by cell
4. Visualize the estimated PDF using histogram and KDE plots
