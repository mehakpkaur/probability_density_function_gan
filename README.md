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
- r = 102317094
  
This nonlinear transformation introduces oscillatory behavior, making the PDF of `z` unknown and analytically intractable.
## Transformation Parameters
The nonlinear transformation applied to the NO₂ concentration data is defined as:
z = x + a_r * sin(b_r * x)
For this implementation, the parameters derived are:
- **a_r = 2.5**
- **b_r = 1.5**


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
- Takes a one-dimensional noise vector sampled from a standard normal distribution `N(0,1)`
- Maps the noise to synthetic samples of the transformed variable `z_f`
### Discriminator
- Receives both real samples `z` and generated samples `z_f`
- Outputs a probability indicating whether the input sample is real or generated

The generator and discriminator are trained simultaneously in an adversarial manner.  The generator attempts to produce samples that resemble the real data distribution, while the discriminator learns to distinguish between real and generated samples.

---

## Training Process
During training, the discriminator is optimized to correctly classify real and fake samples, while the generator is optimized to fool the discriminator.  
Since GAN training involves random initialization and stochastic sampling, slight variations in the learned distribution across different runs are expected.


---

## PDF Estimation from GAN Samples
After training the GAN:
1. A large number of samples are generated from the trained generator
2. The probability density function is approximated using:
   - **Kernel Density Estimation (KDE)** for smoother visualization

These methods provide a numerical and visual approximation of the learned probability density.

---
## Kernel Density Estimation (KDE)
Kernel Density Estimation provides a smoother and more continuous approximation of the learned probability density function.

![KDE PDF](ass4/gan_pdf_kde.png)


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
