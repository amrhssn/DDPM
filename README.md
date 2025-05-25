# DDPM

The denoising diffusion probabilistic model (DDPM) can be viewed as a Markovian hierarchical
variational autoencoder, where the forward encoder is a fixed linear Gaussian model, and 
we are interested in learning the reverse decoder. 
The idea is to progressively add Gaussian noise to the input data in the forward process
until it is indistinguishable from the standard Gaussian noise, and learn a denoising model 
in the backward process to reconstruct the input from the noise.

Specifically, given the Markov property, the forward process can be factorized as 
$q(x_{1:T}|x_0)= \prod_{t=1}^T q(x_t|x_{t-1})$, where $x_0$ is the input and $x_{1:T}$ are 
the latent variables given the input. We have 
$q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)$, and 
$\alpha_t = 1-\beta_t$ where $\beta_t$ is the noise level added at each layer of the forward 
encoder. 

The join distribution of the reverse decoder is 
$p(x_{0:T})=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$ where $p(x_T)=\mathcal{N}(x_T|0, I)$. 
Our goal is to learn the parameters $\theta$ by minimizing the variational lower bound which
is derived from the forward and reverse processes.


---
## 1. Noise Schedule

### Default (Linear)  

$$ 
β_t = β_{\rm start} + \frac{t-1}{T-1}\,(β_{\rm end}-β_{\rm start}),\quad
$$
where $β_{\rm start}=10^{-4},\,β_{\rm end}=0.02,\,T=1000.$

$$
α_t = 1-β_t,
\quad
\bar α_t = \prod_{i=1}^t α_i,
$$


### Best Practice (Cosine)  

$$
\bar α_t
= \frac{\cos\!\Bigl(\tfrac{t/T + s}{1 + s}\,\tfrac\pi2\Bigr)^2}
       {\cos\!\Bigl(\tfrac{s}{1 + s}\,\tfrac\pi2\Bigr)^2},
\quad
α_t = \frac{\bar α_t}{\bar α_{t-1}},
\quad
s=0.008,\;T=1000.
$$

### Cache  
Precompute arrays of  

$$
\sqrt{α_t},\,
\sqrt{\bar α_t},\,
\sqrt{1-\bar α_t},\,
\sigma_q(t)=\sqrt{\frac{(1-α_t)(1-\bar α_{t-1})}{1-\bar α_t}}.
$$

---

## 2. Forward Process

### Definition  

$$
q(x_t\mid x_0)=\mathcal{N}\bigl(x_t;\sqrt{\bar α_t}\,x_0,\,(1-\bar α_t)\,I\bigr).
$$  

### Sampling  

$$
x_t = \sqrt{\bar α_t}\,x_0 \;+\;\sqrt{1-\bar α_t}\,ε_0,\quad ε_0\sim\mathcal N(0,I).
$$

---

## 3. Ground-Truth Posterior

### Posterior 

$$
    q(x_{t-1}\mid x_t,x_0)
    = \mathcal{N}\bigl(x_{t-1};\mu_q(x_t,x_0),\,\sigma_q^2(t)\,I\bigr),
$$

where

$$ 
    \mu_q
    = \frac{\sqrt{α_t}(1-\bar α_{t-1})}{1-\bar α_t}\,x_t
      + \frac{\sqrt{\bar α_{t-1}}(1-α_t)}{1-\bar α_t}\,x_0.
$$

---

## 4. Denoising Network: Predict $ε_0$

### Network 

$\hat ε_θ(x_t,t)$ predicts the original noise $ε_0$.

### Reverse mean  

$$
    μ_θ(x_t,t)
    = \frac{1}{\sqrt{α_t}}\Bigl(x_t - \frac{1-α_t}{\sqrt{1-\bar α_t}}\,\hat ε_θ(x_t,t)\Bigr).
$$

---

## 5. Loss Function: Full ELBO in Three Terms

The full Evidence Lower Bound (ELBO) decomposes into three loss terms:

- **Reconstruction term**:

$$
\mathbb{E}_{q(x_1|x_0)} \left[ \log p_θ(x_0 | x_1) \right]
$$

- **Prior matching term**:

$$
-\text{KL}(q(x_T|x_0) \| p(x_T)) = 
-\text{KL}(\mathcal{N}(x_T; \sqrt{\bar α_T} x_0, (1 - \bar α_T) I) \| \mathcal{N}(0, I))
$$

- **Denoising matching term**:

$$
-\sum_{t=2}^{T} \mathbb{E}_{q(x_t|x_0)} 
\left[ \text{KL}(q(x_{t-1} | x_t, x_0) \| p_θ(x_{t-1} | x_t)) \right]
$$

### **Best Practices and Reparameterization**

Instead of directly modeling the above KL terms, the following reparameterized surrogate 
loss is used:

Assuming the predicted noise $\hat ε_θ(x_t, t)$ and 
original noise $ε_0 \sim \mathcal{N}(0, I)$, the denoising matching term becomes:

$$
\mathcal{L}_{\text{simple}}(θ) =
\mathbb{E}_{t \sim \mathcal{U}[1,T],\, ε_0 \sim \mathcal{N}(0,I),\, x_0 \sim \text{data}} \left[
\left\| ε_0 - \hat ε_θ(x_t, t) \right\|^2
\right]
$$

This corresponds to:

$$
\mathcal{L}_{\text{simple}}(θ) = \mathbb{E}_t \left[
\lambda_t \cdot \| ε_0 - \hat ε_θ(x_t, t) \|^2
\right]
$$

### **Best Practice Loss Weighting**

$$
q(x_t \mid x_0) = 
\mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}\right)
$$

- Signal: $\mu = \sqrt{\bar{\alpha}_t} x_0$
- Noise variance: $\sigma^2 = 1 - \bar{\alpha}_t$

We have **signal-to-noise ratio (SNR)= $\frac{\mu^2}{\sigma^2}$**. So at timestep $t$:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

And we have:

$$
\lambda_t = \frac{1}{2} \left( \text{SNR}(t-1) - \text{SNR}(t) \right)=
\frac{1}{2}(\frac{\hat{\alpha}_{t-1}}{1 - \hat{\alpha}_{t-1}} - \frac{\hat{\alpha}_t}{1-\hat{\alpha}_t})
$$

---
## 6. Sampling Process After Training

Once the denoising network $\hat\epsilon_\theta(x_t, t)$ is trained, 
we generate new samples via one of two methods:

### 6.1 Default (Ancestral) DDPM Sampling

1. **Initialization**  
   Sample

   $$
   x_T \sim \mathcal{N}(0, I)\,.
   $$

2. **Iterative Reverse Steps**

   For $t = T, T-1, \dots, 1$:

   1. Predict noise:
   
      $$
      \hat\epsilon = \hat\epsilon_\theta(x_t, t)\,.
      $$
   
   2. Compute reverse mean:
   
      $$
      \mu_\theta(x_t, t)
      = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\,\hat\epsilon\Bigr).
      $$
   
3. Add stochasticity:

$$ 
x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\tilde\beta_t}\;z_t,  \quad  z_t \sim \mathcal{N}(0, I),
$$
   
  where

$$
\tilde\beta_t
= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t.
$$

**Output**: $x_0$ is the final generated sample, approximately following $p(x_0)$.