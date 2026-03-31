# Discovery of Hidden Symmetries and Conservation Laws

**Google Summer of Code 2026 | ML4SCI Evaluation Tasks**

**Author:** Harshvardhan Saini
**Organization:** ML4SCI

---

## 📌 Overview

This repository contains the complete implementation of the evaluation tasks for the **Discovery of Hidden Symmetries and Conservation Laws** project.

Moving beyond simple task completion, this codebase serves as an empirical **Proof of Concept** for my GSoC 2026 proposal:

> *Discovering continuous symmetries as Latent Vector Fields (Lie Algebra Generators) and building Symmetry-Aware Networks via Tangent Regularization.*

Using rotated MNIST digits as a sandbox, this repository demonstrates how complex pixel-space transformations can be unrolled into continuous, discoverable manifolds within the latent space of generative models.

---

## 🚀 Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/HarshSaini10/E2E_Tasks.git
cd E2E_Tasks
pip install -r requirements.txt
```

---

## 🧪 Specific Tasks: Symmetry Discovery Pipeline

### 🔹 Task 1: Dataset Preparation & Latent Space Creation

**Objective:**
Rotate MNIST digits ('1' and '2') in 30° increments and learn a continuous latent representation.

**Methodology:**

* Variational Autoencoder (VAE)
* ResNet-18 encoder (modified for grayscale)
* Transpose convolution decoder

**Key Finding:**
The VAE transforms pixel-space rotations into a structured **circular ("swirl") manifold** in latent space.

<!-- Optional -->

<!-- ![Latent Space Swirl](./path_to_image/swirl.png) -->

---

### 🔹 Task 2: Supervised Symmetry Discovery

**Objective:**
Learn a latent mapping corresponding to 30° rotation.

**Methodology:**

* Train MLP on latent vector pairs
* Input-output pairs: (θ → θ + 30°)

**Result:**
Model successfully learns discrete transitions along symmetry manifold.

---

### 🔹 Task 3: Unsupervised Symmetry Discovery

**Objective:**
Discover symmetry without labels.

**Core Idea:**
Infinitesimal transformation:

```
z' = z + ε g(z)
```

**Methodology:**

* Learn generator ( g(z) )
* Logit-preserving loss
* Orthogonality + normalization constraints

**Key Finding:**
The learned **vector field aligns tangentially to latent manifold**, proving:

> Continuous symmetries can be extracted as **Lie Algebra generators** from latent space.

<!-- Optional -->

<!-- ![Vector Field](./path_to_image/flow.png) -->

---

## 🌟 Bonus Task: Rotation Invariant Network

### 🔹 Key Idea: Tangent Regularization

Instead of data augmentation, enforce invariance in latent space.

**Transformation:**

```
z_rot = z + ε g(z)
```

**Loss Function:**

```
L = || f(z) - f(z + ε g(z)) ||²
```

### 🔹 Result

* Classifier becomes **rotation-invariant**
* Decision boundaries align with symmetry manifold
* No augmentation required

---

## 🔬 Key Insight

The pipeline shows:

* Latent spaces encode **continuous symmetry structure**
* Symmetries can be **learned, not hardcoded**
* Vector fields ≈ **Lie algebra generators**

---

## 🚀 Connection to GSoC 2026

This work directly extends to:

* Lorentz symmetry discovery in physics datasets
* CMS jet data representations
* Groups like:

  * SO(2) (rotations)
  * SO(1,3)+ (Lorentz group)

### 🔹 Impact

* Better generalization
* Data-efficient models
* Physics-aware architectures
* Unsupervised symmetry discovery

---

## 📌 Next Steps

1. Create folder structure:

   ```
   specific_tasks/
   bonus_task/
   ```

2. Move notebooks accordingly

3. Add `invariant_training.py`

4. Save this as `README.md`

5. Push to GitHub

---

## 🙌 Final Note

This repository is designed to be:

* Clean
* Reproducible
* Research-aligned

It bridges **deep learning + symmetry discovery + physics**, forming the foundation for a high-impact research direction.

---
