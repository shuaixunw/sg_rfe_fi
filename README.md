# Statistically Guided and Interaction-Aware Recursive Feature Elimination (IA-RFE)

**Authors:** Shuaixun Wang, Mingkai Liu  
**Affiliation:** Imperial College London  
**Manuscript:** *Statistically Guided and Interaction-Aware Recursive Feature Elimination for Stable Prognostic Modeling in Traumatic Brain Injury*  
**Status:** Accepted for publication in *IEEE Journal of Biomedical and Health Informatics (JBHI)*  

---

## 🔍 Overview

This repository provides the official implementation of **IA-RFE (Interaction-Aware Recursive Feature Elimination)**, a feature-selection framework designed to enhance **stability, interpretability, and statistical rigor** in biomedical prognostic modeling.

The method integrates:

- **GPUTreeSHAP Interaction Values** for quantifying nonlinear feature interactions.
- **Network Importance (NI)** to unify main and interactive effects.
- **DeLong Statistical Test** to guide recursive elimination — ensuring that removed features do not cause a statistically significant degradation in ROC-AUC.
- **Dynamic α-adaptation** to control interaction weighting automatically.

The approach is particularly developed for **Traumatic Brain Injury (TBI)** prognosis but can be generalized to other clinical prediction tasks.

---

## 🧠 Method Summary

IA-RFE iteratively eliminates features based on a **dual-criterion strategy**:

1. **Interaction-Aware Scoring:**  
   Using the SHAP interaction tensor from GPU-accelerated XGBoost, we compute a *network importance* score for each feature:  
   \[
   NI_i = \text{main}_i + \alpha \sum_{j \ne i} |\text{inter}_{ij}|
   \]
   where \(\alpha\) is adaptively determined from the median interaction strength ratio.

2. **Statistical Safeguard (DeLong Test):**  
   Before removing a feature, the model compares AUC distributions before and after deletion using the **DeLong test**.  
   A feature is removed **only if p ≥ 0.05**, ensuring no significant performance loss.

This design achieves **statistically safe pruning** while maintaining high prognostic performance and cross-fold stability.

---

## ⚙️ Key Features

- ✅ GPU-accelerated SHAP interaction computation (via **GPUTreeShap**).  
- ✅ Adaptive α to balance main and interactive effects.  
- ✅ Cross-validated DeLong test for statistical guidance.  
- ✅ Bootstrapped AUC confidence intervals.  
- ✅ Fully reproducible pipeline with modular helper functions.

---

## 📁 Repository Structure

