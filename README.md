<div align="center">
  <div>
  <h1>Enhancing Plasticity for First Session Adaptation Continual Learning</h1> 

[![Static Badge](https://img.shields.io/badge/PLASTIC-arXiv-red)](https://arxiv.org/abs/2310.11482)


  </div>

</div>

<div align="center">

<div>
    <a href='https://iemprog.github.io/' target='_blank'>Imad Eddine MAROUF</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.it/citations?user=YfzgrDYAAAAJ&hl=en' target='_blank'>Subhankar Roy</a><sup>2</sup>&emsp;
    <a href='https://stelat.eu/' target='_blank'>Stéphane Lathuilière</a><sup>3</sup>
    <a href='https://enzotarta.github.io/' target='_blank'>Enzo Tartaglione</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup>Télécom-Paris, Institut Polytechnique de Paris, France &emsp;
<sup>2</sup>University of Bergamo, Italy&emsp;
<sup>3</sup>Inria at University Grenoble Alpes, France&emsp;
</div>
</div>


The code repository for "[Enhancing Plasticity for First Session Adaptation Continual Learning](https://arxiv.org/abs/2310.11482)" in PyTorch.

📣 Accepted as a conference paper at CoLLAs 2025

## Abstract
The integration of large pre-trained models (PTMs) into Class-Incremental Learning (CIL) has facilitated the development of compute-efficient strategies such as First-Session Adaptation (FSA), which fine-tunes the model solely on the first task while keeping it frozen for subsequent tasks. Although effective in homogeneous task sequences, these approaches struggle when faced with the heterogeneity of real-world task distributions. We introduce Plasticity-Enhanced Test-Time Adaptation in Class-Incremental Learning (PLASTIC), a method that reinstates plasticity in CIL while preserving model stability. PLASTIC leverages Test-Time Adaptation (TTA) by dynamically fine-tuning LayerNorm parameters on unlabeled test data, enabling adaptability to evolving tasks and improving robustness against data corruption. To prevent TTA-induced model divergence and maintain stable learning across tasks, we introduce a teacher-student distillation framework, ensuring that adaptation remains controlled and generalizable. Extensive experiments across multiple benchmarks demonstrate that PLASTIC consistently outperforms both conventional and state-of-the-art PTM-based CIL approaches, while also exhibiting inherent robustness to data corruptions


<p align="center">
  <img src="public/pipeline_plot.png" alt="PLASTIC Design" />
</p>

🚀 **Code Coming Soon!** 🚀  
  
## 📂 Code & Models  
The code will be released soon. Stay tuned! 
