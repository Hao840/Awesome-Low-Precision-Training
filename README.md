# Awesome Low-Precision Training
A collection of research papers on low-precision training for foundation models, organized by numerical representation formats. The collection also includes quantization-aware training techniques for large language models.

\*🤖 *indicates the corresponding paper contains experiments using LLM.*

## Contents
  - [**Fixed-Point and Integer-Based Methods**](#Fixed-Point-and-Integer-Based-Methods)
    - [**Fixed-Point**](#Fixed-Point)
    - [**Integer**](#Integer)
      - [**General methods**](#General-methods)
      - [**Optimizer-state-targeted methods**](#Optimizer-state-targeted-methods)
      - [**Communication-targeted methods**](#Communication-targeted-methods)
    - [**Binary**](#Binary)
  - [**Floating-Point-Based Methods**](#Floating-Point-Based-Methods)
  - [**Customized Format-Based Methods**](#Customized-Format-Based-Methods)
  - [**Quantization-Aware Training Methods**](#Quantization-Aware-Training-Methods)

## Fixed-Point and Integer-Based Methods

### Fixed-Point
-  [**2024 | arXiv**] Trainable Fixed-Point Quantization for Deep Learning Acceleration on FPGAs [[📄 paper](http://arxiv.org/abs/2401.17544)]
-  [**2021 | arXiv**] A Simple and Efficient Stochastic Rounding Method for Training Neural Networks in Low Precision [[📄 paper](http://arxiv.org/abs/2103.13445)]
-  [**2020 | ICML**] Multi-Precision Policy Enforced Training (MuPPET): A precision-switching strategy for quantised fixed-point training of CNNs [[📄 paper](https://arxiv.org/abs/2006.09049)]
-  [**2019 | ICML**] SWALP: Stochastic Weight Averaging in Low-Precision Training [[📄 paper](https://arxiv.org/abs/1904.11943)] [[💻 code](https://github.com/stevenygd/SWALP)]
-  [**2019 | NeurIPS**] Backprop with Approximate Activations for Memory-efficient Network Training [[📄 paper](https://arxiv.org/abs/1901.07988v1)] [[💻 code](https://github.com/ayanc/blpa)]
-  [**2019 | ICLR**] Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm [[📄 paper](https://arxiv.org/abs/1812.11732)]
-  [**2018 | ICLR**] Mixed Precision Training of Convolutional Neural Networks Using Integer Operations [[📄 paper](https://arxiv.org/abs/1802.00930v2)]
-  [**2018 | ICCD**] Training Neural Networks with Low Precision Dynamic Fixed-Point [[📄 paper](https://ieeexplore.ieee.org/document/8615717)]
-  [**2017 | IJCNN**] FxpNet: Training a Deep Convolutional Neural Network in Fixed-Point Representation [[📄 paper](https://ieeexplore.ieee.org/document/7966159)]
-  [**2015 | ICML**] Deep Learning with Limited Numerical Precision [[📄 paper](https://arxiv.org/abs/1502.02551)]
-  [**2015 | ICLR workshop**] Training Deep Neural Networks With Low Precision Multiplications [[📄 paper](https://arxiv.org/abs/1412.7024)]

### Integer

#### General methods

-  [**2025 | TPAMI | 🤖**] Latent Weight Quantization for Integerized Training of Deep Neural Networks [[📄 paper](https://www.computer.org/csdl/journal/tp/2025/04/10834560/23mYTMVTReM)]
-  [**2025 | FCS**] Efficient Deep Neural Network Training via Decreasing Precision With Layer Capacity [[📄 paper](https://link.springer.com/content/pdf/10.1007/s11704-024-40669-3.pdf)]
-  [**2025 | arXiv | 🤖**] Accurate INT8 Training Through Dynamic Block-Level Fallback [[📄 paper](https://arxiv.org/abs/2503.08040)]
-  [**2024 | ICML | 🤖**] Jetfire: Efficient and Accurate Transformer Pretraining With INT8 Data Flow and Per-Block Quantization [[📄 paper](https://github.com/thu-ml/Jetfire-INT8Training)]
-  [**2024 | ICML**] AMPA: Adaptive Mixed Precision Allocation for Low-Bit Integer Training [[📄 paper](https://proceedings.mlr.press/v235/ding24b.html)]
-  [**2024 | ICASSP**] Activation Compression of Graph Neural Networks using Block-wise Quantization with Improved Variance Minimization [[📄 paper](https://arxiv.org/abs/2309.11856)] [[💻 code](https://github.com/saintslab/i-Exact)]
-  [**2024 | arXiv | 🤖**] Direct Quantized Training of Language Models with Stochastic Rounding [[📄 paper](https://arxiv.org/abs/2412.04787)] [[💻 code](https://github.com/KYuuto1006/DQT)]
-  [**2024 | arXiv**] Towards Accurate and Efficient Sub-8-Bit Integer Training [[📄 paper](https://arxiv.org/abs/2411.10948)]
-  [**2024 | arXiv**] HLQ: Fast and Efficient Backpropagation via Hadamard Low-rank Quantization [[📄 paper](http://arxiv.org/abs/2406.15102)]
-  [**2024 | arXiv | 🤖**] Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients [[📄 paper](http://arxiv.org/abs/2401.17544)] [[💻 code](https://github.com/VITA-Group/Q-GaLore)]
-  [**2023 | NeurIPS**] Stable and Low-Precision Training for Large-Scale Vision-Language Models [[📄 paper](https://arxiv.org/abs/2304.13013)]
-  [**2023 | NeurIPS**] Training Transformers With 4-Bit Integers [[📄 paper](https://arxiv.org/abs//2306.11987)] [[💻 code](https://github.com/haochengxi/Train_Transformers_with_INT4)]
-  [**2023 | ICML**] Few-bit Backward: Quantized Gradients of Activation Functions for Memory Footprint Reduction [[📄 paper](https://proceedings.mlr.press/v202/novikov23a/novikov23a.pdf)] [[💻 code](https://github.com/skolai/fewbit)]
-  [**2023 | CoLLAs**] Hadamard Domain Training with Integers for Class Incremental Quantized Learning [[📄 paper](https://arxiv.org/abs/1803.03383)] [[💻 code](https://github.com/Intelligent-Microsystems-Lab/QuantizedCIL)]
-  [**2022 | ICML**] GACT: Activation Compressed Training for Generic Network Architectures [[📄 paper](https://proceedings.mlr.press/v162/liu22v/liu22v.pdf)] [[💻 code](https://github.com/LiuXiaoxuanPKU/GACT-ICML)]
-  [**2022 | Neurocomputing**] Towards Efficient Full 8-bit Integer DNN Online Training on Resource-limited Devices without Batch Normalization [[📄 paper](http://arxiv.org/abs/2105.13890)]
-  [**2022 | TPDS**] NITI: Training Integer Neural Networks Using Integer-only Arithmetic [[📄 paper](https://arxiv.org/abs/2009.13108)] [[💻 code](https://github.com/wangmaolin/niti)]
-  [**2021 | ICML**] ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training [[📄 paper](http://proceedings.mlr.press/v139/chen21z/chen21z.pdf)] [[💻 code](https://github.com/ucbrise/actnn)]
-  [**2021 | ICLR**] CPT: Efficient Deep Neural Network Training via Cyclic Precision [[📄 paper](http://arxiv.org/abs/2101.09868)] [[💻 code](https://github.com/RICE-EIC/CPT)]
-  [**2021 | ICLR**] EXACT: Scalable Graph Neural Networks Training via Extreme Activation Compression [[📄 paper](https://openreview.net/forum?id=vkaMaq95_rX)] [[💻 code](https://github.com/zirui-ray-liu/Exact)]
-  [**2021 | AAAI**] Distribution Adaptive INT8 Quantization for Training CNNs [[📄 paper](https://arxiv.org/abs/2102.04782)]
-  [**2021 | Neurocomputing**] Training and Inference for Integer-Based Semantic Segmentation Network [[📄 paper](https://arxiv.org/abs/2011.14504)] [[💻 code](https://github.com/MarkYangjiayi/Semantic-Quantization)]
-  [**2021 | CVPR workshop**] In-Hindsight Quantization Range Estimation for Quantized Training [[📄 paper](http://arxiv.org/abs/2105.04246)]
-  [**2020 | NeurIPS**] FracTrain: Fractionally Squeezing Bit Savings Both Temporally and Spatially for Efficient DNN Training [[📄 paper](https://arxiv.org/abs/2012.13113)] [[💻 code](https://github.com/RICE-EIC/FracTrain)]
-  [**2020 | NeurIPS**] A Statistical Framework for Low-bitwidth Training of Deep Neural Networks [[📄 paper](https://arxiv.org/abs/2010.14298)] [[💻 code](https://github.com/cjf00000/StatQuant)]
-  [**2020 | CVPR**] Towards Unified INT8 Training for Convolutional Neural Network [[📄 paper](https://arxiv.org/abs/1912.12607)]
-  [**2020 | CVPR**] Fixed-Point Back-Propagation Training [[📄 paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Fixed-Point_Back-Propagation_Training_CVPR_2020_paper.pdf)]
-  [**2020 | Neural Networks**] Training High-Performance and Large-Scale Deep Neural Networks with Full 8-bit Integers [[📄 paper](https://arxiv.org/abs/1909.02384)] [[💻 code](https://github.com/yang-yk/wageubn)]
-  [**2019 | NeurIPS**] Dimension-Free Bounds for Low-Precision Training [[📄 paper](https://papers.NeurIPS.cc/paper/2019/file/d4cd91e80f36f8f3103617ded9128560-Paper.pdf)]
-  [**2018 | NeurIPS**] Scalable Methods for 8-bit Training of Neural Networks [[📄 paper](https://papers.NeurIPS.cc/paper/2018/file/e82c4b19b8151ddc25d4d93baf7b908f-Paper.pdf)] [[💻 code](https://github.com/eladhoffer/quantized.pytorch)]
-  [**2018 | ICLR**] Training and Inference with Integers in Deep Neural Networks [[📄 paper](https://arxiv.org/abs/1802.04680)] [[💻 code](https://github.com/boluoweifenda/WAGE)]
-  [**2018 | ECCV**] Value-aware Quantization for Training and Inference of Neural Networks [[📄 paper](https://arxiv.org/abs/1804.07802)]
-  [**2018 | arXiv**] Training Deep Neural Network in Limited Precision [[📄 paper](https://arxiv.org/abs/1810.05486)]
-  [**2018 | arXiv**] High-Accuracy Low-Precision Training [[📄 paper](https://arxiv.org/abs/1803.03383)]
-  [**2017 | ICML**] The ZipML Framework for Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning [[📄 paper](https://arxiv.org/abs/1611.05402)] [[💻 code](https://github.com/IST-DASLab/smart-quantizer)]
-  [**2017 | NeurIPS**] Training Quantized Nets: A Deeper Understanding [[📄 paper](https://arxiv.org/abs/1706.02379)]
-  [**2016 | arXiv**] DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients [[📄 paper](https://arxiv.org/abs/1606.06160)] [[💻 code](https://github.com/hpi-xnor/BMXNet-v2)]
-  [**2016 | arXiv**] Convolutional Neural Networks using Logarithmic Data Representation [[📄 paper](https://arxiv.org/abs/1603.01025)]

#### Optimizer-state-targeted methods

-  [**2024 | NeurIPS | 🤖**] MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence [[📄 paper](https://arxiv.org/abs/2405.15593)] [[💻 code](https://github.com/IST-DASLab/MicroAdam)]
-  [**2024 | NeurIPS**] 4-bit Shampoo for Memory-Efficient Network Training [[📄 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/e5b4633454cb2174779d294ccda02318-Paper-Conference.pdf)] [[💻 code](https://github.com/Sike-Wang/low-bit-Shampoo)]
-  [**2024 | EMNLP | 🤖**] Exploring Quantization for Efficient Pre-Training of Transformer Language Models [[📄 paper](https://aclanthology.org/2024.findings-emnlp.787/)]
-  [**2024 | arXiv | 🤖**] Memory-Efficient 4-bit Preconditioned Stochastic Optimization [[📄 paper](https://arxiv.org/pdf/2412.10663)]
-  [**2023 | NeurIPS | 🤖**] Memory Efficient Optimizers with 4-bit States [[📄 paper](https://arxiv.org/abs/2309.01507)] [[💻 code](https://github.com/thu-ml/low-bit-optimizers)]
-  [**2023 | arXiv | 🤖**] QFT: Quantized Full-Parameter Tuning of Llms With Affordable Resources [[📄 paper](https://arxiv.org/abs/2310.07147)]
-  [**2022 | ICLR | 🤖**] 8-bit Optimizers via Block-wise Quantization [[📄 paper](https://arxiv.org/abs/2110.02861v2)]

#### Communication-targeted methods

-  [**2025 | TPAMI | 🤖**] LoCo: Low-Bit Communication Adaptor for Large-Scale Model Training [[📄 paper](https://arxiv.org/pdf/2407.04480)] [[💻 code](https://github.com/XingyuXie/LoCo)]
-  [**2024 | NeurIPS | 🤖**] SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training [[📄 paper](https://arxiv.org/pdf/2410.15526)] [[💻 code](https://github.com/bytedance/SDP4Bit)]
-  [**2024 | NSDI | 🤖**] THC: Accelerating Distributed Deep Learning Using Tensor Homomorphic Compression [[📄 paper](https://www.usenix.org/system/files/nsdi24-li-minghao.pdf)] [[💻 code](https://github.com/sophiali06/byteps_thc)]
-  [**2023 | ICML | 🤖**] Quantized Distributed Training of Large Models with Convergence Guarantees [[📄 paper](https://proceedings.mlr.press/v202/markov23a/markov23a.pdf)]
-  [**2023 | arXiv | 🤖**] ZeRO++: Extremely Efficient Collective Communication for Giant Model Training [[📄 paper](https://arxiv.org/pdf/2306.10209)]
-  [**2022 | NeurIPS | 🤖**] Fine-tuning Language Models over Slow Networks using Activation Quantization with Guarantees [[📄 paper](https://arxiv.org/pdf/2206.01299)]
-  [**2020 | NeurIPS**] Adaptive Gradient Quantization for Data-Parallel SGD [[📄 paper](https://proceedings.neurips.cc/paper/2020/file/20b5e1cf8694af7a3c1ba4a87f073021-Paper.pdf)] [[💻 code](https://github.com/Tabrizian/learning-to-quantize)]
-  [**2019 | ICML**] DOUBLESQUEEZE: Parallel Stochastic Gradient Descent with Double-pass Error-Compensated Compression [[📄 paper](https://proceedings.mlr.press/v97/tang19d.html)]
-  [**2019 | NeurIPS**] Double Quantization for Communication-Efficient Distributed Optimization [[📄 paper](https://papers.nips.cc/paper_files/paper/2019/hash/ea4eb49329550caaa1d2044105223721-Abstract.html)]
-  [**2017 | NeurIPS**] QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding [[📄 paper](https://arxiv.org/abs/1610.02132)]
-  [**2015 | arXiv**] 8-Bit Approximations for Parallelism in Deep Learning [[📄 paper](https://arxiv.org/abs/1511.04561)]

### Binary

-  [**2024 | arXiv**] 1-Bit FQT: Pushing the Limit of Fully Quantized Training to 1-bit [[📄 paper](https://arxiv.org/abs/2408.14267)] [[💻 code](https://github.com/Gaochang-bjtu/1-bit-FQT)]
-  [**2023 | ICLR | 🤖**] Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam [[📄 paper](https://arxiv.org/pdf/2202.06009)]
-  [**2023 | NeurIPS**] Birder: Communication-Efficient 1-bit Adaptive Optimizer for Practical Distributed DNN Training [[📄 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7c72fcd7b6bffc3864c7152ab5a2dd83-Paper-Conference.pdf)]
-  [**2023 | TECS**] Enabling Binary Neural Network Training on the Edge [[📄 paper](https://arxiv.org/pdf/2102.04270)] [[💻 code](https://github.com/awai54st/Enabling-Binary-Neural-Network-Training-on-the-Edge)]
-  [**2022 | HiPC**] 1-bit LAMB: Communication Efficient Large-Scale Large-Batch Training with LAMB’s Convergence Speed [[📄 paper](https://arxiv.org/abs/2104.06069)]
-  [**2022 | DAC**] Sign Bit is Enough: A Learning Synchronization Framework for Multi-hop All-reduce with Ultimate Compression [[📄 paper](https://arxiv.org/abs/2204.06787)]
-  [**2021 | ICML**] 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed [[📄 paper](https://arxiv.org/abs/2102.02888)]
-  [**2019 | ICML**] Error Feedback Fixes SignSGD and other Gradient Compression Schemes [[📄 paper](https://arxiv.org/abs/1901.09847)] [[💻 code](https://github.com/epfml/error-feedback-SGD)]
-  [**2018 | ICML**] Quantized Neural Networks: Training Neural Networks With Low Precision Weights and Activations [[📄 paper](http://www.jmlr.org/papers/volume18/16-456/16-456.pdf)]
-  [**2016 | ECCV**] XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks [[📄 paper](https://arxiv.org/abs/1603.05279)] [[💻 code](https://github.com/allenai/XNOR-Net)]
-  [**2015 | NeurIPS**] Binaryconnect: Training Deep Neural Networks With Binary Weights During Propagations [[📄 paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/3e15cc11f979ed25912dff5b0669f2cd-Paper.pdf)]
-  [**2014 | Interspeech**] 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs [[📄 paper](https://www.isca-archive.org/interspeech_2014/seide14_interspeech.pdf)]

## Floating-Point-Based Methods

-  [**2025 | ICLR | 🤖**] COAT: Compressing Optimizer States and Activation for memory efficient FP8 Training [[📄 paper](https://arxiv.org/abs/2410.19313)] [[💻 code](https://github.com/NVlabs/COAT)]
-  [**2025 | arXiv | 🤖**] Optimizing Large Language Model Training Using FP4 Quantization [[📄 paper](https://arxiv.org/abs/2501.17116)]
-  [**2025 | arXiv | 🤖**] Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models [[📄 paper](https://arxiv.org/abs/2502.11458)]
-  [**2024 | arXiv | 🤖**] DeepSeek-V3 Technical Report [[📄 paper](https://arxiv.org/abs/2412.19437)]
-  [**2024 | arXiv | 🤖**] Scaling FP8 Training to Trillion-Token LLMs [[📄 paper](https://arxiv.org/abs/2409.12517)]
-  [**2024 | ICML workshop | 🤖**] Scalify: Scale Propagation for Efficient Low-Precision LLM Training [[📄 paper](https://arxiv.org/abs/2407.17353)] [[💻 code](https://github.com/graphcore-research/jax-scalify)]
-  [**2024 | DATE**] A Stochastic Rounding-Enabled Low-Precision Floating-Point MAC for DNN Training [[📄 paper](https://arxiv.org/abs/2404.14010)]
-  [**2023 | NeurIPS workshop | 🤖**] Training and Inference of Large Language Models Using 8-Bit Floating Point [[📄 paper](https://arxiv.org/abs/2309.17224)]
-  [**2023 | arXiv | 🤖**] FP8-LM: Training FP8 large language models [[📄 paper](https://arxiv.org/abs/2310.18313)]
-  [**2022 | ACT**] Campo: Cost-Aware Performance Optimization for Mixed-Precision Neural Network Training [[📄 paper](https://www.usenix.org/system/files/atc22-he.pdf)]
-  [**2022 | arXiv**] 8-Bit Numerical Formats for Deep Neural Networks [[📄 paper](https://arxiv.org/abs/2206.02915)] [[💻 code](https://github.com/chengchingwen/DLFP8Types.jl)]
-  [**2022 | arXiv | 🤖**] FP8 Formats for Deep Learning [[📄 paper](https://arxiv.org/abs/2209.05433)]
-  [**2022 | arXiv**] Accuracy Booster: Enabling 4-bit Fixed-point Arithmetic for DNN Training [[📄 paper](https://arxiv.org/abs/2211.10737)]
-  [**2021 | ICLR**] Neural Gradients Are Near-Lognormal: Improved Quantized and Sparse Training [[📄 paper](https://arxiv.org/abs/2006.08173)]
-  [**2020 | IJCAI**] Reducing Underflow in Mixed Precision Training by Gradient Scaling [[📄 paper](https://www.ijcai.org/proceedings/2020/404)]
-  [**2020 | ICLR**] Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks [[📄 paper](https://arxiv.org/abs/2001.05674)]
-  [**2020 | NeurIPS**] Ultra-Low Precision 4-bit Training of Deep Neural Networks [[📄 paper](https://papers.NeurIPS.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf)]
-  [**2019 | ICLR**] Accumulation Bit-Width Scaling For Ultra-Low Precision Training Of Deep Networks [[📄 paper](https://arxiv.org/abs/1901.06588)]
-  [**2019 | NeurIPS**] Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks [[📄 paper](https://proceedings.neurips.cc/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf)]
-  [**2019 | arXiv**] Mixed Precision Training With 8-bit Floating Point [[📄 paper](https://arxiv.org/abs/1905.12334)]
-  [**2019 | arXiv**] A Study of BFLOAT16 for Deep Learning Training [[📄 paper](https://arxiv.org/abs/1905.12322)]
-  [**2018 | ICLR**] Mixed Precision Training [[📄 paper](https://arxiv.org/abs/1710.03740)]
-  [**2018 | NeurIPS**] Training Deep Neural Networks with 8-bit Floating Point Numbers [[📄 paper](https://arxiv.org/abs/1812.08011)]
-  [**2018 | NeurIPS**] Training DNNs with Hybrid Block Floating Point [[📄 paper](https://arxiv.org/abs/1804.01526)]

## Customized Format-Based Methods

-  [**2025 | arXiv | 🤖**] Training LLMs with MXFP4 [[📄 paper](https://arxiv.org/abs/2502.20586)]
-  [**2025 | arXiv**] Oscillation-Reduced MXFP4 Training for Vision Transformers [[📄 paper](https://arxiv.org/abs/2502.20853)]
-  [**2023 | arXiv | 🤖**] Microscaling Data Formats for Deep Learning [[📄 paper](https://arxiv.org/abs/2310.10537)]
-  [**2022 | TCADICS**] Exploring the Potential of Low-bit Training of Convolutional Neural Networks [[📄 paper](https://arxiv.org/abs/2006.02804)]
-  [**2020 | TC**] Evaluations on Deep Neural Networks Training Using Posit Number System [[📄 paper](https://ieeexplore.ieee.org/document/9066876)]
-  [**2019 | CoNGA**] Posits: The Good, the Bad and the Ugly [[📄 paper](https://dl.acm.org/doi/abs/10.1145/3316279.3316285)]
-  [**2019 | JETCAS**] FloatSD: A New Weight Representation and Associated Update Method for Efficient Convolutional Neural Network Training [[📄 paper](https://ieeexplore.ieee.org/document/8693838)]
-  [**2017 | NeurIPS**] Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks [[📄 paper](https://arxiv.org/abs/1711.02213)]
-  [**2017 | SFI**] Beating Floating Point at its Own Game: Posit Arithmetic [[📄 paper](https://www.superfri.org/index.php/superfri/article/view/137)]

## Quantization-Aware Training Methods

-  [**2025 | arXiv | 🤖**] Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models? [[📄 paper](https://arxiv.org/abs/2502.11895)]
-  [**2025 | arXiv | 🤖**] QuEST: Stable Training of LLMs with 1-Bit Weights and Activations [[📄 paper](https://arxiv.org/abs/2502.05003)] [[💻 code](https://github.com/IST-DASLab/QuEST)]
-  [**2025 | arXiv** ] Stabilizing Quantization-Aware Training by Implicit-Regularization on Hessian Matrix [[📄 paper](https://arxiv.org/abs/2503.11159)]
-  [**2024 | ACL | 🤖**] LLM-QAT: Data-Free Quantization Aware Training for Large Language Models [[📄 paper](https://arxiv.org/abs/2305.17888)] [[💻 code](https://github.com/facebookresearch/LLM-QAT)]
-  [**2024 | ACL | 🤖**] BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation [[📄 paper](https://arxiv.org/abs/2402.10631)] [[💻 code](https://github.com/DD-DuDa/BitDistiller)]
-  [**2024 | arXiv | 🤖**] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models [[📄 paper](https://arxiv.org/abs/2407.11062)] [[💻 code](https://github.com/OpenGVLab/EfficientQAT)]
-  [**2024 | arXiv | 🤖**] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits [[📄 paper](https://arxiv.org/abs/2402.17764)] [[💻 code](https://github.com/microsoft/BitNet)]
-  [**2023 | arXiv | 🤖**] BitNet: Scaling 1-bit Transformers for Large Language Models [[📄 paper](https://arxiv.org/abs/2310.11453)] [[💻 code](https://github.com/microsoft/BitNet)]
