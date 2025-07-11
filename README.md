# ⚡A Wide & Shallow Network Tailored for Infrared Small Target Detection⚡

## Abstract
Designing a lightweight yet competitive model is always a challenging problem across the entire computer vision community—Infrared Small Target Detection (IRSTD) is no exception. Our proposed model, WSNet, is specifically designed to address this challenge. Rather than relying on deeper architectures or elaborate fusion mechanisms, WSNet achieves competitive performance while dramatically reducing both computational overhead and memory consumption. The core innovation of WSNet lies in its highly efficient network architecture and the Width Extension Module (WEM), which systematically expands the network’s width to enhance feature representation—providing a more effective and lightweight alternative to simply increasing network depth. Furthermore, we introduce a Channel-Spatial Hybrid Attention (CSHA) module, which effectively suppresses irrelevant noise and highlights salient features crucial for small target detection. To the best of our knowledge, WSNet is the most lightweight model currently available in the IRSTD domain, with only 0.054M parameters—hundreds of times fewer than state-of-the-art models—and a computational cost of just 1.050G FLOPs. Extensive experiments across multiple benchmark datasets demonstrate that WSNet not only matches the performance of leading methods but also achieves significantly faster inference speed, making it a viable real-time solution for embedded and resource-constrained applications.

## Contributions
* Our WSNet demonstrates that detecting infrared small targets does not require a very deep network architecture. With only 0.054M parameters and 1.050G FLOPs, WSNet stands out as the most lightweight model in the field of IRSTD.
* We propose a Width Extension Module (WEM) to enhance feature representation capabilities by expanding the model’s width.
* We present a Channel-Spatial Hybrid Attention (CSHA) module that enables the model to focus on useful details and filter out irrelevant noise, improving detection performance. 
* Extensive experiments on multiple benchmark datasets, including SIRST, NUDT-SIRST, and IRSTD-1K, demonstrate that WSNet achieves performance comparable to existing SOTA models, while offering significantly faster inference speed. Moreover, WSNet can be deployed directly on resource-constrained devices such as CPUs, where it still supports real-time detection—making it highly suitable for real-time applications and large-scale deployment in practical IRSTD scenarios.

## Datasets
We used the SIRST, NUDT-SIRST, IRSTD-1K for both training and test. 
Please first download the datasets via [Google Drive](https://drive.google.com/file/d/1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2/view?usp=sharing), and place the 3 datasets to the folder `./datasets/`.
* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── SIRST
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST.txt
  │    │    │    ├── test_SIRST.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── ...  
  ```
<be>

The original links of these datasets:
* SIRST &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* NUDT-SIRST &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* IRSTD-1K &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

## Commands
### Commands for converting to operation pathing
  ```
  $ cd /root/WSNet
  ```
  where '/root/' denotes your current path.
### Commands for training
* **Run **`train.py`** to perform network training in single GPU and multiple GPUs. Example for training [model_name] on [dataset_name] datasets:**
  ```
    $ python train.py --model_names WSNet --dataset_names SIRST
  ```
### Commands for test
* **Run **`test.py`** to perform network inference and evaluation. Example for test [model_name] on [dataset_name] datasets:**
  ```
  $ python test.py --model_names WSNet --dataset_names SIRST
  ```
### Commands for inference only with images
* **Run **`inference.py`** to inference only with images. Examples:**
  ```
  $ python inference.py --model_names WSNet --dataset_names SIRST
  ```
### Commands for parameters/FLOPs calculation
* **Run **`cal_params.py`** for parameters and FLOPs calculation. Examples:**
  ```
  $ python cal_params.py --model_names WSNet
  ```

## Acknowledgement
* This code and repository layout is highly borrowed from [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). Thanks to Xinyi Ying.
