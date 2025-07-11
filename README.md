# ⚡Wide and Deep Learning for Infrared Small Target Detection⚡

## Abstract
Recent advancements in infrared small target detection (IRSTD) have predominantly relied on increasing computational resources and memory, posing significant challenges for real-world, large-scale deployment. To overcome these challenges, we propose WDNet, a lightweight IRSTD model that strikes an optimal balance between network's width and depth, effectively enhancing detection performance while minimizing computational cost and resource consumption. Unlike traditional model structure design strategies, WDNet introduces a Width Extension Module (WEM), which enhances the model's feature representation capability by expanding the network’s width in a structured manner, rather than simply increasing the network's depth. Additionally, we customize a Channel-Spatial Hybrid Attention (CSHA) module that enables the model to filter out noisy, irrelevant information while focusing on important features. To the best of our knowledge, WDNet is the most lightweight model in the field of IRSTD, with only 0.054M parameters—over a hundred times smaller than the SOTA models—and just 1.050G FLOPs. Extensive experiments conducted on multiple datasets demonstrate that our approach not only matches but often exceeds the performance of the SOTA models. Moreover, WDNet's inference speed is several times faster than the previous best model, making it a real-time detector suitable for embedded applications. This efficiency, combined with high detection performance, makes WDNet a compelling solution for practical, large-scale IRSTD deployment in real-world scenarios.

## Contributions
* We introduce WDNet, an extremely lightweight IRSTD model, with only 0.054M parameters and 1.050G FLOPs. WDNet is significantly smaller than existing SOTA models, offering a dramatic reduction in computational cost.
* We propose a Width Extension Module (WEM) to expand the model’s width, enhancing feature representation capabilities through different receptive field size.
* We present a Channel-Spatial Hybrid Attention (CSHA) module that enables the model to focus on useful details and filter out irrelevant noise, improving detection performance.
* Extensive experiments on multiple datasets, such as SIRST, NUDT-SIRST, and IRSTD-1K, demonstrate that WDNet not only matches but often surpasses the performance of existing SOTA models. Furthermore, WDNet’s inference time is among the fastest, making it suitable for real-time detection and large-scale deployment in real-world IRSTD applications.

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
  $ cd /root/WDNet
  ```
  where '/root/' denotes your current path.
### Commands for training
* **Run **`train.py`** to perform network training in single GPU and multiple GPUs. Example for training [model_name] on [dataset_name] datasets:**
  ```
    $ python train.py --model_names WDNet --dataset_names SIRST
  ```
### Commands for test
* **Run **`test.py`** to perform network inference and evaluation. Example for test [model_name] on [dataset_name] datasets:**
  ```
  $ python test.py --model_names WDNet --dataset_names SIRST
  ```
### Commands for inference only with images
* **Run **`inference.py`** to inference only with images. Examples:**
  ```
  $ python inference.py --model_names WDNet --dataset_names SIRST
  ```
### Commands for parameters/FLOPs calculation
* **Run **`cal_params.py`** for parameters and FLOPs calculation. Examples:**
  ```
  $ python cal_params.py --model_names WDNet
  ```

## Acknowledgement
* This code and repository layout is highly borrowed from [IRSTD-Toolbox](https://github.com/XinyiYing/BasicIRSTD). Thanks to Xinyi Ying.
