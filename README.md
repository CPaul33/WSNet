# ⚡A Wide & Shallow Network Tailored for Infrared Small Target Detection⚡

## Highlights
**What are the main findings?**
* Extremely Lightweight Model: WSNet achieves state-of-the-art efficiency with only
0.054 M parameters and 1.050 G FLOPs, making it the lightest model to date in the field
of Infrared Small Target Detection (IRSTD).
* Wide and Shallow Architecture: Contrary to conventional deep networks, WSNet adopts
a wide and shallow design, which is more suitable for infrared images that lack rich
semantic information. Excessive depth leads to performance degradation in IRSTD.
* Superior Performance-Speed Trade-off: WSNet achieves competitive detection accuracy
(e.g., highest IoU on SIRST, and best Pd on NUDT-SIRST) while offering the fastest
inference speed (up to 146 FPS on GPU, 30 FPS on CPU).

**What is the implication of the main finding?**
* Practical Deployment in Resource-Limited Environments: WSNet’s lightweight design
and real-time CPU compatibility enable its deployment in embedded systems, drones,
and portable infrared devices, where computational resources are limited but low-
latency detection is critical.
* Paradigm Shift in IRSTD Architecture Design: The success of a wide and shallow
network challenges the prevailing “deeper is better” assumption in deep learning
for IRSTD, encouraging the community to reconsider architecture tailoring based on
domain-specific characteristics.

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
