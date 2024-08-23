# An Improved Sample Selection Framework for Learning with Noisy Labels

<h5 align="center">

*Qian Zhang, Yi Zhu, Ming Yang, Ge Jin, Yingwen Zhu, Yanjun Lu, Yu Zou1, Qiu Chen*

[[PLOS ONE]](https://doi.org/10.1371/journal.pone.0309841)
[[License: MIT License]](https://github.com/LanXiaoPang613/SOS/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [An Improved Sample Selection Framework for Learning with Noisy Labels](https://doi.org/10.1371/journal.pone.0309841).

**Abstract**
Deep neural networks have powerful memory capabilities, yet they frequently suffer from overfitting to noisy labels, leading to a decline in classification and generalization performance. To address this issue, sample selection methods that filter out potentially clean labels have been proposed. However, there is a significant gap in size between the filtered, possibly clean subset and the unlabeled subset, which becomes particularly pronounced at high-noise rates. Consequently, this results in underutilizing label-free samples in sample selection methods, leaving room for performance improvement. This study introduces an enhanced sample selection framework with an oversampling strategy (SOS) to overcome this limitation. This framework leverages the valuable information contained in label-free instances to enhance model performance by combining an SOS with state-of-the-art sample selection methods. We validate the effectiveness of SOS through extensive experiments conducted on both synthetic noisy datasets and real-world datasets such as CIFAR, WebVision, and Clothing1M. The source code for SOS will be made available at https://github.com/LanXiaoPang613/SOS.

![SOS Framework](./framework.tiff)

[//]: # (<img src="./framework.tiff" alt="SOS Framework" style="margin-left: 10px; margin-right: 50px;"/>)

## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
python Train_cifar_sos.py --r 0.4 --noise_mode 'asym' --lambda_u 30 --data_path './data/cifar10/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
python Train_cifar_sos.py --r 0.5 --noise_mode 'sym' --lambda_u 30 --data_path './data/cifar10/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```


## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
xxxx
```

## Acknowledgement

* [UNICON](https://github.com/nazmul-karim170/UNICON-Noisy-Label): Inspiration for the basic framework.
* [LongReMix](https://github.com/filipe-research/LongReMix): Inspiration for the oversampling strategy.
