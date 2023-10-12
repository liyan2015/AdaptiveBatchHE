# AdaptiveBatchHE

<!-- start intro -->

This repository provides the implementation of the paper ["Adaptive Batch Homomorphic Encryption for Joint Federated Learning in Cross-Device Scenarios"](https://ieeexplore.ieee.org/document/10275042), which is published in IEEE INTERNET OF THINGS JOURNAL. In this paper, we propose an adaptive batch HE framework for cross-device FL, which determines cost-efficient and sufficiently secure encryption strategies for clients with heterogeneous data and system capabilities.

Our framework consists of the following three key components:
  
<p align="center">
<img src="fig/framework.jpg" align="center" width="85%"/>
</p>

<!-- end intro -->

## 1. Clustering of Clients based on Sparsity of CNNs

<!-- start sparsity -->

<p align="center">
<img src="fig/sparsity.jpg" align="center" width="40%"/>
</p>

The code in the folder [CNN Sparisty](https://github.com/liyan2015/AdaptiveBatchHE/tree/main/CNN%20Sparisty) is for determining the sparsity vectors of clients.

`federated_main.py` is the main function.

The input is the path of the dataset.

<!-- end sparsity -->

## 2. Selection of HE Key Size for Each Client based on Fuzzy Logic

<!-- start fuzzy -->

<p align="center">
<img src="fig/fuzzyworkflow.jpg" align="center" width="100%"/>
</p>

The code in the folder [fuzzy logic](https://github.com/liyan2015/AdaptiveBatchHE/tree/main/fuzzy%20logic) is for determining the HE key size of clients.

`fuzzy_logic_main.py` is the main function.

There are three inputs: `input_NS`, `input_TR`, and `input_CC`.

Their values are between 0 and 1.

<!-- end fuzzy -->

## 3. Accuracy-lossless Batch Encryption and Aggregation

<!-- start batch -->

<p align="center">
<img src="fig/batchencry_server_client.jpg" align="center" width="100%"/>
</p>

The code in the folder [batch encryption](https://github.com/liyan2015/AdaptiveBatchHE/tree/main/batch%20encryption) is for accuracy-lossless batch encryption and aggregation of model parameters for FL training.

`federated_experiment_main.py` is the main function.

The details of the three components are shown in the paper.

<!-- end sparsity -->

## Prerequisites

To run the code, it needs some libraies:

- Python >= 3.8
- Pytorch >= 1.10
- torchvision >= 0.11
- phe >= 1.5

Our environment is shown in the file, named `environment.yaml`.

## Citing

<!-- start citation -->

If you use this repository, please cite:
```bibtex
@article{han2023adaptiveBatchHE,
  title={Adaptive Batch Homomorphic Encryption for Joint Federated Learning in Cross-Device Scenarios},
  author={Han, Junhao and Yan, Li},
  journal={IEEE Internet of Things Journal},
  volume={Early Access},
  year={2023},
  publisher={IEEE}
}
```

<!-- end citation -->

