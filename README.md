AdaptiveBatchHE

This repository is for our IEEE IOT-J 2023 paper "Adaptive Batch Homomorphic Encryption for Joint Federated Learning in Cross-Device Scenarios". 

Our framework consists of the following three key components.

<div align=center>

![framework](./fig/framework.PNG)

</div>

## Clustering of Clients based on Sparsity of CNNs

The code in the folder cnn sparsity is used to obtain the sparse vectors of the client.

<div align=center>

![sparsity](./fig/sparsity.PNG)

</div>

## Selection of HE Key Size for Each Client based on Fuzzy Logic

The code in the folder fuzzy logic is used to obtain the key size of the client.

![fuzzy](./fig/fuzzy.PNG)

## Accuracy-lossless Batch Encryption and Aggregation

The code in the folder batch encryption is used to train model.

![batch](./fig/batch.PNG)

The details are shown in the paper.





