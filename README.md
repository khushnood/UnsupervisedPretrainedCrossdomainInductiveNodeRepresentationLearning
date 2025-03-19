# Cross-Domain Inductive Applications with Unsupervised (Dynamic) Graph Neural Networks(GNN): Leveraging Siamese GNN and Energy-Based PMI Optimization
This project aims to develope pretrained GNN model which learn parameters in unsupervised settings on a smaller graphs and can be used to generate node embedding for any scale. 


## Abstract: 
The existing body of work in graph embedding has mainly focused on Graph Neural Network (GNN) models that work on transduction settings, meaning the model can be used only for the graph on which it has been trained. This limitation restricts the applicability of GNN models, as training them on large graphs is a very expensive process. Furthermore, there is a research gap in applying these models to cross-domain inductive prediction, where training on a single smaller graph and using it on larger or different domain graphs is desired. Therefore, considering these problems, this study aims to propose a novel GNN model that not only works on inductively generating node representations within the same domain but also across different domains.
To achieve this, we have considered state-of-the-art Graph Neural Networks (GNNs), such as Graph Convolutional Networks, Graph Attention Networks, Graph Isomorphism Networks, and Position-Aware Graph Neural Networks. Additionally, to learn parameters from smaller graphs, we have developed a Siamese Graph Neural Network and trained it using a novel loss function specifically designed for Graph Siamese Neural Networks. Furthermore, to address real-world sparse graphs, we have provided TensorFlow code that operates efficiently on sparse graphs, making it spatially optimized for larger-scale graph applications. 
To evaluate the performance of our model, we have used five real-world dynamic graphs. Additionally, we have trained our model on a smaller dataset in an unsupervised manner. The pretrained model is then used to generate inter- and intra-domain graph node representations. Our framework is robust, as any state-of-the-art GNN method can be utilized to exploit the Siamese neural network framework and learn parameters based on the proposed energy-based cost.

**Keywords**: DynamicNodeEmbedding, GraphRepresentationLearning, Unsupervised node representation learning, Pre-trained GNN.

## Please cite the following paper if you use this code. 
#### K. Abbas, S. Dong, A. Abbasi et al., Cross-Domain Inductive Applications with Unsupervised (Dynamic) Graph Neural Networks(GNN): Leveraging Siamese GNN and Energy-Based PMI Optimization, Physica D (2025), doi: https://doi.org/10.1016/j.physd.2025.134632.

## Requirements  
- Python >= 3.7  
- TensorFlow 2.5 compatible  

### Install Dependencies  
Run the following command to install the required packages:  
```bash
pip install scipy==1.7.3 networkx==2.3 tqdm==4.40.0 pandas==1.3.2 Keras==2.3.1 matplotlib==3.5.2 torch==1.9.0 node2vec==0.4.3 sklearn==0.0 prettytable qc-procrustes
```

## Usage  

### Train the Model from Scratch  
Use the following command to train the model without a pre-trained checkpoint:  
```bash
python main.py --datasets 'THCN' --pretrained 0 --model_path 'pretrainedModel_sparse/pretrainedModel.ckpt'
```

### Train Using a Pretrained Model  
To fine-tune the model using a pre-trained checkpoint, run:  
```bash
python main.py --datasets 'THCN' --pretrained 1 --model_path 'pretrainedModel_sparse/pretrainedModel.ckpt'
```

### Transfer Learning  
For transfer learning, use the following command:  
```bash
python main.py --datasets 'THCN' --pretrained transfer --model_path 'pretrainedModel_sparse/pretrainedModel.ckpt'
```

## Custom Datasets  
To use your own dataset, refer to the function `load_dataset()` in:  
```python
loader/dataset_loader.py
```

## Parallel Execution  
To speed up embedding generation, you can enable parallel execution:  
```bash
python main.py --datasets 'THCN' --pretrained 0 --model_path 'pretrainedModel_sparse/pretrainedModel.ckpt' --ncpu 10
```
**Note:** Ensure that your system has enough RAM to handle parallel processing across different snapshots.
