# Cross-Domain Inductive Applications with Unsupervised (Dynamic) Graph Neural Networks(GNN): Leveraging Siamese GNN and Energy-Based PMI Optimization
This project aims to develope pretrained GNN model which learn parameters in unsupervised settings on a smaller graphs and can be used to generate node embedding for any scale. 

### K. Abbas, S. Dong, A. Abbasi et al., Cross-Domain Inductive Applications with Unsupervised (Dynamic) Graph Neural Networks(GNN): Leveraging Siamese GNN and Energy-Based PMI Optimization, Physica D (2025), doi: https://doi.org/10.1016/j.physd.2025.134632.

## Abstract: 
The existing body of work in graph embedding has mainly focused on Graph Neural Network (GNN) models that work on transduction settings, meaning the model can be used only for the graph on which it has been trained. This limitation restricts the applicability of GNN models, as training them on large graphs is a very expensive process. Furthermore, there is a research gap in applying these models to cross-domain inductive prediction, where training on a single smaller graph and using it on larger or different domain graphs is desired. Therefore, considering these problems, this study aims to propose a novel GNN model that not only works on inductively generating node representations within the same domain but also across different domains.
To achieve this, we have considered state-of-the-art Graph Neural Networks (GNNs), such as Graph Convolutional Networks, Graph Attention Networks, Graph Isomorphism Networks, and Position-Aware Graph Neural Networks. Additionally, to learn parameters from smaller graphs, we have developed a Siamese Graph Neural Network and trained it using a novel loss function specifically designed for Graph Siamese Neural Networks. Furthermore, to address real-world sparse graphs, we have provided TensorFlow code that operates efficiently on sparse graphs, making it spatially optimized for larger-scale graph applications. 
To evaluate the performance of our model, we have used five real-world dynamic graphs. Additionally, we have trained our model on a smaller dataset in an unsupervised manner. The pretrained model is then used to generate inter- and intra-domain graph node representations. Our framework is robust, as any state-of-the-art GNN method can be utilized to exploit the Siamese neural network framework and learn parameters based on the proposed energy-based cost.

**Keywords**: DynamicNodeEmbedding, GraphRepresentationLearning, Unsupervised node representation learning, Pre-trained GNN.

## Please cite the following paper if you use this code. 

**Requirements**
pip install networkx==2.3  numpy==1.19.5 tqdm==4.40.0 pandas==1.3.2 Keras==2.3.1 matplotlib==3.5.2  node2vec==0.4.3 sklearn==0.0 qc-procrustes  scipy==1.7.3 prettytable


python main.py

run with default parameter..see main.py for cmd arguments.
