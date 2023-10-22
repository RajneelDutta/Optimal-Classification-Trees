# Optimal-Classification-Trees

This repository contains code and results for my dissertation project on facial image classification using Convolutional Neural Networks (CNNs) and Optimal Classification Trees (OCTs). The goal is to leverage the feature learning capabilities of CNNs and the interpretability of tree-based models like OCTs to achieve both high accuracy and model transparency.
Description

The repository includes:

- Python code to extract features from facial images using a pre-trained VGG16 CNN
- Implementation of Optimal Classification Trees (OCT) for optimizing tree-based classification
- Hyperparameter tuning experiments on OCT depth, regularization, etc.
- Novel bagged ensemble OCT models for improved performance
- Benchmark datasets: Extended Yale B, UCI data
- Comparative analysis with other classifiers like Random Forests

The dissertation aims to address the tradeoff between accuracy and interpretability in facial image classification through innovative integration of deep learning for feature extraction and tree-based optimization techniques for transparent decisions.

## Getting Started

The code requires Python and common data science libraries like Numpy, Pandas, Scikit-Learn, etc. Gurobi is used for solving the OCT optimization problems.

## Results

Key results include:

- CNN-OCT pipeline achieves reasonable accuracy, though limited by compute constraints
- Bagged OCT models demonstrate accuracy comparable to Random Forests
- Analysis provides insights into balancing performance vs interpretability

## Reference

Rajneel Dutta, Bridging the Accuracy-Interpretability Divide: A study on novel frameworks for Precise and Interpretable Classification, Lancaster University, 2023
Contributing

Contributions and improvements to the code are welcome! Please open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

I would like to thank my supervisor Nicos Pavlidis for his guidance and support.

 
