# ML-QEM
Machine Learning for Practical Quantum Error Mitigation [[arXiv](https://arxiv.org/abs/2309.17368)]

### Table of Contents

##### For Users

1.  [Installation](./docs/installation_guide.md)
2.  [Instructions for Use](#instructions-for-use)
3.  [Demos](./docs/demos)
4.  [Source Data](#source-data-for-figures)
5.  [Generic Decorator for Qiskit Estimator Primitives](#Generic-Decorator-for-Qiskit-Esitmator-Primitives)
6.  [Citation](#citation)
7.  [How to Give Feedback](#how-to-give-feedback)
8.  [Contribution Guidelines](#contribution-guidelines)
9.  [References and Acknowledgements](#references-and-acknowledgements)
10.  [License](#license)

---------------------------------------------------------------------------------------------------

### Overview

Quantum computers are actively competing to surpass classical supercomputers, but quantum errors remain their chief obstacle. The key to overcoming these on near-term devices has emerged through the field of quantum error mitigation, enabling improved accuracy at the cost of additional runtime. In practice, however, the success of mitigation is limited by a generally exponential overhead. Can classical machine learning address this challenge on today's quantum computers? Here, through both simulations and experiments on state-of-the-art quantum computers using up to 100 qubits, we demonstrate that machine learning for quantum error mitigation (ML-QEM) can drastically reduce overheads, maintain or even surpass the accuracy of conventional methods, and yield near noise-free results for quantum algorithms. We benchmark a variety of machine learning models -- linear regression, random forests, multi-layer perceptrons, and graph neural networks -- on diverse classes of quantum circuits, over increasingly complex device-noise profiles, under interpolation and extrapolation, and for small and large quantum circuits. These tests employ the popular digital zero-noise extrapolation method as an added reference. We further show how to scale ML-QEM to classically intractable quantum circuits by mimicking the results of traditional mitigation results, while significantly reducing overhead. Our results highlight the potential of classical machine learning for practical quantum computation.

---------------------------------------------------------------------------------------------------

### Instructions for Use
We provide two datasets and notebooks for demonstration. The [first demo](./docs/demos/demo1_rf_mimic_zne_100q_twirl.ipynb) shows our ML-QEM method mimicking digital ZNE + Pauli twirling on a 100Q TFIM Trotter circuit. The [second demo](./docs/demos/emo2_ising_4q_hardware_plot.ipynb) shows our ML-QEM mitigating the expectation values of a 4Q TFIM Trotter circuit on real hardware, outperforming digital ZNE.

Other notebooks (with prefix "hXX", e.g., h01_mbd.ipynb), python scripts, and datasets can be found in this [folder](./docs/tutorials). Specifically, circuit-level features and MLP models can be found in [mlp.py](./docs/tutorials/mlp.py), and GNN models can be found in [gnn.py](./docs/tutorials/gnn.py).

----------------------------------------------------------------------------------------------------

### Source Data for Figures
The [Excel sheets](https://github.com/qiskit-community/blackwater/blob/c36d50f2831979ebce66c3d1c5f4b34d24af2840/docs/paper_figures/ML-QEM%20Source%20data.xlsx) contain the source data for the Figures in our paper. Datasets can be found and loaded using the [script](./docs/paper_figures/plot.ipynb).

----------------------------------------------------------------------------------------------------

### Generic Decorator for Qiskit Esitmator Primitives
We also introduce a generic decorator for Qiskit Estimator primitives. This decorator converts the Estimator into a LearningEstimator, granting it the capability to perform a postprocessing step. During this step, we apply the mitigation model to the noisy expectation value to produce the final mitigated results. This manipulation maintains the object types, so we can utilize the LearningEstimator as a regular Estimator primitive within the entire Qiskit application and algorithm ecosystem without any further modifications.

The decorator requires a model as one of the arguments. We provide several default models, such as Scikit-learn, PyTorch, and TensorFlow, and examples of how to train and use them. The source code, documentation, and examples can be found in the main branch.

----------------------------------------------------------------------------------------------------

### Citation

Using the code, please cite the paper ([arXiv link here](https://arxiv.org/abs/2309.17368)):
```
@article{ML-QEM,
  title={Machine Learning for Practical Quantum Error Mitigation}, 
  author={Haoran Liao and Derek S. Wang and Iskandar Sitdikov and Ciro Salcedo and Alireza Seif and Zlatko K. Minev},
  year={2023},
  journal={arXiv:2309.17368},
}
```

----------------------------------------------------------------------------------------------------

### How to Give Feedback

We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/IceKhan13/blackwater/issues) in the repository


----------------------------------------------------------------------------------------------------

### Contribution Guidelines

For information on how to contribute to this project, please take a look at our [contribution guidelines](./CONTRIBUTING.md).


----------------------------------------------------------------------------------------------------

## References and Acknowledgements
[1] Qiskit https://qiskit.org/ \
[2] Qiskit-terra https://github.com/Qiskit/qiskit-terra \
[3] PyTorch https://pytorch.org/ \
[4] PyTorch geometric https://pytorch-geometric.readthedocs.io/en/latest/ \
[5] [Zlatko Minev](https://github.com/zlatko-minev) for :water_polo: :ocean:

----------------------------------------------------------------------------------------------------

### License
[Apache License 2.0](./LICENSE)
