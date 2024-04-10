# ML-QEM

### Table of Contents

##### For Users

1.  [Installation](./docs/installation_guide.md)
2.  [Instructions for Use](#instruction-for-use)
3.  [Demos](./docs/demo)
4.  [Citation](#citation)
5.  [How to Give Feedback](#how-to-give-feedback)
6.  [Contribution Guidelines](#contribution-guidelines)
7.  [References and Acknowledgements](#references-and-acknowledgements)
8.  [License](#license)

---------------------------------------------------------------------------------------------------

### Instructions for Use
We provide two datasets and notebooks for demonstration. The [first demo](./docs/demos/demo1_rf_mimic_zne_100q_twirl.ipynb) shows our ML-QEM method mimicking digital ZNE + Pauli twirling on a 100Q TFIM Trotter circuit. The [second demo](./docs/demos/emo2_ising_4q_hardware_plot.ipynb) shows our ML-QEM mitigating the expectation values of a 4Q TFIM Trotter circuit on real hardware, outperforming digital ZNE.

Other notebooks (with prefix "hXX", e.g., h01_mbd.ipynb), python scripts, and datasets can be found in this [folder](./docs/tutorials).

Specifically, circuit-level features and MLP models can be found in [mlp.py](./docs/tutorials/mlp.py), and GNN models can be found in [gnn.py](./docs/tutorials/gnn.py).

----------------------------------------------------------------------------------------------------

### Citation

Using the code please consider citing:
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
