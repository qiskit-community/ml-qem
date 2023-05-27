---
title: 'BlackWater: library.rst for solving quantum computing problems using machine learning'
tags:
  - quantum computing
  - qiskit
  - machine learning
authors:
  - name: Iskandar Sitdikov
    orcid: 0000-0002-6809-8943
    corresponding: true 
    affiliation: 1
affiliations:
 - name: IBM Quantum, T.J. Watson Research Center, Yorktown Heights, NY 10598, USA
   index: 1
date: 22 May 2023
bibliography: paper.bib

---

# Summary

BlackWater is an open-source machine learning library designed to address 
the challenges of solving quantum computing problems. By integrating 
cutting-edge machine learning techniques with quantum computing abstractions, 
BlackWater provides a comprehensive set of tools for researchers 
and practitioners in the field. This library enables the application of 
classical machine learning algorithms to quantum computing, bridging 
the gap between the two domains and accelerating the development of 
quantum applications.

# Statement of need

The field of quantum computing presents an array of complex challenges, 
ranging from efficient unitary synthesis to error mitigation and correction. 
These challenges require innovative approaches to unlock the full 
potential of quantum computers. Recent advances in machine learning 
(ML) have showcased their ability to tackle some of the toughest problems 
across various domains. Recognizing this potential, there is a pressing 
need to leverage classical ML techniques in the field of quantum computing.

# Contents

To enable the application of classical machine learning (ML) algorithms to 
quantum computing problems, we have implemented a general set of 
encoders and decoders in our project. These components are designed to 
bridge the gap between quantum computing abstractions and ML-compatible 
formats, such as numpy arrays.

The encoders extract relevant features and characteristics from quantum 
objects, such as quantum states, circuits, or gates, and encode them 
into numerical formats suitable for ML algorithms. 

Additionally, we provide a library of pre-built models for classical 
ML and deep learning, along with dedicated environments for 
reinforcement learning (RL) algorithms. 

# Acknowledgements

...

# References