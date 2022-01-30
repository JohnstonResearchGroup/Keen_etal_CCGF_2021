# Keen_etal_CCGF_2021
Source code and data files for the manuscript "Hybrid quantum-classical approach for coupled-cluster Green's function theory."

Reference: arXiv:2104.06981.

Title: Hybrid quantum-classical approach for coupled-cluster Green's function theory

Authors: Trevor Keen, Bo Peng, Karol Kowalski, Pavel Lougovski, and Steven Johnston. 

Abstract: The three key elements of a quantum simulation are state preparation, time evolution, and measurement. While the com- plexity scaling of dynamics and measurements are well known, many state preparation methods are strongly system-dependent and require prior knowledge of the system’s eigenvalue spectrum. Here, we report on a quantum-classical implementation of the coupled-cluster Green’s function (CCGF) method, which replaces explicit ground state preparation with the task of applying unitary operators to a simple product state. While our approach is broadly applicable to a wide range of models, we demonstrate it here for the Anderson impurity model (AIM). The method requires a number of T gates that grows as $O(N^5)$ per time step to calculate the impurity Green’s function in the time domain, where N is the total number of energy levels in the AIM. For comparison, a classical CCGF calculation of the same order would require computational resources that grow as $O(N^6)$ per time step.
