# Project-ACA
This program implements two ways to perform source separation using the techniques from Nonnegative Matrix Factorization as the course project of Audio Content Analysis.

By treating V, the power spectrum of the mixed input signal obtained by Short-Time Fourier Transform, as the product of a set of basis vectors W and some activation coefficients H, we manage to implement two different iterative algorithms to perform source separation based on Nonnegative Matrix Factorization. One of the algorithms uses W learnt from the training set to decompose the power spectrum of mixed input, while the other learns the vector when decomposing V. We combine audio samples from different genders to create a mixed input, apply the algorithms and compare the results with the original samples produced by single gender.

All the samples (both training and testing) are from TIMIT dataset. Average results (in dB): SDR-1.48, SIR-4.39, SAR-7.24 (separated speeches from male speakers using order 30).
