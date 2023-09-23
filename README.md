# Purpose
Pose classification with EMG signals is helpful for a lot of prosthetic applications. This project aims to use data from a single user to correctly classify hand gestures. Because EMG signals are noisy and dependent on the user, how sensors are placed on a user, and the environment, being able to create a model from a small amount of data that can accurately classify hand positions can be helpful for devices and systems that need to be personalized to a user. This project will explore generating data with GANs to see if it can improve classification accuracy. This will also experiment with using different number of input data channels as the number of electrodes used to collect data can vary depending on use case.

# Dataset
Using [putEMG](https://biolab.put.poznan.pl/putemg-dataset/) from [Poznan University of Technology Biomedical Engineering and Biocybernetics Team](https://biolab.put.poznan.pl/) open source dataset downloaded from [here](https://github.com/biolab-put/putemg-downloader)

# Methods
Using 6 features extrated from time domain to classify hand gesture class. Comparing this to using GANs to generate sythetic data to see how accuracy, sensitivity, and specificity vary. Both will be benchmarked against KNN, SVM, logistic classifier, and Neural Network. Similar methods proposed by [Electromyogram-Based Classification of Hand and Finger Gestures Using Artificial Neural Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8749583/pdf/sensors-22-00225.pdf)

<!-- # Real Features
Support vector, logistic, and random forest classifiers were used to classify hand gestures from features extracted from EMG signals. Accuracy is as follows: 

|    |   SVM |   LOR |   RF |\n|---:|------:|------:|-----:|\n|  0 |     |    |  |  -->

# Synthetic Features

# Running the Code
1. Download PUT-EMG data to "Data-HDF5" folder
2. Get features by running `bash .sh` # TODO
3. Benchmark for real data with `bash benchmark_all.sh real` # TODO
4. Benchmark for synth and real data with `bash benchmark_all.sh synth` # TODO
5. 

# Things to try later
- Explore using Wavelet transforms to extract features
    -  This could find features that better indicate hand classification but could introduce a lot of latency into real world systems because transforms are pretty computationally heavy
-

