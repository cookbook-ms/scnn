# simplicial_convolutional_neural_networks
This is the repository (updating from time to time) for the paper simplicial convolutional neural network (https://arxiv.org/abs/2110.02585). Some descriptions are detailed as below.  

# to run the code

 1. download the code and data
 2. run the below to train and test the SNN Ebli20 and our SCNN for a 10% and 20% missing value cases, which output the results in folders [`./experiments/output_10`](./experiments/output_10) and [`./experiments/output_20`](./experiments/output_20)
    
    ```sh
    cd simplicial_convolutional_neural_networks
    # this is to train and test the SNN Ebli20 for a 10% missing value case
    python .experiments/impute_citations1.py .data/s2_3_collaboration_complex ./experiments/output_10 150250 10 
    # this is to train and test the SNN Ebli20 for a 20% missing value case
    python .experiments/impute_citations1.py .data/s2_3_collaboration_complex ./experiments/output_20 150250 20
    # this is to train and test our SCNN for a 10% missing value case
    python .experiments/impute_citations2.py .data/s2_3_collaboration_complex ./experiments/output_10 150250 10 
    # this is to train and test our SCNN for a 20% missing value case
    python .experiments/impute_citations2.py .data/s2_3_collaboration_complex ./experiments/output_20 150250 20 
    # likewise for the cases with other percentages of missing values 
    ```

# file description 
1. [`.data`](.data) folder
- folder [`.data`](.data) contains the code (from s2_1_xxx to s2_7_xxx) to generate the data. See https://github.com/stefaniaebli/simplicial_neural_networks for a detailed description.
- Specifically, folder [`.data/s2_3_collaboration_complex`](.data/s2_3_collaboration_complex) includes 10 realizations of the data for each percentage of missing values, and the collaboration complex data, including the boundaries, cochains, and the Hodge Laplacians. 

2. [`.experiments`](.experiments) folder
- [`.experiments/impute_citations1.py`](.experiments/impute_citations1.py) is the SNN Ebli20 implementation, and [`.experiments/impute_citations2.py`](.experiments/impute_citations2.py) is our SCNN
- [`.experiments/accuracy_evaluation.py`](.experiments/accuracy_evaluation.py) is the evaluation code for the mean and the variance of the imputation results with both architectures, averaged over 10 realizations
- [`.experiments/output_analysis.py`](.experiments/output_analysis.py) draws the training loss and test loss for both architectures given a specific missing value 
- [`.experiments/output_10`](.experiments/output_10) folder stores the training and test output files for the missing value 10% for both architectures, and the other cases are stored in [`.experiments/output_xx`](.experiments/output_xx) folders

3. [`.scnn`](scnn) folder contains the simplicial convolutional layers for both architectures
- [`.scnn/scnn.py`](.scnn/scnn.py) is the convolutional layer for both architectures and we considered the Chebyshev implementation of the higher-order simplicial filter implementation as in [`.scnn/chebyshev.py`](.scnn/chebyshev.py)  

