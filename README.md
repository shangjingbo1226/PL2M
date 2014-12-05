A Parallel and Efficient Algorithm for Learning to Match (PL2M)
====

Build
----

Simply type "make" in the terminal line or follow the commands in the Makefile.

Usage
----

pl2m_train is used to train models based on train/test files and user/item feature matrix files, which needs 5 parameters:

     [usage] <configure-file> <user-feature-matrix> <item-feature-matrix> <train> <test> <model-output-prefix>

pl2m_infer is used to make predictions based on trained models, which needs 3 parameters:

     [usage] <test> <particular-model> <prediction-output-file>

For more details, please check the example.

Configure File
----

Please check the "examples/pcf.conf" for details.

Format of the Matrix File
----
Spare Matrix for user features and item features.

For each line, first come with the id of user/item. Then followed by the number of non-zero features. Features are described by their indices and values, separated by a colon. Here is an example for user 123, who have 5 non-zero features in total.

    123 5 123:1 1:0.5 10:0.5 11:0.5 20:0.5
    
For more details, please check the example.

Format of Train/Test file
----
Classical 3/4 columns, where weight of this instance is optional (1 default). However, in one train/test files, the columns in each line should be same.

    user item rate [weight]
    
For more details, please check the example.

Example
----

1. run the buildFeatMat.py in the folder "data".

        python buildFeatMat.py

which will generate 4 four files in the folder "data", which are the train/test files and user/item feature matrix files.

2. get into the folder "example", create a folder "models".

        mkdir models
    
3. run the predefined script.

        ./run.sh

which includes both training and testing procedures.

Reference
----

If you are using this toolkit for some research, please cite the following papers.

[1]  Shang, J., Chen, T., Li, H., Lu, Z., & Yu, Y. (2014). A Parallel and Efficient Algorithm for Learning to Match. In Data Mining (ICDM), 2014 IEEE 14th International Conference. IEEE.

There is also a long version of this paper available on arXiv.


