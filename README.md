# CSCI-6360-Project02 : TransRegression, Perceptron, Neuralnet-3L and Neuralnet-XL in ScalaTion and Keras

This project covers implementations of TransRegression, Perceptron, Neuralnet-3L and Neuralnet-XL in Scalation and Keras , over 10 datasets downloaded from the UCI Machine Learning Repository. The datasets include:

1. Auto MPG   (Instances: 406, Attributes: 8)
2. Beijing PM2.5 Dataset   (Instances: 43824, Attributes: 13)
3. Concrete Compressive Strength Dataset   (Instances: 1030, Attributes: 9)
4. Real Estate Valuation Dataset   (Instances: 414, Attributes: 7)
5. Parkinson's Tele Monitoring   (Instances: 5875, Attributes: 26)
6. GPS Trajectories   (Instances: 163, Attributes: 15)
7. Appliances Energy Prediction   (Instances: 19735, Attributes: 29)
8. Combined Cycle Powerplant   (Instances: 9568, Attributes: 4)
9. CSM Dataset   (Instances: 217, Attributes: 12)
10. Naval Propulsion Dataset   (Instances: 11934, Attributes: 16)

This apart, the user also gets the option to run the models on their own datasets, by mentioning the correct path to that file in either of the environments. 

## Getting Started
These instructions describe the prerequisites and steps to get the project up and running.

### Prerequisites
This project has the following requirements for Scalation:
* Scala 2.12.8 +
* Java 8
* sbt_1.0 +

### Usage
After cloning the repository, to generate the  R<sup>2</sup> - R<sub>bar</sub><sup>2</sup> - R<sub>CV</sub><sup>2</sup> plots, one can navigate to the Scalation folder which contains the build.sbt file. Here, open the terminal and run the command:`sbt run`
This will build the Scalation project, and the user will get a prompt to select from the 10 datasets. The user can enter his choice by enterining a number between '1' to '10', each corresponding to the respective dataset. 

If the user wishes to use this project for their own dataset, they will have to enter '11' as their choice, which will prompt them to enter the path of their dataset (in CSV format). However, there are a few guidelines for the dataset that the user chooses to experiment on:
* it has to be a numeric dataset (data-encoding hasn't been implemented yet!)
* the first column of the dataset needs to be the 'Y' attribute.
If the user chooses to add their own dataset to the list, they will have to navigate one step back, to the `/data` directory and move the dataset there. The naming convention followed in the project is, "x.csv" where 'x' is the choice that the user inputs.

To check the Scala script, the user will have to navigate to `Scalation/src/main/scala/Perceptron/perceptron.scala`

The user can check the Python implementations saved as functions in the Jupyter Notebook, saved in the `python` sub-directory.

### Contributors
See [CONTRIBUTORS](https://github.com/Jayant1234/ANNs/blob/master/CONTRIBUTORS.md) file for more details.

### Authors
* [Aashish Yadavally](https://github.com/aashishyadavally)
* [Jayant Parashar](https://github.com/Jayant1234)
* [Akash Saurabh](https://github.com/ascoolakash)

