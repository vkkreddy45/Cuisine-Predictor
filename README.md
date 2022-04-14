# cs5293sp22-project2

### Text Analytics Project 2
### Author: Vudumula Kranthi Kumar Reddy


# About

In this Project the main goal is to create applications that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. Concepts that I have used for implementing this project are text analytics and Knowledge of python.


### Table of Contents

**[Required Packages](#required-packages)**<br>
**[Function Description](#function-description)**<br>
**[How to run the project](#how-to-run-the-project)**<br>
**[How to run the test cases](#how-to-run-the-test-cases)**<br>
**[Bugs and Assumptions](#bugs-and-assumptions)**<br>
**[References](#references)**<br>


# Required Packages

The below mentioned are the packages used for this Project:
* json
* nltk
* os
* pandas
* sklearn


# Function Description

1. prediction.py:

    This File contains all the functions that has been used for predicting the cuisine type.
    
    a. Readdata():
    
        This function is used to read the json files.
        
    b. preprocess():
    
        This function is used to clean the read data using normalaization techniques like stemming, lemataization etc. In this function firstly we append all the ingredients data into a list from a data frame and then we convert all the characters to lower. Then we check if there are any digits and replace them as empty and then using nltk we tokenize the data. Finally, we check if there are any stop words and then remove then by not appending them into a list.
        
     c. vectorization(): 
     
         This function is used to vectorize all the data that is returned from the previous function. In this function we store the first index vectorized data in to one variable as this is the vectorized input that we are using as input and then we store the remaining data into other variable which we using the next function.
         
     d. model():
     
         This function is used to build a model, create a pipeline and for train_test_split. In this function we first build the model with SVC() and create a pipeline. Then we train, test and split the data. Then we fit the data into the created pipeline and find the cuisine.
         
     e. TopnRecipe():
     
         This function is used to find the closest cuisines. In this function we find the top n cuisines with the cosine similarity score.
         
     f. print_final_output():
     
        This function is used to print the output in json format. For that purpose in this function we use json library and dump the output data into it.
        
     g. write_to_file():
     
         In this Function we are trying to write the entire output into a file by creating a new file.
         
     h. start():
     
         This function is used to run all the defined functions at once where it takes the required arguments form the command line.
                 
2. project2.py:

    This is the main function were the argument parsing is done.
    
     a. main(args): 
  
         In this Function, we check whether the passed flags from the command line are present or not and. If the flags are true then we fetch the start function in the prediction file. This start function is the one where the entire prediction process is done.
                   
3. test/test_data.py:

    This file contains all the test cases for the above mentioned functions, to check whether every function is running correctly or not.
    
    a. test_readdata():
    
        In this Function, I am trying to check whether the inputed file is being passed or None.
        
    b. test_preprocessing():
    
        In this Function, I am checking whether the result is a list or not and checking it is not None.
     
    c. test_vector():
    
        In this Function, I am trying to check that the count of vectors is greater than 0.
        
    d. test_model():
    
        In this Function, I am trying to check that the result is a dictionary.
      
    e. test_topnrecip():
    
        In this Function, I am checking that the result is greater than 0 and not equal to None.
         
* scores.json:

    The predicted output is written into this json file.
    
* data.json:

    This is used as sample data for testing the test cases.
    
* yummly.json:

    This is the given dataset for implementing this project.
    
    
# How to run the project

```
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies" 
```

# How to run the test cases

```
pipenv run python -m pytest
```


# Bugs and Assumptions

* When checked for this model accuracy rate it was 78% (approx.). By this we can say that one may/may not get the correct prediction all the times.
* One might face problem when the passed file is not of .json extension.
* One might get errors when the normalization (cleaning of data) is not done correctly.
* One might have trouble when the normalized list is not vectorized properly.
* one may get errors when the packages are not installed properly.
* one might face problem when the passsed x and y values of train_test_split vary with respect to their shapes or row, column size.        


# References

* [Replacing Digits](https://stackoverflow.com/questions/19084443/replacing-digits-with-str-replace)
* [Sklearn Libraries Examples](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)
* [Argument Parsing](https://stackoverflow.com/questions/25778813/how-to-read-multiple-command-line-parameters-with-same-flag-in-python)
* [Label Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [Cosine Similarity](https://gist.github.com/pgolding/fdf74a3e8e797fad0391befd5a906ddb)
* [Json dump Format](https://stackoverflow.com/questions/37398301/json-dumps-format-python)

For Detailed Information about the references check out Collaborators File. 
