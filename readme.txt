The first time the classifier runs, it needs to train. training files (reviews) should be placed in the positive and negative folder. 

training data(computations) are stored in a persistent file called training_data.pk in the base folder. This means that if you move the py file you also need to move the pickle file or create a new one with the same name. The file consists of the numbers used to calculate the probability that a review belongs to one particular class, and its purpose is to remove the need to train everytime the classifier runs.  

HOW TO RUN

1. Open python file NaiveBayesClassifier.py in a python enviroment.


2. At the end of the code (under the "CODE RUNNING" comment, after line 300 or so), an instance of our classifier is created.
   The test_all_reviews() function is used to make predictions on every text file in the test folder.  
   The test(filename) function makes a prediction on the parameter text-file. This file must be in the base folder of our classifier.   


