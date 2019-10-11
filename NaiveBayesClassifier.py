import os
import pickle
from timeit import default_timer as timer
import math
import pathlib
from collections import Counter


# variable that stored training data. This variable of initialized in the initialize(function). I
# t is a persistent variable that stored data between runs using pickle.
data = []

# this variable contains all the types of predictions made during our testing. The data can then be used
# to calculate accuracy, precision, recall, and F-measure.
evaluation_data = {}

negativeReviews = []  # a list containing names of all the text-files of negative training reviews.
positiveReviews = []  # a list containing names of all the text-files of positive training reviews.
negativeWords = []    # a list containing all words in all negative training reviews.
positiveWords = []    # a list containing all words in all positive training reviews.
stopWords = ['is', 'a', 'of', 'the', 'it', 'and', 'for', 'with', 'be', 'in',
             'this', 'an', 'to', 'has', 'that', 'she', 'he', 'it', 'i', 'was',
             'as', 'are', 'on', 'his', 'her', 'at', 'have']

# stopWords are words that will be ignored when collecting words for training pruposes.
# These are typically very commonly-used words.

# The following variables are used to calculate the probability of a review being of a certain class.
# These field are initialized in the initialize() function.

combinedWords = []        # The total amount of words in all scanned reviews.
UniqueWords = []          # The total amount of UNIQUE words in all scanned reviews.
PositiveAmount = 0        # The amount of positive reviews processed.
NegativeAmount = 0        # The amount of negative reviews processed.
positiveFrequencies = {}  # a dictionary containing the pairs: (positive word, frequency of this word)
negativeFrequencies = {}  # a dictionary containing the pairs: (negative word, frequency of this word)
PProb = 0.0               # The probability that the class is POSITIVE
NProb = 0.0               # The probability that the class is NEGATIVE

#  The function below collects all words of the parameter type POSITIVE or NEGATIVE
#  and puts them into a list depending on the class.

def train(classType):
    print("\n")
    print("TRAINING THE CLASSIFIER")
    print("\n")
    global positiveReviews
    global positiveWords
    global negativeReviews
    global negativeWords

    if classType == "positive":
        reviews = positiveReviews
        wordlist = positiveWords
    if classType == "negative":
        reviews = negativeReviews
        wordlist = negativeWords
    counter = 0
    for filename in os.listdir(classType+'/'):
        try:
            if filename.endswith(".txt"):
                reviews.append(str(filename))
                counter += 1
                with open(classType+'/' + filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        for word in line.split():
                            if word.lower() not in stopWords:
                                wordlist.append(word)
        except Exception as e:
            raise e
            print("No files found!")
    print('In total', counter, classType, 'reviews were scanned')
    return wordlist


# This function sets the numbers for how many positive/negative and total unique words there are.
# It also finds the probability for each class.
# Finally it finds the frequencies of all positive and negative words and stores them in dictionaries.
# These numbers are used when calculating the probability of the class of review in the function class_probabilities
# Some of the variables are being divided by 10000 to avoid having the probabilities being either too small or too big.
# This division does not seem to affect the accuracy at all, which is good.
# It is just meant to make us avoid the inf or 0.0 probability results.

def initialize():
    global data
    global combinedWords
    global PositiveAmount
    global NegativeAmount
    global UniqueWords
    global PProb
    global NProb
    global positiveFrequencies
    global negativeFrequencies

    training_data = 'training_data.pk'

    with open(training_data, 'rb') as fi:
        try:
            data = pickle.load(fi)
        except EOFError:
            print("empty training data")


    print("Do we have training data?", has_data())

    # Do we have the required data to make predictions? If not we have to train out classifier by collecting words
    # from the training reviews and deriving data from this.
    if not has_data():
        train('positive')
        train('negative')
        # adds the required data to a persistent data variable.
        data.append(positiveWords + negativeWords)
        data.append((len(positiveWords))/10000)
        data.append((len(negativeWords))/10000)
        data.append((len(set(combinedWords)))/10000)
        data.append(len(negativeReviews) / len(negativeReviews + positiveReviews))
        data.append(len(positiveReviews) / len(negativeReviews + positiveReviews))
        data.append(Counter(positiveWords))
        data.append(Counter(negativeWords))
        # make the data variable persistent
        with open(training_data, 'wb') as fi:
            # dump your data into the file
            pickle.dump(data, fi)

    # if we DO have the data variable, we can simply use its content to get the required data.
    combinedWords = data[0]
    PositiveAmount = data[1]
    NegativeAmount = data[2]
    UniqueWords = data[3]
    NProb = data[4]
    PProb = data[5]
    positiveFrequencies = data[6]
    negativeFrequencies = data[7]


def has_data():
    global data
    if data:
        return True
    else:
        return False


initialize()


class NaiveBayesClassifier():
    # This function is responsible for calculating the actual probability that will be used
    # to determine which class the file will be predicted to belong to.
    def class_probabilities(self, filename, type, evaulation):
        global stopWords
        if evaulation == True:
            filename = "test/" + filename
        if type == "positive":
            amount = PositiveAmount
            ClassProb = PProb
            frequency = positiveFrequencies
        elif type == "negative":
            amount = NegativeAmount
            ClassProb = NProb
            frequency = negativeFrequencies
        result = 1
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                for word in line.split():
                    if word.lower() not in stopWords:
                        result *= ((frequency[word]) + 1) / (amount + UniqueWords)
        return result * ClassProb


    # This function assigns a class to a review based on the greatest of the two probabilities: positive and negative.
    # It also uses the test_accuracy function to find out if our prediction were correct or not.
    # The parameter evaluation excepts a boolean value that

    def max_prob(self, filename, evaluation):
        positive = self.class_probabilities(filename, "positive", evaluation)
        negative = self.class_probabilities(filename, "negative", evaluation)
        if positive > negative:
            # print('This review is POSITIVE. The probability was:', positive)
            decision = "positive"
        elif positive < negative:
            # print('This review is NEGATIVE. The probability was:', negative)
            decision = "negative"
        else:
            print("The probabilities were exactly the same!. "
                  "This is likely because the probabilities ended up too small for Python to handle. "
                  "This review will be discarded from test results.")
            return -1
        if evaluation == False:
            print("The prediction of", filename, "is", decision)
            return 0
        if not ((positive == negative) and (evaluation == True)):
            testResult = self.test_accuracy(filename, decision)
            if testResult == 0:
                print("Test Result: True ", decision +"\n")
                return 0
            elif testResult == 1:
                print("Test Result: False ", decision +"\n")
                return 1
            elif testResult == 2:
                print("Test Result: True ", decision +"\n")
                return 2
            elif testResult == 3:
                print("Test Result: False ", decision +"\n")
                return 3

    #  This function tests if the class-prediction assigned corresponds to the actual class of the review.
    #  It does so my comparing the prediction to the numbers at the end of the text files.

    def test_accuracy(self, filename, decision):
        if decision == "positive":
            if(filename.endswith("_10.txt") or filename.endswith("_9.txt")
             or filename.endswith("_8.txt") or filename.endswith("_7.txt")):
                return 0
            else:
                return 1
        elif decision == "negative":
            if(filename.endswith("_1.txt") or filename.endswith("_2.txt")
            or filename.endswith("_3.txt") or filename.endswith("_4.txt")):
                return 2
            else:
                return 3

    # This function tests all reviews in the test folder.
    # It also keeps track of how many predictions were correct, and it finds the overall accuracy percentage of our test.

    def test_all_reviews(self):
        # This timer will count how many seconds it takes to execute the entire program. It ends at the end of the document.
        start = timer()

        global evaluation_data
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        discarded = 0

        for filename in os.listdir('test/'):
            try:
                if filename.endswith(".txt"):
                        print("Now scanning:", filename, "Number in queue:", truePositive + trueNegative + falsePositive + falseNegative)
                        result = self.max_prob(filename, True)
                        if result == 0:
                            truePositive += 1
                        elif result == 1:
                            falsePositive += 1
                        elif result == 2:
                            trueNegative += 1
                        elif result == 3:
                            falseNegative += 1
                        else:
                            discarded += 1
                        print("\n")
            except Exception as e:
                raise e
                print("No files found!")

        evaluation_data["truePositive"] = truePositive
        evaluation_data["trueNegative"] = trueNegative
        evaluation_data["falsePositive"] = falsePositive
        evaluation_data["falseNegative"] = falseNegative
        evaluation_data["discarded"] = discarded

        end = timer()

        self.evaluate()
        print("\n")
        print("Total Time elapsed when testing:", int(end - start), "seconds.")

    def evaluate(self):
        global evaluation_data
        e = evaluation_data
        discarded = e["discarded"]
        totalPredictions = e["truePositive"] + e["trueNegative"] + e["falsePositive"] + e["falseNegative"]
        positivePrecision = (int(e["truePositive"] / (e["truePositive"] + e["falsePositive"]) * 100))
        negativePrecision = (int(e["trueNegative"] / (e["trueNegative"] + e["falseNegative"]) * 100))
        positiveRecall = (int(e["truePositive"] / (e["truePositive"] + e["falseNegative"]) * 100))
        negativeRecall = (int(e["trueNegative"] / (e["trueNegative"] + e["falsePositive"]) * 100))
        positiveFMeasure = int((2 * positivePrecision * positiveRecall) / (positivePrecision + positiveRecall))
        negativeFMeasure = int((2 * negativePrecision * negativeRecall) / (negativePrecision + negativeRecall))
        accuracy = (int(((e["truePositive"] + e["trueNegative"]) / totalPredictions) * 100))

        print(totalPredictions + discarded, "reviews were scanned in total.")
        print("Of these,", discarded, "reviews were discarded because of class probabilities were inf or 0.0")
        print('This means', totalPredictions, 'reviews were actually tested')
        print(e["truePositive"] + e["trueNegative"], "of them were predicted correctly.")
        print(e["falsePositive"] + e["falseNegative"], "of them were predicted incorrectly.")
        print("Total positive precision was: ", positivePrecision, "percent")
        print("Total negative precision was: ", negativePrecision, "percent")
        print("Total positive recall was: ", positiveRecall, "percent")
        print("Total negative recall was: ", negativeRecall, "percent")
        print("Positive F-measure: ", positiveFMeasure, "percent")
        print("Negative F-measure: ", negativeFMeasure, "percent")
        print("Overall prediction Accuracy was:", accuracy, 'percent.')



    def test(self, filename):
        try:
            for file in os.listdir():
                if file.endswith(".txt"):
                    if file == filename:
                        self.max_prob(filename, False)
                        break
        except Exception as e:
            raise e
            print("No files found!")


    def hello(self):
        print("hello")


# CODE RUNNING
clf = NaiveBayesClassifier()

# TEST ALL REVIEWS IN TEST FOLDER
clf.test_all_reviews()

# TEST ONE REVIEW IN THE BASE FOLDER. CHANGE PARAMETER TO TEST ON A DIFFERENT FILE
clf.test("test.txt")





