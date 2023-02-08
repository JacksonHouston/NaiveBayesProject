# CSCI 460 Naive Bayes Programming Project

Please implement a Naive Bayes classifier in Python3 and apply it to the mushroom data sets provided: mushroom-training.data, and mushroom-testing.data. You are to train using just the training set, but provide classification accuracy for both the training and testing data sets after learning has occurred. Apply the classifier to these datasets first using no m-estimate, then consider how using *m = 1* affects the accuracy.

In addition to providing the code and results just described, please provide thorough and well-considered answers to the following questions:
*  Is the naive Bayes assumption valid and correct here? If not, does it matter here?
*  Did using an *m*-estimate help improve the classifier performance? Why or why not?

You may use and extend the python code I provide to you if you choose, or you may implement your own code from scratch to complete this assignment. Please submit your answers in the BlackBoard assignment submission field text box, but I will grade your source code by pulling from your GitHub repository.  So make sure it is pushed by the due date.

Please make sure your code is documented sufficiently so that it is easy to know how read and execute your program.  Do not collaborate with other students, and do not use other people's (or AI) code off of the Internet -- other than what I give you.

## Using the Provided Python Code
I have provided a python module called *dataset.py*. This program implements a simple class for reading in data in the format I have provided, and it also includes some convenience functions for helping with counting. As a hint, I suggest you look at method of the *Dataset* class called *selectSubset()*.

The module is already fairly capable as written. Indeed, assuming both *dataset.py* and mushroom-training.data are in the current directory, you can enter a Python3 command interpreter and type the following to get a sense for what is possible:

```
import dataset

D = dataset.Dataset("mushroom-training.data")
print("Attributes in the data set are: ", D.attributes.keys())

selectionCriteria = {"cap-shape":"b", "class":"p"}
print("There are", len(D.instances), "instances in total")
print("There are", len(D.selectSubset(selectionCriteria)), \
      " poisonous examples with a bell-shaped cap")
```

Read through the internal documentation in *dataset.py* to understand it’s use. Remember that you can always type help(dataset.Dataset) in the interpreter if the module has been imported. Please report any problems with the code to me.

## File Format
The file format is a very simple format to be used for a variety of classification (supervised learning) and clustering (unsupervised learning) tasks. It can represent categorical, ordinal (integer), or numeric (real) data. Attributes are named, and for classification tasks, one attribute is typically called "class". In such cases, the dataset will create a new attribute for each instance called "assigned-class", though no value is bound for that attribute for any instance by default. The "class" attribute is the true concept class value as given in the data file, while the "assigned-class" attribute is intended to facilitate the assignment of a class value by some classifier.

There are two types of lines expected in a data file:
1. Lines specifying attribute information;
2. Lines specifying instance data.

In the first case, lines must be in the form:
```
<attribute-name> : <attribute-type> : <attribute-values>
```

The attribute name can be any string that does not contain a ":" character, but they must be unique for each attribute. Attribute type can be "cat", "ord", or "num" for categorical, ordinal, or numeric data, respectively. In the case of categorical data, the attribute values specify all the allowable values in that category. In the case of ordinal or numeric data, the Dataset file reader will establish a range based on the minimal and maximal values in the list provided.

In the second case, lines must be in the form:
```
<value-1>, <value-2>, ..., <value-n>
```

There must be the same number of values in the line as their are attribute lines in the file. The position of the lines does not matter, but the order does. That is, the code will assume that the values are ordered in the same order as attributes are defined in the file. The instance data is stored in the data set as a list of dictionaries. Each instance is a dictionary keyed by attribute name.

**NOTE:** Data set files are processed line-by-line. Since the code and data were created for Unix platforms, Windows users may have issues with newlines. If you are unable to resolve these, let me know, and I’ll see if I can provide alternative forms of the data with DOS-style new- line/linefeed.


## Do Your Own Coding
To be clear:  You are responsible for implementing this. You *cannot* coordinate or communicate with others, copy code off the Internet, or use an AI tool. This is *your own* work. Copying source code without permission and acknowledgement is considered *plagiarism*.
