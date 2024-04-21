EXERCISE 4)

NOTES: I downloaded Keras to get the mnist dataset, so you won't find the csv file in this folder.

ANSWER TO THE QUESTION:

--------------------------------------------------------------------------------------------------------------------------------------------
The best classifier turned out to be the self-implemented One-vs-All classifier. But the difference between the built in OVO and the self
implemented OVA is very small. 95.6 % accuracy for the OVO and 95.9 % for the OVA.

When looking at the 2 matrices, there are no big differences in classification errors. In fact, the differences between OVO and OVA are
minimal. Some lables (1, 4, 6, 7, 9) have an equal number of correct prediction and other lable's correct prediction differ by 5 predictions at max (1 case: label 5).