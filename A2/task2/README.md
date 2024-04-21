# EXPLANATION OF BEST FIT DEGREE

The best fit degree depends on how the datapoints are scattered. If the resulting scatterplot
forms a cluster of a more or less linear shape, the lowest degree (1) is the best option. If
the scattered training data forms a U - shape, degree 2 is a good option. In this exercise, most
of the times, the scattered data results in a wave-like shape that has the tendency to go 
upwards (left to right). I implemented the MSE function to find out which degree is the best fit
and the result is amost always either degree 4 or degree 5. This is the case because a polynomial
graph of 4th degree has a w-like shape and sometimes the data is scrambled in a way that the n-shaped curve
has a slightly upward-pointing tip in the bottom-left corner of the graph. Degree 5 is a best fit when there
is a slightly downward-pointing tip in the top-right corner of the graph in addition to the slightly 
downward-pointing tip in the bottom left corner of the graph. Degree 4 and 5 are the best degrees for both,
training and test data.