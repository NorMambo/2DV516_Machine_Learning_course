EXERCISE 5)

--------------------------------------------------------------------------------------------------------------------------------------------

Label 0 (T-SHIRT) gets often confused with label 6 (SHIRT)

Label 1 (TROUSER) does not get confused with other clothes very often.

Label 2 (PULLOVER) gets often confused with label 0, 4 and 6 (T-SHIRT, COAT and SHIRT) (confusion with COAT yields most of the errors).

Label 3 (DRESS) gets often confused with label 0 and 4 (T-SHIRT and COAT) (confusion with COAT yields most of the errors)

Label 4 (COAT) gets often confused with label 2 (PULLOVER).

Label 5 (SANDAL) gets often confused with label 7 and 9 (SNEAKER and ANKLE BOOT) (confusion with SNEAKER yields most of the errors). 

Label 6 (SHIRT) gets often confused with label 0, 2 and 4 (T-SHIRT, PULLOVER AND COAT) (confusion with T-SHIRT yields most of the errors).

Label 7 (SNEAKER) gets often confused with lable 9 (ANKLE BOOT).

Label 8 (BAG) does not get confused with other clothes very often.

Label 9 (ANKLE BOOT) sometimes gets confused (SNEAKER).

--------------------------------------------------------------------------------------------------------------------------------------------

The model has an accuracy of around 85 %, meaning that it could be improved by finding better hyperparameters.

I ran GridSearchCV twice and the second time it took me approximately 2.5 hours to get a result. Therefore I didn't run it again (I was a bit short on time..).
Instead, I tried modifying some of the parameters manually, but I didn't find a satisfying alternative. So I decided to use the ones I got using GridSearchCV.


