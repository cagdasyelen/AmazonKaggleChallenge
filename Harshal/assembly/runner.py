from gradient_booster import Xgboost
from random_forest import RandomForest


## LOADER DIRECTORY
dataDir = 'data_final'

############## PART 2 #################
# 1. Xgboost - Gradient Boosting Machine
print 'Beginning Extreme Gradient Boosting...'
out_file = './data_final/part_2/xgboost.csv'
Kfold = 5
booster = Xgboost(Kfold, dataDir)
train_acc = booster.get_train_accuracy()
print "Train Accuracy :%f \n" %(train_acc)
booster.store_classification_result(out_file)
print 'Extreme Gradient Boosting Completed. Labels saved. Moving Forward'

############## PART 3 #################
# 1. Xgboost - Gradient Boosting Machine
print 'Beginning Extreme Gradient Boosting...'
out_file = './data_final/part_3/xgboost.csv'
'''Same Code as above'''
booster.store_classification_result(out_file)
print 'Extreme Gradient Boosting Completed. Labels saved. Moving Forward'

# 2. Random Forest - Entropy Index
print 'Beginning Random Forest Classfication (ENTROPY BASED) ...'
out_file = './data_final/part_3/rf_entropy.csv'
Kfold = 10
forest = RandomForest(Kfold,dataDir,'entropy')
train_acc = forest.get_train_accuracy()
print "Train Accuracy Random Forest (Entropy):%f \n" %(train_acc)
forest.store_classification_result(out_file)
print 'Random Forest Classfication Completed. Labels saved. Moving Forward.'


# 3. Random Forest - Gini Index
print 'Beginning Random Forest Classfication (GINI BASED) ...'
out_file = './data_final/part_3/rf_gini.csv'
Kfold = 10
forest = RandomForest(Kfold,dataDir,'gini')
train_acc = forest.get_train_accuracy()
print "Train Accuracy Random Forest (Gini) :%f \n" %(train_acc)
forest.store_classification_result(out_file)
print 'Random Forest Classfication Completed. Labels saved. All done.'
