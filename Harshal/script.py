from gradient_booster import Xgboost
from logistic_regressor import LogRes
from random_forest import RandomForest
from extra_trees import ExtraTrees

def main():
    ##### Part 1: Vanilla Logistic Regression ##### Private Score : 50%
    # USE SHOBHIT's CODE
    ##### Part 2: XGBoost
    '''
    Kfold = 5
    dataDir = "data"
    booster = Xgboost(Kfold, dataDir)
    train_acc = booster.get_train_accuracy()
    print "Train Accuracy :%f \n" %(train_acc)
    booster.store_classification_result()
    '''
    ##### Part 3: Ensembler
    print 'Beginning Ensembling....'
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    dataDir = 'data'

    # 1. Logistic Regressor - balanced - Sparse Greedy Features
    print 'Beginning Logistic Regression ...'
    out_file = './data/ensemble_results/lr_unbalanced_sparse.csv'
    clf = LogRes(train_file, test_file, out_file)
    clf.logistic_regression()
    print 'Logstic Regression Completed. Labels saved. Moving Forward.'

    # 2. Random Forest - Entropy Index
    print 'Beginning Random Forest Classfication (ENTROPY BASED) ...'
    out_file = './data/ensemble_results/rf_entropy.csv'
    Kfold = 10
    forest = RandomForest(Kfold,dataDir,'entropy')
    train_acc = forest.get_train_accuracy()
    print "Train Accuracy Random Forest (Entropy):%f \n" %(train_acc)
    forest.store_classification_result(out_file)
    print 'Random Forest Classfication Completed. Labels saved. Moving Forward.'

    # 3. Random Forest - Gini Index
    print 'Beginning Random Forest Classfication (GINI BASED) ...'
    out_file = './data/ensemble_results/rf_gini.csv'
    Kfold = 10
    forest = RandomForest(Kfold,dataDir,'gini')
    train_acc = forest.get_train_accuracy()
    print "Train Accuracy Random Forest (Gini) :%f \n" %(train_acc)
    forest.store_classification_result(out_file)
    print 'Random Forest Classfication Completed. Labels saved. Moving Forward.'

    # 4. Extra Trees - Entropy Index
    print 'Beginning Extremely Randomized RF i.e Extra Trees (ENTROPY BASED)...'
    out_file = './data/ensemble_results/extra_forest_entropy.csv'
    Kfold = 10
    trees = ExtraTrees(Kfold,dataDir,'entropy')
    train_acc = trees.get_train_accuracy()
    print "Train Accuracy Extra Trees (Entropy):%f \n" %(train_acc)
    trees.store_classification_result(out_file)
    print 'Extra Trees (ENTROPY) Completed. Labels saved. Moving Forward.'

    # 5. Extra Trees - Gini Index
    print 'Beginning Extremely Randomized RF i.e Extra Trees (GINI BASED)...'
    out_file = './data/ensemble_results/extra_forest_gini.csv'
    Kfold = 10
    trees = ExtraTrees(Kfold,dataDir,'gini')
    train_acc = trees.get_train_accuracy()
    print "Train Accuracy Extra Trees (Gini):%f \n" %(train_acc)
    trees.store_classification_result(out_file)
    print 'Extra Trees (GINI) Completed. Labels saved. Moving Forward.'
    
    # 6. Xgboost - Gradient Boosting Machine
    print 'Beginning Extreme Gradient Boosting...'
    out_file = './data/ensemble_results/xgboost.csv'
    Kfold = 7
    booster = Xgboost(Kfold, dataDir)
    train_acc = booster.get_train_accuracy()
    print "Train Accuracy :%f \n" %(train_acc)
    booster.store_classification_result(out_file)
    print 'Extreme Gradient Boosting Completed. Labels saved'

    print 'All models trained and test labels saved'
    print 'All done... Exiting...'
    print 'Exited'

if __name__ == "__main__":
    main()
