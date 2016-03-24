from logistic_regression import LogReg
from gradient_booster import Xgboost

def main():
    ##### Part 1: Vanilla Logistic Regression ##### Private Score : 50%
    '''
    lg = LogReg("data")
    train_acc = lg.get_train_accuracy()
    print "Train Accuracy :%f \n" %(train_acc)
    lg.store_classification_result()
    '''
    ##### Part 2: XGBoost
    Kfold = 5
    dataDir = "data"
    booster = Xgboost(Kfold, dataDir)
    train_acc = booster.get_train_accuracy()
    print "Train Accuracy :%f \n" %(train_acc)
    booster.store_classification_result()


if __name__ == "__main__":
    main()
