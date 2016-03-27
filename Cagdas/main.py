from c1_logistic_regression import LogReg
from c2_stacking import XgBase



def main():
    '''
    #c1
    lg = LogReg()
    training_accuracy = lg.get_training_accuracy()
    print "Train Accuracy :%f \n" %(training_accuracy)
    lg.store_result()
    '''
    #c2
    xg = XgBase()
    training_accuracy = xg.get_training_accuracy()
    print "Train Accuracy :%f \n" %(training_accuracy)
    xg.store_result()



if __name__ == "__main__":
    main()


