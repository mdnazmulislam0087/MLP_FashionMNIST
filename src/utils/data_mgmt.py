import tensorflow as tf





def get_data(validation_datasize):
    """ It is used to load the dataset

    Returns:
        tuple: Train , valid and test sets
    """
    fashion_mnist=tf.keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test,y_test)=fashion_mnist.load_data()
    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize]/255.0 , X_train_full[validation_datasize:]/255.0
    y_valid, y_train= y_train_full[:validation_datasize], y_train_full[validation_datasize:]


    # scale the test set as well
    X_test = X_test / 255.0
    
    return (X_train, y_train) , (X_valid, y_valid), (X_test, y_test)
  
