# This script grabs the data, formats the data, and measures the accuracy of three different model types on an out of sample set
# This script plays with an OLS model, a Tensorflow neural network, and a pyrenn neural network
import numpy as np
import sklearn
import math
import statsmodels.api as sm
import matplotlib.pyplot as pl
import keras
import pyrenn

# Importing the raw data as strings
raw_data_from_csv = np.genfromtxt("Breast_Cancer.csv", delimiter=",", dtype='str')
raw_data_from_csv = raw_data_from_csv[1:,:]

# Changing sex and smoker values from strings to binary values
raw_data_from_csv[raw_data_from_csv == 'M'] = '1'
raw_data_from_csv[raw_data_from_csv == 'B'] = '0'

# The explanatory variables are put in X_data and the response variable is put in Y_data
X_data = raw_data_from_csv[:,2:].astype(np.float)
Y_data = raw_data_from_csv[:,1].astype(np.float)

# This for loop runs a model trained on the first half of the data on out of sample data from the second half and then vice versa
for j in range(0,2):

    if j == 0:
        # The sample is split in two, the second half of which will create the model, the first of which will measure out of sample accuracy
        X_test = X_data[0:math.ceil(X_data.shape[0]/2),:]
        Y_test = Y_data[0:math.ceil(Y_data.shape[0]/2),]
        X_train = X_data[math.ceil(X_data.shape[0]/2):,:]
        Y_train = Y_data[math.ceil(Y_data.shape[0]/2):,]
    else:
        # The sample is split in two, the first half of which will create the model, the second of which will measure out of sample accuracy
        X_train = X_data[0:math.ceil(X_data.shape[0]/2),:]
        Y_train = Y_data[0:math.ceil(Y_data.shape[0]/2),]
        X_test = X_data[math.ceil(X_data.shape[0]/2):,:]
        Y_test = Y_data[math.ceil(Y_data.shape[0]/2):,]
    
    # A constant is added to the data for a linear fit
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    # The training data is fit via OLS
    # t values are low for some variables but removing these variables from the OLS model doesn't increase performance on out of sample data
    linear_model = sm.OLS(Y_train,X_train).fit()
    print(linear_model.summary())
    linear_predictions = linear_model.predict(X_test)
    # A scatter of OLS model claims predictions vs. actual claims
    pl.scatter(linear_predictions.T,Y_test.T)
    pl.show()
    
    # Data is scaled for neural network training
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train[:,1:])
    X_train = scaler.transform(X_train[:,1:])
    X_test = scaler.transform(X_test[:,1:])


    # A second neural network target is created. First is Malignent, second is benign. Helps with model accuracy.
    Y_train = np.column_stack((Y_train,1-Y_train))
    
    # A neural network via Keras on Tensorflow.
    # Tested with many different neural network structures
    # Using hidden layers with 20 neurons each, 'relu' activation functons to start, 'softmax' activation function on the output layer to constrain between 0-1, 'adam' optimizer measured via 'categorical_crossentropy' for the categorical output
    # A validation set uses 35% of the training data to prevent overfitting
    # The predictions are averaged across many different models with different training and validation sets
    for i in range(0,10):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(20, input_dim=30, activation='relu'))
        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    	
        model.fit(X_train, Y_train, epochs=1000, verbose=0, callbacks=[earlystopping], validation_split=0.35, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=1, validation_steps=5)
    	
        if i == 0:
            nn_predictions = model.predict(X_test)
        else:
            nn_predictions = (nn_predictions*(i)+model.predict(X_test))/(i+1)
    
        print(i)
        
        # Prints the correlation between the average of model outputs and the targets and then the correlation between the most recent model output and the targets
        print(np.corrcoef(nn_predictions[:,0].T,Y_test.T)[0,1]**2)
        print(np.corrcoef(model.predict(X_test)[:,0].T,Y_test.T)[0,1]**2)
    
    nn_predictions = nn_predictions[:,0]
    
    # A scatter of Tensorflow model claims predictions vs. actual claims
    pl.scatter(nn_predictions.T,Y_test.T)
    pl.show()
    
    # Creating a neural network using the Levenberg-Marquardt backpropagation training function
    # Used for quick descent training and possibly a more accurate prediction
    # Fewer hidden layers and less nodes are used due to a larger propensity to overfit
    # Cannont use a validation set for early stopping in pyrenn so these two lines are used to find convergence
    # Seems to converge around 10 epoches. Should stop early at 10 epoches to avoid overfitting on a small dataset
    Y_train = Y_train[:,0]
    net = pyrenn.CreateNN([30,5,1])
    pyrenn.train_LM(X_train.T, Y_train.T, net,verbose=1,k_max=20)
    
    # The predictions are averaged across many different trained models
    for i in range(0,10):
        print(i)
        net = pyrenn.CreateNN([30,5,1])
        pyrenn.train_LM(X_train.T, Y_train.T, net,verbose=0,k_max=10)
        if i == 0:
            LM_predictions = pyrenn.NNOut(X_test.T, net)
        else:
            LM_predictions = (LM_predictions*(i)+pyrenn.NNOut(X_test.T, net))/(i+1)
        
        print(i)
        
        # Prints the correlation between the average of model outputs and the targets and then the correlation between the most recent model output and the targets
        print(np.corrcoef(LM_predictions.T,Y_test.T)[0,1]**2)
        print(np.corrcoef(pyrenn.NNOut(X_test.T, net).T,Y_test.T)[0,1]**2)
    
    # A scatter of Pyrenn model claims predictions vs. actual claims
    pl.scatter(LM_predictions.T,Y_test.T)
    pl.show()
    
    # Prints accuracy measures for each model
    print('The OLS model R^2 is' ,np.corrcoef(linear_predictions.T,Y_test.T)[0,1]**2)
    linear_predictions[linear_predictions>=0.5] = 1
    linear_predictions[linear_predictions<0.5] = 0
    print('The percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(linear_predictions.T-Y_test.T)))
    print('The Tensorflow model R^2 is' ,np.corrcoef(nn_predictions.T,Y_test.T)[0,1]**2)
    nn_predictions[nn_predictions>=0.5] = 1
    nn_predictions[nn_predictions<0.5] = 0
    print('The percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(nn_predictions.T-Y_test.T)))
    print('The Pyrenn model R^2 is' ,np.corrcoef(LM_predictions.T,Y_test.T)[0,1]**2)
    LM_predictions[LM_predictions>=0.5] = 1
    LM_predictions[LM_predictions<0.5] = 0
    print('The percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(LM_predictions.T-Y_test.T)))
    
    # If the second round through the for loop, stacks the forecast predictions
    if j == 0:
        all_linear_predictions = linear_predictions
        all_nn_predictions = nn_predictions
        all_LM_predictions = LM_predictions
    else:
        all_linear_predictions = np.hstack((all_linear_predictions,linear_predictions))
        all_nn_predictions = np.hstack((all_nn_predictions,nn_predictions))
        all_LM_predictions = np.hstack((all_LM_predictions,LM_predictions))
        
# Accuracy measures for each model in total (includes forecasted first half and second half of data)
# The linear model R^2 = 68.49%, percent diagnosed correctly = 91.92%
# The Tensorflow model R^2 ~ 78.5%, percent diagnosed correctly ~ 94.5%
# The Pyrenn model R^2 ~ 80.0%, percent diagnosed correctly ~ 95.0%   
print('The overall OLS model R^2 is' ,np.corrcoef(all_linear_predictions.T,Y_data.T)[0,1]**2)
print('The overall percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(all_linear_predictions.T-Y_data.T)))
print('The overall Tensorflow model R^2 is' ,np.corrcoef(all_nn_predictions.T,Y_data.T)[0,1]**2)
print('The overall percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(all_nn_predictions.T-Y_data.T)))
print('The overall Pyrenn model R^2 is' ,np.corrcoef(all_LM_predictions.T,Y_data.T)[0,1]**2)
print('The overall percent of cases diagnosed correctly is ' ,1-np.mean(np.abs(all_LM_predictions.T-Y_data.T)))


# Accuracy measurements vary when only averaging over 10 trained models. Averaging over 100+ trained models will be more consistent
# This small dataset causes overfitting easily and hinders accuracy. The easiest way to increase the accuracy of each model would be to add to the sample size
# The neural networks are 5% to 10% more accurate!