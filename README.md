# Ml-Library-I
 Ml-Library-I for WoC 2.0
This library contains separate classes containing functions implementing different machine learning.
1) LinReg
    Here only linear hypothesis dependent on all input features of input data is considered.
    In constructor, input data is preprocessed and weight matrix is initialised.
    -Constructor:
        object_name = LinReg(data_in, data_out, lamb, l_rate)
            *data_in->                  input training data_in
                                        (in form of Array, list OR matrix containing each training example in a new row)
            *data_out->                 output training data ( 1-D array, series or list)
            *lamb->                     L2 regularisation coefficient, default to zero
            *l_rate->                   Learning rate of algorithm, default to 0.01data_in

    -y_pred()->                         Prediction function for calculating predicted values of output.
    -y_cost()->                         Cost function for calculating Mean Squared Error ie. ** J_train**
    -update()->                         Function to update the weight matrix according to process of Gradient Descent.
    -train(epoch)->                     Training function employing traditional gradient descent.
                                        epoch= No. of iterations to be made while training the model. Default to 1000.
    -batch_train(epoch)->               Training function employing batch gradient descent.
                                        epoch= No. of iterations to be made while training the model. Default to 100.
    -cost_cv()->                        Cross validation cost function for calculating Mean Squared Error ie. **J_cv**.
    -test(test_data_in, test_data_out)->Function accepting test input and output data to check the accuracy of model.
        *test_data_in->                 input testing data
                                        (in form of Array, list OR matrix containing each training example in a new row)
        *test_data_out->                output testing data
                                        ( 1-D array, series or list)
    -y_pred_test->                      Function for predicting the output corresponding to the testing data
    -cost_test()->                      Returns the ** RMS error ** in test output.
    -model_cost_vs_epoch(lr,ur,step)    Function which plots graph of training cost vs epoch(in train() function )
                *lr->                   Lower range of epoch
                *ur->                   Upper range of epoch
                *step->                 Increment in epoch
    -predict1(x)->                      Function implements the model on 1D data example ** one at a time **


2)LogReg
    -Constructor:
        object_name= LogReg(LinReg(data_in, data_out, lamb, l_rate))
            *data_in->                          input training data_in
                                                (in form of Array, list OR matrix containing each training example in a
                                                new row)
            *data_out->                         output training data
                                                ( 1-D array, series or list in case of single-class output)
                                                ( 2-D array,list or dataframe consisting of output data with each
                                                example a new row)
            *lamb->                             L2 regularisation coefficient.
                                                default set to 0.
            *l_rate->                           Learning rate of algorithm,
                                                default set to 0.01.
    A) Linear Hypothesis:
        -y_pred_lin()->                             Prediction function for linear hypothesis based model.

        -y_cost_lin()->                             Cost function for linear hypothesis with L2 regularisation.
        -update_lin()->                             Function to update the weight matrix according to process of
                                                    Gradient Descent.
        -train_model_lin(epoch)->                   Training function employing traditional gradient descent.
                                                    epoch= No. of iterations to be made while training the model.
                                                    Default to 1000.
        -batch_train_lin(epoch)->                   Training function employing batch gradient descent.
                                                    epoch= No. of iterations to be made while training the model.
                                                    Default to 100.
        -cost_cv_lin()                              Cross validation cost function for calculating Mean Squared Error
                                                    ie. **J_cv**.



    B) Quadratic Hypothesis
        -y_pred_quad()->                            Prediction function for quadratic hypothesis based model.
        -y_cost_quad()->                            Cost function for qquadratic hypothesis with L2 regularisation.
        -update_quad()->                            Function to update the weight matrix according to process of
                                                    Gradient Descent. *
        -train_model_quad(epoch)->                  Training function employing traditional gradient descent.
                                                    epoch= No. of iterations to be made while training the model.
                                                    Default to 1000.
        -batch_train_quad(epoch)->                  Training function employing batch gradient descent.
                                                    epoch= No. of iterations to be made while training the model.
                                                    Default to 100.
        -cost_cv_quad()                             Cross validation cost function for calculating Mean Squared Error
                                                    ie. **J_cv**.

        -test_model(test_data_in,test_data_out)->   **COMMON TO BOTH HYPOTHESES**
                                                    Function accepting for testing model over data.
                *test_data_in->                     input testing data
                                                    (in form of Array, list OR matrix containing each training example
                                                    in a new row)
                *test_data_out->                    output testing data
                                                    (in form of Array, list OR matrix containing each training example
                                                    in a new row)
        -y_test_pred()->                            **COMMON TO BOTH HYPOTHESES**
                                                    Function for predicting the output corresponding to the testing data
        -y_test_accu()->                            **COMMON TO BOTH HYPOTHESES**
                                                    Gives % accuracy of model over training dataset
        -model_accu_vs_epoch(lr, ur, step)->        **COMMON TO BOTH HYPOTHESES**
                                                    Function which plots graph of model accuracy over trainiing dataset
                                                    vs epoch(in train() function)
                *lr->                               Lower range of epoch
                *ur->                               Upper range of epoch
                *step->                             Increment in epoch
