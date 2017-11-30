#The reticulate package provides an R interface to Python modules, classes, and 
#functions
install.packages('reticulate')
require(reticulate)



install.packages('devtools')


#installing keras 
#It will first install tensorflow then keras.


 


#loading keras in R 
library(keras)
#The R interface to Keras uses TensorFlow as it’s underlying computation engine.
#So we need to install Tensorflow engine
install_tensorflow()

#For installing a gpu version of Tensorflow
install_tensorflow(gpu = T)




#loading the keras inbuilt mnist dataset
data<-dataset_mnist()
?dataset_mnist #MNIST database of handwritten digits


#Training Data
train_x<-data$train$x
train_y<-data$train$y

#Test Set
test_x<-data$test$x
test_y<-data$test$y

#converting a 2D array into a 1D array for feeding 
#into the MLP and normalising the matrix
train_x <- array(as.numeric(train_x), dim = c(dim(train_x)[[1]], 784))
test_x <- array(as.numeric(test_x), dim = c(dim(test_x)[[1]], 784))

train_x <- train_x / 255
test_x <- test_x / 255



cat(dim(train_x)[[1]], 'train samples\n')#60000 train examples
cat(dim(test_x)[[1]], 'test samples\n')#10000 test examples


#convert class vectors to binary class matrices
train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)

#Now defining a keras MLP sequential model containing a linear stack of layers
model <- keras_model_sequential()



model %>% 
  #Input layer-256 units
  #Add a densely-connected NN layer to an output
  layer_dense(units=256,activation="relu",input_shape=c(784))  %>%
  #dropout layer to prevent Overfitting
  layer_dropout(rate=0.3) %>%
  
 
  #Apply an activation function to an output.
  #Relu can only be used for Hidden layers
  layer_dense(units = 128,activation = "relu") %>%
  layer_dropout(rate=0.3) %>%
  

  layer_dense(units=10,activation="softmax") 
  #softmax activation for Output layer which computes the probabilities for the classes
  



#Model's summary
summary(model)


#
model %>%
  compile(loss ="categorical_crossentropy",
          optimizer = "adam",
          metrics= c("accuracy"))


  

history<-model %>% fit(train_x, train_y, epochs = 10, batch_size = 128,
                       callbacks = callback_tensorboard(log_dir = "logs/run_b"),
                       validation_split = 0.3) #train on 80% of train set and will evaluate 


summary(history)
history$params
history$metrics 

plot(history,labels=T)
which.min(history$metrics$acc)

plot(x = history$metrics$acc,y = history$metrics$loss,
     pch=19,col='red',type='b',
     ylab="Error on trining Data",xlab="Accuracy on Training Data")
title("Plot of accuracy vs Loss")
legend("topright",c("Epochs"),col="red",pch=19)


#Evaluating model on the Test dataset
score <- model %>% 
  evaluate(test_x,test_y,batch_size=128)
