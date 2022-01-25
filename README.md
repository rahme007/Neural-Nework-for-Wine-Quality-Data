<img src="https://www.dailyhostnews.com/wp-content/uploads/2018/07/Python-featured-2100x1200.jpg" width="80" height="50">

# Neural-Nework-for-Wine-Quality-Data
This project deals with the quality assessment of wine using different features studied by wine makers in Portugal. For each wine they measured 11 numerical chemical attributes (e.g., pH values), and they asked three or more “experts” to assign a “quality value” (0 to 10), recording the median score from the experts.

## Introduction
The features of wine quality determine the quality of the wine (0 -10). There are 6497 samples, and one more attribute (‘type’) will be added to each sample, to identify if the sample is from a red (type = 1) or a white (type = 0) wine. The project has been divided into two segments:

1.	**Regression Model** [(red & white together), target = quality (0 to 10)]

    The neural network for solving this regression problem will have:
       <ul>
        <li>12 inputs in the first layer (the 11 original attributes AND the new ‘type’ attribute)</li>
        <li>1 output processing element in the output layer, since the result will be a single number (0 to 10)</li>
      </ul>

2.	**Classification Model** [target = classification of 3 quality levels  (low 3 or 4 ; medium 5, 6, or 7 ; high  8 or 9 )]

      ALTERNATIVE TARGET VECTOR, ‘level’ is created which indicates the “quality level of each sample from the original 64971 target array.

      [ NOTE – It was found out that none of the original ‘quality’ scores had values 0, 1, 2 or 10. That is, in practice, the numerical values in the ‘quality column are greater than or equal to 3 and lower than or equal to 9]

      The table below shows the quality level of each sample

| **‘quality’** value of the sample | **‘level’** for the sample | Meaning |
| :---: | :---: |:---:|
| 3 or 4 | 1 | **LOW** quality **LEVEL** |
| 5, 6 or 7 | 2 | **MEDIUM** quality **LEVEL** |
| 8 or 9 | 3 | **HIGH** quality **LEVEL** |

To implement this, the 64971 ‘level’ array will be ‘one-hot-encoded’ to yield a 6497 x 3 array, where each of the 6497 samples will have as target a 3-element array where 2 of them will be 0 and one will be 1 (This is done as it was performed for the Reuters classification example in the book, p. 79., Listing 3.14)
Therefore, the network to perform classification of the wines into the 3 levels of quality will have;
<ul>
<li>12 inputs in the first layer (the 11 original attributes AND the new ‘type’ attribute)</li>
<li>3 processing elements in the output layer. Ideally one of them would turn to 1 and two would remain at 0, to indicate that the input sample belongs to one of the 3 levels of quality. </li>
</ul>
In summary the project contains two models:
<ol type = "i">
<li>	regression model:  regmodl
<li>	classification model:  clasmodl
</ol>

## [PART I] : Getting the data as Numpy arrays
The Jupyter Notebook Wine_Quality_Prep.ipynb retrieves the original winequality-white.csv and winequality-red.csv directly from the UCI repository, originally as Pandas DataFrames, and:
<ol type = "a">
  <li> Adds the ‘type’ information (red = 1, white = 0) as the 13th column in both Dataframes </li>
  <li> Appends  the red DataFrame to the white DataFrame, yielding a single DataFrame, wines, with 6497 samples </li>
  <li> Uses sklearn’s train_test_split() to create TRAINING [TR]  and VALIDATION [TT] SPLITS – Note that although the name of the function is train_TEST_ … , we will be using the results for training [TR], AND VALIDATION [TT]. </li>
  IMPORTANT NOTE: In this project we are not setting aside “test data [ts]”. All available data will be used for effective training (changing weights), [tr]. or for obtaining validation metrics [tt], needed for you to EVALUATE the performance of tentative models and guide your iterative improvement of the model in each task. 
The data is split into 75% (4872) for training [TR], and 25% (1625) for validation [TT]. Therefore, you will be using “simple hold-out validation” in this project.
  <li> The train splits will be returned from train_test_split() as Pandas DataFrames. The notebook converts them to Numpy arrays (to be used in Keras). </li>
  <li> Finally, the notebook will use the already available target (rank-1) vectors (one for training, one for validation) to create two additional target arrays, where the quality levels have been mapped as indicated in the table above (1 = LOW, 2 = MEDIUM, 3 = HIGH) and one-hot-encoded, so there is a 3-value target for each pattern. – These additional target arrays are used for the classification model. </li>
  
</ol>

## [Part II]: The Regression Model
1. A simple model (regmodl1) is developed that does better than “a baseline,” and it contains two hidden layers such as:
<ul>
  <li>The input layer consists of 12 processing elements as there are 12 attributes.</li>
  <li>The first hidden layer contains 16 processing elements, and the activation function is selected as ‘relu’</li>
  <li>The second hidden layer contains 16 processing elements, and the activation is selected as ‘relu’.</li>
  <li>The output layer contains one processing element since it predicts the quality of the wine from 0-10.</li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘mse’ (mean of squared error).</li>
</ul>
After running the fit method of the regression model with training and validation data, combine plot of training and validation losses per epoch is analyzed. Fig. 1 shows the plot of the training and validation loss.
<br>

![image](https://user-images.githubusercontent.com/98129458/151060683-ed00282b-cd09-4c1b-b229-cefa6bcbd743.png)

<p align='center'>Fig. 1: Regression Model 1 (regmodl1) Plot </p>
<br>
<br>
Here the model seems to converge in a fixed loss after 49 epochs both for training and validation samples. Note that the validation data presented here is noisy and it has been smoothened by the exponential moving average of the previous points. This model is expensive as it requires 50 epochs to converge.

2. A better model (regmodl2) has been designed that actually overfits the data. The model has 3 hidden layers.
<ul>
  <li>•	The input layer consists of 12 processing elements as there are 12 attributes. </li>
  <li>The first hidden layer contains 64 processing elements and the activation function is selected as ‘relu’. </li>
  <li>The second hidden layer contains 64 processing elements, and the activation function is selected as ‘relu’. </li>
  <li>The third hidden layer consists of 64 processing elements, and the activation function is chosen as ‘relu’. ‘relu’ is used to avoid vanishing gradient. </li>
  <li>The output layer contains one processing element since it predicts the quality of the wine from 0-10. </li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘mse’ (mean of squared error). </li>
</ul>

**Reason for changing the model from regmodl1 to regmodl2:** regmodl1 has been modified to regmodl2 by adding one more hidden layer. Also, the processing elements have been changed to 64 for each layer. The change has been made to observe the overfitting property of the model. Fig. 2 shows training and validation loss of the new regression model for the wine assessment. From the figure, the changed model seems to overfit after 36 epochs. Therefore, the model has learned the required knowledge for the regression within 36 epochs. It implies that the model can be stopped learning after 36 epochs. Otherwise, it will start memorizing the data.

![image](https://user-images.githubusercontent.com/98129458/151061401-9f1bc99f-bca8-482f-9374-5828edf17912.png)

Fig. 2: Regression Model 2 (regmodl2) Plot 
<br>
<br>
3. After observing the iteration of the performance of the model regmodl2, it is reasonable to tune the hyperparameter (i.e., epoch in this case) to arrive at regmodl3. From Fig. 2, it has been observed that the regmodl2 will start to memorize the data after 36 epochs. Hence, regmodl3 is the final regression model which considers stopping training after 36 epochs. This is the best model to solve the regression task. The regmodl3 also contains three hidden layers.
<ul>
  <li>The input layer consists of 12 processing elements as there are 12 attributes. </li>
  <li>The first hidden layer contains 64 processing elements and the activation function is selected as ‘relu’. </li>
  <li>The second hidden layer contains 64 processing elements, and the activation function is selected as ‘relu’. </li>
  <li>The third hidden layer consists of 64 processing elements, and the activation function is chosen as ‘relu’. ‘relu’ is used to avoid vanishing gradient. </li>
  <li>The output layer contains one processing element since it predicts the quality of the wine from 0-10. </li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘mse’ (mean of squared error). </li>
 </ul>
 
 ![image](https://user-images.githubusercontent.com/98129458/151062986-090c08d2-4539-4bcf-a024-b3b6e9656e6f.png)
 
 Fig. 3: Regression Model 3 (regmodl3) Plot 
 <br>
 <br>
 **Reason for changing the model from regmodl2 to regmodl3:** the model regmodl2 shows significant overfitting after 36 epochs, and it starts to memorize the data from the 37th epoch. The best model regmodl3 would be the one that does not overfit and requires an optimal amount of epochs. In that case, rogmodl3 has been stopped after 36 epochs. Furthermore, the training loss and validation loss are approximately the same in this epoch. Fig. 3 displays the combined plot of training and validation loss. 

## [Part III]: The Classification Model
1. A simple model is developed that does than a ‘baseline’, classmodl1 and it contains 1 hidden layer only.
<ul>
  <li>The input layer consists of 12 processing elements as there are 12 attributes.</li>
  <li>The first hidden layer contains 64 processing elements and the activation function is selected as ‘relu’</li>
  <li>The output layer contains 3 processing element since it predicts the quality level 1,2 or 3 (one-hot-encoded). ‘softmax’ is used as the activation function for this layer.</li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘categorical_crossentropy’.</li>
 </ul>
After running the fit method of the regression model with training and validation data, combine plot of training and validation losses per epoch is analyzed. Fig. 4 shows the plot of the training and validation loss. The plot is very noisy, and the validation loss fluctuates from 0.6 to 0.8. Note that both training and validation data presented here is noisy, and it has been smoothened by the exponential moving average of the previous points.

![image](https://user-images.githubusercontent.com/98129458/151063640-e18d409b-6134-4654-bff2-a62f29a107e3.png)

Fig. 4: Classification Model 1 (clasmodl1) Plot 
<br>
<br>
2. A better model (clasmodl2) is developed that actually overfits. The model contains 3 hidden layers.
<ul>
  <li>The input layer consists of 12 processing elements as there are 12 attributes.</li>
  <li>The first hidden layer contains 64 processing elements and the activation function is selected as ‘relu’</li>
  <li>The second hidden layer contains 64 processing elements and the activation function is selected as ‘relu’.</li>
  <li>The third hidden layer consists of 64 processing elements and the activation function is chosen as ‘relu’. ‘relu’ is used to avoid vanishing gradient. </li>
  <li>The output layer contains 3 processing elements since it predicts the quality level 1,2 or 3 (one-hot-encoded). ‘softmax’ is used as the activation function for this layer.</li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘categorical_crossentropy’.</li>
 </ul>
 
 **Reason for changing the model from clasmodl1 to clasmodl2:** the model clasmodl has been modified to clasmodl2 by adding two more hidden layers. From the perspective of validation loss, the network begins overfitting almost right away, after just one epoch, and overfits much more severely. Due to the noisy plot of training and validation data, it is obvious to add more layers in order to observe the loss for hyperparameter tuning. Thus, the model has been changed to clasmodl2 which is a better model. Fig. 5 shows the combined plot of losses occurred during training and validation for each epoch. From the figure, it is evident that the model starts overfitting drastically after 30 epochs. The validation loss starts to increase rapidly after 30 epochs. The increase of training loss is also observed in that plot.  
 
 ![image](https://user-images.githubusercontent.com/98129458/151064538-8039e58c-1066-4a8a-bf8c-4bf05c606a31.png)
 <br>
 Fig. 5: Classification Model 2 (clasmodl2) Plot 
 <br>
 <br>
 3. The iterative observations of the performance in the clasmodl2 reveal that the overfitting of the model needs to be prevented by tuning the stopping criteria for the epochs. The new clasmodl3 has been developed with epochs stopped at 30. Furthermore, the model has used ‘Dropout’ to enhance the generalization.
 <ul>
  <li>The input layer consists of 12 processing elements as there are 12 attributes.</li>
  <li>The first hidden layer contains 64 processing elements and the activation function is selected as ‘relu’</li>
  <li>Dropout layer is introduced with dropout rate of 0.5.</li>
  <li>The second hidden layer contains 64 processing elements and the activation function is selected as ‘relu’.</li>
  <li>Dropout layer is introduced with dropout rate of 0.5.</li>
  <li>The third hidden layer consists of 64 processing elements and the activation function is chosen as ‘relu’. ‘relu’ is used to avoid the vanishing gradients.</li>
  <li>The output layer contains 3 processing element since it predicts the quality level 1,2 or 3 (one-hot-encoded). ‘softmax’ is used as the activation function for this layer.</li>
  <li>The Optimizer is selected as ‘rmsprop’. </li>
  <li>The loss function is selected as ‘categorical_crossentropy’.</li>
  </ul>
  Note that, the dropout layer after third hidden layer had also been added to experiment the improvement of the performance. However, the performance was found degraded and thus the layer had been removed.
  
<br>
<br>

**Reason for changing the model from clasmodl2 to clasmodl3:** the model clasmodl2 has been modified to clasmodl3 to find a best model for this multi class classification problem. As the classification model in section III.2   starts memorizing rapidly after 30 epochs, the training has been stopped after 30 epochs in the new model clasmodl3. It is also noticeable that clasmodl2 is severely noisy and tries to memorize the data. It is also seen from the plot of Fig. 5 that, the training loss starts to increase as the number of epochs incremented. Therefore, in order to reduce the overfitting and the noise to the network, dropout layers have been added to minimize the memorization and enhance the generalization. Fig. 6 shows the combined plot of the training and validation loss of the new model. The plot indicates significant reduction of noise and memorization. It is also prominent that the validation loss is lower than the training loss. Because usually dropout is triggered when training the samples but disengaged when validation samples are being evaluated.

![image](https://user-images.githubusercontent.com/98129458/151065434-51342c0a-fe6b-48f6-8a1f-62dbd32cfa06.png)

<br>
Fig. 6: Classification Model 3 (clasmodl3) Plot 

## Conclusion
The regression model for the wine quality assessment is analyzed, and it has been observed that the loss starts memorizing after a certain number of the epoch. The first regression model remodl1 does well, which is reasonably expected due to the model's simplicity. However, the model is expensive since it requires 50 epochs to converge at the desired level. The performance of the second regression model regmodl2 is also expected since the necessary condition of the model is overfitting. The network shows an overfitting phenomenon after 36 epochs.  The third regression model is the best model (regmodl3), which is enhanced with the hyperparameter (i.e., number of epochs=36). The regression model can be further improved by adding more training data.
<br>
<br>
On the other hand, the first classification model is simple, and the combined loss of plot is very noisy. As the model is developed with only one hidden layer containing 64 processing elements, the performance is well expected. The second classification model (clasmodl2) is a significant improvement over the first model. This model performs well with the validation set though it starts to overfit after 30 epochs. This is quite expected since the model development is focused on memorization in this sub-section. Still, the clasmodl2 has some noisy output regarding to validation set. The third classification model (clasmodl3) is the best model because it focuses both in epochs and reduction of memorization. The number of epochs has been decreased based on the previous experience with clasmodl2 model. Moreover, the noisy data observed in the second model has been greatly reduced due to effect of dropout mechanism. The dropout layer significantly reduces the overfitting of the network and enhances the generalization.

