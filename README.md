# Employee-Attrition
An ML model used to predict the attrition of an employee from a given dataset created by IBM Data Scientists which stores employee details.
Introduction

Employee Attrition is a very influential factor in deciding the annual profit earned by an organization. Loss of talented employees is a major issue faced by business leaders within such organizations. Retaining a good employee can boost business in many ways. The work is done efficiently and the quality of work is not compromised, having a good employee as a company representative leaves a good impression on clients, and major projects are completed according to client needs. This is very profitable for a company in the long run and gives returns that are much higher than the employee’s annual salary. 

Looking at the factors mentioned above, losing an employee due to reasons like dissatisfaction in the workplace is undesirable. Some factors that cause dissatisfaction are, not getting credit for work done by them, feeling underappreciated, heavy workload or lack of incentives or bonuses. Our project aims to spread awareness about these problems and in later stages, we can even give suggestions to the HR department of the companies that use our model to avoid losing employees for the reasons mentioned above. 

The proposed approach suggests the use of supervised machine learning techniques to train our model and the technique that exhibits the best results shall be used to predict the attrition of an employee. Based on the results obtained, features that are most important in determining attrition are also discussed. The study recommends a technique to select the most important features to predict the employee’s decisions. Consequently, the proposed work will allow business leaders to change the scenario of their workplace and build a safe space for their employees. Figure 1 describes the 

                             
   Figure 1 Factors Influencing Employee Attrition
Literature Review

	Turnover of employee count is an issue that is often overlooked  when calculating annual profit earned by a company. It refers to the rate at which employees leave an organization over a given period. The loss of a single hardworking employee causes at least twice the annual salary of the employee. Numerous studies found that employees who quit the organization do that because of job satisfaction. In a study by Yadav et. al(2019), it was concluded that incurred by the HR department in recruiting and training new employees is much higher than an employee’s annual salary. 
The study stated various challenges faced by hiring managers and talked about the various categories of employee attrition. The prediction was done by comparison of various Machine Learning models’ performance when it came to the reliable features in this prediction. These features were identified by RFECV (Recursive Feature Elimination with Cross Validation). Models like Logistic Regression, SVM, Random Forest, Decision Tree and AdaBoost were analyzed, where AdaBoost and Random Forest gave the best results. The features that were majorly analyzed were average monthly hours, satisfaction level, number of projects and last evaluation. The results showed how this trend can be prevented by increasing employee satisfaction levels and other factors, and how preventing it is very beneficial for the company’s future.
	
	In 2020, Jain  et. al [1], stated that employee retention could be achieved only when employee appraisal and satisfaction rates were higher. The results showed that features like satisfaction level, number of projects and work accidents contributed most towards an employee’s attrition. To the processed data, the support vector machine (SVM), decision trees (DT), and random forest (RF) algorithms were applied. Random Forest gave the best results, an accuracy of 99% was seen and it was checked through the standard confusion matrix. 

	Finally, Naive-Bayes, Logistic Regression, Multi-layer Perceptron Classifier and K-Nearest Neighbours(KNN) are also useful models for employee attrition studies. In a study by Yedida & Reddy (2018), the KNN classifier gave the best results on a dataset pulled from Kaggle. AUC and ROC curves were used to check the general predictiveness of the model. 



Research Gap
The project’s primary goal is to predict Employee Attrition. which can be very helpful to business leaders. An employee attrition system is designed to manage and reduce the rate of employee turnover in an organisation. We implement a machine learning algorithm which provides a more accurate prediction of Employee Attrition. Employee Attrition is important as it affects the productivity and profitability of the company.

Methodology 
3.1 Project Planning
When preparing to implement the project development of a machine learning-based employee attrition system, there are a number of steps that need to be taken. 
The following steps are taken:
1. Data gathering and Preprocessing
The fundamental phase in the machine learning pipeline is data gathering for training the ML model. The accuracy of the predictions provided by ML systems is only as good as the training data. The dataset chosen was created by IBM Data Scientists based on Age, Monthly Income, Distance From Home, Job Role etc.
Data preprocessing is the process of transforming raw data into a format that can be easily analysed and interpreted. In this stage, the dataset is modified according to the results given out by the feature selection techniques used. The techniques are Chi-Square and Correlation Matrix. 
The chi-square test evaluates the relationship between two or more categorical variables. Correlation Matrix evaluates the relationship between two or more continuous variables. 
2. Model Selection and Training
The model chosen depends on the qualities of the data as well as the particular specifications of the current issue. 
The supervised learning algorithms groups chosen by us include KNN, Logistic Regression, Decision Tree, Random Forest, Multilayer Perceptron and XGBoost.  
KNN- It is a non-parametric algorithm that makes predictions based on the similarities between a new sample and existing data points in the training set. The KNN algorithm first selects a value for K, which represents the number of nearest neighbours to consider when making a prediction. Then, for a new data point, the algorithm finds the K closest data points in the training set based on a chosen distance metric, such as Euclidean distance.
Logistic Regression- Logistic regression is a statistical model used to predict the probability of a binary outcome based on one or more predictor variables. It is a type of regression analysis that is commonly used in machine learning and data analysis applications.: Logistic regression can handle both categorical and continuous predictor variables, which makes it a versatile tool for analysing a wide range of data types.
Decision Tree- It is a tree-structured classifier, where internal nodes stand in for a dataset's features, branches for the decision-making process, and each leaf node for the classification result. Decision trees can be used with ensemble techniques like gradient boosting and random forests to enhance performance and lessen overfitting. Decision trees are a suitable option for big data applications since they can handle enormous datasets with many attributes. This is helpful when the relationship between the input features and the target variable is complex or challenging to model using a linear function.
Random Forest - Random Forest is a popular machine learning algorithm used for both classification and regression problems. It is a decision tree-based algorithm that generates multiple decision trees and combines their predictions to produce a final prediction. The benefit of using Random Forest is that it can handle both categorical and continuous variables and can be used to model complex non-linear relationships between variables. This algorithm’s results do not exhibit overfitting, which means that it can generalise well to new data. Additionally, it can identify the relative importance of each input variable, which can be useful in feature selection.
Multilayer Perceptron-A Multilayer Perceptron (MLP) is a type of feedforward artificial neural network which is used in machine learning and pattern recognition applications. It is a supervised learning algorithm that is used for both classification and regression problems.
XGBoost- XGBoost is an optimised gradient boosting algorithm which is designed to be highly scalable and efficient, which makes it well-suited to large and complex datasets. The XGBoost algorithm works by iteratively training decision trees on the residual errors of the previous trees. During each iteration, the algorithm assigns weights to the training examples based on their previous misclassification rate. This process continues until a user-defined stopping criterion is reached or the desired number of trees is reached.
3. Model Validation 
This is the process of assessing a machine learning model's performance using a different collection of data, called the validation dataset. The model's capacity to generalise to new data is assessed using the validation datasets, which differ from the training datasets. 
4. Model Optimization 
Model optimisation is the process of fine-tuning a machine learning model's hyper-parameters to enhance performance. Hyper-parameters are parameters that must be established prior to training since they cannot be learned during training. For this model's optimisation, Grid Search and Gradient Descent have been used:
In this method, a grid of hyper-parameters is established, and the model is trained and assessed for each set of hyper-parameters in the grid. Based on how well the model performs, the ideal set of hyper-parameters is chosen.
By iteratively modifying a model's parameters in the opposite direction as the gradient, the optimization process known as gradient descent is used in machine learning to minimise the cost function of a model. The gradient is the slope of the cost function, and the algorithm can converge to the best set of parameters by going in the direction of a negative gradient.

                         
Implementation
This paper discusses colourful literacy styles to prognosticate employee attrition rates. This section outlines the proposition behind each machine learning algorithm we've used and the results along with the data pre-processing way. The data was taken from a data set handed by IBM. There are two class markers yes and no, labelled 1 and 0 independently. The dataset had 1471 data points, each labelled yes or no.
 
4.1  Methodology
The following steps can be used to summarise the technique of an Employee Attrition Prediction System:


1. Data Gathering: 
We have trained and tested our models by gathering data from features in IBM-HR-Employee-Attrition-dataset.csv, including Business Travel, Daily Rate, Department, Education, Employee Count, Gender,  Job Role, Monthly Income and  Overtime.
For the attrition prediction system, we took into account Business Travel, MothlyIncome, Job Satisfaction, Number of Companies Worked, Over Time, Years Since the Last Promotion and various other values because these variables have a big impact on an employee leaving the company. The most important feature in our dataset is Monthly Income. According to the p-value plot,  the features Business Travel and Performance Rating  were dropped. We collected the other features and their values and then did the required pre-processing.
2. Data Pre-Processing:  
Machine learning data preprocessing involves splitting up our datasets into a training set and a test set. It is one of the most important data pretreatment stages since it allows us to improve the functionality of our machine-learning model. Our dataset is divided into training and testing data in a 75:25 ratio. We have used the StandardScaler() function in the Python Sklearn module to standardize the data values into a standard format before encoding the categorical data into numerical data. Only data values that adhere to the Normal Distribution can be standardized.
A correlation matrix is simply a table which displays the correlation coefficients for different variables. The correlation between any two variables is shown in each cell. The value ranges from -1 to 1. We use the corr() method on data-frames to generate a correlation matrix for a given dataset. It is a powerful tool to summarize a large dataset and to identify and visualize patterns in the given data. 
The chi-square distribution is a probability distribution that is commonly used in statistics to test for the independence of two categorical variables. In feature selection, it is used to identify the most important features that are relevant to the target variable.
In our project, we have used two techniques, namely Correlation Matrix and Chi-Square Distribution for the feature selection process. Then using those features, we trained and tested our models that produced their respective accuracy scores.
3. Hyperparameter Tuning: 
The goal of hyperparameter tuning was to find the hyperparameters that lead to the best performance of the model on the validation set. This was done by training and evaluating the model with different combinations of hyperparameters, and selecting the set of hyperparameters that performed the best. There are several methods for hyperparameter tuning, including grid search, random search, Bayesian optimisation, and gradient-based optimisation.
4. Model Building
The modelling process consists of selecting models that are based on various machine learning techniques. The goal of the project is to identify the best performance classiﬁer for the problem. The classiﬁcation algorithms taken into consideration are: 
A. K-Nearest Neighbors (KNN): 
The k-nearest neighbours algorithm, also known as KNN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. The KNN algorithm classifies new data based on the class of the k nearest neighbours. To use the KNN algorithm, you first need to select a value for k (the number of nearest neighbours to consider) and a distance metric (such as Euclidean, Manhattan distance etc ) to measure the similarity between instances. When a new instance is presented to the algorithm, it computes the distance between the new instance and all the instances in the training set. The k instances with the smallest distances are then selected as the nearest neighbours. This paper uses the value of k as 13.

B. Logistic Regression:
Logistic regression is a statistical technique used to analyze the relationship between a dependent variable (binary or categorical) and one or more independent variables (continuous or categorical) in a dataset. It is a type of generalized linear model (GLM) that uses a logistic function to model the probability of the dependent variable. 

Logistic regression is often used with regularization techniques to prevent overfitting. The general form of the model is -


 
                                                …………… (i)


C. Decision tree: 
Decision Tree is a type of machine learning algorithm which is a tree-structured model that makes decisions by recursively partitioning the data into subsets based on the values of input features. During training, the algorithm selects the best features to split the data at each node based on a criterion such as information gain or Gini impurity. The process of building a decision tree continues until a stopping criterion is met, such as a maximum depth or a minimum number of samples per leaf.

D. Random forest: 
Random forest is a type of ensemble learning method used in machine learning for classification, regression, and other tasks. The basic idea behind the random forest algorithm is to build a large number of decision trees, each trained on a different subset of the data and a random subset of features. During training, the algorithm randomly selects a subset of features at each node of the decision tree and then chooses the best feature among the selected subset to split the data.

This helps to reduce the correlation among the trees and improve the diversity of the ensemble. To predict a new instance, each decision tree in the random forest independently predicts the class of the instance, and the final prediction is made by taking the majority vote of the individual tree predictions.

E. Multi-layer perceptron(MLP) Classifier: 
The MLP classifier is a powerful machine learning model that can handle complex non-linear relationships in data. The MLP classifier consists of an input layer, one or more hidden layers, and an output layer. Each layer consists of several neurons, which are connected to neurons in the next layer through weighted connections. During training, the weights of the connections between the neurons are adjusted to minimize the error between the predicted output and the true output.

F. Extreme Boosting Tree: 
Extreme Gradient Boosting (XGBoost) is a popular machine learning algorithm for classification and regression problems. It is an ensemble learning method that combines multiple weak learners, in this case, decision trees, to create a strong learner. XGBoost is based on the gradient boosting algorithm and adds enhancements to improve its performance, including regularization, parallel processing, and a unique split-finding algorithm. The algorithm starts with a single decision tree and then iteratively adds more trees to the ensemble. In each iteration, the algorithm calculates the gradient of the loss function concerning the current prediction and fits a new tree to the negative gradient of the loss function.

4.2 Results

	We divided our datasets in a 75:25 ratio in data preprocessing in a train-test split. We used our test data to measure the accuracy of different models and we got a good accuracy on each dataset on each model.

	The resultant accuracy score, prediction score, recall score and F1 score produced after using two techniques,i.e. correlation matrix and chi-square distribution are as follows: 


algo
accuracy







Table 1: Correlation Matrix Model Results

Model used
Accuracy 
score
Precision
score
Recall score
F1 score
KNN
0.837
0.333
0.017
0.032
Logistic Regression
0.842
0.667
0.034
0.065
Random Forest
0.87
0.923
0.203
0.333
Decision Tree
0.821
0.387
0.203
0.267
MLP Classifier
0.84
nan
0.0
nan
Extreme Boosting Tree
0.861
0.633
0.322
0.427




Table 2: Chi-Square Model Results

Model used
Accuracy 
score
Precision
score
Recall score
F1 score
KNN
0.842
0.533
0.136
0.216
Logistic Regression
0.856
0.571
0.407
0.475
Random Forest
0.859
0.818
0.153
0.257
Decision Tree
0.84
nan
0.0
nan


4.3 Discussions

Thus, as mentioned above, we discarded two features i.e. Performance Rating and Business Travel from our dataset. Thereafter, we continued our important feature selection process with two techniques which were Correlation Matrix and Chi-Square Distribution.
According to the scores obtained by the various models used in the above two tables, we come to the conclusion that:

When we used a correlation matrix for feature selection, we trained six models and the model that gave the best performance was Random Forest, with an accuracy of 87% which is a quite high accuracy score. The second best-trained model according to the dataset was the Extreme Boosting Tree, which achieved an accuracy of 86.1%. The least-performing model was Decision Tree with an accuracy of 82.1%.

When we used chi-square distribution for feature selection, we trained 4 models and the model that gave the best performance here was also Random Forest, with an accuracy of 85.9%. The second best-trained model according to the dataset was Logistic Regression, which achieved an accuracy of 85.6%, which was only slightly lesser than Random Forest. The least-performing model in this method too was Decision Tree with an accuracy of 84%.

Feature Importance for each model: 


			  Figure 2.1 Correlation Model Feature Importance  


                       Figure 2.2 Chi-Square Model Feature Importance  

In the above figure1 and figure 2, we find that regardless of the feature selection technique used, features like Monthly Income, Age and Over Time have more importance than other features. They can be given more precedence in future to predict the attrition of an employee for effective prediction. 

Conclusion and Future Scope of the Project
5.1   Conclusion
In conclusion, the model designed performs well when Correlation Matrix is used for feature selection, compared to the Chi-Square test. When using Correlation Matrix, the Random Forest algorithm’s performance is the best with an accuracy of 87%. In terms of the F1 score, Logistic Regression under the Chi-Square test gives the best F1 score of 0.475 and its accuracy is 85.6%, which is the best among the models used in that technique. Therefore, both models are good for us in their respective use cases. This project has a lot of scope for improvement, the first step being rigorous training of the model on actual datasets and the next one will be giving these results to businesses to avoid losing valuable employees.
5.2 Future Scope
The scope of this project is immense. First of all, the important features and best-performing model could be used to create a web application where a user can give data specific to an employee and the model can predict their attrition. Secondly, these results can be used by HR to counsel the valuable employees who are facing some issues in the company and help them out. This can be done by giving incentives, reducing the workload or just a token of appreciation. This study can change the profitability of a business to a huge extent.

References
S. S. Alduayj and K. Rajpoot, "Predicting Employee Attrition using Machine Learning," 2018 International Conference on Innovations in Information Technology (IIT), Al Ain, United Arab Emirates, 2018, pp. 93-98, doi: 10.1109/INNOVATIONS.2018.8605976.

Fallucchi, Francesca, et al. "Predicting employee attrition using machine learning techniques." Computers 9.4 (2020): 86.

S. Yadav, A. Jain and D. Singh, "Early Prediction of Employee Attrition using Data Mining Techniques," 2018 IEEE 8th International Advance Computing Conference (IACC), Greater Noida, India, 2018, pp. 349-354, doi: 10.1109/IADCC.2018.8692137.

R. Jain and A. Nayyar, "Predicting Employee Attrition using XGBoost Machine Learning Approach," 2018 International Conference on System Modeling & Advancement in Research Trends (SMART), Moradabad, India, 2018, pp. 113-120, doi: 10.1109/SYSMART.2018.8746940.

Jain, Praphula Kumar, Madhur Jain, and Rajendra Pamula. "Explaining and predicting employees’ attrition: a machine learning approach." SN Applied Sciences 2 (2020): 1-11.

Yedida, Rahul & Reddy, Rahul & Vahi, Rakshit & Jana, Rahul & Gv, Abhilash & Kulkarni, Deepti. (2018). Employee Attrition Prediction.
