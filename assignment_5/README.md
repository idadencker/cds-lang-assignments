# Evaluating Environmental impact


## Introduction
This program contains a notebook that will load all carbon files containing information of the CO2 emissions from the previous 4 assignments. In the notebook results from the files are visualised to determine which assignments and which specific task are particularly CO2 demanding. The plots are saved in the out folder. <br>
The files are based on and created using the CodeCarbon package which estimate and track carbon emissions produced by the cloud or personal computing resources used to execute some code. More information on the CodeCarbon package are available [here](https://codecarbon.io/). <br>
The emissions are tracked by subtasks to be able to determine what, if any, tasks produce more CO2 emissions than other. However also an overall emission estimate is calculated to evaluate the overall CO2 emission estimate for each assignment. <br>
The results from implementing CodeCarbon are summarised and discussed.


## Repository overview 
The repository consists of:
- 1 README.md file
- out folder for holding the csv CO2 emission files as well as plots visualising the findings
- src folder containing a notebook for visualising results 


## Summary and discussion
First of all, looking at the task-specific emissions, it is evident that very big differences in demanded computing resources do exist. Two different tasks stand out as the far most demanding tasks: Extracting the emotions on the Game of Thrones dataset from assignment 4 with a total emission of approximately 0.024 kilograms of CO2 as well as fitting the neural network model on the fake or real news dataset from assignment 2 with a total emission of 0.010 kilograms. The rest of the tasks falls below an emission of 0.002 kilograms. The explanation for the emission-heavy task of extracting emotions can be linked to the very large dataset of 23.911 lines to be analysed in terms of emotions. The model used in the assignment is a distilled version of the RoBERTa model. The RoBERTa model is an architecturally complex transformer-based model using multiple layers consisting of self-attention mechanisms and neural networks. Working through the multilayered model to extract an emotion for each of more than 23 thousand sentences will entail a big need for computational resources.<br>
Though the distilled version of the RoBERTa model is computationally faster than the RoBERTa model the emissions reports conclude that it is still resource demanding compared to other tasks. <br>
The second most CO2 demanding task involves fitting the neural network on the fake or real news dataset. Similarly, we are dealing with a large dataset of 6335 articles. Furthermore, neural networks are known to be computationally expensive due to the multilayered deep-learning architecture, however highly dependent on the complexity of the model. In this case hyperparameter tuning was implemented totaling 810 fits and some of the more computationally heavy models fits could e.g. include a maximum of 100 neurons in the hidden layer. To reduce computational resources we could limit the number of parameters considered for hyperparameter tuning or disregard the fine-tuning process completely, however on the possible expense of accuracy. <br>
In accordance with the task-specific emissions, analysing the total emission per assignment lists assignment 4 with a total emission of 0.024 kilograms of CO2 and assingment 2 with 0.010 kilograms as the most CO2 demanding assignments. Assignment 3 proves to be the least demanding of the four. Opposite to the other assignments, assignment 3 is not looping through and extracting information on a whole dataset but rather is limited to only include texts from a specific artist from the Spotify dataset. This contributes to lowering the computational resources. <br>
As it also appears on the CodeCarbon [documentation](https://mlco2.github.io/codecarbon/faq.html), the emissions are estimates and must be taken with some precautions. Many factors are at play when quantifying emissions, and it is possible that future improvements can and will be made to better the estimates. However the emissions estimates still provides a good over-all picture of the total CO2 emissions, revealing big differences in computational resources based on tasks and assignment. 

