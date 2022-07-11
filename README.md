README - Overall
----------------------------------------------------------------------------------------------------------------
Our file contains two part: 1) code    2) Result  
  
1. code file contains all the implemented methods and models with coresponding names. Some keys or steps are presented in three jupyter notebook file.  

record.py : Two function to save our experiment result and load to print.

models.py : Related models' implementation.

datasplit.py : Related data generation (IID, unbalanced non-IID, highly degree of non-IID: 1 class, 2 class, 3class)

FL.py : LocalUpdate process is implemented.

aggregation.py : Median, Trimmed mean and Krum.

model_attack.py : Implement the model attack.

  
2) Result file contains the experiment results (accuracy) in .txt file and checkpoint file and .pth are all in the folder.  
  
Some figure and experiments result are reserved in the jupyter notebook file.  

README - Federated Learning
-----------------------------------------------------------------------------------------------------------------
Standard_FL.ipynb : The Federated Learning and the experiments about impact factors are in the code file.
The detailed implementation is in the file with detailed instruction and explanation.




README - Data Poisoning Attacks
-----------------------------------------------------------------------------------------------------------------

Data_Poisoning_Attack.ipynb : Data poisoning attack process based on standard federated learning.

The data poisoning attacks against federated learning has been implemented inside the federated training section. Detailed attack settings such as defining malicious participant percentage, injecting poisoned data, as well as manipulating attack timing could be explicitly noticed within the federated training block, along with clear comments explaining the purpose of codes.  

Several helper functions are defined to help with the analysis of data poisoning attacks.  
The “targeted_label_flipping_attack(poison_set, source, target)” function is used to implement targeted label flipping attack, it flips all data points in the training set with source class label into the target class label.  
The “global_model_accuracy(model, loader, device)” function helps calculate the global model’s testing accuracy.  
The “class_recall(model, loader, device, class_X)” function helps calculate the prediction accuracy of one specific class.  
The “accuracy_array_list(acc_array)” function helps convert an array of accuracies into a list of accuracies in the form of percentage.  
The “float_format(accuracies)” function helps format each float accuracy with 4 digits after radix point.  
  
Several txt files are included, they record the global model’s validation/testing accuracies or the source class’s recalls during federated learning.  
Several pre-trained model checkpoints are also included, they can be directly loaded into the notebook and used for model evaluation (validation and testing).  


README - Model Poisoning Attacks
-----------------------------------------------------------------------------------------------------------------

Model_Poisoning_Attack.ipynb : Model poisoning attack process based on standard federated learning with different type of aggregation.
 The detailed implementation is in the file with detailed instruction and explanation.  
 
