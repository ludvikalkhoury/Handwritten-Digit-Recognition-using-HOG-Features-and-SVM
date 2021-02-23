# Handwritten-Digit-Recognition-using-HOG-Features-and-SVM
In this repositoty, I will provide a MatLab and a Python for handwritten digit recognition using HOG features and SVM. Both MatLab and Python codes have the same stucture and divided into three (3) sections:  
**STEP 1: Data Preparation**  
**STEP 2: HOG Features Calculation**  
**STEP 3: Set up and run SVM**  


### **STEP 1: Data Preparation**  
In the first section of the code, the MNIST dataset [1] is loaded. The dataset is divided into a training set and a testing test along with their labels. The total number of digits in the training and testing set is 60000 and 10000, respectively. The labels are the ten (10) digits (0 to 9). In MatLab, each digit is represented by a vector of 784 elements. The 784-element vector will be resized later in the code to form a 28x28 pixel image. In Python, the the resizing step is skipped since each digit is represented by a 28x28 pixel image. 
 
### **STEP 2: HOG Features Calculation**  
Histogram of Oriented Gradients (HOG) feature vectors [2] are computed from each 28x28 pixel image. Each vector is comprised of 324 elements. The whole 324-element feature vector will be used to train the Support Vector Machine (SVM) later. 
 
### **STEP 3: Set up and run SVM** 
Support Vector Machine (SVM) [3] is the multi-class classifier I employed in this example to classify the handwritten digits. I used the one-vs-one method [4] and a linear kernel. Confusion matrix is computed at the end of the code from the Test dataset (10000 digits).

### **Confusion Matrix** 
I included the confusion matrices computed using the MatLab and the Python code. Note that the difference is due to the different algorithms that MatLab and Python use in order to compute the optimal weights used in SVM. 


## Reference:  
[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.  
[2] McConnell, R. K. (1986). *U.S. Patent No. 4,567,610*. Washington, DC: U.S. Patent and Trademark Office.  
[3] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, 20(3), 273-297.  
[4] Alpaydin, E. (2020). *Introduction to machine learning*. MIT press.


## Keywords:  
Support Vector Machine (SVM), Python, MatLab, Machine Learning (ML), Artificial Intelligence (AI), Handwritten Digit Recognition, Histogram of Oriented Gradients (HOG) Features, MNIST Dataset.  
