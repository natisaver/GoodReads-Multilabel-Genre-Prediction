# GoodReads-Multilabel-Genre-Prediction
## A multilabel genre predictor of books for DSAI Project

In this project, we predict the genres (self-curated top 30) that a book can be classified into based on the plot description, brightness (luminance) of cover images and numRatings. The data set is obtained from [Zenodo](https://zenodo.org/record/4265096/files/books_1.Best_Books_Ever.csv?download=1). 

<img height=400 src="./Images/title.png"/>

##### Table of Contents  
[Files](### Jupyter Notebook Files:)  
[Overview](###Project Overview:)  
...snip...    
<a name="headers"/>
## Headers

### Jupyter Notebook Files:
1. Data Preprocessing, Scraping & Cleaning [link to ipynb1](https://github.com/natisaver/GoodReads-Multilabel-Genre-Prediction/blob/main/Notebooks/1_Data_Preprocessing.ipynb)
2. Exploratory Analysis [link to ipynb2](https://github.com/natisaver/GoodReads-Multilabel-Genre-Prediction/blob/main/Notebooks/2_EDA.ipynb)
3. Models & Results [link to ipynb3](https://zenodo.org/record/4265096/files/books_1.Best_Books_Ever.csv?download=1)

###Project Overview:
---
<img height=400 src="./Images/overview2.png"/>
---

`MinMax Scaling` was utilised on X data.

Three techniques are used to classify the movies into various multi-labels:
* **Binary Relevance**: This consists of fitting one classifier per class, via `OneVsRest`. For each classifier, the class is fitted against all the other classes. The union of all classes that were predicted is taken as the multi-label output. `Logistic Regression` Machine Learning Model was used.

* **Label Powerset**: In this approach, we transform the multi-label problem to a multi-class problem with 1 multi-class classifier trained on all unique label (genre) combinations found in the training data. Each plot in the test data set is classified into one of these unique combinations. `Naive Bayes` algorithm was used.

* **Label Powerset with Clustering**: We use clustering technique (`k-means`) to reduce the number of possible classes into a manageable number. `Linear SVC` was then used.


---

Models were finally evaluated for their `F1-Scores`.

| Pipeline | PRecision |  Recall | F1-Score |
| ------------- | ------------- | ------------- | ------------- |
| TF-IDF + Binary Relevance + Logistic Regression  | 0.65  | 0.26  | 0.29 |
| TF-IDF + Label Powerset + Naive Bayes  | 0.58 | 0.42 | 0.43 | 
| TF-IDF + Label Powerset with Clustering + Linear SVC  | 0.29 | 0.26 | 0.27 |

References:
- https://scikit-learn.org/stable/modules/multiclass.html
---
by Nathaniel, Marcus & Yan Chi
