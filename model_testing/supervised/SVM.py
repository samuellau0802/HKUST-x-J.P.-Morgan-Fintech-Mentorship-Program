import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
sns.set() # use seaborn plotting style
from imblearn.under_sampling import NearMiss

def SVM_vect_tfidf(X_train, X_test, y_train, y_test, sample_method=0):
    # Build the model
    if sample_method == 0:
        model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])
    elif sample_method == 1:
        model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('smote', SMOTE(random_state=12)),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])
    elif sample_method == 2:
        model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('nm', NearMiss(random_state=12)),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])

    # Train the model using the training data
    model.fit(X_train, y_train)
    # Predict the symbols of the test data
    y_pred = model.predict(X_test)
    # accuracy
    #print("The accuracy is {}".format(accuracy_score(y_test, y_pred)))
    
    # plot the confusion matrix
    mat = confusion_matrix(y_test, y_pred)
    #sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=set(y_train),yticklabels=set(y_train))
    #plt.xlabel("true labels")
    #plt.ylabel("predicted label")
    #plt.show()

    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred,target_names=set(y_train)), mat
