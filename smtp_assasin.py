import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

# Paths to the directories containing the emails
s_ep = os.path.join("trn", "s_ep") # Change of keywords after your own files and dirs you desire
v_ep = os.path.join("trn", "v_ep") # Change of keywords after your own files and dirs you desire

# List of file directories and corresponding labels
l_fd = [(s_ep, 0), (v_ep, 1)] # Change of keywords after your own files and dirs you desire
e_cs = []
l_bs = []


# Function to read email content and labels
def cve_sh_trfm(s_ep, v_ep, l_fd, e_cs, l_bs): 
    for cfs, lbs in l_fd:
        files = os.listdir(cfs)
        for file in files:
            file_path = os.path.join(cfs, file)
            try:
                with open(file_path, "r") as current_file:
                    eml_ctt = current_file.read().replace("\n", "")
                    eml_ctt = str(eml_ctt)
                    e_cs.append(eml_ctt)
                    l_bs.append(lbs)
            except:
                pass


# Splitting the dataset and training the model
def vary_trmd(cve_sh_trfm, s_ep, v_ep, l_fd, e_cs, l_bs):
    x_train, x_test, y_train, y_test = train_test_split(
        e_cs, l_bs, test_size=0.4, random_state=17
    )
    nlp_followed_by_dt = Pipeline(
        [
            ("vect", HashingVectorizer(input="eml_ctt", ngram_range=(1, 4))),
            ("tfidf", TfidfTransformer(use_idf=True)),
            ("dt", tree.DecisionTreeClassifier(class_weight="balanced")),
        ]
    )
    nlp_followed_by_dt.fit(x_train, y_train)

    # Predicting and evaluating the model
    y_test_predict = nlp_followed_by_dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_test_predict)
    confusion = confusion_matrix(y_test, y_test_predict)

    # Writing accuracy and confusion matrix to files
    with open("accuracy_score.txt", "w") as acc_file:
        acc_file.write("Accuracy: " + str(accuracy))

    with open("confusion_matrix.txt", "w") as conf_file:
        conf_file.write("Confusion Matrix:\n" + str(confusion))
