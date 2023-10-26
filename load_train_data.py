# -*- coding: utf-8 -*-
# @Author: Hannah Shader
# @Date:   2023-10-18 12:02:15
# @Last Modified by:   Hannah Shader
# @Last Modified time: 2023-10-26 18:35:52
import os
import pandas as pd
import string
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import sklearn.linear_model
import nltk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")


# Preprocessing Function
def text_list(text_list):
    new_text_list = []
    for sentence in text_list:
        new_sentence = ""
        tokens = nltk.word_tokenize(sentence)
        for token, pos in nltk.pos_tag(tokens):
            if not token.isupper() and len(token) != 1:
                token = token.lower()
            token = "".join(char for char in token if char not in string.punctuation)
            new_sentence += token + " "
        new_text_list.append(new_sentence.strip())
    return new_text_list


# Error on Training Data for each classifier
# Not needed for classification, but used to visualize error for training
# vs. validation data
def get_training_auroc_for_classifier(
    clf_list, dataframe, preprocessed_texts, tr_values_list
):
    new_col_values = []
    for clf, preprocessed_tr_text_list in zip(clf_list, preprocessed_texts):
        embedding_instance = get_bert_embedding(preprocessed_tr_text_list)
        clf.fit(embedding_instance, tr_values_list)

        # Make predictions on the training data
        train_predictions = clf.predict(embedding_instance)

        # Calculate ROC AUC score on the training data
        roc_auc = roc_auc_score(tr_values_list, train_predictions)
        new_col_values.append(roc_auc)

    dataframe["AUROC testing"] = new_col_values
    return dataframe


# Get
def preprocess_text_list(text_list):
    lemmatizer = WordNetLemmatizer()
    new_text_list = []
    for sentence in text_list:
        new_sentence = ""
        tokens = nltk.word_tokenize(sentence)
        for token, val in nltk.pos_tag(tokens):
            token = lemmatizer.lemmatize(token)
            new_sentence += token + " "
        new_text_list.append(new_sentence.strip())
    return new_text_list


def get_bert_embedding(sentence_list, pooling_strategy="cls"):
    embedding_list = []
    for nn, sentence in enumerate(sentence_list):
        if (nn % 100 == 0) & (nn > 0):
            print("Done with %d sentences" % nn)

        # Tokenize the sentence and get the output from BERT
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Take the embeddings from the last hidden state (optionally, one can use pooling techniques for different representations)
        # Here, we take the [CLS] token representation as the sentence embedding
        last_hidden_states = outputs.last_hidden_state[0]

        # Pooling strategies
        if pooling_strategy == "cls":
            sentence_embedding = last_hidden_states[0]
        elif pooling_strategy == "mean":
            sentence_embedding = torch.mean(last_hidden_states, dim=0)
        elif pooling_strategy == "max":
            sentence_embedding, _ = torch.max(last_hidden_states, dim=0)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        embedding_list.append(sentence_embedding)
    return torch.stack(embedding_list)


if __name__ == "__main__":
    data_dir = "data_reviews"
    x_train_df = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    x_test_df = pd.read_csv(os.path.join(data_dir, "x_test.csv"))

    original_tr_text_list = x_train_df["text"].values.tolist()
    original_tr_values_list = y_train_df["is_positive_sentiment"].values.tolist()

    combined = [
        (original_tr_text_list[i], original_tr_values_list[i])
        for i in range(len(original_tr_text_list))
    ]

    # Step 2: Sort the combined array based on the values from the first array
    random.shuffle(combined)

    # Step 3: Reconstruct the sorted arrays
    shuffled_tr_text_list = [x[0] for x in combined]
    shuffled_tr_values_list = [x[1] for x in combined]

    # getting heldout
    # 60 data points is 10 percent
    heldout_tr_text_list = shuffled_tr_text_list[:60]
    heldout_tr_values_list = shuffled_tr_values_list[:60]

    tr_text_list = shuffled_tr_text_list[60:]
    tr_values_list = shuffled_tr_values_list[60:]
    new_tr_text_list = preprocess_text_list(tr_text_list)
    embeddings = get_bert_embedding(new_tr_text_list)

    max_feat_list = [1827]
    chosen_C_values = [0.01, 0.1, 1, 10, 100]

    combinations = []
    for max_feat in max_feat_list:
        for c in chosen_C_values:
            combinations.append([max_feat, c])

    hyperparam_list = []
    models = []
    roc_auc_scores = []
    preprocessed_text_lists = []

    results_df = pd.DataFrame(columns=["max_feat", "C", "AUROC validation"])
    roc_auc_scores = []

    for combination in combinations:
        max_feat = combination[0]
        c = combination[1]

        tr_text_list = preprocess_text_list(tr_text_list)
        preprocessed_text_lists.append(tr_text_list)
        clf = sklearn.linear_model.LogisticRegression(C=c, max_iter=20)

        scores = cross_val_score(
            clf,
            embeddings,
            tr_values_list,
            cv=5,
            scoring="roc_auc",
        )

        mean = np.mean(scores)
        results_df = results_df.append(
            {"max_feat": max_feat, "C": c, "AUROC validation": mean},
            ignore_index=True,
        )
        hyperparam_list.append([max_feat, c])
        models.append(clf)
        roc_auc_scores.append(np.mean(scores))
        mean_auroc = np.mean(roc_auc_scores)
        print("mean is: ", mean_auroc)
        print("c is:", c)
        print("max_feat is:", max_feat)

    # get the variables back to the chosen model
    print("roc_auc_scores is: ", roc_auc_scores)
    max_index = max(range(len(roc_auc_scores)), key=lambda i: roc_auc_scores[i])
    print("max_index is:", max_index)
    clf = models[max_index]
    max_feat = hyperparam_list[max_index][0]
    c = hyperparam_list[max_index][1]
    print("c is:", c)
    tr_text_list = preprocess_text_list(tr_text_list)

    # get a datafram to store data
    results_df = get_training_auroc_for_classifier(
        models, results_df, preprocessed_text_lists, tr_values_list
    )
    grouped = results_df.groupby("C").mean().reset_index()

    # Plot the accuracy vs. C value
    plt.figure(figsize=(10, 6))
    plt.plot(
        grouped["C"], grouped["AUROC validation"], marker="o", label="AUROC Validation"
    )
    plt.plot(grouped["C"], grouped["AUROC testing"], marker="o", label="AUROC Testing")
    plt.xlabel("C Value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. C Value")
    plt.xscale("log")  # Use a logarithmic scale for C values (since they vary widely)
    plt.legend()
    plt.grid(True)
    plt.show()

    # reset heldout data back to all values shuffled
    tr_text_list = shuffled_tr_text_list
    tr_values_list = shuffled_tr_values_list
    new_tr_text_list = preprocess_text_list(tr_text_list)
    embeddings = get_bert_embedding(new_tr_text_list)

    # retrain model with all values
    tr_text_list = preprocess_text_list(tr_text_list)
    preprocessed_text_lists.append(tr_text_list)
    clf = sklearn.linear_model.LogisticRegression(C=c, max_iter=20)

    # fit the data
    clf.fit(embeddings, tr_values_list)
    te_text_list = x_test_df["text"].values.tolist()

    # get FP and FN values for report from heldout data
    predictions = clf.predict(embeddings)
    print(predictions)
    print(tr_values_list)
    false_positive_indices = [
        i for i in range(60) if predictions[i] == 1 and tr_values_list[i] == 0
    ]
    false_negative_indices = [
        i for i in range(60) if predictions[i] == 0 and tr_values_list[i] == 1
    ]

    # display these FP and FN values
    print("false positive indices are:")
    print(false_negative_indices)
    print("false negative indices are:")
    print(false_negative_indices)
    print("false positive are:")
    for i in range(len(false_positive_indices)):
        print(tr_text_list[false_positive_indices[i]])
    print("false negatives are:")
    print(false_negative_indices)
    for i in range(len(false_negative_indices)):
        print(tr_text_list[false_negative_indices[i]])

    # predict on test data
    new_te_text_list = preprocess_text_list(te_text_list)
    test_embeddings = get_bert_embedding(new_te_text_list)
    te_text_list = preprocess_text_list(te_text_list)

    predictions = clf.predict_proba(test_embeddings)
    positive_class_probs = predictions[:, 1]

    output_file = "yproba1_test.txt"
    np.savetxt(output_file, positive_class_probs, fmt="%s")
