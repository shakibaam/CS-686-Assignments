
from collections import defaultdict
import math


def generate_word_dict(file_name):
    """
    Generates a dictionary mapping word IDs to words from a given file.

    Args:
        file_name (str): The name of the file containing words.

    Returns:
        dict: A dictionary where keys are word IDs (starting from 1) and values are the corresponding words.
    """
    word_dict = {}
    with open(file_name, "r") as file:
        for word_id, word in enumerate(file, start=1):  # Start enumeration from 1 (since word IDs start from 1)
            word_dict[word_id] = word.strip()  # Strip any whitespace or newline characters
    
    return word_dict

def generate_words_vectors(file_name,train = True):
    """
    Generates a dictionary mapping document IDs to lists of word IDs from a given file.

    Args:
        file_name (str): The name of the file containing document ID and word ID pairs, one per line.
        train (bool): A flag indicating whether the function is processing training data (True) or test data (False).

    Returns:
        dict: A dictionary where keys are document IDs and values are lists of word IDs associated with those documents.
              If a document ID is in the predefined list of missing documents (for training or testing), its value will be an empty list.
    """
     
    # Initialize a dictionary to hold docID and list of wordIDs
    doc_word_dict = defaultdict(list)
    missing_docs_train = [1017]
    missing_docs_test = []
    with open(file_name, "r") as file:
        
        for line_number, line in enumerate(file, start=1):
            
            doc_id, word_id = map(int, line.split())
           
           
            doc_word_dict[doc_id].append(word_id)
            if train:
                if doc_id in missing_docs_train:
                    doc_word_dict[doc_id] = []
            else:
                if doc_id in missing_docs_test:
                    doc_word_dict[doc_id] = []
            
    return doc_word_dict

def generate_label_vectors(file_name):
    """
    Generates a dictionary mapping document IDs to their corresponding labels from a given file.

    Args:
        file_name (str): The name of the file containing labels, one per line.

    Returns:
        dict: A dictionary where keys are document IDs (starting from 1) and values are the corresponding labels.
    """
    label_dict = {}
    
    with open(file_name, "r") as file:
        for doc_id, line in enumerate(file, start=1):
            label = int(line.strip())
            label_dict[doc_id] = label
    return label_dict

def class_prior_probabilty(labels):
    """
    Calculates the prior probabilities for each class label based on the provided labels.

    Args:
        labels (dict): A dictionary where keys are document IDs and values are the corresponding class labels.

    Returns:
        dict: A dictionary where keys are class labels and values are their prior probabilities.
    """
   
    # # Initialize a dictionary to count occurrences of each class label
    label_counts = {}
    # Get the total number of documents (labels)
    total_documents = len(labels)
    
    # Iterate through each label to count occurrences
    for label in labels.values():
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Calculate prior probabilities for each class
    prior_probabilities = {label: count / total_documents for label, count in label_counts.items()}
    
    return prior_probabilities

def filter_docs_by_label(labels, target_label):
    """
    Filters the document IDs based on a specified target label.

    Args:
        labels (dict): A dictionary where keys are document IDs and values are the corresponding class labels.
        target_label: The label to filter documents by.

    Returns:
        list: A list of document IDs that have the specified target label.
    """
    
    filtered_docs = [doc_id for doc_id, label in labels.items() if label == target_label]
    return filtered_docs

def compute_word_given_class_probabilities(wordID,docs_in_class,doc_word_dict_train):
     """
    Computes the probability of a word given a specific class based on the training documents.

    Args:
        wordID (int): The ID of the word for which the probability is being calculated.
        docs_in_class (list): A list of document IDs that belong to the specified class.
        doc_word_dict_train (dict): A dictionary where keys are document IDs and values are lists of word IDs in each document.

    Returns:
        float: The probability of the word given the class.
    """
     # Number of documents in the class
     num_docs_in_class = len(docs_in_class)
     # Count the number of documents in the class that contain the specified word
     docs_with_word = [doc_id for doc_id in docs_in_class if wordID in doc_word_dict_train.get(doc_id,[])]
     # Calculate the probability using Laplace smoothing
     probability = (len(docs_with_word) + 1) / (num_docs_in_class + 2)
    
     return probability

def precompute_word_probabilities(word_dict, atheism_docs, books_docs, doc_word_dict_train):
    """
    Precomputes the probabilities of each word given the two classes (atheism and books).

    Args:
        word_dict (dict): A dictionary where keys are word IDs and values are the corresponding words.
        atheism_docs (list): A list of document IDs that belong to the 'atheism' class.
        books_docs (list): A list of document IDs that belong to the 'books' class.
        doc_word_dict_train (dict): A dictionary where keys are document IDs and values are lists of word IDs in each document.

    Returns:
        dict: A dictionary containing the probabilities of each word for both classes.
    """

    # Initialize a dictionary to hold probabilities for both classes
    probs = {1: {}, 2: {}}
    for wordID in word_dict:
        # Compute the probability of the word given the atheism class
        probs[1][wordID] = compute_word_given_class_probabilities(
            wordID, atheism_docs, doc_word_dict_train)
        # Compute the probability of the word given the books class
        probs[2][wordID] = compute_word_given_class_probabilities(
            wordID, books_docs, doc_word_dict_train)
    return probs

def compute_posterior_probabilty(docID,label,doc_word_dict_train,word_dict,docs_in_class,class_prior,probs):
    """
    Computes the posterior probability of a document belonging to a specific class.

    Args:
        docID (int): The ID of the document for which the probability is being calculated.
        label (int): The class label for which the posterior probability is computed.
        doc_word_dict_train (dict): A dictionary where keys are document IDs and values are lists of word IDs in each document.
        word_dict (dict): A dictionary where keys are word IDs and values are the corresponding words.
        docs_in_class (list): A list of document IDs that belong to the specified class.
        class_prior (float): The prior probability of the class.
        probs (dict): A dictionary containing the probabilities of each word given the classes.

    Returns:
        float: The posterior probability of the document given the class.
    """
    vocabulary = set(word_dict.keys())
    words_in_doc = set(doc_word_dict_train.get(docID,[]))
   
    # score = class_prior
    score = math.log2(class_prior)
    for wordID in vocabulary:
        
        word_likelihood = probs[label][wordID]
        
        if wordID in words_in_doc:
            score += math.log2(word_likelihood)
            
        else:
            
            score += math.log2(1 - word_likelihood)

    return score
   
    
def predict_class(docID, doc_word_dict_train, word_dict, atheism_docs, books_docs, prior_probabilities,probs):
    """
    Predicts the class label for a given document based on its content.

    Args:
        docID (int): The ID of the document for which the class is being predicted.
        doc_word_dict_train (dict): A dictionary where keys are document IDs and values are lists of word IDs in each document.
        word_dict (dict): A dictionary where keys are word IDs and values are the corresponding words.
        atheism_docs (list): A list of document IDs that belong to the 'atheism' class.
        books_docs (list): A list of document IDs that belong to the 'books' class.
        prior_probabilities (dict): A dictionary containing prior probabilities for each class.
        probs (dict): A dictionary containing the probabilities of each word given the classes.

    Returns:
        int: The predicted class label (1 for atheism, 2 for books).
    """
    
    # Compute posterior probabilities for both classes
    score_class_1 = compute_posterior_probabilty(docID, 1, doc_word_dict_train, word_dict, atheism_docs, prior_probabilities[1],probs)
    score_class_2 = compute_posterior_probabilty(docID, 2, doc_word_dict_train, word_dict, books_docs, prior_probabilities[2],probs)
    
    # Return the class with the higher score
    return 1 if score_class_1 > score_class_2 else 2

def compute_accuracy(doc_word_dict_test, test_labels, word_dict, atheism_docs, books_docs, prior_probabilities,probs):
    
    correct_predictions = 0
    total_documents = len(doc_word_dict_test)
    
    # Predict the class for each document in the test set and compare with true labels
    for docID in doc_word_dict_test:
        predicted_label = predict_class(docID, doc_word_dict_test, word_dict, atheism_docs, books_docs, prior_probabilities,probs)
        # print(predicted_label)
        true_label = test_labels[docID]
        if predicted_label == true_label:
            correct_predictions += 1
    
    # Calculate accuracy as a percentage
    accuracy = (correct_predictions / total_documents) * 100
    return accuracy


def compute_discriminative_words(word_dict, atheism_docs, books_docs, doc_word_dict_train):
    """
    Computes and prints the 10 most discriminative word features between two classes.

    Args:
        word_dict (dict): A dictionary where keys are word IDs and values are the corresponding words.
        atheism_docs (list): A list of document IDs that belong to the 'atheism' class.
        books_docs (list): A list of document IDs that belong to the 'books' class.
        doc_word_dict_train (dict): A dictionary where keys are document IDs and values are lists or sets of word IDs in each document.
    """
    discriminative_scores = []
    
    for wordID in word_dict.keys():
        
        prob_word_given_label1 = compute_word_given_class_probabilities(wordID, atheism_docs, doc_word_dict_train)
        prob_word_given_label2 = compute_word_given_class_probabilities(wordID, books_docs, doc_word_dict_train)
        
       
        log_diff = abs(math.log2(prob_word_given_label1) - math.log2(prob_word_given_label2))
        
        
        discriminative_scores.append((word_dict[wordID], log_diff))
    
   
    discriminative_scores.sort(key=lambda x: x[1], reverse=True)
    
   
    print("The 10 most discriminative word features:")
    for i in range(10):
        word, score = discriminative_scores[i]
        print(f"{i+1}. Word: '{word}', Discriminative Score: {score:.4f}")
 

if __name__ == '__main__':

    doc_word_dict_train = generate_words_vectors('p1/trainData.txt',train = True) 
    doc_word_dict_test = generate_words_vectors('p1/testData.txt',train = False)
    train_labels = generate_label_vectors('p1/trainLabel.txt')
    test_labels = generate_label_vectors('p1/testLabel.txt')
    word_dict = generate_word_dict('p1/words.txt')

    prior_probabilities = class_prior_probabilty(train_labels)
    print(prior_probabilities)
    
    atheism_docs = filter_docs_by_label(train_labels, 1)  # Get docs with label 1 (e.g., atheism)
    
    books_docs = filter_docs_by_label(train_labels, 2) 

    probs = precompute_word_probabilities(word_dict, atheism_docs, books_docs, doc_word_dict_train)
   

accuracy_train = compute_accuracy(doc_word_dict_train, train_labels, word_dict, atheism_docs, books_docs, prior_probabilities,probs)
print("Accuracy on the train set:", accuracy_train)

accuracy_test = compute_accuracy(doc_word_dict_test, test_labels, word_dict, atheism_docs, books_docs, prior_probabilities,probs)
print("Accuracy on the test set:", accuracy_test)

compute_discriminative_words(word_dict, atheism_docs, books_docs, doc_word_dict_train)