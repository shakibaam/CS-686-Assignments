from collections import defaultdict
import math
import heapq
# import graphviz
from matplotlib import pyplot as plt
from itertools import count


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
    missing_docs_train = [55, 101, 148, 159, 176, 194, 284, 299, 327, 348, 350, 378, 469, 515, 540, 549, 564, 584, 666, 700, 749, 764, 768, 801, 810, 892, 970, 980, 1021, 1051, 1081, 1154, 1224, 1266, 1272, 1343, 1357, 1406, 1436, 1453]
    missing_docs_test = [10, 106, 163, 174, 175, 215, 221, 236, 245, 258, 296, 319, 355, 367, 390, 393, 406, 411, 442, 451, 539, 610, 622, 659, 661, 670, 696, 697, 708, 715, 725, 936, 938, 942, 950, 1004, 1069, 1102, 1148, 1154, 1184, 1206, 1211, 1291, 1303, 1306, 1349, 1413, 1437, 1440]
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
    # Open and read the file line by line
    with open(file_name, "r") as file:
        for doc_id, line in enumerate(file, start=1):
            label = int(line.strip())
            label_dict[doc_id] = label
    return label_dict

def compute_total_entropy(labels):
    """
    Computes the total entropy of a set of labels.


    Args:
        labels (dict): A dictionary where keys are document IDs and 
                       values are the corresponding labels (1 for 
                       'atheism', 2 for 'books').

    Returns:
        float: The total entropy of the labels. If there are no documents, 
               the function returns 1 (indicating maximum uncertainty).
               The entropy is calculated using the formula:
               I(E) = -Σ(P(y) * log2(P(y))) for each label present.
    """
   
    num_atheism = sum(1 for label in labels.values() if label == 1)
    num_books = sum(1 for label in labels.values() if label == 2)
    total_docs = len(labels)
    I_E_total = 1
   
    if(total_docs==0): #Empty set entropy is 1
        return 1

    P_y_atheism = num_atheism/total_docs
    P_y_book = num_books/total_docs

   
    if(P_y_atheism != 0 and P_y_book != 0):
        
        I_E_total= -1 * P_y_atheism * math.log2(P_y_atheism) -1 * P_y_book * math.log2(P_y_book)
    elif(P_y_atheism == 0):
        
        I_E_total= 0
    elif(P_y_book == 0):
        
         I_E_total= 0

    return I_E_total


# Information Gain Method one
def compute_information_gain_one(wordID,doc_word_dict_train,train_labels):
    """
    Computes the information gain for a specific word ID based on the training data.

    Args:
        wordID (int): The ID of the word for which information gain is being calculated.
        doc_word_dict_train (dict): A dictionary mapping document IDs to lists of word IDs for the training dataset.
        train_labels (dict): A dictionary mapping document IDs to their corresponding labels.

    Returns:
        tuple: A tuple containing:
            - Information gain (float): The calculated information gain for the specified word.
            - docs_with_word_dic_wordID (dict): A dictionary of documents containing the specified word.
            - docs_without_word_dic_wordID (dict): A dictionary of documents not containing the specified word.
            - docs_with_word_dict_labels (dict): A dictionary of labels for documents containing the specified word.
            - docs_without_word_dict_labels (dict): A dictionary of labels for documents not containing the specified word.
    """
    
    # split documents with and without that word
    docs_with_word = [doc_id for doc_id, words in doc_word_dict_train.items() if wordID in words]
    # Get all document IDs from doc_word_dict_train
    all_docs = set(doc_word_dict_train.keys())
    # Fill docs_without_word by finding documents that aren't in docs_with_word
    docs_without_word = list(all_docs - set(docs_with_word))
    


    # Create a dictionary mapping document IDs to their corresponding labels for documents containing the word
    docs_with_word_dict_labels = {doc_id: train_labels[doc_id] for doc_id in docs_with_word}
    # Create a dictionary mapping document IDs to their corresponding labels for documents not containing the word
    docs_without_word_dict_labels = {doc_id: train_labels[doc_id] for doc_id in docs_without_word}
    # Create a dictionary mapping document IDs to their word lists for documents containing the word
    docs_with_word_dic_wordID = {doc_id:words for doc_id, words in doc_word_dict_train.items() if doc_id in docs_with_word}
    # Create a dictionary mapping document IDs to their word lists for documents not containing the word
    docs_without_word_dic_wordID = {doc_id:words for doc_id, words in doc_word_dict_train.items() if doc_id in docs_without_word}

    

    I_E_total = compute_total_entropy(train_labels)
    I_E_1 = compute_total_entropy(docs_with_word_dict_labels) # I(E1) for docs with words
    I_E_2 = compute_total_entropy(docs_without_word_dict_labels) # I(E2) for docs without words
    
    I_E_split = 0.5 * I_E_1 + 0.5 * I_E_2

    Information_Gain = I_E_total - I_E_split

    return Information_Gain,docs_with_word_dic_wordID,docs_without_word_dic_wordID,docs_with_word_dict_labels,docs_without_word_dict_labels

# Information Gain Method two
def compute_information_gain_two(wordID,doc_word_dict_train,train_labels):
    """
    Computes the information gain for a specific word ID based on the training data.

    Args:
        wordID (int): The ID of the word for which information gain is being calculated.
        doc_word_dict_train (dict): A dictionary mapping document IDs to lists of word IDs for the training dataset.
        train_labels (dict): A dictionary mapping document IDs to their corresponding labels.

    Returns:
        tuple: A tuple containing:
            - Information gain (float): The calculated information gain for the specified word.
            - docs_with_word_dic_wordID (dict): A dictionary of documents containing the specified word.
            - docs_without_word_dic_wordID (dict): A dictionary of documents not containing the specified word.
            - docs_with_word_dict_labels (dict): A dictionary of labels for documents containing the specified word.
            - docs_without_word_dict_labels (dict): A dictionary of labels for documents not containing the specified word.
    """
    

    # split documents with and without that word
    docs_with_word = [doc_id for doc_id, words in doc_word_dict_train.items() if wordID in words]
    # Get all document IDs from doc_word_dict_train
    all_docs = set(doc_word_dict_train.keys())
    # Fill docs_without_word by finding documents that aren't in docs_with_word
    docs_without_word = list(all_docs - set(docs_with_word))

    #labels of docs with and wothout word
    labels_with_word = [train_labels[doc_id] for doc_id in docs_with_word]
    labels_without_word = [train_labels[doc_id] for doc_id in docs_without_word]
    
    # Create a dictionary mapping document IDs to their corresponding labels for documents containing the word
    docs_with_word_dict_labels = {doc_id: train_labels[doc_id] for doc_id in docs_with_word}
    # Create a dictionary mapping document IDs to their corresponding labels for documents not containing the word
    docs_without_word_dict_labels = {doc_id: train_labels[doc_id] for doc_id in docs_without_word}
    
    # Create a dictionary mapping document IDs to their word lists for documents containing the word
    docs_with_word_dic_wordID = {doc_id:words for doc_id, words in doc_word_dict_train.items() if doc_id in docs_with_word}
    # Create a dictionary mapping document IDs to their word lists for documents not containing the word
    docs_without_word_dic_wordID = {doc_id:words for doc_id, words in doc_word_dict_train.items() if doc_id in docs_without_word}
    

    
    I_E_total = compute_total_entropy(train_labels)
    I_E_1 = compute_total_entropy(docs_with_word_dict_labels) # I(E1) for docs with words
    I_E_2 = compute_total_entropy(docs_without_word_dict_labels) # I(E2) for docs without words
    
    I_E_split = (len(labels_with_word)/len(train_labels)) * I_E_1 + (len(labels_without_word)/len(train_labels)) * I_E_2

    Information_Gain = I_E_total - I_E_split

    return Information_Gain,docs_with_word_dic_wordID,docs_without_word_dic_wordID,docs_with_word_dict_labels,docs_without_word_dict_labels 
    
def find_best_words(word_dict,doc_word_dict_train,train_labels,IG_option = 1):
    """
    Finds the word with the highest information gain.

    Args:
        word_dict (dict): A dictionary mapping word IDs to words.
        doc_word_dict_train (dict): A dictionary mapping document IDs to lists of word IDs for the training dataset.
        train_labels (dict): A dictionary mapping document IDs to their corresponding labels.
        IG_option (int, optional): An option to choose the method for calculating information gain.
                                    1 for using compute_information_gain_one, 
                                    2 for using compute_information_gain_two. Default is 1.

    Returns:
        tuple: A tuple containing:
            - best_info_gain (float): The highest information gain found among the words.
            - best_word (int): The ID of the word that provides the highest information gain.
            - docs_with_word_dic_wordID (dict): A dictionary of documents containing the best word.
            - docs_without_word_dic_wordID (dict): A dictionary of documents not containing the best word.
            - docs_with_word_dict_labels (dict): A dictionary of labels for documents containing the best word.
            - docs_without_word_dict_labels (dict): A dictionary of labels for documents not containing the best word.
    
    The function iterates through each word in the word_dict, calculates the information gain 
    for each word using the specified method (IG_option), and keeps track of the word that 
    yields the highest information gain. It returns the best information gain along with 
    relevant document and label dictionaries for further processing in the decision tree.
    """
    best_word = None
    best_info_gain = -1
    docs_with_word_dic_wordID = {}
    docs_without_word_dic_wordID = {}
    docs_with_word_dict_labels = {}
    docs_without_word_dict_labels = {}

    for wordID in word_dict:
        # print(word_dict[wordID])
        if(IG_option == 1):
       
            IG,docs_with_word_wordID,docs_without_word_wordID,docs_with_word_labels,docs_without_word_labels = compute_information_gain_one(wordID,doc_word_dict_train,train_labels)
        else:
            IG,docs_with_word_wordID,docs_without_word_wordID,docs_with_word_labels,docs_without_word_labels = compute_information_gain_two(wordID,doc_word_dict_train,train_labels)
        
        if IG>best_info_gain:
            best_info_gain = IG
            best_word = wordID
            docs_with_word_dic_wordID = docs_with_word_wordID
            docs_without_word_dic_wordID = docs_without_word_wordID
            docs_with_word_dict_labels = docs_with_word_labels
            docs_without_word_dict_labels = docs_without_word_labels

    return best_info_gain, best_word ,docs_with_word_dic_wordID ,docs_without_word_dic_wordID ,docs_with_word_dict_labels, docs_without_word_dict_labels


class Node:
    def __init__(self, dataset_with_word,dataset_without_word, docs_with_word_labels, docs_without_word_labels, point_estimate, best_feature, info_gain,whole_dataset = None):
        """
        Initialize the node with the following:
        - dataset: The subset of data assigned to this node 
        - point_estimate: The majority class label or a summary of labels 
        - best_feature: The best feature to split on 
        - info_gain: The information gain of splitting on the best feature 
        """
        self.dataset = whole_dataset # Store the entire dataset for reference
        self.dataset_with_word = dataset_with_word  # Store the subset of data containing the specified word
        self.dataset_without_word = dataset_without_word # Store the subset of data not containing the specified word
        self.docs_with_word_labels = docs_with_word_labels # Store labels for documents that contain the specified word
        self.docs_without_word_labels = docs_without_word_labels # Store labels for documents that do not contain the specified word
        self.point_estimate = point_estimate  
        self.best_feature = best_feature  # Best feature to split on 
        self.info_gain = info_gain  # Information gain 
        self.left = None  # Left child
        self.right = None  # Right child

    def __repr__(self):
        return f"Node(best_feature={self.best_feature}, info_gain={self.info_gain}, point_estimate={self.point_estimate})"
    
    

def desicion_tree(doc_word_dict_train,doc_word_dict_test,train_labels,test_labels,word_dict,IG_option = 1):
    
    """
    Constructs a decision tree based on the training data and evaluates its performance.

    Args:
        doc_word_dict_train (dict): A dictionary mapping document IDs to lists of word IDs for the training dataset.
        doc_word_dict_test (dict): A dictionary mapping document IDs to lists of word IDs for the test dataset.
        train_labels (dict): A dictionary mapping document IDs to their corresponding labels for the training dataset.
        test_labels (dict): A dictionary mapping document IDs to their corresponding labels for the test dataset.
        word_dict (dict): A dictionary mapping word IDs to words.
        IG_option (int, optional): An option to choose the method for calculating information gain.
                                    1 for using compute_information_gain_one, 
                                    2 for using compute_information_gain_two. Default is 1.

    Returns:
        tuple: A tuple containing:
            - root_node (Node): The root node of the constructed decision tree.
            - train_accuracies (list): A list of training accuracies after each split.
            - test_accuracies (list): A list of testing accuracies after each split.
            - nodes_added (list): A list of the number of nodes added after each split.
    """
    priority_queue = [] # Initialize a priority queue to manage nodes based on information gain
    Max_internal_nodes = 100 # Set the maximum number of internal nodes to be created in the decision tree

    train_accuracies = [] # List to store training accuracies after each split
    test_accuracies = [] # List to store testing accuracies after each split
    nodes_added = [] # List to track the number of nodes added after each split
    unique_counter = count() # Create a counter to generate unique identifiers for nodes

    # Find the best word to split on based on information gain using the training data
    initial_best_info_gain, initial_best_word,docs_with_word_dic_wordID ,docs_without_word_dic_wordID ,docs_with_word_dict_labels, docs_without_word_dict_labels = find_best_words(word_dict,doc_word_dict_train, train_labels,IG_option)
    
    print(word_dict[initial_best_word])
    print(initial_best_info_gain)

    # Compute the majority class (point estimate) for the root node 
    point_estimate = max(set(train_labels.values()), key=list(train_labels.values()).count)

  # Create the root node
    root_node = Node(
    whole_dataset = doc_word_dict_train ,   
    dataset_with_word= docs_with_word_dic_wordID,
    dataset_without_word = docs_without_word_dic_wordID,
    docs_with_word_labels = docs_with_word_dict_labels,
    docs_without_word_labels =  docs_without_word_dict_labels,
    point_estimate = point_estimate,  # Majority class in the dataset
    best_feature = initial_best_word,  # Best word to split on
    info_gain = initial_best_info_gain  # Information gain for the split
)
       
    
    
    # Insert the root node into the priority queue (negative information gain to simulate max-heap)
    heapq.heappush(priority_queue, (-initial_best_info_gain,0, root_node))

    node_count = 0  # To track internal nodes
    counter = 0
  

    while counter < Max_internal_nodes and priority_queue:
      
        # Pop the node with the highest information gain (top of the max-heap)
        neg_info_gain, node_count, current_node = heapq.heappop(priority_queue)
        
        info_gain = -neg_info_gain  # Restore positive info gain

        dataset_with_word = current_node.dataset_with_word
        dataset_without_word = current_node.dataset_without_word
        docs_with_word_labels = current_node.docs_with_word_labels
        docs_without_word_labels = current_node.docs_without_word_labels
        
        # If information gain is zero or negative, stop splitting
     
        if info_gain <= 0.0:
            
            continue

        # Find the best word to split on for the left child node based on information gain
        left_info_gain, left_best_word, left_docs_with_word_dic_wordID, left_docs_without_word_dic_wordID, left_docs_with_word_labels, left_docs_without_word_labels  = find_best_words(word_dict, dataset_with_word, docs_with_word_labels,IG_option)
       

        # Point estimate (majority class) for the left child node
        left_point_estimate = max(set(docs_with_word_labels.values()), key=list(docs_with_word_labels.values()).count)

        # Create the left child node
        # Add the left child node to the priority queue with the following parameters:
        # - negative information gain (-left_info_gain) to ensure the queue behaves like a max-heap
        # - a unique identifier generated by next(unique_counter) to maintain the order of nodes
        # - the left child node itself (left_node)
        left_node = Node(
            dataset_with_word= left_docs_with_word_dic_wordID,
            dataset_without_word = left_docs_without_word_dic_wordID,
            docs_with_word_labels = left_docs_with_word_labels,
            docs_without_word_labels =  left_docs_without_word_labels,
            point_estimate=left_point_estimate,
            best_feature=left_best_word,
            info_gain=left_info_gain
        )

        # Find the best word to split on for the right child node based on information gain
        right_info_gain, right_best_word, right_docs_with_word_dic_wordID, right_docs_without_word_dic_wordID, right_docs_with_word_labels, right_docs_without_word_labels  = find_best_words(word_dict, dataset_without_word, docs_without_word_labels,IG_option)

        # Point estimate (majority class) for the right child node
        right_point_estimate = max(set(docs_without_word_labels.values()), key=list(docs_without_word_labels.values()).count)

        # Create the right child node
        right_node = Node(
            dataset_with_word= right_docs_with_word_dic_wordID,
            dataset_without_word = right_docs_without_word_dic_wordID,
            docs_with_word_labels = right_docs_with_word_labels,
            docs_without_word_labels =  right_docs_without_word_labels,
            point_estimate=right_point_estimate,
            best_feature=right_best_word,
            info_gain=right_info_gain
        )

        # Attach the left and right child nodes to the current node
        current_node.left = left_node
        current_node.right = right_node

        heapq.heappush(priority_queue, (-left_info_gain, next(unique_counter), left_node))
       
        heapq.heappush(priority_queue, (-right_info_gain, next(unique_counter), right_node))
        node_count += 2

        # Calculate accuracy after each split
        train_accuracy = compute_accuracy(root_node, doc_word_dict_train, train_labels)
        test_accuracy = compute_accuracy(root_node, doc_word_dict_test, test_labels)
        
        print(f"after split {current_node.best_feature}")
        print(f"After node {counter}:")
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

        # Store accuracy values and node count for plotting
        train_accuracies.append(train_accuracy * 100)
        test_accuracies.append(test_accuracy * 100)
        nodes_added.append(counter + 1)
        counter +=1
        

    # Return the root node and the accuracy lists for plotting
    return root_node, train_accuracies, test_accuracies, nodes_added

def print_tree(node,depth=0):
        if node is None:
            return
        
        # Print the current node's details (best feature, info gain, etc.)
        indent = "  " * depth  # Indentation based on the depth of the node in the tree
        print(f"{indent}Node at depth {depth}:")
        print(f"{indent}  Best Feature (Word ID): {node.best_feature}")
        # Print the associated word if word_dict is provided
        if word_dict is not None and node.best_feature in word_dict:
         print(f"{indent}  Associated Word: {word_dict[node.best_feature]}")  
        print(f"{indent}  Information Gain: {node.info_gain}")
        print(f"{indent}  Point Estimate (Majority Class): {node.point_estimate}")
        
        # Recursively print the left and right children
        if node.left:
            print(f"{indent}  Left Child:")
            print_tree(node.left, depth + 1)
        
        if node.right:
            print(f"{indent}  Right Child:")
            print_tree(node.right, depth + 1)

def predict(node, doc_word_dict,docID):
    """
    Predicts the label for a given document based on the decision tree.

    Args:
        node (Node): The current node in the decision tree.
        doc_word_dict (dict): A dictionary mapping document IDs to lists of word IDs.
        docID (int): The ID of the document for which the prediction is to be made.

    Returns:
        int: The predicted label (majority class) for the specified document.
    
    """
    
    current_node = node
    
    while current_node.left or current_node.right:
        if current_node.best_feature in doc_word_dict[docID]:  # If the document contains the best feature (word)
            current_node = current_node.left  # Traverse the left child
        else:
            current_node = current_node.right  # Traverse the right child
    
    return current_node.point_estimate  # Return the majority class (point estimate) at the leaf node

def compute_accuracy(root_node, doc_word_dict, true_labels):
    """
    Computes the accuracy of the decision tree's predictions.

    Args:
        root_node (Node): The root node of the decision tree.
        doc_word_dict (dict): A dictionary mapping document IDs to lists of word IDs.
        true_labels (dict): A dictionary mapping document IDs to their corresponding true labels.

    Returns:
        float: The accuracy of the predictions as a proportion of correct predictions to total documents.
    
    """
    correct_predictions = 0
    total_docs = len(doc_word_dict)

    for doc_id, word_lists in doc_word_dict.items():
        predicted_label = predict(root_node, doc_word_dict,doc_id)
        true_label = true_labels[doc_id]
        if predicted_label == true_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_docs
    return accuracy





if __name__ == '__main__':

    doc_word_dict_train = generate_words_vectors('dataset/trainData.txt',train = True) 
    doc_word_dict_test = generate_words_vectors('dataset/testData.txt',train = False)
    train_labels = generate_label_vectors('dataset/trainLabel.txt')
    test_labels = generate_label_vectors('dataset/testLabel.txt')
    word_dict = generate_word_dict('dataset/words.txt')
    # print((word_dict[2992]))

    tree, train_accuracies, test_accuracies, nodes_added = desicion_tree(doc_word_dict_train,doc_word_dict_test,train_labels,test_labels,word_dict,IG_option=2)
    print_tree(tree)
    

    

    # Plotting the results
    plt.plot(nodes_added, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(nodes_added, test_accuracies, label='Testing Accuracy', marker='x')
    plt.xlabel('Number of Nodes Added')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy vs. Number of Nodes Added')
    plt.legend()
    plt.grid(True)
    plt.show()

  

    
    
    
    
    