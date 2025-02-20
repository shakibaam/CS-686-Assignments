import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):


    # Define the column names based on the description
    column_names = ['S', 'F', 'Dg', 'G', 'D']
    data = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
    return data

def initilaze_CPT():
    # Initialize the Conditional Probability Table (CPT) with prior probabilities
    first_CPT = {
       'D':np.array([0.5,0.25,0.25]), # D= No, Mild, Severe
       'G': np.array([0.1,0.9]), # G = T,F
       'S_with_G': np.array([[0.01,0.99], # D = No, G = T
                             [0.1,0.9],    # D = Mild, G = T
                             [0.12,0.88]  # D = Severe, G = T
                             ]),
        'S_without_G':  np.array([[0.1,0.9], # D = No, G = F
                                  [0.75,0.25],# D = Mild, G = F
                                  [0.85,0.15] # D = Severe, G = F
                             ]) ,
        'F': np.array([[0.05,0.95], # D= No
                       [0.9,0.1],    # D= Mild
                       [0.2, 0.8]   # D= Severe
                       ]),
        'Dg' : np.array([[0.03,0.97], # D= No
                         [0.15,0.85],  # D= Mild
                         [0.95,0.15]
                         ])

                                  
    }
    return first_CPT

def add_noise_to_CPT(CPT , delta,seed=None):
    
    new_CPT = {}
    

    for key , probs in CPT.items():
      
       # Generate random noise within the range [0, delta] with the same shape as the probabilities
      noise = np.random.uniform(0, delta, probs.shape)
    
      if probs.ndim == 1:  # Check if the probabilities are 1D (prior probabilities)
        noise = np.random.uniform(0, delta, len(probs))
        noisy_probs = (probs + noise) / (1 + np.sum(noise))
      elif probs.ndim == 2:  # Check if the probabilities are 2D (conditional probabilities)
        noise = np.random.uniform(0, delta, probs.shape)
        noisy_probs = (probs + noise) / (1 + np.sum(noise, axis=1, keepdims=True))
        
      
      new_CPT[key] = noisy_probs

    return new_CPT 
         
def Expectation_step(training_data,noisy_CPT):
    # Initialize weights for each data point
    weights = np.zeros((len(training_data), 3))  
    for i, (_, row) in enumerate(training_data.iterrows()): 
        if row['D'] in [0, 1, 2]: # Check if the observed D state is valid (No, Mild, Severe)
            weights[i, :] = 0  # Initialize all weights to 0
            weights[i, row['D']] = 1  # Set the weight for the observed D state to 1
        else: 
            numerators = []  # List to store the numerators for weight calculation
           
            for j, D in enumerate(['No', 'Mild', 'Severe']):

                # Compute S_prob based on the value of G and S
                if(row['G'] == 0):
                    if(row['S']==1):
                        S_prob = noisy_CPT['S_without_G'][j,0] # Probability of S given D and G=0
                    else: S_prob = noisy_CPT['S_without_G'][j,1] # Probability of S given D and G=0
                else:
                    if(row['S']==1):
                        S_prob = noisy_CPT['S_with_G'][j,0]
                    else: S_prob = noisy_CPT['S_with_G'][j,1]
                

                # Compute F_prob
                if(row['F']==1):
                    F_prob = noisy_CPT['F'][j,0] # Probability of S given D and G=1
                else:
                    F_prob = noisy_CPT['F'][j,1] # Probability of S given D and G=1
                
                
                #Compute Dg_prob
                if(row['Dg']==1):
                    Dg_prob = noisy_CPT['Dg'][j,0]
                else:
                    Dg_prob = noisy_CPT['Dg'][j,1]
                
                
                # Prior prob D
                prior_D_prob = noisy_CPT['D'][j]
              
                numerator = S_prob * F_prob * Dg_prob * prior_D_prob
                numerators.append(numerator)

            # Compute denominator 
            denominator = sum(numerators) 

            # Compute weights w_ij (normalize each numerator)
            weights[i, :] = [num / denominator for num in numerators]
           

    return weights       

def Maximazation_step(training_data,weights,CPT):

    #Update P_D
    CPT['D'] = np.mean(weights, axis=0)

    #Update S_probs
    for j, D in enumerate(['No', 'Mild', 'Severe']):
        for g in [0, 1]:  # Loop through states of G

            # Filter data points where G == g
            filter_data_g = (training_data['G'] == g)
            filter_weights_g = weights[filter_data_g, j]
            
            normalization_factor_g  = np.sum(filter_weights_g)
            if(g==0):
                
                CPT['S_without_G'][j,0] = (np.sum(filter_weights_g * (training_data.loc[filter_data_g, 'S'] == 1)))/normalization_factor_g
                
                CPT['S_without_G'][j,1] = (np.sum(filter_weights_g * (training_data.loc[filter_data_g, 'S'] == 0)))/normalization_factor_g
            else:
                
                CPT['S_with_G'][j,0] = (np.sum(filter_weights_g * (training_data.loc[filter_data_g, 'S'] == 1)))/normalization_factor_g
                
                CPT['S_with_G'][j,1] = (np.sum(filter_weights_g * (training_data.loc[filter_data_g, 'S'] == 0)))/normalization_factor_g

    #Update F_probs
    for j, D in enumerate(['No', 'Mild', 'Severe']):
        
        filter_weights_f = weights[:, j]
        normalization_factor_f = np.sum(filter_weights_f)
        
        CPT['F'][j,0] = (np.sum(filter_weights_f * (training_data['F'] == 1)))/normalization_factor_f
        
        CPT['F'][j,1] = (np.sum(filter_weights_f * (training_data['F'] == 0)))/normalization_factor_f
    
    # Update Dg_probs
    for j, D in enumerate(['No', 'Mild', 'Severe']):
        
        filter_weights_dg = weights[:, j]
        normalization_factor_dg = np.sum(filter_weights_dg)
        
        CPT['Dg'][j,0] = (np.sum(filter_weights_dg * (training_data['Dg'] == 1)))/normalization_factor_dg
        
        CPT['Dg'][j,1] = (np.sum(filter_weights_dg * (training_data['Dg'] == 0)))/normalization_factor_dg
    
    return CPT

def calculate_likelihood(training_data, CPT, weights):
    # Extract features and precompute probabilities
    S_prob = np.where(
        training_data['G'] == 0,
        CPT['S_without_G'][:, training_data['S'].values],
        CPT['S_with_G'][:, training_data['S'].values]
    )

    F_prob = CPT['F'][:, training_data['F'].values]
    Dg_prob = CPT['Dg'][:, training_data['Dg'].values]
    prior_D_prob = CPT['D'][:, np.newaxis]

    # Compute likelihood for all rows and states of D
    likelihoods = S_prob * F_prob * Dg_prob * prior_D_prob * weights.T

    # Sum likelihoods over states of D 
    total_likelihood = np.sum(np.sum(likelihoods, axis=0))

    return total_likelihood




def EM(training_data, delta_value,seed=None,treshhold = 0.01):
    initial_CPT = initilaze_CPT()
    CPT = add_noise_to_CPT(initial_CPT,delta_value,seed)
    prev_likelihood = float('inf')
   
    
    while(True):
        
        weights = Expectation_step(train_df,CPT)
        
        current_likelihood = calculate_likelihood(training_data, CPT, weights)
        # Check convergence
        if abs(current_likelihood - prev_likelihood) < treshhold:
            break
        CPT = Maximazation_step(train_df,weights,CPT)
        prev_likelihood = current_likelihood
    
    return CPT

def predict(test_data,CPT):
    predictions = []
    for _, row in test_data.iterrows():
        posterior_probs = []
        for j, D in enumerate(['No', 'Mild', 'Severe']):

            # Compute S_prob
            if(row['G'] == 0):
                if(row['S']==1):
                    S_prob = CPT['S_without_G'][j,0]
                else: S_prob = CPT['S_without_G'][j,1]
            else:
                if(row['S']==1):
                    S_prob = CPT['S_with_G'][j,0]
                else: S_prob = CPT['S_with_G'][j,1]
            # Compute F_prob
            if(row['F']==1):
                F_prob = CPT['F'][j,0]
            else:
                F_prob = CPT['F'][j,1]
            
            #Compute Dg_prob
            if(row['Dg']==1):
                Dg_prob = CPT['Dg'][j,0]
            else:
                Dg_prob = CPT['Dg'][j,1]
            
            # Prior prob D
            prior_D_prob = CPT['D'][j]

            posterior_probs.append(S_prob * F_prob * Dg_prob * prior_D_prob)
        
        #Normalize
        posterior_probs = np.array(posterior_probs)
        posterior_probs /= np.sum(posterior_probs)

        # Predict the class with the highest posterior probability
        max_posterior = np.argmax(posterior_probs)
        predictions.append(max_posterior)
    
    return predictions

def calculate_accuracy(test_data, pred_labels):
    true_labels = test_data['D']
    accuracy = np.mean(true_labels == pred_labels)
    return accuracy

def plot_accuracy_comparison(delta_values):
    # Initialize lists to store results
    before_em_means = []
    before_em_stds = []
    after_em_means = []
    after_em_stds = []
    
    # For each delta value
    for i, delta in enumerate(delta_values):
        before_em_accuracies = []
        after_em_accuracies = []
        
        # Run multiple trials for each delta
        for trial in range(20):  # 20 trials
            print("{} of {}".format(trial,delta))

            # Use current time + trial number as seed to ensure different randomization
            seed = (i * 100 + trial) % (2**32 - 1)  
            # Initialize CPT and add noise
            initial_CPT = initilaze_CPT()
            noisy_CPT = add_noise_to_CPT(initial_CPT, delta,seed)
            
            # Compute accuracy before EM
            predictions_before = predict(test_df, noisy_CPT)
            acc_before = calculate_accuracy(test_df, predictions_before)
            print("Delat: {} acc_before: {}".format(delta,acc_before))
            before_em_accuracies.append(acc_before)
            
            # Run EM and compute accuracy after
            trained_CPT = EM(train_df, delta,seed)
            predictions_after = predict(test_df, trained_CPT)
            acc_after = calculate_accuracy(test_df, predictions_after)
            print("Delat: {} acc_after: {}".format(delta,acc_after))
            after_em_accuracies.append(acc_after)
            print('**********************')
        
        # Calculate mean and std for this delta value
        before_em_means.append(np.mean(before_em_accuracies))
        before_em_stds.append(np.std(before_em_accuracies))
        after_em_means.append(np.mean(after_em_accuracies))
        after_em_stds.append(np.std(after_em_accuracies))
        
        print(f"Delta: {delta:.2f} completed")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy before EM with error bars
    plt.errorbar(delta_values, before_em_means, yerr=before_em_stds, 
                label='Before EM', marker='o', capsize=5, capthick=1, 
                elinewidth=1, markersize=5, color='blue', alpha=0.7)
    
    # Plot accuracy after EM with error bars
    plt.errorbar(delta_values, after_em_means, yerr=after_em_stds, 
                label='After EM', marker='s', capsize=5, capthick=1, 
                elinewidth=1, markersize=5, color='red', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Delta (δ)')
    plt.ylabel('Prediction Accuracy')
    plt.title('Test Set Prediction Accuracy vs. Delta')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('em_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    delta_values = np.linspace(0, 4, 20)
    train_df = load_data('a4datasets/traindata.txt')
    test_df = load_data('a4datasets/testdata.txt')
    initial_CPT = initilaze_CPT()
    noisy_cpt = add_noise_to_CPT(initial_CPT,0)

    initial_predictions = predict(test_df, noisy_cpt)
    initial_accuracy = calculate_accuracy(test_df, initial_predictions)
    print(initial_accuracy)

    trained_CPT = EM(train_df, 0)
    predictions_after = predict(test_df, trained_CPT)
    after_accuracy = calculate_accuracy(test_df, predictions_after)
    print(after_accuracy)

    plot_accuracy_comparison(delta_values)

   

 
    
    
    

