"""Implementation of Problem 2, Assignment 1, CS486, Fall 2024.

Please replace 'pass  #(TODO) complete this function' with your own
implementation. Please do not change other parts of the code, including the
function signatures; however, feel free to add imports.
"""
import model
import math


def decode(model: model.LanguageModel, k: int = 2, n: int = 3) -> str:
    """Lowest-cost-first search with frontier size limit.

    Args:
        model: A language model.
        k: the frontier size limit. Default is 2.
        n: the length of the desired sentence. Default is 3.

    Returns:
        The decoded text, where words are separated by a single space.

    Hints:
        1. model.vocab() gives you the vocabulary.
        2. model.prob(w: str, c: List[str]) gives you the probability of word
           w given previous context c.
    """
    pass  # (TODO) complete this function

    
    c = '' # Previous Context
    frontier = {c:0}
    vocabs = model.vocab() 
    
    while(frontier):
        new = {} # New nodes produced after expanding the current node
        c = min(frontier, key= lambda x: frontier[x]) # Select the node with the lowest cost from the frontier
        current_cost = frontier.pop(c) # Get the current node's cost and remove it from the frontier for expansion
        
        # Check if the current sentence has reached the target length, if so return the sentence
        if len(c.split()) == n: 
            return c
            

        # Iterate through the vocabulary and compute the cost of transitioning from current node     
        for v in vocabs:
            if (model.prob(v,c.split())!=0):
               
                if(c!=''):
                    new[str(c) + ' '+ str(v)] = current_cost - math.log(model.prob(v,c.split())) 
                else:  new[str(c) + str(v)] = current_cost - math.log(model.prob(v,c.split()))

        # Update the frontier with the new sentences and their respective costs        
        frontier.update(new)
        # print(frontier)
       
        # Sort the frontier and remove extra nodes until reaching the frontier size limit
        frontier = dict(sorted(frontier.items(), key=lambda item: item[1])[:k])
        # print(f"Frontier (limited to {k}): {frontier}")

        
    # If frontier becomes empty and no sentence with length n found
    return 'No sentence of length n found'
          



if __name__ == '__main__':
    
    m = model.LanguageModel()
    
    if decode(m, 2, 3) == 'a dog barks' and decode(m, 1, 3) == 'a cat meows':
        print('success!')
    else:
        print('There might be something wrong with your implementation; please double check.')