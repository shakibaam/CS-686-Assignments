import random
import numpy as np
import sys
import matplotlib.pyplot as plt


class Sender:
    """
    A Q-learning agent that sends messages to a Receiver

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = range(num_sym)
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros((grid_rows, grid_cols, num_sym)) # Your code here!

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state

        :param state: the state the agent is acting from, in the form (x,y), which are the coordinates of the prize
        :type state: (int, int)
        :return: The symbol to be transmitted (must be an int < N)
        :rtype: int
        """
        # Your code here!
        current_x , current_y = state
        if random.uniform(0, 1) < self.epsilon:
           
          
           action = random.choice(self.actions)
            
        else:
            
            action = np.argmax(self.q_vals[current_y, current_x])
          
            
            

        return action

    def update_q(self, old_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted, in the form (x,y), which are the coordinates
                          of the prize
        :type old_state: (int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
    
        # Your code here!
        old_x, old_y = old_state
        # self.q_vals[old_x, old_y, action] += self.alpha * (reward - self.q_vals[old_x, old_y,action])
        self.q_vals[old_y, old_x, action] += self.alpha * (reward - self.q_vals[old_y, old_x,action])
        



class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = [0,1,2,3] # Note: these correspond to [up, down, left, right]
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros((num_sym,grid_rows,grid_cols,len(self.actions))) # Your code here!

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state
        :param state: the state the agent is acting from, in the form (m,x,y), where m is the message received
                      and (x,y) are the board coordinates
        :type state: (int, int, int)
        :return: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
        :rtype: int
        """
        # Your code here!
        current_message,current_x, current_y = state

        if random.uniform(0,1) < self.epsilon:

           
            
            action = random.choice(self.actions)
    
            
        else:
         
            action = np.argmax(self.q_vals[current_message,current_y, current_x])
            
            
        
        return action

        

    def update_q(self, old_state, new_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted in the form (m,x,y), where m is the message received
                          and (x,y) are the board coordinates
        :type old_state: (int, int, int)
        :param new_state: the state the agent entered after it acted
        :type new_state: (int, int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # Your code here!
        old_message, old_x, old_y = old_state
        new_message, new_x, new_y = new_state
       
        current_q = self.q_vals[old_message,old_y, old_x,action]
  
        max_future_q = np.max(self.q_vals[new_message, new_y, new_x, :])

      
        self.q_vals[old_message,old_y, old_x,action] +=  self.alpha * (reward + self.discount * max_future_q - current_q)
        



def get_grid(grid_name:str):
    """
    This function produces one of the three grids defined in the assignment as a nested list

    :param grid_name: the name of the grid. Should be one of 'fourroom', 'maze', or 'empty'
    :type grid_name: str
    :return: The corresponding grid, where True indicates a wall and False a space
    :rtype: list[list[bool]]
    """
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid


def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    """
    Produces the new position after a move starting from (posn_x,posn_y) if it is legal on the given grid (i.e. not
    out of bounds or into a wall)

    :param posn_x: The x position (column) from which the move originates
    :type posn_x: int
    :param posn_y: The y position (row) from which the move originates
    :type posn_y: int
    :param move_id: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
    :type move_id: int
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :return: The new (x,y) position if the move was legal, or the old position if it was not
    :rtype: (int, int)
    """
    moves = [[0,-1],[0,1],[-1,0],[1,0]]
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result


def run_episodes(sender: Sender, receiver: Receiver, grid: list[list[bool]], num_ep: int, delta: float):
    """
    Runs the reinforcement learning scenario for the specified number of episodes
    :param sender: The Sender agent
    :param receiver: The Receiver agent
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :param num_ep: The number of episodes
    :param delta: The chance of termination after every step of the receiver
    :return: A list of the reward received by each agent at the end of every episode
    """
    reward_vals = []

    # Episode loop
    for ep in range(num_ep):
        # Set receiver starting position
        receiver_x = 2
        receiver_y = 2

        # Choose prize position
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))

        # Initialize new episode
        sender_action = sender.select_action((prize_x, prize_y))  # Sender acts
        discounted_reward = 0
        step = 0  # Time step for discounting
        terminate = False

        # Receiver loop
        while not terminate:
            receiver_action = receiver.select_action((sender_action, receiver_x, receiver_y))
            receiver_new_x, receiver_new_y = legal_move(receiver_x, receiver_y, receiver_action, grid)

            immediate_reward = 1 if (receiver_new_x == prize_x and receiver_new_y == prize_y) else 0
            
            # Check if the episode should terminate
            if receiver_new_x == prize_x and receiver_new_y == prize_y:
                discounted_reward = (discount ** step) * immediate_reward
                terminate = True
            elif np.random.uniform(0, 1) < delta:
                discounted_reward = 0  # Early termination, no reward
                terminate = True

            receiver.update_q((sender_action, receiver_x, receiver_y), (sender_action, receiver_new_x, receiver_new_y), receiver_action, immediate_reward)

            receiver_x = receiver_new_x
            receiver_y = receiver_new_y
            step += 1
        
        sender.update_q((prize_x, prize_y), sender_action, immediate_reward)  # Update sender's Q-value with accumulated reward
        receiver.alpha -= (receiver.alpha_i - receiver.alpha_f) / num_ep  # Update receiver's alpha
        sender.alpha -= (sender.alpha_i - sender.alpha_f) / num_ep  # Update sender's alpha

        reward_vals.append(discounted_reward)

    return reward_vals


def Q2():
    print("############Q2###############")
    ####Q2####
    num_learns_episodes = [10, 100, 1000, 10000, 50000, 100000]
    epsilons = [0.01, 0.1, 0.4]
    
    num_tests = 10
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    grid_name = 'fourroom'
    grid = get_grid(grid_name)

    eps_results = {0.01: [], 0.1: [], 0.4: []}

    for ep in epsilons:
        eps_results[ep] = []
        for Nep in num_learns_episodes:
            rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                print("Starting learning phase of epsilon {}, NP {} trial {}...".format(ep,Nep,test))
                learn_rewards = run_episodes(sender, receiver, grid, Nep, delta)
                print("Learning phase complete.")
                print(np.mean(learn_rewards))
                print("Starting testing of epsilon {}, NP {} trial {}...".format(ep,Nep,test))
            
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0 
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0

                test_rewards = run_episodes(sender, receiver, grid, 1000, delta)
                print(np.mean(test_rewards))
                rewards.append(np.mean(test_rewards))  # Average discounted reward for this test
               
       
            
            # Compute mean and standard deviation of rewards
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            eps_results[ep].append((Nep, mean_reward, std_reward))

    plt.figure(figsize=(10, 6))
    for ep, data in eps_results.items():
        print(eps_results[ep])
        neps, means, stds = zip(*data)
        plt.errorbar(np.log10(neps), means, yerr=stds, label=f"ϵ = {ep}")
    plt.xlabel("log10(Nep)")
    plt.ylabel("Average Discounted Reward")
    plt.title("Average Discounted Reward vs log10(Nep)")
    plt.legend()
    plt.grid()
    plt.savefig("average_discounted_reward.png")
    plt.show()


def Q2_policy():

    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    grid_name = 'fourroom'
    grid = get_grid(grid_name)

    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, 100000, 0.1, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, 100000, 0.1, discount)

    learn_rewards = run_episodes(sender, receiver, grid, 100000, delta)
    fig, axs = plt.subplots(1, num_signals + 1, figsize=(20, 5))

    # Visualize receiver policy for each message
    for message in range(num_signals):
        receiver_policy = np.empty_like(grid, dtype=str)
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x]:
                    receiver_policy[y][x] = 'W'  # Wall
                else:
                    # Choose the best action at this state
                    action = np.argmax(receiver.q_vals[message, y, x, :])
                    if(action == 0):
                        receiver_policy[y][x] = '↑'
                    elif(action == 1): receiver_policy[y][x] = '↓'
                    elif(action == 2): receiver_policy[y][x] = '←'
                    elif(action==3):receiver_policy[y][x] = '→'
            

        axs[message].imshow(grid, cmap='gray', alpha=0.3)
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                axs[message].text(x, y, receiver_policy[y][x], ha='center', va='center', fontsize=10)
        axs[message].set_title(f"Receiver Policy (Message {message})")
        axs[message].axis('off')

    # Visualize sender policy
    sender_policy = np.empty_like(grid, dtype=str)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x]:
                sender_policy[y][x] = 'W'  # Wall
            else:
                message = np.argmax(sender.q_vals[y, x, :])
                sender_policy[y][x] = str(message)  # Sender chooses the message

    axs[-1].imshow(grid, cmap='gray', alpha=0.3)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            axs[-1].text(x, y, sender_policy[y][x], ha='center', va='center', fontsize=10)
    axs[-1].set_title("Sender Policy")
    axs[-1].axis('off')

    plt.tight_layout()
    plt.show()

def Q3():
    print("############Q3###############")
    ####Q3####
    num_learns_episodes = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    N_values = [2, 4, 10]  # Different N values
    discount = 0.95
    delta = 1 - discount
    ep = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    grid_name = 'fourroom'
    grid = get_grid(grid_name)

    n_results = {}
    for N in N_values:
        n_results[N] = []
        for Nep in num_learns_episodes:
            rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                receiver = Receiver(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                print("Starting learning phase of N {}, NP {} trial {}...".format(N,Nep,test))
                learn_rewards = run_episodes(sender, receiver, grid, Nep, delta)
                print("Learning phase complete.")

                print("Starting testing phase of N {}, NP {} trial {}...".format(N,Nep,test))
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0 
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0

                test_rewards = run_episodes(sender, receiver, grid, 1000, delta)
                
                rewards.append(np.average(test_rewards))  # Average discounted reward for this test

            # Compute mean and standard deviation of rewards for this Nep
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            n_results[N].append((Nep, mean_reward, std_reward))
        
        
    plt.figure(figsize=(10, 6))
    for N, results in n_results.items():
        neps, means, stds = zip(*results)
        plt.errorbar(np.log10(neps), means, yerr=stds, label=f"N = {N}", capsize=3)
    plt.xlabel("log(Nep)")
    plt.ylabel("Average Discounted Reward")
    plt.title("Average Discounted Reward vs Nep for Different N")
    plt.xscale("linear")  
    plt.legend()
    plt.grid()
    plt.savefig("Q3_average_discounted_reward_linear.png")
    plt.show()

def Q4():
    print("############Q4###############")
    ####Q4####
    num_learns_episodes = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    # num_signals = 4
    N_values = [2, 3, 5]  # Different N values
    discount = 0.95
    delta = 1 - discount
    ep = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    grid_name = 'maze'
    grid = get_grid(grid_name)

    n_results = {}
    for N in N_values:
        n_results[N] = []
        for Nep in num_learns_episodes:
            rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                receiver = Receiver(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                print("Starting learning phase of N {}, NP {} trial {}...".format(N,Nep,test))
                learn_rewards = run_episodes(sender, receiver, grid, Nep, delta)
                print("Learning phase complete.")

                print("Starting testing phase of N {}, NP {} trial {}...".format(N,Nep,test))
            
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0 
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                test_rewards = run_episodes(sender, receiver, grid, 1000, delta)
                print("Testing phase complete.")
                rewards.append(np.average(test_rewards))  # Average discounted reward for this test

            # Compute mean and standard deviation of rewards for this Nep
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            n_results[N].append((Nep, mean_reward, std_reward))
        
        
    plt.figure(figsize=(10, 6))
    for N, results in n_results.items():
        neps, means, stds = zip(*results)
        plt.errorbar(np.log10(neps), means, yerr=stds, label=f"N = {N}", capsize=3)
    plt.xlabel("log(Nep)")
    plt.ylabel("Average Discounted Reward")
    plt.title("Average Discounted Reward vs Nep for Different N")
    plt.xscale("linear")  
    plt.legend()
    plt.grid()
    plt.savefig("Q4_average_discounted_reward_linear.png")
    plt.show()

def Q5():
    print("############Q5###############")
    ####Q5####
    num_learns_episodes = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10
    N_values = [1]  
    discount = 0.95
    delta = 1 - discount
    ep = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    grid_name = 'empty'
    grid = get_grid(grid_name)

    n_results = {}
    for N in N_values:
        n_results[N] = []
        for Nep in num_learns_episodes:
            rewards = []
            for test in range(num_tests):
                # Initialize agents
                sender = Sender(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                receiver = Receiver(N, len(grid), len(grid[0]), alpha_init, alpha_final, Nep, ep, discount)
                print("Starting learning phase of N {}, NP {} trial {}...".format(N,Nep,test))
                learn_rewards = run_episodes(sender, receiver, grid, Nep, delta)
                print("Learning phase complete.")

                print("Starting testing phase of N {}, NP {} trial {}...".format(N,Nep,test))

                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0 
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0

                test_rewards = run_episodes(sender, receiver, grid, 1000, delta)
                
                rewards.append(np.average(test_rewards))  # Average discounted reward for this test

            # Compute mean and standard deviation of rewards for this Nep
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            n_results[N].append((Nep, mean_reward, std_reward))
        
        
    plt.figure(figsize=(10, 6))
    for N, results in n_results.items():
        neps, means, stds = zip(*results)
        plt.errorbar(np.log10(neps), means, yerr=stds, label=f"N = {N}", capsize=3)
    plt.xlabel("log(Nep)")
    plt.ylabel("Average Discounted Reward")
    plt.title("Average Discounted Reward vs Nep for Different N")
    plt.xscale("linear")  
    plt.legend()
    plt.grid()
    plt.savefig("Q5_average_discounted_reward_linear.png")
    plt.show()




if __name__ == "__main__":
    # You will need to edit this section to produce the plots and other output required for hand-in

    # Define parameters here
    num_learn_episodes = 100000
    num_test_episodes = 1000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01

    # Initialize agents
    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

    # Print experiment setup
    print("Running experiment with the following parameters:")
    print(f"Grid: {grid_name}, Episodes: {num_learn_episodes} (learn), {num_test_episodes} (test)")
    print(f"Signals: {num_signals}, Discount: {discount}, Epsilon: {epsilon}")
    print(f"Alpha: {alpha_init} -> {alpha_final}")

    # Learn# Learn
    print("Starting learning phase...")
    learn_rewards = run_episodes(sender, receiver, grid, num_learn_episodes, delta)
    # print(learn_rewards)
    print("Learning phase complete.")

    # Test
    print("Starting testing phase...")
    sender.epsilon = 0.0
    sender.alpha = 0.0
    sender.alpha_i = 0.0
    sender.alpha_f = 0.0
    receiver.epsilon = 0.0
    receiver.alpha = 0.0
    receiver.alpha_i = 0.0
    receiver.alpha_f = 0.0
    test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
    print("Testing phase complete.")



    # Print results
    print("Average reward during learning: " + str(np.average(learn_rewards)))
    print("Average reward during testing: " + str(np.average(test_rewards)))
   

    Q2()
    # Q2_policy()
    # Q3()
    # Q4()
    # Q5()






