import random
import math

import numpy
import sklearn.tree
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class EpsilonFunc:
    def __init__(self, a=0.5, min=0.00, max=1, constant=1., set_to_zero_under=0.005):
        assert(0 < a < 1)
        self.t = 0
        self.a = a
        self.value = constant
        self.min = min
        self.max = max
        self.zero = set_to_zero_under

    def update(self):
        self._update()
        self.t += 1
        self.value = max(min(self.value, self.max), self.min)
        if self.zero and 0 < self.value <= self.zero:
            self.value = 0

    def _update(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.value)


class EpsilonLinear(EpsilonFunc):
    def _update(self):
        self.value -= self.a


class EpsilonCosine(EpsilonFunc):
    def _update(self):
        self.value = math.cos(self.a * self.t)


class EpsilonExpDecay(EpsilonFunc):
    def _update(self):
        self.value = math.exp(-self.a * self.t)


class Constant(EpsilonFunc):
    def _update(self):
        pass


def label_encode_state(inputs, actions):
    dummies = {}
    for i, values in inputs.items():
        if i == 'light':
            values = ['red', 'green']
        dummies[i] = preprocessing.LabelEncoder()
        dummies[i].fit(values)

    dummies['actions'] = preprocessing.LabelEncoder()
    dummies['actions'].fit(actions)
    return dummies


def get_state_for_label(labels, _s):
    return (labels['actions'].inverse_transform([int(_s[0])])[0],  # waypoint
            labels['light'].inverse_transform([int(_s[1])])[0],  # light
            labels['oncoming'].inverse_transform([int(_s[2])])[0],  # oncoming
            labels['left'].inverse_transform([int(_s[3])])[0],  # left
            labels['right'].inverse_transform([int(_s[4])])[0])


def get_labels_for_state(labels, _s):
    return numpy.array([labels['actions'].transform([_s[0]])[0],  # waypoint
                        labels['light'].transform([_s[1]])[0],  # light
                        labels['oncoming'].transform([_s[2]])[0],  # oncoming
                        labels['left'].transform([_s[3]])[0],  # left
                        labels['right'].transform([_s[4]])[0]])

DT = DecisionTreeClassifier()


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=None, alpha=0.5, let_drive_on_red=True, estimate_unknown=True,
                 estimate_for_safety=True, min_to_estimate=0):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.guess_for_safety = estimate_for_safety
        self.let_it_drive_on_red = let_drive_on_red
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self._epsilon = epsilon or EpsilonLinear(0.01)   # Random exploration factor
        self._alpha = alpha or EpsilonExpDecay(0.01)      # Learning factor
        self.state_labels = label_encode_state(env.valid_inputs, env.valid_actions)
        self.estimate = estimate_unknown
        self.min_to_estimate = min_to_estimate


        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed

    @property
    def epsilon(self):
        return self._epsilon.value

    @property
    def alpha(self):
        return self._alpha.value

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########{'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        return state

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        return max(self.Q[state].values())

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        if state not in self.Q:
            self.Q[state] = {action: dict(visited=False, value=0.) for action in self.valid_actions}

        return

    def get_max_action_for_state(self, state):
        action = 'na'
        for state_action in self.Q[state]:
            if action is 'na' or self.Q[state][state_action]['value'] > self.Q[state][action]['value']:
                action = state_action

        if not self.Q[state][action]['visited'] and action != self.next_waypoint and \
                not self.Q[state][self.next_waypoint]['visited']: # force exploration if waypoint is also unknown
            action = self.next_waypoint

        return action

    def _get_label_action(self, l):
        return self.state_labels['actions'].inverse_transform([int(l)])[0]

    def estimate_action(self, state):

        X = numpy.zeros(shape=(len(self.Q) * len(self.valid_actions), 5))
        Y = numpy.zeros(shape=(len(self.Q) * len(self.valid_actions), 1))
        index = 0
        negative_actions = [a for a in self.Q[state] if self.Q[state][a]['value'] < 0]
        for s in self.Q:
            for _a in [_a for _a in self.Q[s] if self.Q[s][_a]['value'] > 0 and _a not in negative_actions]:
                X[index] = get_labels_for_state(self.state_labels, s)
                Y[index] = self.state_labels['actions'].transform([_a])
                index += 1

        X = X[:index]
        Y = Y[:index]

        if len(X):
            DT.fit(X[:, int(self.guess_for_safety):], Y)
            return self._get_label_action(
                DT.predict([get_labels_for_state(self.state_labels, state)[int(self.guess_for_safety):]]))

        else:
            return 'na'

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        def get_first_not_visited():
            """ use waypoint if not visited, otherwise select the first element not visited yet """
            if not (self.Q[state][self.next_waypoint]['visited']):  # forced exploration
                return self.next_waypoint
            else:
                no_visited = [a for a in self.valid_actions if not self.Q[state][a]['visited']]
                return random.choice(no_visited) if no_visited else 'na'

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        if self.learning:
            if not self.let_it_drive_on_red and state[1] == 'red':
                action = None
            else:
                if random.random() < self.epsilon:
                    print "######## exploring..."
                    _action = get_first_not_visited()
                    action = _action if _action != 'na' else random.choice(self.valid_actions)
                else:

                    action = self.get_max_action_for_state(state)

                    # if value for action is negative and we have not visited the waypoint let's try it
                    if self.estimate and self.Q[state][action]['value'] <= self.min_to_estimate:
                        _action = self.estimate_action(state)
                        print ("DT Action: {} - q state: {}".format(_action, self.Q[state]))
                        if _action != 'na':
                            action = _action

            self.Q[state][action]['visited'] = True
            self._epsilon.update()
            self._alpha.update()
            print "Q State: {} - action: {}".format(self.Q[state], action)
        else:
            action = self.get_max_action_for_state(state)

        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
 
        return action

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        self.Q[state][action]['value'] += self.alpha * (reward -self.Q[state][action]['value'])
        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    # rewards=dict(penalty=1., minor_violation=5, major_violation=20., minor_accident=40., major_accident=80.,
    #            correct_move=2, correct_move_red=2., incorrect_move=1.))

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,
                             learning=True,
                             alpha=EpsilonExpDecay(0.0002, max=0.8, min=0.5),
                             epsilon=EpsilonExpDecay(0.0025),
                             estimate_unknown=False,
                             estimate_for_safety=True,
                             let_drive_on_red=True,
                             min_to_estimate=0.25)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    import time
    t0 = time.time()
    sim.run(n_test=20, tolerance=0.005)
    print ("\n\n######### total time: " + str((time.time() - t0) ))


if __name__ == '__main__':
    run()
