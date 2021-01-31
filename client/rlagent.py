import ast
import random

from agent import Agent
import matplotlib.pyplot as plt


class RLAgent(Agent):

    def __init__(self, gamma=0.9, alpha=1., epsilon=1.):
        super().__init__()
        self.rewards = self.get_rewards()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = {(x, y): {"north": 0, "south": 0, "east": 0, "west": 0}
                        for x in range(self.maxCoord[0]) for y in range(self.maxCoord[1])}
        # self.q_table[self.goalNodePos] = {"north": 100, "south": 100, "east": 100, "west": 100}
        self.targets = self.get_targets()
        self.table = None

    def get_targets(self):
        """Return the targets defined in the world."""
        msg = self.c.execute("info", "targets")
        res = ast.literal_eval(msg)
        # test
        # print('Received targets:', res)
        return res

    def goal_reached(self):
        """Returns whether or the agent's current tile contains a goal"""
        return self.current_position() == self.goalNodePos

    def target_reached(self):
        """Returns whether or the agent's current tile contains a target"""
        current_position = self.current_position()
        return bool(self.targets[current_position[0]][current_position[1]])

    def get_rewards(self):
        """Fetches the reward map from the server"""
        msg = self.c.execute("info", "rewards")
        rewards = ast.literal_eval(msg)
        # test
        print('Received map of obstacles:', rewards)
        return rewards

    def reward_at(self, coords):
        """Returns the value of the reward for the tile at position coords[0], coords[1]"""
        return self.rewards[coords[0]][coords[1]]

    def mark_visited(self, coords, color="coral"):
        """Tells the server to paint with color the tile at coords[0], coords[1]"""
        super().mark_visited(coords, color)

    def execute_random_action(self, no_turning_back=True):
        """Chooses a legal, more or less random action,
        tells the server to execute it, and returns the identifying string.
        The randomness depends on the agent's epsilon and
        whether or not the agent can return to the previous tile (no_turning_back)."""
        current_position = self.current_position()

        directions = [direction for direction in ("north", "south", "east", "west")
                      if self.is_visitable(*self.step(current_position, direction))]
        if all((no_turning_back,
                len(directions) > 1,
                self.opposite_direction(self.last_direction) in directions)):
            directions.remove(self.opposite_direction(self.last_direction))

        if random.random() < self.epsilon:
            direction = random.choice(directions)
        else:
            direction = max(self.q_table[current_position], key=self.q_table[current_position].get)

        self.c.execute("command", direction)
        self.last_direction = direction
        return direction

    def walk_randomly_until_goal_reached(self, episodic, no_turning_back=True, visualization=False):
        """Executes an entire learning episode. The agent explores the world randomly and updates its Q-table.
        Returns the agent's path from the starting point to the goal,
        along with the actions taken and the rewards found, in the form of
        [(state1, state2, action1, reward1), (state2, state3, action2, reward2), ...]

        If not episodic, the agent updates its Q-table every time it takes an action.
        """
        path = []
        while not self.goal_reached() and not self.target_reached():
            position_before_action = self.current_position()
            action = self.execute_random_action(no_turning_back)
            position_after_action = self.current_position()
            reward = self.reward_at(position_after_action)

            path.append((position_before_action, position_after_action, action, reward))
            if not episodic:
                self.update_q_table(position_before_action, position_after_action, action, reward, visualization)
            self.mark_visited(position_after_action)

        return path

    def go_home(self):
        """Tells the server to place the agent in it's starting position."""
        self.c.execute("command", "home")
        self.last_direction = None
        self.mark_visited(self.current_position())

    def clean_path(self, path):
        """Tells the server to remove the marks from the states in path."""
        for coord in set(path):
            self.unmark(coord[0])

    def update_q_table(self, previous_state, current_state, action, reward, visualization=False):
        """Updates the agent's Q-table, according to the Q-learning update formula.
        Learning rate alpha and discount rate gamma are controlled by the agent and defined in the init function.
        If visualization is enabled, this also updates the agent's matplotlib table."""
        q_of_s_a = self.q_table[previous_state][action]
        self.q_table[previous_state][action] = (
            q_of_s_a + self.alpha * (reward + self.gamma * max(self.q_table[current_state].values()) - q_of_s_a)
        )

        if visualization:
            rev_previous_state = (previous_state[1]+1, previous_state[0])
            self.table.get_celld()[rev_previous_state].get_text().set_text(
                str(round(self.q_table[previous_state]["north"])) + " " +
                str(round(self.q_table[previous_state]["east"])) + "\n" +
                str(round(self.q_table[previous_state]["west"])) + " " +
                str(round(self.q_table[previous_state]["south"]))
            )
            plt.draw()
            plt.pause(.000001)

    def update_q_table_from_path(self, path, visualization=False):
        """Iterates though a path and updates the agent's Q-table.
        Path should take the form [(state1, state2, action1, reward1), (state2, state3, action2, reward2), ...].
        If visualization is enabled, this also updates the agent's matplotlib table."""
        for step in reversed(path):
            previous_state, current_state, action, reward = step
            self.update_q_table(previous_state, current_state, action, reward, visualization)

    def mark_arrow(self, coords, direction):
        """Tells the server to place an arrow pointing in direction at tile coords[0], coords[1]."""
        self.c.execute("marrow", "{type},{row},{column}".format(type=direction, column=coords[0], row=coords[1]))

    def display_policy(self):
        """Displays the arrow representation of the policy derived from the Q-table"""
        for entry in self.q_table.items():
            state, directions = entry
            max_direction = max(directions.items(), key=lambda x: x[1])[0]
            if directions[max_direction] != 0:
                self.mark_arrow(state, max_direction)

    def initialize_visualization(self):
        """Sets up the state necessary for matplotlib visualization, i.e. turns on matplotlib interactive mode and
        creates a matplotlib table representing the Q-table."""
        val1 = [i for i in range(self.maxCoord[0])]
        val2 = [i for i in range(self.maxCoord[1])]
        val3 = [[str(round(self.q_table[(c, r)]["north"])) + " " +
                 str(round(self.q_table[(c, r)]["east"])) + "\n" +
                 str(round(self.q_table[(c, r)]["west"])) + " " +
                 str(round(self.q_table[(c, r)]["south"])) for c in range(self.maxCoord[0])] for r in
                range(self.maxCoord[1])]
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_axis_off()
        self.table = ax.table(
            cellText=val3,
            rowLabels=val2,
            colLabels=val1,
            rowColours=["palegreen"] * 10,
            colColours=["palegreen"] * 10,
            cellLoc='center',
            loc='upper left')

        for cell in self.table.get_celld():
            self.table.get_celld()[cell].get_text().set_rotation(-45)

        ax.set_title('Q-table', fontweight="bold")

        cell_dict = self.table.get_celld()
        for i in range(0, len(val1)):
            for j in range(1, len(val3) + 1):
                cell_dict[(j, i)].set_height(.1)

        plt.show()
        plt.pause(.0001)
        input()

    def q_learn(self, episodes, episodic=True, no_turning_back=True, visualization=False):
        """Executes the Q-learning algorithm.
        Number of episodes defined by the user.
        Learning rate, discount factor and probability of random action defined in the agent's init function.
        Q-table update may be episodic or stepwise.
        May be visualized using matplotlib."""
        if visualization:
            self.initialize_visualization()

        for episode in range(episodes):
            path = self.walk_randomly_until_goal_reached(
                episodic,
                no_turning_back,
                visualization
            )
            self.clean_path(path)
            if episodic:
                self.update_q_table_from_path(path, visualization)
            self.go_home()
            self.display_policy()

        if visualization:
            plt.show()


# STARTING THE PROGRAM:
def main():
    print("Starting client!")
    agent = RLAgent(gamma=0.9, alpha=1, epsilon=1)
    if agent.get_connection() != -1:
        agent.q_learn(
            episodes=50,
            episodic=True,
            no_turning_back=False,
            visualization=False
        )
        agent.display_policy()
    input()


if __name__ == "__main__":
    main()
