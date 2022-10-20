import numpy as np
import matplotlib.pyplot as plt

# set style for grid world
plt.style.use('ggplot')


class WarehouseAgent:
    """
    The Warehouse Agent Environment
    Inspired from Sokoban

    The game is played on a board of squares, where each square is a floor or a wall. Some floor squares
    contain boxes, and some floor squares are marked as storage locations. The task is to make the
    warehouse agent push the box to reach its intended destination.
    The warehouse agent is confined to the board and may move horizontally or vertically onto empty
    squares (never through walls or boxes). The agent can move a box by walking up to it and pushing it to
    the square beyond. Boxes cannot be pulled, and they cannot be pushed to squares with walls or other
    boxes. The number of boxes equals the number of storage locations. The puzzle is solved when all boxes
    are placed at storage locations.

    Dimension: The grid world has a dimension of size 6 x 7

    State Space: Each row-column cell in the grid denotes the state of that cell
    [For eg, the agent in the image is located at row no. 2 and col no. 3 which translates to state (1,2)]

    Action Space: The agent can move in four directions: UP, DOWN, LEFT, RIGHT. It can only push boxes
    forward and can’t move them in any other direction. If the agent pushes the box to the edge of the wall,
    then it can not push the box out of it - certain actions are irreversible

    Reward: Agent gets a reward of -1 when the box is not at the storage location. The agent receives a
    reward of 0 when the box is at the goal location.

    Terminating conditions: If either the box reaches the goal location or it gets stuck at a place where it
    cannot get out.

    """

    def __init__(self):
        """
        Initializing the environment
        """
        # setting grid dimensions
        self.GRID_DIM = [6, 7]

        # setting wall locations
        self.walls = [[i, 0] for i in range(4)] + [[0, i] for i in range(1, 7)]
        self.walls += [[3, 1], [3, 2], [4, 2], [5, 2], [5, 3], [5, 4], [5, 5], [4, 5], [3, 5], [3, 6], [2, 6], [1, 6]]

        # setting initial state                  #### SET INITIAL STATE HERE ####
        self.agent_position = np.asarray([1, 2])
        self.box_location = np.asarray([4, 3])
        self.goal_location = np.asarray([3, 1])
        self.obs = np.asarray([self.agent_position, self.box_location, self.goal_location])

    def step(self, action="up"):
        """
        Function to control and evaluate the agents' action

        Args:
          action: pass on the action which the agent needs to take at that time step
        Returns:
          new_state: the new state agent reaches after taking the action
          reward: the reward obtained on taking the action
          done: boolean value to determine if episode terminating condition is reached
        """

        if action == "up":

            # get above location coordinates (2)
            up_loc = (self.agent_position - np.asarray([1, 0])).tolist()
            up_loc_2 = (up_loc - np.asarray([1, 0])).tolist()

            # if there are walls above or there is a box beyond which is a wall, do not move
            if (up_loc[::-1] in self.walls) or (up_loc_2[::-1] in self.walls and up_loc == self.box_location.tolist()):
                return self.obs, -1.0, 0, None

            # if there are no walls, or if the box can be pushed, update the locations
            else:
                if up_loc != self.box_location.tolist():
                    self.agent_position = np.asarray(up_loc)
                else:
                    self.agent_position = np.asarray(up_loc)
                    self.box_location = np.asarray(up_loc_2)

        if action == "down":

            # get below location coordinates (2)
            down_loc = (self.agent_position + np.asarray([1, 0])).tolist()
            down_loc_2 = (down_loc + np.asarray([1, 0])).tolist()

            # if there are walls below or there is a box beyond which is a wall, do not move
            if (down_loc[::-1] in self.walls) or (down_loc_2[::-1] in self.walls and down_loc == self.box_location.tolist()):
                return self.obs, -1.0, 0, None

            # if there are no walls, or if the box can be pushed, update the locations
            else:
                if down_loc == self.box_location.tolist():
                    self.agent_position = np.asarray(down_loc)
                    self.box_location = np.asarray(down_loc_2)
                else:
                    self.agent_position = np.asarray(down_loc)

        if action == "left":

            # get location coordinates to the left (2)
            left_loc = (self.agent_position - np.asarray([0, 1])).tolist()
            left_loc_2 = (left_loc - np.asarray([0, 1])).tolist()

            # if there are walls below or there is a box beyond which is a wall, do not move
            if (left_loc[::-1] in self.walls) or (left_loc_2[::-1] in self.walls and left_loc == self.box_location.tolist()):
                return self.obs, -1.0, 0, None

            # if there are no walls, or if the box can be pushed, update the locations
            else:
                if left_loc == self.box_location.tolist():
                    self.agent_position = np.asarray(left_loc)
                    self.box_location = np.asarray(left_loc_2)
                else:
                    self.agent_position = np.asarray(left_loc)

        if action == "right":

            # get location coordinates to the right (2)
            right_loc = (self.agent_position + np.asarray([0, 1])).tolist()
            right_loc_2 = (right_loc + np.asarray([0, 1])).tolist()

            # if there are walls below or there is a box beyond which is a wall, do not move
            if (right_loc[::-1] in self.walls) or (right_loc_2[::-1] in self.walls and right_loc == self.box_location.tolist()):
                return self.obs, -1.0, 0, None

            # if there are no walls, or if the box can be pushed, update the locations
            else:
                if right_loc == self.box_location.tolist():
                    self.agent_position = np.asarray(right_loc)
                    self.box_location = np.asarray(right_loc_2)
                else:
                    self.agent_position = np.asarray(right_loc)

        # define state space
        self.obs = np.asarray([self.agent_position, self.box_location, self.goal_location])

        # check if the box is further movable
        if self.__check_box() == 0:
            return self.obs, -1.0, 0, None
        # check if the box is cornered and cannot be moved
        elif self.__check_box() == -1:
            return self.obs, -1.0, 1, None
        # check if the box reached its goal location
        else:
            return self.obs, 0.0, 1, None

    def __check_box(self):
        """
        Private function to check if the box reached the corner or a goal location

        Returns:
          1 if box reached the goal location
          0 if box is still movable
          -1 if box is cornered and cannot be moved
        """

        # if goal reached
        if self.box_location.tolist() == self.goal_location.tolist():
            return 1

        # define a vector that checks for walls in all 4 directions
        occ_vec = [0, 0, 0, 0]

        # get the coordinates of the box surroundings
        box_surroundings = self.box_location[::-1] + np.asarray([[0, 1], [1, 0], [-1, 0], [0, -1]])

        # check if the surroundings coincide with the wall, if so, then update the occ_vec vector
        for is_occ, direction in enumerate(box_surroundings.tolist()):
            if any(i == direction for i in self.walls):
                occ_vec[is_occ] = 1

        # if adjacent directions are covered by walls (corners), or if there are
        # more than 2 walls around, return immovable
        if (occ_vec in [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]) or np.sum(occ_vec) > 2:
            return -1

        # if box still movable
        else:
            return 0

    def reset(self):
        """
        Function to reset the environment at the end of each episode to its initial state configuration

        Returns:
          state: the state of the environment reset to its initial conditions
        """

        # reset agent and box locations
        self.agent_position = [1, 2]
        self.box_location = [4, 3]

        # define state space
        self.obs = np.asarray([self.agent_position, self.box_location, self.goal_location])

        return self.obs

    def render(self):
        """
          Function to get the simulation of the warehouse agent system
        """

        # define x and y axis
        grid_x_axis = np.linspace(1, 6, 6)
        grid_y_axis = np.linspace(1, 7, 7)

        # define figure and plot
        fig = plt.figure(figsize=(9, 11))
        ax = plt.gca()

        # agent box and storage (movable)
        agent = plt.Circle((self.agent_position[1] + 0.5, self.agent_position[0] + 0.5), 0.26, color='g', alpha=0.8)
        box = plt.Rectangle((self.box_location[1] + 0.2, self.box_location[0] + 0.2), 0.6, 0.6, color="brown",
                            alpha=0.8)

        # wall/storage (immovable)
        storage = plt.Rectangle((self.goal_location[1] + 0.2, self.goal_location[0] + 0.2), 0.6, 0.6, color="blue",
                                alpha=0.15)
        storage_circle = plt.Circle((self.goal_location[1] + 0.5, self.goal_location[0] + 0.5), 0.13, color="blue",
                                    alpha=0.3)

        for wall in self.walls:
            wall_box = plt.Rectangle((wall[0], wall[1]), 1, 1, color="black", alpha=0.85)
            ax.add_patch(wall_box)

        # add all patches
        ax.add_patch(storage)
        ax.add_patch(storage_circle)
        ax.add_patch(agent)
        ax.add_patch(box)

        # additional plot settings
        plt.xticks(ticks=grid_x_axis, labels=grid_x_axis)
        plt.yticks(ticks=grid_y_axis, labels=grid_y_axis)
        ax.invert_yaxis()
        plt.show()