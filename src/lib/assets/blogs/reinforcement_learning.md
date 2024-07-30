---
title: Maze Solving Robot with Reinforcement Learning
author: Akshay Ballal
stage: live
image: https://res.cloudinary.com/dltwftrgc/image/upload/v1690576074/Blogs/reinforcement_learning/cover_image_jamm6z.jpg
description:  Discover the power of Reinforcement Learning (RL) in this beginner-friendly blog. Learn how an Agent interacts with an Environment, aiming to maximize rewards over time. Explore the concept of 'Return' and the use of discounting factor 'gamma' for continuous tasks. Build a maze-solving algorithm and visualize results with Pygame. Perfect RL's basics in this 'Hello World' project!
date: 07-28-2023
---

In this blog we will use the Value Iteration Algorithm to solve a maze and will implement the algorithm and simulate it with PyGame. This is what we will be making.

<img width = 600 src = "https://res.cloudinary.com/dltwftrgc/image/upload/v1690719895/Blogs/reinforcement_learning/demo_fvdswm.gif">

You can find the code here
Git Repo: https://github.com/akshayballal95/reinforcement_learning_maze.git

## Setting up the Project 

Let us start by first creating a virtual environment using the following command

```bash
python -m venv venv
```

and activate it using the following command for windows
```bash
venv\Scripts\activate.bat
```

and for mac
```bash
source venv\bin\activate
```

Next, install the dependencies

```bash
pip install -r requirements.txt
```

---
## Creating the Maze Class

Create a new file called `maze.py` and import the libraries in it

``` python
import numpy as np
import random
from typing import Tuple
```

Next instantiate the Maze class and define an `__init__` function

```python
class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        self.goal = (23, 20)
        self.number_of_tiles = SIZE
        self.tile_size = MAZE_HEIGHT // self.number_of_tiles
        self.walls = self.create_walls(level)
        self.goal_pos = goal_pos
        self.state = self.get_init_state(level)
        self.maze = self.create_maze(level)

        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )

```

The maze is created based on the `level` provided as input, and the goal position within the maze is specified by the `goal_pos` parameter. The class also contains attributes and methods related to solving the maze using reinforcement learning techniques.

- `self.state_values` : A NumPy array of size `self.number_of_tiles x self.number_of_tiles` filled with zeros. This array will be used to store the estimated values of each state during the reinforcement learning process.
- `self.policy_probs` : A NumPy array of size `self.number_of_tiles x self.number_of_tiles x 4`, where 4 represents the number of possible actions (up, down, left, right). Each entry in this array is initialized with `0.25`, which means initially, each action is equally likely in each state. This array will be used to store the action probabilities for each state during the reinforcement learning process.

---
## Creating the Maze

```python
   def create_maze(self, level):
        maze = []
        walls = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == " ":
                    maze.append((row, col))
                elif level[row][col] == "X":
                    walls.append((row,col))
        return maze, walls

    def get_init_state(self, level):
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)
```

The provided code snippet contains two methods for the maze environment:

1. `create_maze(self, level)`: This method takes the `level` parameter, which is a list of strings representing the maze layout. It iterates through each cell in the `level` and identifies the positions of maze tiles and walls based on the characters " " and "X", respectively. It appends the positions of maze tiles as `(row, col)` tuples to the `maze` list and the positions of walls to the `walls` list. Finally, it returns a tuple containing two lists: the `maze` list with the positions of maze tiles and the `walls` list with the positions of walls.
    
2. `get_init_state(self, level)`: This method also takes the `level` parameter, which represents the maze layout. It iterates through each cell in the `level` and searches for the character "P". When it finds "P" in the maze, it immediately returns the position `(row, col)` of "P" as the initial state for the maze-solving process. The function stops searching once it finds "P" and returns the position as a tuple `(row, col)`.

These two methods are used to set up the maze environment, define the initial state, and obtain the positions of walls and maze tiles for further processing and maze-solving algorithms.

---
## Take Steps and Compute Reward

```python
def compute_reward(self, state: Tuple[int, int], action: int):
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal_pos)

    def step(self, action):
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def simulate_step(self, state, action):
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal
        return next_state, reward, done

    def _get_next_state(self, state: Tuple[int, int], action: int):
        if action == 0:
            next_state = (state[0], state[1] - 1)
        elif action == 1:
            next_state = (state[0] - 1, state[1])
        elif action == 2:
            next_state = (state[0], state[1] + 1)
        elif action == 3:
            next_state = (state[0] + 1, state[1])
        else:
            raise ValueError("Action value not supported:", action)
        if (next_state[0], next_state[1]) not in self.walls:
            return next_state
        return state
```

This code snippet defines several methods related to a maze-solving environment using reinforcement learning. Let's explain each method briefly:

1. `compute_reward(self, state: Tuple[int, int], action: int)`: This method calculates the reward for taking an `action` from the current `state` in the maze. The reward is a negative value (-1.0) if the `state` is not equal to the `goal_pos`, otherwise, it is zero. The goal is to minimize the total reward while navigating the maze.
    
2. `step(self, action)`: This method takes a `step` in the maze environment, executing the given `action`. It returns a tuple containing the `next_state`, the `reward` obtained from the `compute_reward` method for the current state and action, and a `done` flag indicating whether the agent has reached the `goal` state or not.
    
3. `simulate_step(self, state, action)`: Similar to the `step` method, this method simulates a step in the maze environment for a given `state` and `action`, returning the `next_state`, `reward`, and `done` flag without updating the actual environment state.
    
4. `_get_next_state(self, state: Tuple[int, int], action: int)`: This is a private method used internally to get the `next_state` given the current `state` and the chosen `action`. It calculates the next state based on the direction of the action (0: left, 1: up, 2: right, 3: down) and checks if the next state is not a wall (not present in the `walls` list). If the next state is a wall, it returns the current state, meaning the agent stays in the same position.

---
### Solving the Maze

```python
    def solve(self, gamma=0.99, theta=1e-6):
	   
		delta = float("inf")
	
		while delta > theta:
			delta = 0
			for row in range(self.number_of_tiles):
				for col in range(self.number_of_tiles):
					if (row, col) not in self.walls:
						old_value = self.state_values[row, col]
						q_max = float("-inf")
	
						for action in range(4):
							next_state, reward, done = self.simulate_step(
								(row, col), action
							)
							value = reward + gamma * self.state_values[next_state]
							if value > q_max:
								q_max = value
								action_probs = np.zeros(shape=(4))
								action_probs[action] = 1
	
						self.state_values[row, col] = q_max
						self.policy_probs[row, col] = action_probs
	
						delta = max(delta, abs(old_value - self.state_values[row, col]))

```

The `solve` method represents the implementation of the Value Iteration algorithm to find an optimal policy for the maze environment.

Let's go through the method step by step:

1. Initialization:
    
    - `gamma`: The discount factor for future rewards (default value: 0.99).
    - `theta`: The threshold for convergence (default value: 1e-6).
    - `delta`: A variable initialized to infinity, which will be used to measure the change in state values during the iteration.
2. Iterative Value Update:
    
    - The method uses a while loop that continues until the change in state values (`delta`) becomes smaller than the specified convergence threshold (`theta`).
    - Within the loop, the `delta` variable is reset to 0 before each iteration.
3. Value Iteration:
    
    - For each cell in the maze (excluding walls), the method calculates the Q-value (q_max) for all possible actions (up, down, left, right).
    - The Q-value is determined based on the immediate reward obtained from the `simulate_step` method and the discounted value of the next state's value (gamma * self.state_values[next_state]).
    - The action with the maximum Q-value (q_max) is selected, and the corresponding policy probabilities (action_probs) are updated to reflect that this action has the highest probability of being chosen.
4. Updating State Values and Policy Probabilities:
    
    - After calculating the Q-values for all actions in a given state, the state value (self.state_values[row, col]) for that state is updated to the maximum Q-value (q_max) among all actions.
    - The policy probabilities (self.policy_probs[row, col]) for that state are updated to have a high probability (1.0) for the action that resulted in the maximum Q-value and 0 probability for all other actions.
5. Convergence Check:
    
    - After updating the state values and policy probabilities for all states in the maze, the method checks whether the maximum change in state values (`delta`) during this iteration is smaller than the convergence threshold (`theta`).
    - If the condition is met, the loop terminates, and the maze-solving process is considered converged.

The process of value iteration is repeated until the state values converge, and an optimal policy is determined, maximizing the expected cumulative reward for navigating the maze from the initial state to the goal state.

---
## Simulating the Robot with PyGame

Let's now simulate our bot by using the Maze data in PyGame and building the walls, placing the treasure and moving the player. Create a new python script called `main.py` and add this code to it. 

```python
import pygame
import numpy as np
from maze import Maze
import threading

# Constants
GAME_HEIGHT = 600
GAME_WIDTH = 600
NUMBER_OF_TILES = 25
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700
TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES

# Maze layout
level = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X XXXXXXXX          XXXXX",
    "X XXXXXXXX  XXXXXX  XXXXX",
    "X      XXX  XXXXXX  XXXXX",
    "X      XXX  XXX        PX",
    "XXXXXX  XX  XXX        XX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "X  XXX      XXXXXXXXXXXXX",
    "X  XXX  XXXXXXXXXXXXXXXXX",
    "X         XXXXXXXXXXXXXXX",
    "X             XXXXXXXXXXX",
    "XXXXXXXXXXX      XXXXX  X",
    "XXXXXXXXXXXXXXX  XXXXX  X",
    "XXX  XXXXXXXXXX         X",
    "XXX                     X",
    "XXX         XXXXXXXXXXXXX",
    "XXXXXXXXXX  XXXXXXXXXXXXX",
    "XXXXXXXXXX              X",
    "XX   XXXXX              X",
    "XX   XXXXXXXXXXXXX  XXXXX",
    "XX    XXXXXXXXXXXX  XXXXX",
    "XX        XXXX          X",
    "XXXX                    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]

env = Maze(
    level,
    goal_pos=(23, 20),
    MAZE_HEIGHT=GAME_HEIGHT,
    MAZE_WIDTH=GAME_WIDTH,
    SIZE=NUMBER_OF_TILES,
)
env.reset()
env.solve()

SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700

TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES


# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
pygame.display.set_caption("Maze Solver")  # Set a window title

surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
clock = pygame.time.Clock()
running = True

# Get the initial player and goal positions
treasure_pos = env.goal_pos
player_pos = env.state


def reset_goal():
    # Check if the player reached the goal, then reset the goal
    if env.state == env.goal_pos:
        env.reset()
        env.solve()


# Game loop
while running:
    # Start a new thread
    x = threading.Thread(target=reset_goal)
    x.daemon = True
    x.start()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the surface
    surface.fill((27, 64, 121))

    # Draw the walls in the maze
    for row in range(len(level)):
        for col in range(len(level[row])):
            if level[row][col] == "X":
                pygame.draw.rect(
                    surface,
                    (241, 162, 8),
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

    # Draw the player's position
    pygame.draw.rect(
        surface,
        (255, 51, 102),
        pygame.Rect(
            player_pos[1] * TILE_SIZE,
            player_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=3,
    )

    # Draw the goal position
    pygame.draw.rect(
        surface,
        "green",
        pygame.Rect(
            env.goal_pos[1] * TILE_SIZE,
            env.goal_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=TILE_SIZE,
    )

    # Update the screen
    screen.blit(
        surface, ((SCREEN_HEIGHT - GAME_HEIGHT) / 2, (SCREEN_WIDTH - GAME_WIDTH) / 2)
    )
    pygame.display.flip()

    # Get the action based on the current policy
    action = np.argmax(env.policy_probs[player_pos])

    # Move the player based on the action
    if (
        action == 1
        and player_pos[0] > 0
        and (player_pos[0] - 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] - 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 3
        and player_pos[0] < NUMBER_OF_TILES - 1
        and (player_pos[0] + 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] + 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 0
        and player_pos[1] > 0
        and (player_pos[0], player_pos[1] - 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] - 1)
        env.state = player_pos
    elif (
        action == 2
        and player_pos[1] < NUMBER_OF_TILES - 1
        and (player_pos[0], player_pos[1] + 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] + 1)
        env.state = player_pos

    x.join()

    # Control the frame rate of the game
    clock.tick(60)

# Quit Pygame when the game loop is exited
pygame.quit()

```

Here's a step-by-step explanation of the code:

1. Import libraries and create constants:
    
    - The code imports the required libraries, including Pygame, NumPy, and a custom `Maze` class.
    - Several constants are defined to set up the maze environment, screen dimensions, and tile sizes.
2. Create the maze environment:
    
    - The maze layout is defined using a list of strings called `level`.
    - An instance of the `Maze` class is created, passing the `level` and goal position as parameters.
    - The `env` object is then used to reset the maze and find the optimal policy using the `solve` method.
3. Initialize Pygame:
    
    - Pygame is initialized, and a game window is created with a specified caption.
4. The main game loop:
    
    - The program enters a loop that runs until the `running` variable becomes `False`.
    - The `reset_goal` function is called in a separate thread to check if the player has reached the goal and reset the goal if needed.
5. Drawing the maze:
    
    - The maze is drawn on the screen by iterating through the `level` list and drawing tiles based on wall characters ("X").
    - The player's position is drawn as a rectangle filled with a specific color (red) and centered in the maze cell.
    - The goal position is drawn as a green rectangle with rounded corners.
6. Update the game state and player movement:
    
    - The player's action is determined based on the current policy (`env.policy_probs`) at the player's position. The action with the highest probability is chosen using `np.argmax`.
    - Based on the chosen action, the player's position is updated if it is a valid move (not hitting walls).
    - The player's position in the `env` object is updated as well.
7. Frame rate control:
    
    - The game loop is controlled with a frame rate of 60 frames per second using `clock.tick(60)`.
8. Exiting the game:
    
    - The game loop exits when the `running` variable becomes `False`, typically triggered by the user closing the game window.
    - Pygame is closed with `pygame.quit()`.

That's all. Now you can run `main.py`.

Hope this was useful for you guys to understand the basics of reinforcement learning. 

<div style="display: flex; gap:10px; align-items: center">
<img width ="90" height="90" src  = "https://res.cloudinary.com/dltwftrgc/image/upload/t_Facebook ad/v1683659009/Blogs/AI_powered_game_bot/profile_lyql45.jpg" >
<div style = "display: flex; flex-direction:column; gap:10px; justify-content:space-between">
<p style="padding:0; margin:0">my website: <a href ="http://www.akshaymakes.com/">http://www.akshaymakes.com/</a></p>
<p  style="padding:0; margin:0">linkedin: <a href ="https://www.linkedin.com/in/akshay-ballal/">https://www.linkedin.com/in/akshay-ballal/</a></p>
<p  style="padding:0; margin:0">twitter: <a href ="https://twitter.com/akshayballal95">https://twitter.com/akshayballal95/</a></p>
</div>
</div>