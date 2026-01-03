import random
from typing import Tuple, List, Optional

class Maze:
    """Generates and manages a random maze environment."""
    
    def __init__(self, width: int = 10, height: int = 10, wall_probability: float = 0.3):
        """
        Initialize maze with random walls.
        
        Args:
            width: Maze width
            height: Maze height
            wall_probability: Probability of a cell being a wall (0.0 to 1.0)
        """
        self.width = width
        self.height = height
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        
        # Generate random walls
        for y in range(height):
            for x in range(width):
                if random.random() < wall_probability:
                    self.grid[y][x] = True
        
        # Ensure start and target are not walls
        self.start_pos = (0, 0)
        self.target_pos = (width - 1, height - 1)
        self.grid[self.start_pos[1]][self.start_pos[0]] = False
        self.grid[self.target_pos[1]][self.target_pos[0]] = False
        
        # Ensure a path exists (simplified - just clear a basic path)
        self._ensure_path()
    
    def _ensure_path(self):
        """Ensure at least one path exists from start to target using a diagonal staircase pattern."""
        x, y = self.start_pos
        target_x, target_y = self.target_pos

        # Create a diagonal staircase path to avoid clearing entire edges
        # This alternates between moving right and down
        while x < target_x or y < target_y:
            # Clear current position
            self.grid[y][x] = False

            # Move right if we haven't reached target x
            if x < target_x:
                x += 1
                self.grid[y][x] = False

            # Move down if we haven't reached target y
            if y < target_y:
                y += 1
                self.grid[y][x] = False
    
    def is_wall(self, x: int, y: int) -> bool:
        """Check if position contains a wall."""
        if not self.is_valid_position(x, y):
            return True  # Out of bounds treated as wall
        return self.grid[y][x]
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within maze bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_sensors(self, x: int, y: int) -> dict:
        """
        Get sensor readings for all four directions.
        Grid boundaries are treated as walls.
        
        Returns:
            Dictionary with keys: 'up', 'down', 'left', 'right'
            Values are True if wall/boundary detected, False if clear
        """
        return {
            'up': self.is_wall(x, y - 1),
            'down': self.is_wall(x, y + 1),
            'left': self.is_wall(x - 1, y),
            'right': self.is_wall(x + 1, y)
        }
    
    def display(self, agent_pos: Tuple[int, int] = None, orientation: Optional[str] = None):
        """
        Display the maze in console.
        
        Args:
            agent_pos: Current position of the agent (x, y)
            orientation: Agent's facing direction ('north', 'east', 'south', 'west')
        """
        # Orientation symbols (using ASCII for Windows compatibility)
        orientation_symbols = {
            'north': '^',  # pointing up
            'east': '>',   # pointing right
            'south': 'v',  # pointing down
            'west': '<'    # pointing left
        }
        
        print("\n" + "=" * (self.width * 2 + 1))
        for y in range(self.height):
            row = "|"
            for x in range(self.width):
                if agent_pos and (x, y) == agent_pos:
                    if orientation and orientation in orientation_symbols:
                        row += orientation_symbols[orientation] + " "
                    else:
                        row += "A "  # Agent without orientation
                elif (x, y) == self.target_pos:
                    row += "T "  # Target
                elif self.grid[y][x]:
                    row += "# "  # Wall
                else:
                    row += ". "  # Empty
            row += "|"
            print(row)
        print("=" * (self.width * 2 + 1))
        
        # Display agent position and orientation
        if orientation:
            print(f"Agent: {agent_pos} (facing {orientation}), Target: {self.target_pos}\n")
        else:
            print(f"Agent: {agent_pos}, Target: {self.target_pos}\n")
