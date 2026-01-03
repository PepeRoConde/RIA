"""
Turtle-based visualization for maze navigation.
Handles all UI rendering separately from the agent logic.
"""

import turtle
from typing import Tuple, Optional, Set
from .maze import Maze


class MazeTurtleUI:
    """Turtle graphics interface for visualizing maze navigation."""

    # Colors
    WALL_COLOR = "#2c3e50"
    PATH_COLOR = "#ecf0f1"
    START_COLOR = "#2ecc71"
    TARGET_COLOR = "#e74c3c"
    AGENT_COLOR = "#3498db"
    DISCOVERED_COLOR = "#bdc3c7"
    AGENT_PATH_COLOR = "#f39c12"

    def __init__(self, maze: Maze, cell_size: int = 40):
        """
        Initialize turtle UI.

        Args:
            maze: Maze object to visualize
            cell_size: Size of each cell in pixels
        """
        self.maze = maze
        self.cell_size = cell_size

        # Setup turtle screen
        self.screen = turtle.Screen()
        self.screen.title("Soar Maze Navigator")
        self.screen.setup(
            width=maze.width * cell_size + 100,
            height=maze.height * cell_size + 150
        )
        self.screen.tracer(0)  # Turn off auto-update for faster drawing

        # Create turtles for different drawing tasks
        self.maze_drawer = turtle.Turtle()
        self.maze_drawer.hideturtle()
        self.maze_drawer.speed(0)

        self.agent_turtle = turtle.Turtle()
        self.agent_turtle.shape("triangle")
        self.agent_turtle.color(self.AGENT_COLOR)
        self.agent_turtle.shapesize(1.5)
        self.agent_turtle.speed(0)

        self.info_writer = turtle.Turtle()
        self.info_writer.hideturtle()
        self.info_writer.speed(0)
        self.info_writer.penup()

        # Track discovered cells for visualization
        self.discovered_cells: Set[Tuple[int, int]] = set()
        self.agent_path = []

        # Draw initial maze
        self.draw_maze()

    def draw_maze(self):
        """Draw the static maze grid."""
        self.maze_drawer.clear()
        self.maze_drawer.penup()

        # Calculate offset to center maze
        offset_x = -self.maze.width * self.cell_size / 2
        offset_y = self.maze.height * self.cell_size / 2

        # Draw each cell
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                cell_x = offset_x + x * self.cell_size
                cell_y = offset_y - y * self.cell_size

                # Determine cell color
                if self.maze.grid[y][x]:
                    color = self.WALL_COLOR
                elif (x, y) == self.maze.start_pos:
                    color = self.START_COLOR
                elif (x, y) == self.maze.target_pos:
                    color = self.TARGET_COLOR
                else:
                    color = self.PATH_COLOR

                self._draw_cell(cell_x, cell_y, color)

        self.screen.update()

    def _draw_cell(self, x: float, y: float, color: str):
        """Draw a single cell at given position."""
        self.maze_drawer.goto(x, y)
        self.maze_drawer.fillcolor(color)
        self.maze_drawer.begin_fill()

        for _ in range(4):
            self.maze_drawer.forward(self.cell_size)
            self.maze_drawer.right(90)

        self.maze_drawer.end_fill()

    def _grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to screen coordinates (center of cell)."""
        offset_x = -self.maze.width * self.cell_size / 2
        offset_y = self.maze.height * self.cell_size / 2

        screen_x = offset_x + grid_x * self.cell_size + self.cell_size / 2
        screen_y = offset_y - grid_y * self.cell_size - self.cell_size / 2

        return screen_x, screen_y

    def update_agent_position(self, x: int, y: int, orientation: Optional[str] = None):
        """
        Update agent position and orientation.

        Args:
            x: Grid x position
            y: Grid y position
            orientation: 'north', 'east', 'south', 'west', or None
        """
        # Convert to screen coordinates
        screen_x, screen_y = self._grid_to_screen(x, y)

        # Move agent
        self.agent_turtle.penup()
        self.agent_turtle.goto(screen_x, screen_y)

        # Set orientation
        if orientation:
            orientation_angles = {
                'north': 90,
                'east': 0,
                'south': 270,
                'west': 180
            }
            self.agent_turtle.setheading(orientation_angles.get(orientation, 0))

        # Add to path
        if (x, y) not in self.agent_path:
            self.agent_path.append((x, y))

        self.screen.update()

    def mark_discovered(self, discovered: Set[Tuple[int, int]]):
        """
        Mark cells as discovered.

        Args:
            discovered: Set of (x, y) tuples representing discovered cells
        """
        # Find newly discovered cells
        new_cells = discovered - self.discovered_cells

        if new_cells:
            offset_x = -self.maze.width * self.cell_size / 2
            offset_y = self.maze.height * self.cell_size / 2

            for x, y in new_cells:
                # Skip start, target, and walls
                if ((x, y) == self.maze.start_pos or
                    (x, y) == self.maze.target_pos or
                    self.maze.grid[y][x]):
                    continue

                cell_x = offset_x + x * self.cell_size
                cell_y = offset_y - y * self.cell_size

                self._draw_cell(cell_x, cell_y, self.DISCOVERED_COLOR)

            self.discovered_cells = discovered.copy()
            self.screen.update()

    def draw_path(self):
        """Draw the agent's path."""
        if len(self.agent_path) < 2:
            return

        path_turtle = turtle.Turtle()
        path_turtle.hideturtle()
        path_turtle.speed(0)
        path_turtle.color(self.AGENT_PATH_COLOR)
        path_turtle.width(3)
        path_turtle.penup()

        # Draw path
        for i, (x, y) in enumerate(self.agent_path):
            screen_x, screen_y = self._grid_to_screen(x, y)

            if i == 0:
                path_turtle.goto(screen_x, screen_y)
                path_turtle.pendown()
            else:
                path_turtle.goto(screen_x, screen_y)

        path_turtle.penup()
        self.screen.update()

    def update_info(self, stats: dict):
        """
        Update information display.

        Args:
            stats: Dictionary with keys: position, target, steps, discovered, total_cells, orientation
        """
        self.info_writer.clear()

        # Position info text at bottom
        y_offset = -self.maze.height * self.cell_size / 2 - 50
        self.info_writer.goto(0, y_offset)

        # Build info string
        pos = stats.get('position', (0, 0))
        target = stats.get('target', (0, 0))
        steps = stats.get('steps', 0)
        discovered = stats.get('discovered', 0)
        total = stats.get('total_cells', 0)
        orientation = stats.get('orientation')

        info_text = f"Position: {pos} | Target: {target} | Steps: {steps}\n"
        info_text += f"Discovered: {discovered}/{total} cells"

        if orientation:
            info_text += f" | Facing: {orientation}"

        # Write centered text
        self.info_writer.write(
            info_text,
            align="center",
            font=("Arial", 12, "normal")
        )

        self.screen.update()

    def show_completion_message(self, steps: int, discovered: int):
        """Display completion message."""
        message_turtle = turtle.Turtle()
        message_turtle.hideturtle()
        message_turtle.penup()
        message_turtle.goto(0, 0)

        message_turtle.color("green")
        message_turtle.write(
            f"Target Reached!\nSteps: {steps}\nDiscovered: {discovered} cells",
            align="center",
            font=("Arial", 20, "bold")
        )

        self.screen.update()

    def set_speed(self, speed: int):
        """
        Set animation speed.

        Args:
            speed: 0 (fastest) to 10 (slowest), or special values like 0 for instant
        """
        self.agent_turtle.speed(speed)

    def mainloop(self):
        """Start the turtle event loop (blocking)."""
        self.screen.mainloop()

    def close(self):
        """Close the turtle window."""
        try:
            self.screen.bye()
        except:
            pass
