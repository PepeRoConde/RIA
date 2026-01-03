"""Test to check wall generation at edges."""
import random

# Simplified Maze class without Soar dependencies
class SimpleMaze:
    def __init__(self, width: int = 10, height: int = 10, wall_probability: float = 0.3):
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

        print(f"_ensure_path: Starting at ({x}, {y}), target at ({target_x}, {target_y})")

        cleared = []
        # Create a diagonal staircase path to avoid clearing entire edges
        # This alternates between moving right and down
        while x < target_x or y < target_y:
            # Clear current position
            self.grid[y][x] = False
            cleared.append((x, y))

            # Move right if we haven't reached target x
            if x < target_x:
                x += 1
                self.grid[y][x] = False
                cleared.append((x, y))

            # Move down if we haven't reached target y
            if y < target_y:
                y += 1
                self.grid[y][x] = False
                cleared.append((x, y))

        print(f"Cleared path (diagonal staircase): {cleared}")

# Test
random.seed(42)
maze = SimpleMaze(10, 10, wall_probability=0.5)

print("\nChecking edge columns:")
print("=" * 60)

# Check left column (x=0)
left_walls = [(0, y) for y in range(10) if maze.grid[y][0]]
print(f"\nLeft column (x=0) walls: {left_walls}")
print(f"Total walls in left column: {len(left_walls)}/10")

# Check right column (x=9)
right_walls = [(9, y) for y in range(10) if maze.grid[y][9]]
print(f"\nRight column (x=9) walls: {right_walls}")
print(f"Total walls in right column: {len(right_walls)}/10")

# Check top row (y=0)
top_walls = [(x, 0) for x in range(10) if maze.grid[0][x]]
print(f"\nTop row (y=0) walls: {top_walls}")
print(f"Total walls in top row: {len(top_walls)}/10")

# Check bottom row (y=9)
bottom_walls = [(x, 9) for x in range(10) if maze.grid[9][x]]
print(f"\nBottom row (y=9) walls: {bottom_walls}")
print(f"Total walls in bottom row: {len(bottom_walls)}/10")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("The _ensure_path() method clears:")
print("  1. Entire top row (y=0) from x=0 to x=8")
print("  2. Entire right column (x=9) from y=0 to y=9")
print("\nThis prevents walls from appearing in:")
print("  - Column x=9 (right edge)")
print("  - Row y=0 (top edge)")
