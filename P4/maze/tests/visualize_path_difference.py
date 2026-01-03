"""Visualize the difference between old and new path algorithms."""
import random

def visualize_maze(grid, cleared_path, title):
    """Display maze with cleared path highlighted."""
    print(f"\n{title}")
    print("=" * 42)

    height = len(grid)
    width = len(grid[0])

    for y in range(height):
        row = "|"
        for x in range(width):
            if (x, y) in cleared_path:
                row += "· "  # Cleared path
            elif (x, y) == (0, 0):
                row += "S "  # Start
            elif (x, y) == (width-1, height-1):
                row += "T "  # Target
            elif grid[y][x]:
                row += "# "  # Wall
            else:
                row += ". "  # Open space
        row += "|"
        print(row)
    print("=" * 42)
    print(f"Cells cleared by ensure_path: {len(cleared_path)}")


def old_ensure_path(width, height):
    """Old L-shaped path algorithm."""
    cleared = []
    x, y = 0, 0
    target_x, target_y = width - 1, height - 1

    # Clear horizontal
    while x < target_x:
        cleared.append((x, y))
        x += 1

    # Clear vertical
    while y < target_y:
        cleared.append((x, y))
        y += 1

    cleared.append((target_x, target_y))
    return cleared


def new_ensure_path(width, height):
    """New diagonal staircase path algorithm."""
    cleared = []
    x, y = 0, 0
    target_x, target_y = width - 1, height - 1

    while x < target_x or y < target_y:
        cleared.append((x, y))

        if x < target_x:
            x += 1
            cleared.append((x, y))

        if y < target_y:
            y += 1
            cleared.append((x, y))

    return list(set(cleared))  # Remove duplicates


# Generate same maze for both
random.seed(42)
width, height = 10, 10
grid = [[random.random() < 0.4 for _ in range(width)] for _ in range(height)]

# Apply old path
old_path = old_ensure_path(width, height)
old_grid = [row[:] for row in grid]
for x, y in old_path:
    old_grid[y][x] = False

# Apply new path
new_path = new_ensure_path(width, height)
new_grid = [row[:] for row in grid]
for x, y in new_path:
    new_grid[y][x] = False

# Visualize
print("\n" + "=" * 60)
print("PATH GENERATION COMPARISON")
print("=" * 60)
print("\nLegend: S=Start, T=Target, #=Wall, .=Open, ·=Cleared Path")

visualize_maze(old_grid, old_path, "OLD: L-shaped path (clears entire edges)")
visualize_maze(new_grid, new_path, "NEW: Diagonal staircase (minimal clearing)")

# Count edge walls
def count_edge_walls(grid):
    width = len(grid[0])
    height = len(grid)

    left = sum(1 for y in range(height) if grid[y][0])
    right = sum(1 for y in range(height) if grid[y][width-1])
    top = sum(1 for x in range(width) if grid[0][x])
    bottom = sum(1 for x in range(width) if grid[height-1][x])

    return left, right, top, bottom

old_edges = count_edge_walls(old_grid)
new_edges = count_edge_walls(new_grid)

print("\n" + "=" * 60)
print("EDGE WALL COUNTS")
print("=" * 60)
print(f"{'Edge':<15} {'OLD':<15} {'NEW':<15} {'Improvement'}")
print("-" * 60)
print(f"{'Left (x=0)':<15} {old_edges[0]:<15} {new_edges[0]:<15} {'+' if new_edges[0] >= old_edges[0] else ''}{new_edges[0] - old_edges[0]}")
print(f"{'Right (x=9)':<15} {old_edges[1]:<15} {new_edges[1]:<15} {'+' if new_edges[1] >= old_edges[1] else ''}{new_edges[1] - old_edges[1]}")
print(f"{'Top (y=0)':<15} {old_edges[2]:<15} {new_edges[2]:<15} {'+' if new_edges[2] >= old_edges[2] else ''}{new_edges[2] - old_edges[2]}")
print(f"{'Bottom (y=9)':<15} {old_edges[3]:<15} {new_edges[3]:<15} {'+' if new_edges[3] >= old_edges[3] else ''}{new_edges[3] - old_edges[3]}")
print("=" * 60)
print(f"\n✅ Fixed: Edges can now have walls!")
