# Turtle-Based UI for Soar Maze Navigator

## Overview

This update separates the UI visualization from the agent logic and uses Python's `turtle` graphics library for a cleaner, more interactive visualization.

## Changes Made

### 1. Fixed DestroyWME Calls

**Problem**: The original code called `DestroyWME()` directly on WMElement objects:
```python
self.wmes[key].DestroyWME()  # WRONG
output_link.DestroyWME(wme)  # WRONG
```

**Solution**: According to the Soar Python API, `DestroyWME` should be called on the Agent with the WME as a parameter:
```python
self.agent.DestroyWME(self.wmes[key])  # CORRECT
self.agent.DestroyWME(wme)  # CORRECT
```

**Files Modified**:
- `enviroment/agent_interface.py`:
  - Line 108: Fixed WME destruction in `_update_input_link()`
  - Line 219: Fixed command WME destruction in `process_output()`

### 2. New Turtle-Based Visualization

**Created**: `enviroment/turtle_ui.py`

A dedicated UI module that handles all visualization using Python's turtle graphics library:

**Features**:
- Clean separation of UI from agent logic
- Real-time visualization of:
  - Maze grid with walls, start, and target
  - Agent position and orientation (arrow)
  - Discovered cells (highlighted)
  - Agent's path
  - Current statistics (position, steps, discovered cells)
- Configurable cell size and colors
- Completion message when target is reached

**Key Classes**:
- `MazeTurtleUI`: Main UI class with methods:
  - `draw_maze()`: Render the static maze
  - `update_agent_position()`: Update agent location and orientation
  - `mark_discovered()`: Highlight discovered cells
  - `draw_path()`: Show agent's traveled path
  - `update_info()`: Display statistics
  - `show_completion_message()`: Display success message

### 3. New Main Runner Script

**Created**: `run_with_turtle.py`

A complete simulation runner with turtle visualization:

**Usage**:
```bash
# Basic usage (DFS strategy, 10x10 maze)
python run_with_turtle.py

# Custom maze size and strategy
python run_with_turtle.py --strategy naive --width 15 --height 15

# Adjust visualization speed
python run_with_turtle.py --delay 0.05  # Faster
python run_with_turtle.py --delay 0.5   # Slower
python run_with_turtle.py --delay 0     # Instant (no animation)

# Larger cells for better visibility
python run_with_turtle.py --cell-size 50

# More complex maze
python run_with_turtle.py --walls 0.4 --width 20 --height 20
```

**Command-line Options**:
- `--strategy {naive,dfs}`: Navigation strategy (default: dfs)
- `--width WIDTH`: Maze width (default: 10)
- `--height HEIGHT`: Maze height (default: 10)
- `--walls WALLS`: Wall probability 0.0-1.0 (default: 0.3)
- `--max-steps MAX_STEPS`: Maximum steps before timeout (default: 1000)
- `--delay DELAY`: Delay between steps in seconds (default: 0.1)
- `--cell-size CELL_SIZE`: Cell size in pixels (default: 40)

## File Structure

```
maze/
├── enviroment/
│   ├── __init__.py           # Updated to export TurtleUI
│   ├── maze.py               # Maze logic (unchanged)
│   ├── agent_interface.py    # Agent logic (DestroyWME fixed)
│   └── turtle_ui.py          # NEW: Turtle visualization
├── agent/
│   ├── maze-navigator-dfs.soar
│   └── maze-navigator-naive.soar
├── run_with_turtle.py        # NEW: Main runner with turtle UI
└── test_destroy_wme.py       # NEW: Test for DestroyWME fix
```

## Color Scheme

The turtle visualization uses the following colors:
- **Walls**: Dark blue-gray (`#2c3e50`)
- **Open path**: Light gray (`#ecf0f1`)
- **Start position**: Green (`#2ecc71`)
- **Target position**: Red (`#e74c3c`)
- **Agent**: Blue triangle (`#3498db`)
- **Discovered cells**: Gray (`#bdc3c7`)
- **Agent's path**: Orange line (`#f39c12`)

## API Usage Example

You can also use the turtle UI programmatically:

```python
from enviroment import Maze, MazeAgent, MazeTurtleUI

# Create maze
maze = Maze(width=10, height=10, wall_probability=0.3)

# Initialize agent
agent = MazeAgent(maze, strategy="dfs")
agent.initialize_soar()

# Initialize UI
ui = MazeTurtleUI(maze, cell_size=40)

# Update loop
while not agent.has_reached_target():
    agent.run_step()

    # Update visualization
    ui.update_agent_position(agent.x, agent.y,
                             agent.ORIENTATION_NAMES[agent.orientation])
    ui.mark_discovered(agent.discovered)
    ui.update_info(agent.get_stats())

# Show completion
ui.show_completion_message(len(agent.path) - 1, len(agent.discovered))
ui.mainloop()
```

## Troubleshooting

### Soar DLL Loading Issues (Windows)

If you see `ImportError: DLL load failed`, ensure:
1. The Soar installation path in `agent_interface.py` is correct
2. The correct Windows binaries are in the path (win_x86-64 folder)
3. Required DLLs are accessible

### Turtle Window Not Appearing

- Check that Python has tkinter installed: `python -m tkinter`
- On Linux, you may need: `sudo apt-get install python3-tk`

### Slow Performance

- Reduce delay: `--delay 0`
- Decrease cell size: `--cell-size 30`
- Use smaller maze: `--width 8 --height 8`

## Testing

Run the DestroyWME fix test (requires working Soar installation):
```bash
python test_destroy_wme.py
```

## Summary

The refactored code now:
1. ✅ Correctly uses `Agent.DestroyWME(wme)` syntax
2. ✅ Separates UI logic into dedicated module
3. ✅ Uses turtle graphics for clean visualization
4. ✅ Provides flexible command-line interface
5. ✅ Maintains all original functionality
