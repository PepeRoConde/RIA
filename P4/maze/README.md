# Soar Maze Navigator

An intelligent agent built with Soar cognitive architecture that navigates through randomly generated mazes using sensory information and path memory.

## Overview

This project demonstrates the integration of Soar cognitive architecture with a Python environment. A Soar agent uses production rules to make navigation decisions based on:
- **Sensory input**: Wall detection in four directions (up, down, left, right)
- **Position awareness**: Current coordinates and distance to target
- **Memory**: Tracking of previously visited locations

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│              Python Environment                  │
│  ┌──────────────┐        ┌──────────────┐      │
│  │     Maze     │        │    Agent     │      │
│  │  Generator   │◄──────►│  Interface   │      │
│  └──────────────┘        └──────┬───────┘      │
│                                  │              │
└──────────────────────────────────┼──────────────┘
                                   │ SML Interface
┌──────────────────────────────────┼──────────────┐
│              Soar Agent          │              │
│  ┌───────────┐  ┌────────────┐  │              │
│  │Navigation │  │   Memory   │  │              │
│  │   Rules   │  │   Rules    │  │              │
│  └───────────┘  └────────────┘  │              │
└─────────────────────────────────────────────────┘
```

### Python Components

1. **Maze** (`environment/maze.py`)
   - Generates random mazes with configurable dimensions
   - Ensures at least one path exists from start to target
   - Provides wall detection sensors

2. **MazeAgent** (`environment/agent_interface.py`)
   - Manages agent state (position, discovered cells, path)
   - Interfaces with Soar through SML (Soar Markup Language)
   - Updates Soar's input-link with sensory data
   - Processes movement commands from Soar's output-link

### Soar Components

1. **maze-navigator.soar** - Main agent file
   - Initializes agent structures
   - Detects success condition

2. **navigation.soar** - Movement rules
   - Proposes movement in all clear directions
   - Prefers moves toward target (using Manhattan distance)
   - Executes movement commands

3. **memory.soar** - Memory management
   - Tracks visited locations
   - Monitors progress
   - Detects deadlock situations

## Installation

### Prerequisites

- Python 3.7+
- Soar 9.6.2+ (included in tutorial materials)

### Setup

1. Clone or download this project
2. Ensure Soar is installed in the expected location (adjust path in `agent_interface.py` if needed)
3. No additional Python packages required (uses only standard library)

## Usage

### Basic Usage

```bash
python run_simulation.py
```

This runs a 10x10 maze with 25% wall density.

### Command Line Options

```bash
python run_simulation.py --width 15 --height 15 --walls 0.3 --max-steps 500
```

Options:
- `--width INT`: Maze width (default: 10)
- `--height INT`: Maze height (default: 10)
- `--walls FLOAT`: Wall probability 0.0-1.0 (default: 0.25)
- `--max-steps INT`: Maximum steps before stopping (default: 200)
- `--no-visual`: Disable maze visualization

### Example Output

```
============================================================
SOAR MAZE NAVIGATOR SIMULATION
============================================================

Generating 10x10 maze...

Initial Maze:
=====================
|A                 |
|  █   █           |
|      █     █     |
|  █       █       |
|      █           |
|                  |
|  █           █   |
|          █       |
|      █       █   |
|                T |
=====================
Agent: (0, 0), Target: (9, 9)

Starting navigation...
------------------------------------------------------------
Attempting to move: right
Moved right to (1, 0)

Position: (1,0) Distance to target: 17 Explored cells: 2
...

============================================================
SUCCESS! Agent reached the target!
============================================================
```

## How It Works

### Input-Link Structure

The agent receives the following information on each decision cycle:

```
input-link
├── pos-x: current X coordinate
├── pos-y: current Y coordinate
├── target-x: target X coordinate
├── target-y: target Y coordinate
├── sensors
│   ├── up: "wall" or "clear"
│   ├── down: "wall" or "clear"
│   ├── left: "wall" or "clear"
│   └── right: "wall" or "clear"
├── distance-x: horizontal distance to target
├── distance-y: vertical distance to target
├── manhattan-distance: total distance to target
├── discovered-count: number of explored cells
└── at-target: "yes" or "no"
```

### Decision Making

The agent uses a simple but effective strategy:

1. **Proposal**: Propose moves in all directions that aren't blocked by walls
2. **Preference**: Prefer moves that reduce Manhattan distance to target
3. **Memory**: Track visited locations (though not yet used for avoidance)
4. **Execution**: Send movement command through output-link

### Soar Production Rule Example

```soar
sp {maze-navigator*propose*move*right
   "Propose moving right if not blocked"
   (state <s> ^name maze-navigator
              ^io.input-link <il>)
   (<il> ^sensors <sensors>
         ^at-target no)
   (<sensors> ^right clear)
-->
   (<s> ^operator <o> +)
   (<o> ^name move
        ^direction right)}
```

This rule:
- Tests if agent is in the maze-navigator state
- Checks that target hasn't been reached
- Verifies right direction is clear
- Proposes a move operator with direction "right"

## Extension Ideas

### Easy Extensions

1. **Backtracking**: Use visited location memory to avoid revisiting cells
2. **Better visualization**: Add color coding for visited cells
3. **Multiple targets**: Navigate to several targets in sequence

### Moderate Extensions

1. **A\* pathfinding**: Implement proper A\* algorithm in Soar
2. **Dynamic mazes**: Walls that can change during navigation
3. **Obstacles**: Moving obstacles to avoid

### Advanced Extensions

1. **Multi-agent coordination**: Multiple agents navigating the same maze
2. **Learning**: Use Soar's chunking to learn navigation strategies
3. **Partial observability**: Limited sensor range requiring exploration

## Project Structure Details

```
soar-maze-navigator/
├── README.md                    # This file
├── run_simulation.py            # Main execution script
├── environment/
│   ├── __init__.py
│   ├── maze.py                  # Maze generation and management
│   └── agent_interface.py       # Python-Soar interface
└── agent/
    ├── maze-navigator.soar      # Main Soar agent file
    ├── navigation.soar          # Navigation production rules
    └── memory.soar              # Memory management rules
```

## Understanding the Code

### Key Concepts

**Working Memory Elements (WMEs)**: Facts in Soar's working memory
```python
self.wmes['pos-x'] = self.input_link.CreateIntWME("pos-x", self.x)
```

**Production Rules**: If-then rules that match patterns and take actions
```soar
sp {rule-name
   (conditions)
-->
   (actions)}
```

**Operators**: Actions the agent can take
- Proposed based on current state
- Selected based on preferences
- Applied to change the world

**Input/Output Links**: Communication channels
- Input-link: Environment → Agent
- Output-link: Agent → Environment

## Troubleshooting

### "Module not found: Python_sml_ClientInterface"
- Check that Soar is installed
- Verify the path in `agent_interface.py` points to your Soar installation

### "Agent is stuck - no valid moves available"
- This means the agent reached a dead end
- Try reducing wall probability: `--walls 0.2`
- Or increase maze size: `--width 15 --height 15`

### Agent doesn't reach target in max steps
- Increase max steps: `--max-steps 500`
- The simple greedy strategy can get stuck in local minima
- Consider implementing backtracking (see extension ideas)

## Learning Resources

- [Soar Manual](https://soar.eecs.umich.edu/documentation)
- [Soar Tutorial](https://github.com/SoarGroup/Soar/wiki)
- Tutorial materials in this repository (Course01_SoarEssentials)

## License

BSD 2-Clause License (matching Soar license)

## Acknowledgments

Built using Soar cognitive architecture developed by the University of Michigan Soar Group.
