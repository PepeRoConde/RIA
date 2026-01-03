# Wall-Following Strategy

## Overview

The wall-following strategy implements a **left-wall-following algorithm** where the agent navigates the maze by keeping a wall on its left side. The agent **never rotates without moving** - rotation and movement happen together.

## Behavior Rules

The agent follows these priorities in order:

### Priority 1: Move Straight
**Condition:** Front is clear AND left has a wall
**Action:** Move forward (no rotation)
**Reasoning:** Keep following the wall on the left

```
     wall
    ┌────┬────┐
    │    │ ↑  │  ← Agent moves straight
    └────┴────┘
```

### Priority 2: Turn Left and Move
**Condition:** Left is clear (no wall)
**Action:** Turn left + move forward
**Reasoning:** Always prefer to keep a wall on the left

```
    ┌────┬────┐
    │    │    │
    └────┴────┘
         ↑
         Agent turns left and moves
```

### Priority 3: Turn Right and Move
**Condition:** Front blocked AND left blocked BUT right is clear
**Action:** Turn right + move forward
**Reasoning:** Can't go straight or left, so turn right

```
    ┌────┬────┐
    │ ↑  │wall│
    ├────┼────┤
    │wall│    │  ← Agent turns right and moves
    └────┴────┘
```

### Priority 4: Turn Back (180°) and Move
**Condition:** All directions blocked (dead end)
**Action:** Turn 180° (turn right twice) + move backward
**Reasoning:** No other option, must backtrack

```
    ┌────┬────┐
    │wall│wall│
    ├────┼────┤
    │wall│ ↑  │  ← Agent turns 180° and moves
    └────┴────┘
```

## Key Design Principles

### No Rotation Without Movement
Unlike the DFS strategy, the agent **never rotates in place**. Every rotation is immediately followed by movement. This makes the behavior more realistic and efficient.

### Combined Commands
The Soar productions output both `turn` and `move` commands together:
```soar
(<ol> ^turn <turn-cmd>)
(<turn-cmd> ^direction left)
(<ol> ^move <move-cmd>)
(<move-cmd> ^direction forward)
```

### Command Processing Order
The agent interface processes commands in this order:
1. **All turn commands first** (in order)
2. **Then all move commands**

This ensures rotation happens before movement.

## Advantages

### Completeness
The left-wall-following algorithm guarantees finding the exit in any **simply-connected maze** (maze without loops or islands).

### Simplicity
The behavior is deterministic and easy to understand:
- Keep a wall on your left
- Always try to turn left when possible
- Only turn right when necessary

### Efficiency
No wasted rotations - the agent only turns when it needs to move in a different direction.

## Soar Implementation

### Input Link Structure
```
(I3)
  ^sensors S1
    ^front clear/wall
    ^right clear/wall
    ^back clear/wall
    ^left clear/wall
  ^at-target yes/no
  ^orientation north/east/south/west
  ^pos-x <x>
  ^pos-y <y>
  ^target-x <tx>
  ^target-y <ty>
```

### Output Link Commands

**Move straight:**
```
(O2)
  ^move M1
    ^direction forward
```

**Turn left and move:**
```
(O2)
  ^turn T1
    ^direction left
  ^move M1
    ^direction forward
```

**Turn right and move:**
```
(O2)
  ^turn T1
    ^direction right
  ^move M1
    ^direction forward
```

**Turn back (180°) and move:**
```
(O2)
  ^turn T1
    ^direction right
  ^turn T2
    ^direction right
  ^move M1
    ^direction forward
```

## Usage

### Command Line
```bash
# Use wall-follow strategy (default)
python run_with_turtle.py

# Explicitly specify wall-follow
python run_with_turtle.py --strategy wall-follow

# Custom maze size
python run_with_turtle.py --strategy wall-follow --width 15 --height 15

# Adjust speed
python run_with_turtle.py --strategy wall-follow --delay 0.2
```

### Programmatic
```python
from enviroment import Maze, MazeAgent, MazeTurtleUI

# Create maze
maze = Maze(width=10, height=10, wall_probability=0.3)

# Create agent with wall-follow strategy
agent = MazeAgent(maze, strategy="wall-follow")
agent.initialize_soar()

# Run simulation
while not agent.has_reached_target():
    agent.run_step()
```

## Comparison with Other Strategies

| Strategy | Rotation Behavior | Completeness | Path Optimality | Use Case |
|----------|------------------|--------------|-----------------|----------|
| **wall-follow** | With movement only | Yes (simple mazes) | Low | Simple mazes, realistic behavior |
| **dfs** | Independent of movement | Yes | Medium | Complex mazes, exploration |
| **naive** | No rotation | No | Low | Simple navigation |

## Limitations

### Not Optimal
The wall-following strategy does **not** find the shortest path. It explores the maze by following walls, which can lead to long, winding paths.

### Requires Simply-Connected Maze
May fail or loop forever in mazes with:
- **Islands** (disconnected wall sections in the middle)
- **Multiple solutions** with loops

### Doesn't Optimize for Target
The agent doesn't consider the target location - it just follows walls blindly until it reaches the target.

## Files

- **Production rules:** `agent/maze-navigator-wall-follow.soar`
- **Agent interface:** `enviroment/agent_interface.py`
- **Main runner:** `run_with_turtle.py`

## Example Output

```
============================================================
Soar Maze Navigator - WALL-FOLLOW Strategy
============================================================

Generating 10x10 maze (wall probability: 0.3)...
Initializing Soar agent with wall-follow strategy...
Soar agent initialized successfully!
Initializing Turtle graphics...

Starting navigation...

Moved right to (1, 0)
Turned left, now facing north
Moved up to (1, -1)
... (continues until target reached)

============================================================
TARGET REACHED!
============================================================
Steps taken: 45
Cells discovered: 23/100
Time elapsed: 4.50 seconds
============================================================
```

## Visualization

When running with turtle graphics, you'll see:
- **Blue triangle** (agent) pointing in the direction it's facing
- **Orange path** showing where the agent has traveled
- **Gray cells** showing discovered areas
- **Agent follows walls** in a predictable pattern
