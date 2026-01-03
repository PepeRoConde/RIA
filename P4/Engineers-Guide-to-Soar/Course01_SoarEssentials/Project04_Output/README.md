# Project 4 - Output

## New Concepts

* Using the ^output-link structure
* Adding unary preferences when proposing an operator
    * Acceptable
    * Worst
    * Indifferent
* Using the (interrupt) RHS function to pause the agent


## Problem

* Our agent selects suppliers in a particular order, but it does not send any of its results to output.


## Solution

* After the agent has evaluated all suppliers in order, copy the evaluated suppliers to the agent's output-link structure.
    * The output-link is provided automatically by Soar to support output to the environment.
    * Rather than add to the output-link incrementally, we copy final results to the output link all at once so that the environment receives everything in a single output message.
    * (We will address the problem of preserving supplier order in the output message in the next project.)


## Instructions: Lesson and Project

1. Open the lesson [here](Lesson04_Output.pdf).
1. Open the project [agent starter](./agent_starter.soar) code.
1. Edit the project code while following along with the lesson instructions.
1. When you've finished this project, click [here](../Project05_MultiApply) to move on to the next!
