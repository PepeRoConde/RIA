# Project 6 - Substates

## New Concepts

* Hierarchical problem solving in Soar
* Substates
* Naming states
    * Using the "^superstate" WME
    * Using the "run --elaboration" command
* Solving impasses in substate processing
    * Accessing tied operators from a substate
    * Returning results from a substate
    * Ensuring result persistance


## Problem

* Our agent needs to sort suppliers with more complex reasoning than simply sorting by total-score.


## Solution

* Use a substate to evaluate tied suppliers.
    * A substate will allow us to use any number of decision cycles to derive the desired preferences.
    * Substate processing will effectively "pause" the top-level (select-supplier) decision-making until it finds the solution that lets the agent select one (select-supplier) operator over another.


## Instructions: Lesson and Project

1. Open the lesson [here](Lesson06_Substates.pdf).
1. Open the project [agent starter](./agent_starter.soar) code.
1. Edit the project code while following along with the lesson instructions.
1. When you've finished this project, click [here](../Project07_PracticeDebugging/) to move on to the next!
