# Project 5 - Multi Apply

## New Concepts

* Making a custom linked-list with WMEs
* Using an (init) operator
* Using the "require" preference
* Multiple apply rules for a single operator
* Using math RHS Functions


## Problem

* Our agent does not output suppliers with any particular order.
    * It selects suppliers in a particular order, but that order is not represented in the final output.


## Solution

* When the agent selects a supplier, append it to a linked-list data structure (which we'll define) in WM.
* After all suppliers have been added to the list, in order, then copy this list to the output-link.
* (We will not reuse code from previous projects this time, because we will need to make slight changes to the rules we wrote before.)


## Instructions: Lesson and Project

1. Open the lesson [here](Lesson05_MultiApply.pdf).
1. Open the project [agent starter](./agent_starter.soar) code.
1. Edit the project code while following along with the lesson instructions.
1. When you've finished this project, click [here](../Project06_Substates/) to move on to the next!
