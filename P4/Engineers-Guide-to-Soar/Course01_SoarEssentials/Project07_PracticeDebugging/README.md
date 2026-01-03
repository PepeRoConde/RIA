# Project 7 - Practice Debugging

## New Concepts

* Copying down superstate WMEs for efficiency
* Nested condition conjunctions within rules
* Debugging with matches -w
* LHS conjunctions and disjunctions


## Problem

* We have a substate set up,
    * But our agent doesn’t yet use it to sort suppliers with complex reasoning.
    * We still sort only by total-score.
* We also don't have much experience using debugging commands to inspect our code when things go wrong.


## Solution

* Use substate operators to:
    * Iterate over attribute weights given on the input-link
        * In descending order
    * For each weight, sort suppliers by their attribute scores for that weight.
* Get practice using the `matches -w` command for rule debugging!



## Instructions: Lesson and Project

1. Open the lesson [here](Lesson07_CustomSorting.pdf).
1. Open the project [agent starter](./agent_starter.soar) code.
1. Edit the project code while following along with the lesson instructions.
1. When you've finished this project, click [here](../Project08_CodeOrganization/) to move on to the next!
