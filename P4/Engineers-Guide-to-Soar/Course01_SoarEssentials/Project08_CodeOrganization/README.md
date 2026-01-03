# Project 8 - Code Organization

## New Concepts

* Organizing and documenting larger agents
* Augmenting operator structures
* Error catching
* The `cmd`` RHS Function
* Parallel summation
* Negated conjunctions


## Problem

* Our project code is getting fairly large!
* Our agent does not yet handle the case where two weights of the same value.


## Solution

* Adopt better practices for organizing and maintaining a large Soar project!
* When attributes share a weight, prefer by the subtotal of their scores.
    * Let a shared weight trigger an impasse and additional substate.
    * Calculate and return the subtotal from that substate so that regular processing can continue as normal.


## Instructions: Lesson and Project

1. Open the lesson [here](Lesson08_CodeOrganization.pdf).
1. Open the project [agent starter](./agent_starter.soar) code.
1. Edit the project code while following along with the lesson instructions.
1. When you've finished this project, click [here](../Project09_CombiningPreferences/) to move on to the next! (_coming soon_)
