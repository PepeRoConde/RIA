# The Engineer’s Guide to Soar

A 14-part course in programming the open-source software [Soar](https://github.com/SoarGroup/Soar), written by and for engineers.

This training material is itself an open-source project under the [SoarGroup](https://github.com/SoarGroup), functionally serving as an extension of the [Soar](https://github.com/SoarGroup/Soar) project.

Click [here](Course01_SoarEssentials) to get started!


## Course 01 - Soar Esentials Outline

0. Starting Soar
1. Hello World
2. Input Link
3. Operator Basics
4. Output
5. Multi-Apply Logic
6. Substates
7. Practice Debugging
8. Code Organization
9. Combining Preferences
10. SML Part 1: SML Basics
11. SML Part 2: Managing WMEs
12. SML Part 3: Custom Input Classes
13. SML Part 4: Event Handlers


## Open-Source Developer Notes

If updating the version of Soar included in this repo, be sure to test the Python run scripts on both Mac and Windows platforms. In particular, ensure that read and execute permissions are active for the Soar folder in the Mac platform.

Also note that the .bat/.sh scripts in the Soar distro folder are slightly modified (improved) from the form found in the official Soar release so that they can be run from Python code and also so that they do not remove files built for different platforms.
The Soar Cheat Sheet is also added to the Soar distro folder.

If you want to add additional lessons to teach more Soar topics, such as chunking, it is recommended to create a separate Course folder that is appropriate for the theme of that topic, if one doesn't already exist. (e.g., a lesson on SMEM should be under a course like Course02_SoarModules; a lesson on chunking or RL under Course03_SoarLearning, etc.)


## License

This project is released under the BSD 2-Clause License.
