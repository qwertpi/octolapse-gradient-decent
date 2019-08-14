# octolapse-gradient-decent
[Octolapse](https://github.com/FormerLurker/Octolapse) 0.4 parses gcode to perform a linear search to attempt to find an optimal set of snapshot points that have minimal distance from one another, this is a proof of concept for the use of gradient descent to approximate a solution to this optimization problem

## Disclaimer
THIS IS NOT CURRENTLY AFFILIATED OR ENDORSED BY OCTOPRINT BUT FORMERLURKER IS THANKED FOR HIS ASSISTANCE IN THE DEVELOPMENT OF THIS
ONLY LINUX IS SUPPORTED

## Copyright
Copyright © 2019  Rory Sharp All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If you have not received this, see <http://www.gnu.org/licenses/gpl-3.0.html>.

For a summary of the licence go to https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)

## Prerequisites
* [Python 3](https://www.python.org/downloads/)
* A gcode file
* jax `pip3 install jax jaxlib`
* Cython `pip3 install Cython`
* tqdm (for training only) (likely to already be installed) `pip3 install tqdm`

## Usage
-1\. Download this repo  
0\. Install the prerequiestes (`pip3 install --user -U -r requirements.txt`)
### Compiling gcode parser
1\. Run the command `python3 compile_gcode_parser.py  build_ext --inplace`
### Runing the program
2\. Run main.py, answer the prompts provided and look at the `end.gif` file

