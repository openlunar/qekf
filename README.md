# qekf

Prototype q-method extended Kalman filter

* [Open Lunar Foundation](https://www.openlunar.org/)

## Description

This is a prototype q-method extended Kalman filter based on work by
Ainscough, Zanetti, Christian, and Spanos, that was intended for
eventual use on an earth-orbiting cubesat, and perhaps ultimately in a
lunar lander. It only includes two states: attitude and gyroscope bias.

### Note about validation/testing status

This repository includes a simulation that validates the qEKF
implementation in a flatsat on a rotating planet. The simulation for a
tumbling spacecraft in orbit is in branch `tumble_sim_wip` and needs a
few more hours of work to be completed. Unfortunately, that work won't
be by Open Lunar's engineering team. If you want to pick up the work
where we left off, start there.

## Requirements

* Python 3.x
* Numpy
* SciPy
* Matplotlib
* [pyquat](https://github.com/openlunar/pyquat)

## Installation

Clone this repository:

    git clone https://github.com/openlunar/qekf.git

Currently, this tool cannot be installed as a package. You must run it
out of the repository directory.

## Usage

Unit tests can be run using

    python3 setup.py test

The flatsat simulation is run using

    python3 sim_flatsat.py

## Developers

If you find a bug or wish to make a contribution, use the project's
[github issue tracker](https://github.com/openlunar/qekf/issues).

## License

Copyright (c) 2020, John O. "Juno" Woods and Open Lunar Foundation.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.