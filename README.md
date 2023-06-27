# ADICT-MSL
ADICT-MSL tries to forecast the future impact of Climate Tipping Points on GDP and other economic indicators

## Set-up instructions.

- Set-up a virtual environment:
    - Make sure you have a recent release of Python installed (we used Python 3.9), if not download
      from: https://www.python.org/downloads/
    - Download Anaconda: https://www.anaconda.com/products/individual
    - Set-up a virtual environment in Anaconda using Python 3.9.
    - Install the requirements by running the following command:
      ```pip install -r requirements.txt```
    - Copy all software from this repository into a file in the virtual environment.
    
## How to use?
- Adjust settings in generate_config.py, run it to save the configurations.
- Adjust the settings in the main.py file and run it to obtain the required results.
- environment.py includes the simulation environment of the climate tipping points.
- eco_monte_carlo_simulation.py incorporates the simulation of the economic variables.
- regression.py produces the regression of the climate damage function.

## References.

This code is written by Joris Knoester (2023)
