## Attempt at replicating model described in "A Synthetic Genetic Edge Detection Program" (DOI 10.1016/j.cell.2009.04.048)

### Code Structure
The program contains 3 classes Bacteria, Media, and Simulation.

#### Bacteria Class
Groups together data and functions relating to how the bacteria respond to light. Not meant to be instantiated, functions and data meant to be static if python had such functionality

* F_light : Models the ompC promoter, used to describe how AHL and CI production changes in response to light intensity
* F_logic : Models the lux-lambda promoter, describes how Bgal activity changes in response to AHL and CI concentrations.

Stores experimentally determined constants for the functions F_logic and F_light

#### Media Class
Stores Bgal activity, AHL concentration, and CI concentration at each point on the "agar dish" for each frame in time. Has a supporting function to get latest frame

#### Simulation Class
Sets up the simulation parameters and implements the function which will step the simulation forward in time. Implements a dedimensionalizing functions.

1. Define the resolution of spatial mesh of the agar plate in polar coordinates (angle,radius)
2. Define the resoultion of the time mesh for each step in time
3. Define experimental constants e.g Bgal maximum activity, rate of decay of AHL etc.
4. Generate time and spatial mesh
5. Instantiate Media class and set initial conditions (AHL conc. 0 across the plate)
6. Create light mask
7. Start Simulation

### Issues
1. Time step/resolution seems to be unclear in the paper, which mentions t* = 0.0027 = 24 hours but the calculation doesn't add up. Furthermore, ending the simulation at time t* = 0.0027 results in no change at all, although this may be caused by implementing F_logic incorrectly or the AHL update function. The resolution of the time step is important, because there seems to be different results depending on how fine the time step is.
2. If t* becomes too large overflow occurs
3. Values at steady state are not the same as in paper (maximum Bgal of around 35, this script has a maxium Bgal of around 60)

| Timestep t* = 0.000027 | Timestep t* = 0.00027 | Timestep t* = 0.0027 | Time Elapsed |
| ---------------------- | --------------------- | -------------------- | ------------ |
| ![](000027/200_000027.png) | ![](00027/20_00027.png) | ![](0027/2_000027.png) | t* = 0.0054 |

### Misc Information that might help
#### Dedimensionalisation of Radius and Time (dedimR and dedimT in code)
I think dedimensionalisation is a way of lumping together several variables to form a new variable that simplifies downstream equation manipulation.
1. Radius is dedimensionalised(r*) by the equation r* = r/R where R = plate radius (4.25cm) and r = current radius position on mesh
2. Time is dedimensionalised(t*) by the equation t* = tD/R^2 where D = diffusion coefficient of AHL (1.67 * 10^-7), R = plate radius (4.25cm), and t = time elapsed (I believe this is meant to be in seconds)
#### Finite Difference Methods
To make differential equations computable you need to cut the continuous domains (space and time) up into discrete chunks, hence the mesh. Once you have discrete chunks you can express dy/dx = (f(x+h)-f(x))/h where h is a chunk of your domain and approaches 0. This expression of dy/dx is a forward difference because it is taking a step ahead and comparing it against the current. There is also the backward difference dy/dx = (f(x)-f(x-h))/h and central difference dy/dx = (f(x+h)-f(x-h))/(2*h). The central difference method has the smallest error term, which is usually omitted from the expressions. There are also expressions for the second derivative onwards which can be derived from Taylor expansion and higher orders.

