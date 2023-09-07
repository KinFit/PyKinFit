# Python Implementation of the KinFit library  

## Introduction

Kinematic fitting is a powerful tool used in the field of particle physics to improve the precision of measured quantities and to reduce background noise. It is particularly useful in the exclusive analysis of particle reactions. The algorithm aims to estimate the true values of track parameters as close as possible to the measured values while fulfilling a set of constraints given by the kinematics of the reaction under study.

### Mathematical Details

The core of the kinematic fitting algorithm is the minimization of the $\chi^{2}$ function, which is defined as:

$\chi^2 = (y - \eta)^T V^{-1} (y - \eta)$

Here, $y$ represents the vector of measured values, $\eta$ represents the vector of true values, and $V$ is the covariance matrix of the measured values. The minimization is performed iteratively.

#### Lagrange Multipliers

The algorithm employs Lagrange multipliers to handle constraints. The Lagrange function $L$ is formulated as:

$\chi^2 = (y - \eta)^T V^{-1} (y - \eta) + 2\lambda^T f(\eta, \xi)$

Here, $f(\eta, \xi)$ represents the constraint equations, and $\lambda$ are the Lagrange multipliers.

#### Jacobian Matrices

The Jacobian matrices $F_{\eta}$ and $F_{\xi}$ are crucial for the algorithm. They are partial derivatives of the constraint equations with respect to $\eta$ and $\xi$, respectively. These matrices are calculated using PyTorch's automatic differentiation engine.

$F_{\eta} = \frac{\partial f}{\partial \eta}, \quad F_{\xi} = \frac{\partial f}{\partial \xi}$

#### Iterative Minimization

The algorithm iteratively updates the parameters $\eta$ and $\xi$ to minimize $\chi^{2}$. In each iteration, new approximations for $\eta$ and $\xi$ are found, and the $\chi^{2}$ value is updated accordingly.

## Features

- **PyTorch Integration**: This implementation utilizes PyTorch's automatic differentiation tool to find the Jacobian matrices $F_{\eta}$ and $F_{\xi}$.
  
- **Generic Implementation**: The algorithm is designed to be generic. Users can provide "measured" parameters and "unmeasured" parameters (if needed) in any representation (track parameterization) along with the covariance matrix of the measured parameters and the constraint equation(s).

## How It Works

1. **Input Parameters**: The user provides the measured and unmeasured parameters, the covariance matrix of the measured parameters, and the constraint equations.

2. **Iterative Minimization**: The algorithm minimizes the $\chi^{2}$ function iteratively to find the best estimates of the true observables.

3. **Output**: The algorithm outputs an improved set of parameters that fulfill the kinematic constraints.

## Installation
To be able to use the kinematic fitting algorithm, you need to clone the repository and install some dependencies.   

### Using conda  
To install the dependencies using conda, you can create a new environment and install the packages listed in the `environment.yml` file:  
```shell script
# Create a new conda environment and install PyTorch
conda env create -f environment.yml
# Activate the enviroment
conda activate kinematic_fitter_env
```

## Usage

### 1. Application to $\pi^{0} \rightarrow \gamma \gamma$ Process

#### Overview

This tutorial demonstrates the application of kinematic fitting to $\pi^{0}$ decay events. In these events, a $\pi^{0}$ particle decays into two photons. The energies of the two photons are smeared by a constant value. The objective is to correct the smeared energy measurements of the photons using a kinematic constraint.

The tutorial employs ROOT's `TGenPhaseSpace` to generate $\pi^{0}$ particle traveling in the z-direction with 500 MeV/c momentum and decays into two photons. It uses the `KinematicFitter` class to perform the fitting.

#### Requirements

- ROOT
- PyTorch
- tqdm

#### How to Run

Execute the script using Python:

```shell script
python tutorial_1.py -n 10000
```

Where `-n` specifies the number of events.

#### Constraint Equations

The core of the fitting process relies on defining appropriate constraint equations. In the case of $\pi^{0}$ decaying into two photons, we use the invariant mass constraint equation to enforce that the invariant mass of the two photons equals the nominal mass of the $\pi^{0}$.

#### Mathematical Representation

The invariant mass $M_{\gamma \gamma}$ for two particles with energies $E_{1}$ and $E_{2}$ and momenta $\vec{p}_2$ and $\vec{p}_2$ is given by:

$M = \sqrt{(E_1 + E_2)^2 - ||\vec{p}_1 + \vec{p}_2||^2}$

In this tutorial, the constraint equation is defined as:

$M = 2 E_1 E_2  (1 - \cos(\theta)) - m^2 = 0$

Here, $E_{1}$ and $E_{2}$ are the energies of the two photons, $\theta$ is the angle between them, and $m$ is the nominal mass of the pi0 (0.1349766 GeV/c²). This constraint eqaution is implemented as a function in the code. This function must take a `torch` tensor as an argument, which represents the measured parameters (in this case te enrgies of the photons, in addition to the angle between the two photons) and return a `torch` tensor that represents the constraint eqaution(s). All the 'constant' parameters sould be included into this constraint function. In the mass of $\pi^0$ is used in the constraint equation but is not part of the parameters. Therefore, the mass is defined inside the function as a constant parameter.   


```python
def constraint_equations(params: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(0.1349766).double()
    y = torch.zeros(1, dtype=torch.float64)
    y[0] = 2 * params[0] * params[1] * (1 - torch.cos(params[2])) - m * m
    return y
```

#### Kinematic Fitting Procedure

1. **Data Generation**: Generate $\pi^0$ decay events using `TGenPhaseSpace`.
2. **Smearing**: Add Gaussian noise to the energy measurements to simulate a 'simpliefied' detector resolution.
3. **Measured Parameters**: Define the initial parameters for the fit using the smeared energies and the angle between the photon momenta.
4. **Constraint Fitting**: Utilize the `KinematicFitter` class to perform the constrained fit.  
    Before performing the kinematic fit, you need to specify the initial values of the parameters that the fit will adjust.
    
    These parameters are typically derived from the measurements you have. In this tutorial, the parameters we're interested in fitting are:
    $E_1$: The energy of the first photon, which is smeared.
    $E_2$: The energy of the second photon, also smeared.
    $\theta$: The angle between the two photons.
    These parameters are packed into a PyTorch tensor, like so:
    ```python
    parameters = torch.tensor([E_1, E_2, g1.Angle(g2.Vect())], dtype=torch.float64)
    ```
    After defining your initial fitting parameters, the next step is to perform the constrained fit using the `KinematicFitter`` class. Here's how you can initialize and use the fit method in this class:
    ```python
    fitter = KinematicFitter(n_constraints=1, n_parameters=n_params, n_iterations=10)
    ```
    Here, the `n_constraints` is te number of constraints, which is also the size of the `y` tensor defined in `constraint_equations` function. The `n_parameters` is the number of the measured parameters (in this case 3, 2 energies and 1 angle). Finally, `n_iterations` is obobviously the number of iterations.
    Before performing the fit, you also need to set the covariance matrix that represents the uncertainties in the measured parameters:
    ```python
    fitter.set_covariance_matrix(cov_matrix=covariance_matrix)
    ```
    Call the `fit` method on the `fitter` object. You pass in the initial parameters and the constraint equation function to this method:
    ```python
    ok = fitter.fit(measured_params=parameters, constraints=lambda parameters: constraint_equations(parameters))
    ```
    The `lambda` function acts like a wrapper around your actual `constraint_equations` function. It takes the parameters as an argument and simply forwards them to `constraint_equations`.
5. **Results and Plots**: Generate histograms for various observables before and after the kinematic fit.

#### Output Histograms

The tutorial produces the following histograms:

1. `hDiPhotonIMPreFit`: Di-Photon invariant mass before the kinematic fit.
2. `hEnergyResolutionPreFit`: Energy resolution of the 1st photon before the kinematic fit.
3. `hChi2`: Chi-squared values from the fit.
4. `hProb`: Probability values associated with the chi-squared values.
5. `hEnergyResolutionPostFit`: Energy resolution of the 1st photon after the kinematic fit.
6. `hDiPhotonIMPostFit`: Di-Photon invariant mass after the kinematic fit.

To view the histograms, open the generated `plots.root` file using ROOT.

### 1. Application to $p_{beam} + p_{target} \rightarrow p + K^+ + \Lambda (p \pi^-)$ Process

#### Overview
In this second tutorial, we delve into a more advanced use case of the `KinematicFitter`` class. We generate an event where a beam proton $p_beam$ and a target proton $p_target$ combine to produce a proton, a kaon, and a $\Lambda$ baryon. The $\Lambda$ particle then decays to a proton and a pion. We want to use kinematic fitting to improve the measurement of the proton and pion's track parameters under the constraint of invariant mass equal to the nominal $\Lambda$ mass.

#### Requirements

- ROOT
- PyTorch
- tqdm

#### How to Run

Execute the script using Python:

```shell script
python tutorial_2.py -n 10000
```

Where `-n` specifies the number of events.

The tutorial starts by importing required Python and ROOT libraries, then initializes key parameters such as smearing and the covariance matrix. The function `get_track_parameters` is responsible for simulating smeared track parameters for the decay products. Momentum $p$, polar angle $\theta$, and azimuthal angle $\phi$ are the measured track parameters.

The constraint is defined in the function `constraint_equations_1C`. The constraint equation equates the invariant mass of the proton and pion to the known mass of the Lambda baryon: 
```python
y[0] = E**2 - Px**2 - Py**2 - Pz**2 - mass_lambda**2
```
Please note that the constraint equation is written in Spherical coordinates:  
$p_x = p sin\theta cos\phi$    
$p_y = p sin\theta sin\phi$   
$p_z = p cos\theta$  
$E = \sqrt{p^2 + m^2}$  

The tutorial produces the following histograms:

1. `hIMPreFit`: the proton and pion invariant mass before the kinematic fit.
2. `hMomentumResolutionPreFit`: Momentum resolution of the proton before the kinematic fit.
3. `hChi2`: Chi-squared values from the fit.
4. `hProb`: Probability values associated with the chi-squared values.
5. `hMomentumResolutionPostFit`: Momentum resolution of the proton after the kinematic fit.
6. `hIMPostFit`: the proton and pion invariant mass after the kinematic fit.

## References

"[KinFit -- A Kinematic Fitting Package for Hadron Physics Experiments.](https://arxiv.org/pdf/2308.09575.pdf)" arXiv:2308.09575 [physics.data-an], August 2023.

## Citation  
```shell script
@article{Esmail:2023yjg,
    author = {Esmail, Waleed and Rieger, Jana and Taylor, Jenny and Bohman, Malin and Sch\"onning, Karin},
    title = "{KinFit -- A Kinematic Fitting Package for Hadron Physics Experiments}",
    eprint = "2308.09575",
    archivePrefix = "arXiv",
    primaryClass = "physics.data-an",
    month = "8",
    year = "2023"
}
```
