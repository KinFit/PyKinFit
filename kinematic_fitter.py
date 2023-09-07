# Generic imports
import scipy
import numpy as np
from typing import Callable

# PyTorch imports
import torch
from torch.linalg import inv
from torch.autograd.functional import jacobian

class KinematicFitter:
    def __init__(self, n_constraints: int, n_parameters: int, n_iterations: int = 4):
        """
        Initialize a KinematicFitter object for performing kinematic fitting using Lagrange multipliers.

        Parameters:
        -----------
        n_constraints : int
            The number of constraints that the fitting algorithm must satisfy.
        
        n_parameters : int
            The number of parameters (measured variables) to be fitted.
            
        n_iterations : int, optional
            The maximum number of iterations for the fitting process. Default is 4.

        Attributes:
        -----------
        n_constraints : int
            Stores the number of constraints.
            
        num_parameters : int
            Stores the number of parameters.
            
        max_iterations : int
            Stores the maximum number of iterations.
            
        tolerance : float
            The tolerance for checking the convergence of the fitting process. Default is 1e-6.
            
        chi2 : float
            The chi-squared value for the fitting process. Initialized at 1e2.
            
        converged : bool
            A flag indicating whether the fitting process has converged. Initialized at False.
        """        
        self.n_constraints = n_constraints
        self.num_parameters = n_parameters
        self.max_iterations = n_iterations
        self.tolerance = 1e-6
        self.chi2 = 1e2
        self.converged = False

    def set_covariance_matrix(self, cov_matrix: torch.Tensor):
        """
        Sets the covariance matrix for the kinematic fit.

        Parameters
        ----------
        cov_matrix : torch.Tensor
            The covariance matrix to be used for the kinematic fit.
            The shape of the matrix must be (n_parameters, n_parameters).
        """        
        # Check if the shape of the given matrix matches (self.num_parameters, self.num_parameters)
        if cov_matrix.shape != (self.num_parameters, self.num_parameters):
            raise ValueError(f"The covariance matrix must have a shape of ({self.num_parameters}, {self.num_parameters})")

        # Convert the given numpy array to a PyTorch tensor
        self.cov_matrix_tensor = cov_matrix.double()                        

    def fit(self, measured_params: torch.Tensor, constraints: Callable[[torch.Tensor], torch.Tensor]) -> bool:
        """
        Perform the kinematic fit on the given measured parameters subject to the specified constraints.

        Parameters
        ----------
        measured_params : torch.Tensor
            The initial set of measured parameters. The tensor should be 1-dimensional and
            have a shape of (num_parameters,).
            
        constraints : Callable[[torch.Tensor], torch.Tensor]
            A callable function that takes in a tensor of shape (num_parameters,) and returns a tensor
            representing the constraint equations. The output tensor should have a shape of (n_constraints,).

        Returns
        -------
        bool
            Returns True if the algorithm converges, otherwise returns False.
        """        
        # Check if any argument is None
        if measured_params is None or constraints is None:
            raise ValueError("measured_params and constraints must be provided and not set to None.")

        # Check if the arguments are of the correct type
        if not isinstance(measured_params, torch.Tensor):
            raise TypeError(f"{type(measured_params)} must be {torch.Tensor}.")

        if not callable(constraints):
            raise TypeError(f"{type(constraints)} must be callable.")
        
        # Check if the shape of the given array matches (self.num_parameters,)
        if measured_params.shape != (self.num_parameters, ):
            raise ValueError(f"The measured parameters must have a dimension of ({self.num_parameters})")

        # Convert tensors to torch.float64 if they are not
        if measured_params.dtype != torch.float64:
            measured_params = measured_params.to(torch.float64)             

        params = measured_params.requires_grad_(True)

        y     = measured_params.view(-1, 1)
        eta   = y
        V     = self.cov_matrix_tensor
        f     = constraints(params).view(-1, 1)
        jacob = jacobian(constraints, params)
        F_eta = jacob.view(1, -1) if self.n_constraints == 1 else jacob

        for _ in range(self.max_iterations):             
            # Step 1: Define r and S
            r = f + F_eta @ (y - eta)
            S = F_eta @ V @ F_eta.T                                               

            # Step 2: Find the updated unmeasurable variables
            # ...

            # Step 3: Find the updated Lagrange multipliers
            lambda_nu = inv(S) @ r

            # Step 4: Find the updated measurable variables
            eta_nu = y - (V @ F_eta.T @ lambda_nu)

            # Step 5: Find the updated covariance matrix
            G = F_eta.T @ inv(S) @ F_eta
            V_nu = V - (V @ G @ V)

            params_nu = eta_nu.flatten().requires_grad_(True)
            f_nu = constraints(params_nu).view(-1, 1)
                    
            # Step 6: Calculate the updated chi^2
            chi2_nu = lambda_nu.T @ S @ lambda_nu + 2 * (lambda_nu.T @ f_nu)
            chi2_nu = chi2_nu.detach().item()
            
            f = f_nu
            jacob_nu = jacobian(constraints, params_nu)

            if np.abs(chi2_nu - self.chi2) < self.tolerance:
                self.converged = True
                break

            # Step 7: Update
            F_eta     = jacob_nu.view(1, -1) if self.n_constraints == 1 else jacob_nu     
            eta       = eta_nu
            self.chi2 = chi2_nu
            self.cov_matrix_tensor = V_nu            

        self.fitted_measured_params = eta

        return self.converged
    
    def fit_unmeasured(self, measured_params: torch.Tensor, unmeasured_params: torch.Tensor, constraints: Callable[[torch.Tensor], torch.Tensor]) -> bool:
        """
        Perform the kinematic fit on the given measured and unmeasured parameters subject to the specified constraints.

        Parameters
        ----------
        measured_params : torch.Tensor
            The initial set of measured parameters. The tensor should be 1-dimensional and
            have a shape of (num_parameters,).
            
        unmeasured_params : torch.Tensor
            The initial set of unmeasured parameters. The tensor should be 1-dimensional.

        constraints : Callable[[torch.Tensor], torch.Tensor]
            A callable function that takes in a tensor of concatenated measured and unmeasured parameters, 
            and returns a tensor representing the constraint equations. The output tensor should have a shape 
            of (n_constraints,).

        Returns
        -------
        bool
            Returns True if the algorithm converges, otherwise returns False.
        """        
        # Check if any argument is None
        if measured_params is None or unmeasured_params is None or constraints is None:
            raise ValueError("measured_params, unmeasured_params and constraints must be provided and not set to None.")

        # Check if the arguments are of the correct type
        if not isinstance(measured_params, torch.Tensor):
            raise TypeError(f"{type(measured_params)} must be {torch.Tensor}.")
        
        if not isinstance(unmeasured_params, torch.Tensor):
            raise TypeError(f"{type(measured_params)} must be {torch.Tensor}.")        

        if not callable(constraints):
            raise TypeError(f"{type(constraints)} must be callable.")
        
        # Check if the shape of the given array matches (self.num_parameters,)
        if measured_params.shape != (self.num_parameters, ):
            raise ValueError(f"The measured parameters must have a dimension of ({self.num_parameters})")

        # Convert tensors to torch.float64 if they are not
        if measured_params.dtype != torch.float64:
            measured_params = measured_params.to(torch.float64)

        if unmeasured_params.dtype != torch.float64:
            unmeasured_params = unmeasured_params.to(torch.float64)                          

        params = torch.hstack([measured_params, unmeasured_params]).requires_grad_(True)

        y     = measured_params.view(-1, 1)
        xi    = unmeasured_params.view(-1, 1)
        eta   = y
        V     = self.cov_matrix_tensor
        f     = constraints(params).view(-1, 1)
        jacob = jacobian(constraints, params)
        
        F_eta = jacob[:, 0:self.num_parameters]
        F_xi  = jacob[:, self.num_parameters:]                             

        for _ in range(self.max_iterations):             
            # Step 1: Define r and S
            r = f + F_eta @ (y - eta)
            S = F_eta @ V @ F_eta.T                                               

            # Step 2: Find the updated unmeasurable variables
            xi_nu = xi - inv(F_xi.T @ inv(S) @ F_xi) @ F_xi.T @ inv(S) @ r

            # Step 3: Find the updated Lagrange multipliers
            lambda_nu = inv(S) @ r + inv(S) @ F_xi @ (xi_nu - xi)

            # Step 4: Find the updated measurable variables
            eta_nu = y - (V @ F_eta.T @ lambda_nu)

            # Step 5: Find the updated covariance matrix
            G = F_eta.T @ inv(S) @ F_eta
            H = F_eta.T @ inv(S) @ F_xi
            U = F_xi.T @ inv(S) @ F_xi
            #V_nu = V - V @ G @ V + (V @ H @ inv(U) @ H.T @ V)
            V_nu = V - V @ (G - (H @ inv(U) @ H.T)) @ V           

            params_nu = torch.vstack([eta_nu, xi_nu]).flatten().requires_grad_(True)
            f_nu = constraints(params_nu).view(-1, 1)
                    
            # Step 6: Calculate the updated chi^2
            chi2_nu = lambda_nu.T @ S @ lambda_nu + 2 * (lambda_nu.T @ f_nu)
            chi2_nu = chi2_nu.detach().item()
            
            f = f_nu
            jacob_nu = jacobian(constraints, params_nu)

            if np.abs(chi2_nu - self.chi2) < self.tolerance:
                self.converged = True
                break
        
            # Step 7: Update
            F_eta     = jacob_nu[:, 0:self.num_parameters]
            F_xi      = jacob_nu[:, self.num_parameters:]     
            eta       = eta_nu
            xi        = xi_nu
            self.chi2 = chi2_nu
            self.cov_matrix_tensor = V_nu        

        self.fitted_measured_params = eta
        self.fitted_unmeasured_params = xi

        return self.converged    
    
    def getChi2(self):
        """
        Returns the chi-squared value of the kinematic fit.

        Returns
        -------
        float
            The chi-squared value.
        """     
        return self.chi2
    
    def getProb(self):
        """
        Returns the p-value associated with the chi-squared statistic and degrees of freedom.

        Returns
        -------
        float
            The p-value.
        """        
        return scipy.stats.chi2.sf(self.chi2, self.n_constraints)
    
    def get_fitted_measured_params(self):
        """
        Returns the updated measured parameters after the kinematic fit.

        Returns
        -------
        numpy.ndarray
            A 1-dimensional numpy array containing the updated measured parameters.
        """        
        return self.fitted_measured_params.detach().numpy().flatten()
    
    def get_fitted_unmeasured_params(self):
        """
        Returns the updated unmeasured parameters after the kinematic fit.

        Returns
        -------
        numpy.ndarray
            A 1-dimensional numpy array containing the updated unmeasured parameters.
        """         
        return self.fitted_unmeasured_params.detach().numpy().flatten()    
    
    def get_updated_covariance(self):
        """
        Returns the updated covariance matrix after the kinematic fit.

        Returns
        -------
        numpy.ndarray
            A 2-dimensional numpy array representing the updated covariance matrix.
        """        
        return self.cov_matrix_tensor.numpy() 

def template_constraint_equations(params: torch.Tensor) -> torch.Tensor:
    """
    Template function for defining constraint equations for kinematic fitting.

    Parameters
    ----------
    params : torch.Tensor
        A 1-dimensional tensor containing the parameters involved in the constraint equations.
        The shape of the tensor should be (num_parameters,).

    Returns
    -------
    torch.Tensor
        A tensor representing the constraint equation(s). The tensor can either be 1-dimensional
        or 2-dimensional depending on the number of constraints. Each row in a 2D tensor or each
        element in a 1D tensor should represent one constraint equation.

    Notes
    -----
    Users should override this function to define their own constraint equations. The equations
    should be defined using PyTorch operations to enable automatic differentiation.

    Example
    -------
    To represent the constraint equation "x^2 + y^2 - z = 0", where x, y, and z are elements
    of the parameter tensor `params`, one could implement this function as follows:

    ```python
    def my_constraints(params: torch.Tensor) -> torch.Tensor:
        x, y, z = params
        return x**2 + y**2 - z
    ```

    For multiple constraint equations, return them as a tensor:

    ```python
    def my_constraints(params: torch.Tensor) -> torch.Tensor:
        x, y, z = params
        constraint1 = x**2 + y**2 - z
        constraint2 = x + y + z - 10
        return torch.tensor([constraint1, constraint2])
    ```
    """
    pass