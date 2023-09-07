# generic imports
import sys
import argparse
import numpy as np
from tqdm.auto import tqdm
from array import array

# ROOT imports
import ROOT

# torch imports
import torch

# framework imports
from kinematic_fitter import KinematicFitter

# Define the constraint equation(s) as a function
def constraint_equations(params: torch.Tensor) -> torch.Tensor:
    # Constant parameters
    m = torch.tensor(0.1349766).double()

    y = torch.zeros(1, dtype=torch.float64)
    y[0] = 2 *params[0] * params[1] * (1 - torch.cos(params[2])) - m*m

    return y

# Smearing value of the momentum
sigma = 0.004
# The covariance matrix initialization
n_params = 3
covariance_matrix = torch.zeros(n_params,n_params)
for j in range(n_params-1):
    covariance_matrix[j,j] = sigma**2

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, help='number of events', default=10000)
    args = parser.parse_args()
    
    # Initialization of the Phase Space generator
    event = ROOT.TGenPhaseSpace()
    pi0 = ROOT.TLorentzVector()
    # A pion travels in the z-direction with 500 MeV/c momentum 
    # and decays to 2 photons (pi0 -> g g)
    pi0.SetXYZM(0.0, 0.0, 0.500, 0.1349766)
    
    masses = array('d', [0.0, 0.0]) 
    event.SetDecay(pi0, 2, masses) # pi0 -> g g
    
    # Initialize random number generator for noise
    noise = ROOT.TRandom3()

    # Create histograms
    h_01 = ROOT.TH1F("hDiPhotonIMPreFit", ";M_{#gamma #gamma} [GeV/c^{2}]; counts [a.u.]", 100, 0.0, 0.5)
    h_02 = ROOT.TH1F("hEnergyResolutionPreFit", ";E_{smeared} - E_{true}", 100, -0.05, 0.05)
    h_03 = ROOT.TH1F("hChi2", "; #chi^{2}; counts [a.u.]", 100, 0.0, 20.0)
    h_04 = ROOT.TH1F("hProb", "; P(#chi^{2}); counts [a.u.]", 100, 0.0, 1.0)
    h_05 = ROOT.TH1F("hEnergyResolutionPostFit", ";E_{smeared} - E_{true}", 100, -0.05, 0.05)
    h_06 = ROOT.TH1F("hDiPhotonIMPostFit", ";M_{#gamma #gamma} [GeV/c^{2}]; counts [a.u.]", 100, 0.0, 0.5)   
    
    # Event loop
    for _ in tqdm(range(args.n)):
        # Generate an event
        event.Generate()
        # Get the the Di-photons
        p1 = event.GetDecay(0)
        p2 = event.GetDecay(1)
        # Define the "reconstructed" photons 4-momentum
        g1 = ROOT.TLorentzVector()
        g2 = ROOT.TLorentzVector()

        # Smear the photon's energy
        E_1 = p1.E() + noise.Gaus(0, sigma)
        E_2 = p2.E() + noise.Gaus(0, sigma)        
        g1.SetPxPyPzE(p1.Px(), p1.Py(), p1.Pz(), E_1)
        g2.SetPxPyPzE(p2.Px(), p2.Py(), p2.Pz(), E_2)

        # Di-Photon invariant mass
        h_01.Fill((g1 + g2).M())
        # Energy Resolution of the 1st photon before the kinematic fit
        h_02.Fill((g1.E() - p1.E()))
        
        # fit parameters: E_1, E_2, angle
        parameters = torch.tensor([E_1, E_2, g1.Angle(g2.Vect())], dtype=torch.float64)

        # Kinematic Fitter Initialization
        fitter = KinematicFitter(n_constraints=1, n_parameters=n_params, n_iterations=10)
        fitter.set_covariance_matrix(cov_matrix=covariance_matrix)
        ok = fitter.fit(measured_params=parameters, constraints=lambda parameters: constraint_equations(parameters))     
        # Fill histograms with fitting results
        h_03.Fill(fitter.getChi2())
        h_04.Fill(fitter.getProb())
        
         # Energy Resolution of the 1st photon after the kinematic fit
        fitted_params = fitter.get_fitted_measured_params()
        h_05.Fill(fitted_params[0] - p1.E())
        
        # Di-Photon invariant mass after the kinematic fit
        fitted_g1, fitted_g2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
        fitted_g1.SetPxPyPzE(p1.Px(), p1.Py(), p1.Pz(), fitted_params[0])
        fitted_g2.SetPxPyPzE(p2.Px(), p2.Py(), p2.Pz(), fitted_params[1])
        h_06.Fill((fitted_g1 + fitted_g2).M())
             

    # Save histograms to a ROOT file
    output_file = ROOT.TFile("plots.root", "RECREATE")
    h_01.Write()
    h_02.Write()
    h_03.Write()
    h_04.Write()
    h_05.Write()
    h_06.Write()
    output_file.Close()

if __name__ == "__main__":
    main(sys.argv[1:])