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
def constraint_equations_1C(params: torch.Tensor) -> torch.Tensor:
    # Constant parameters
    mass_lambda = torch.tensor(1.115).double()
    mass_proton = torch.tensor(0.938).double()
    mass_pion = torch.tensor(0.139).double()

    y = torch.zeros(1, dtype=torch.float64)

    E = torch.sqrt(params[0]**2 + mass_proton**2) + \
        torch.sqrt(params[3]**2 + mass_pion**2)
    
    Px = params[0] * torch.sin(params[1]) * torch.cos(params[2]) + \
         params[3] * torch.sin(params[4]) * torch.cos(params[5])
    
    Py = params[0] * torch.sin(params[1]) * torch.sin(params[2]) + \
         params[3] * torch.sin(params[4]) * torch.sin(params[5])
    
    Pz = params[0] * torch.cos(params[1]) + \
         params[3] * torch.cos(params[4])
    
    y[0] = E**2 - Px**2 - Py**2 - Pz**2 - mass_lambda**2

    return y

# Assume that: momentum, polar angle and azimuthal angle are the measured track parameters
def get_track_parameters(cand: ROOT.TLorentzVector(), std: list, noise: ROOT.TRandom3()) -> dict:
    p = cand.P() + noise.Gaus(0, std[0])
    polar = np.arccos(cand.Pz()/cand.P()) + noise.Gaus(0, std[1])
    azimuthal = np.arctan2(cand.Py(), cand.Px()) + noise.Gaus(0, std[2])
    
    return {"p": p, "theta": polar, "phi": azimuthal}

# Define the smearing values
smear = [0.02, 0.01, 0.01]
n_params = 6 # 3 for proton and 3 for pion
covariance_matrix = torch.zeros(n_params,n_params)
# Convert the list to a tensor so that mathematical operations can be applied element-wise
smear_tensor = torch.tensor(smear)
# Entries for protons
covariance_matrix[:n_params//2, :n_params//2] = torch.diag(smear_tensor**2)
# Entries for pions
covariance_matrix[n_params//2:, n_params//2:] = torch.diag(smear_tensor**2)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, help='number of events', default=10000)
    args = parser.parse_args()
    
    # Initialize TLorentzVector for beam and target protons
    p_beam = ROOT.TLorentzVector()
    p_target = ROOT.TLorentzVector()
    p_beam.SetXYZM(0, 0, 5, 0.938)  # 5 GeV in z-direction, mass of proton is 0.938 GeV
    p_target.SetXYZM(0, 0, 0, 0.938)  # At rest, mass of proton is 0.938 GeV
    # Combine beam and target to get the total 4-momentum
    ppsys = p_beam + p_target
    
    # Masses of final state particles (proton, kaon, Lambda) in GeV
    masses = array('d', [0.938, 0.493, 1.115])    
            
    # Initialization of the Phase Space generator
    event = ROOT.TGenPhaseSpace()
    event.SetDecay(ppsys, 3, masses) # p_beam + p_target -> p K+ (Lambda -> p pi-)
    
    # Initialize random number generator for noise
    noise = ROOT.TRandom3()

    # Create histograms
    h_01 = ROOT.TH1F("hIMPreFit", ";M_{p#pi^{-}} [GeV/c^{2}]; counts [a.u.]", 120, 1.070, 1.170)
    h_02 = ROOT.TH1F("hMomentumResolutionPreFit", ";E_{smeared} - E_{true}", 100, -0.1, 0.1)
    h_03 = ROOT.TH1F("hChi2", "; #chi^{2}; counts [a.u.]", 100, 0.0, 20.0)
    h_04 = ROOT.TH1F("hProb", "; P(#chi^{2}); counts [a.u.]", 100, 0.0, 1.0)
    h_05 = ROOT.TH1F("hMomentumResolutionPostFit", ";E_{smeared} - E_{true}", 100, -0.1, 0.1)
    h_06 = ROOT.TH1F("hIMPostFit", ";M_{p#pi^{-}} [GeV/c^{2}]; counts [a.u.]", 120, 1.070, 1.170)     
    
    # Event loop
    for _ in tqdm(range(args.n)):
        # Generate an event
        event.Generate()
        # Get the decay products
        pproton = event.GetDecay(0) # Primary proton
        kaon    = event.GetDecay(1)  
        Lambda  = event.GetDecay(2)
        
        # Now let's decay Lambda -> proton + pion
        decay_masses = array('d', [0.938, 0.139])  # Masses of proton and pion
        decay = ROOT.TGenPhaseSpace()
        decay.SetDecay(Lambda, 2, decay_masses)
        decay.Generate()
        
        # Get the decay products of Lambda
        sproton = decay.GetDecay(0) # Secondary proton
        pion    = decay.GetDecay(1)
        
        sproton_params = get_track_parameters(sproton, smear, noise)
        pion_params    = get_track_parameters(pion, smear, noise)
        
        # Assume these particles are measured in a "detector"
        measured_sproton, measured_pion = ROOT.TLorentzVector(), ROOT.TLorentzVector()
        
        measured_sproton.SetXYZM(sproton_params['p'] * np.sin(sproton_params['theta']) * np.cos(sproton_params['phi']),
                                 sproton_params['p'] * np.sin(sproton_params['theta']) * np.sin(sproton_params['phi']),
                                 sproton_params['p'] * np.cos(sproton_params['theta']), 0.938)
        
        measured_pion.SetXYZM(pion_params['p'] * np.sin(pion_params['theta']) * np.cos(pion_params['phi']),
                              pion_params['p'] * np.sin(pion_params['theta']) * np.sin(pion_params['phi']),
                              pion_params['p'] * np.cos(pion_params['theta']), 0.139)        
        
        # pion + proton invariant mass before the kinematic fit
        h_01.Fill((measured_sproton + measured_pion).M())
        # Momentum Resolution of the 1st photon before the kinematic fit
        h_02.Fill(sproton.P() - measured_sproton.P())
        
        # fit parameters: 3 for proton and 3 for pion
        parameters = torch.tensor([measured_sproton.P(), measured_sproton.Theta(), measured_sproton.Phi(),
                                   measured_pion.P(), measured_pion.Theta(), measured_pion.Phi()], dtype=torch.float64)

        # Kinematic Fitter Initialization
        fitter = KinematicFitter(n_constraints=1, n_parameters=n_params, n_iterations=10)
        fitter.set_covariance_matrix(cov_matrix=covariance_matrix)
        ok = fitter.fit(measured_params=parameters, constraints=lambda parameters: constraint_equations_1C(parameters))     
        # Fill histograms with fitting results
        h_03.Fill(fitter.getChi2())
        h_04.Fill(fitter.getProb())
        
        fitted_sproton, fitted_pion = ROOT.TLorentzVector(), ROOT.TLorentzVector()
        
        fitted_params = fitter.get_fitted_measured_params()

        fitted_sproton.SetXYZM(fitted_params[0] * np.sin(fitted_params[1]) * np.cos(fitted_params[2]),
                               fitted_params[0] * np.sin(fitted_params[1]) * np.sin(fitted_params[2]),
                               fitted_params[0] * np.cos(fitted_params[1]), 0.938)
        
        fitted_pion.SetXYZM(fitted_params[3] * np.sin(fitted_params[4]) * np.cos(fitted_params[5]),
                            fitted_params[3] * np.sin(fitted_params[4]) * np.sin(fitted_params[5]),
                            fitted_params[3] * np.cos(fitted_params[4]), 0.139)         
        
        # pion + proton invariant mass after the kinematic fit
        h_05.Fill(sproton.P() - fitted_params[0])
        # Momentum Resolution of the 1st photon after the kinematic fit
        h_06.Fill((fitted_sproton + fitted_pion).M())                     

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