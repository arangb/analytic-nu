import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import awkward as ak

def plot_variables(file_path):
    with uproot.open(file_path) as file:
        tree = file["Events"]
        branches = ["MET_pt", "MET_phi", "MET_CovXY"]
        data = tree.arrays(branches, library="np", cut="MET_pt > 50")

    # Define your specific x-axis labels
    x_labels = ["$p_{T}^{miss}$ [GeV]", r"$\phi(p_{T}^{miss})$ [rad]", "$p_{T}^{miss} Cov(x,y)$ [GeV$^{2}$]"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, var in enumerate(branches):
        axes[i].hist(data[var], bins=50, color='steelblue', edgecolor='black')
        axes[i].set_title(var)
        axes[i].set_xlabel(x_labels[i])  # Applying the specific labels here
        axes[i].set_ylabel("Entries")

    plt.tight_layout()
    plt.show()
    return

def print_variables(file_path="file.root", variables=["MET_pt","MET_phi"]):
    with uproot.open(file_path) as file:
        tree = file["Events"]
        data = tree.arrays(variables, cut="(MET_pt > 50) & (MET_phi > -1.5) & (MET_phi < 1.5)", library="np")
        print(data)
    return

def save_to_csv(file_path="file.root",
                variables=["MET_pt","MET_phi"],
                output_name="filtered_events.csv"):
    with uproot.open(file_path) as file:
        tree = file["Events"]
        # 1. Load as an Awkward Array first
        data = tree.arrays(variables, cut="(MET_pt > 20) & (nJet == 4) & (nMuon == 1)") # & (Jet_btagDeepB > 0.4506) & (Muon_pfRelIso04_all < 0.15)")

    # 2. Process each branch: if it's nested/jagged, take the first element
    processed_data = {}
    for branch in variables:
        # Check if the array is nested (jagged)
        if data[branch].ndim > 1:
            # ak.first returns the first element or None if the array is empty
            processed_data[branch] = ak.to_numpy(ak.firsts(data[branch], axis=1))
        else:
            processed_data[branch] = ak.to_numpy(data[branch])

    # 3. Convert to DataFrame and Save
    df = pd.DataFrame(processed_data)
    df.to_csv(output_name, index=False)
    print(f"Saved to {output_name}. First instances extracted using awkward.")


save_to_csv("ttbar_5k.root",["MET_pt","MET_phi","MET_covXX","MET_covYY","MET_covXY","Muon_pt","Muon_eta","Muon_phi","Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_btagDeepB"],"ttljets_nano_aod.csv")

# Original file with ttbar events:
# RunIISummer20UL17NanoAODv9-2_TTtoLNu2Q-1Jets-smeft_MTT-0to700_TuneCP5_13TeV_madgraphMLM-pythia8-2_NANOAODSIM106X_mc2017.root

#px=pt*np.cos(phi)
#py=pt*np.sin(phi)
#pz=pt*np.sinh(eta)
