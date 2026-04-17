"""
This script is a helper to the apply_flow_to_parquet.py file
where it contain auxiliary functions thta will be called during to apply the flows
"""

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd
import zuko
from typing import Any, Dict, List, Optional, Tuple
import xgboost
import awkward

import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])



#extra imports from 8.py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, RandomSampler
from torchdiffeq import odeint
import torch
import time
import sys
import os
import ot as pot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap





######################################################################################
#                                   Model Classes                                   #  
######################################################################################

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim, hidden_dim=64):
        super().__init__()
        self.condition_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # outputs gamma and beta
        )

    def forward(self, x, cond):
        """
        x: tensor of shape (B, feature_dim)
        cond: tensor of shape (B, cond_dim)
        """
        gamma_beta = self.condition_net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B, feature_dim) each
        return gamma * x + beta


class FiLMNet(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=128, depth=3, use_time=True, dropout_prob=0.0):
        super().__init__()
        self.use_time = use_time
        self.dropout = nn.Dropout(dropout_prob)
        self.input_layer = nn.Linear(input_dim + 1, hidden_dim)  # +1 for time
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, cond_dim) for _ in range(depth)
        ])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(hidden_dim, input_dim)  # adjust if output dim differs

    def forward(self, x, t, cond):
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # ensure shape (B, 1)
        t_repeated = t.expand(-1, 1)  # repeat if needed
        x = torch.cat([x, t_repeated], dim=-1)
        h = F.relu(self.input_layer(x))
        h = self.dropout(h)
        for lin, film in zip(self.hidden_layers, self.film_layers):
            h = F.relu(film(lin(h), cond))
            h = self.dropout(h)
        return self.output_layer(h)
    

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=64, depth=5, use_time=True, dropout_prob=0.0):
        super().__init__()
        self.use_time = use_time
        self.dropout = nn.Dropout(dropout_prob)
        total_input_dim = input_dim + cond_dim + (1 if use_time else 0)

        layers = [nn.Linear(total_input_dim, hidden_dim), Swish(), nn.Dropout(dropout_prob)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, cond):
        if self.use_time:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            x = torch.cat([x, t, cond], dim=-1)
        return self.net(x)
    



def lossplot_train(array_loss_train, array_loss_val, n_epochs, best_epoch, outdir, filename):
    """
    Function to plot the loss over epochs.
    :param array_loss: List of loss values
    :param n_epochs: Number of epochs
    """

    plt.plot(np.linspace(1, n_epochs, n_epochs), array_loss_train, label='Training loss')
    plt.plot(np.linspace(1, n_epochs, n_epochs), array_loss_val, label='Validation loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(outdir, filename))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()



def calculate_corrected_smeared_sigma_m_over_m(mc_df):

    #for key in mc_df.keys():
    #    print(key)

    # Try for tagandprobe framework, else diphoton nomenclature
    try:
        smeared_tag_energy_err   = np.sqrt( mc_df["tag_energyErr_corr"]**2 + (mc_df["tag_rho_smear"]*mc_df["tag_pt"]*np.cosh( mc_df[ "tag_eta" ] ) )**2 )
        smeared_probe_energy_err = np.sqrt( mc_df["probe_energyErr_corr"]**2 + (mc_df["probe_rho_smear"]*mc_df["probe_pt"]*np.cosh( mc_df[ "probe_eta" ] ) )**2 )
    
        corrected_smeared_sigma_m_over_m = (0.5)*np.sqrt( (smeared_tag_energy_err /(  mc_df["tag_pt"]*np.cosh( mc_df[ "tag_eta" ] )  ))**2 +   (smeared_probe_energy_err/(  mc_df["probe_pt"]*np.cosh( mc_df[ "probe_eta" ] )  ))**2   )

    except:
        smeared_tag_energy_err   = np.sqrt( mc_df["pho_lead_energyErr_corr"]**2 + (mc_df["pho_lead_rho_smear"]*mc_df["pho_lead_pt"]*np.cosh( mc_df[ "pho_lead_eta" ] ) )**2 )
        smeared_probe_energy_err = np.sqrt( mc_df["pho_sublead_energyErr_corr"]**2 + (mc_df["pho_sublead_rho_smear"]*mc_df["pho_sublead_pt"]*np.cosh( mc_df[ "pho_sublead_eta" ] ) )**2 )
    
        corrected_smeared_sigma_m_over_m = (0.5)*np.sqrt( (smeared_tag_energy_err /(  mc_df["tag_pt"]*np.cosh( mc_df[ "tag_eta" ] )  ))**2 +   (smeared_probe_energy_err/(  mc_df["probe_pt"]*np.cosh( mc_df[ "probe_eta" ] )  ))**2   )

    return corrected_smeared_sigma_m_over_m



# This class is responsible for the transformation of isolation variables
class Make_iso_continuous:
    def __init__(self, tensor, device, shift=None):

        self.device = device
        self.shift = shift if shift is not None else 0.05

        # Clone the tensor to avoid modifying the original data
        self.before_transform = tensor.copy()

        # Masks to identify zero and non-zero elements
        self.iso_bigger_zero = tensor > 0
        self.iso_equal_zero = tensor == 0

        # Store the number of zero elements
        self.n_zero_events = np.sum(self.iso_equal_zero)

        # Initialize lowest_iso_value (used in the inverse transformation)
        self.lowest_iso_value = 0.0

        # Placeholder for min and max values after log transform (used in inverse scaling)
        self.tensor_min = None
        self.tensor_max = None

    def shift_and_sample(self, tensor):
  
        # Ensure the tensor is on the correct device
        tensor = tensor
        # Masks to identify zero and non-zero elements
        bigger_than_zero = tensor > 0
        tensor_zero = tensor == 0

        # For non-zero elements, shift the values
        tensor[bigger_than_zero] = tensor[bigger_than_zero] + self.shift - self.lowest_iso_value

        # For zero elements, sample from a triangular distribution between 0 and self.shift * 0.98
        if tensor_zero.any():
            sampled_values = np.random.triangular(
                left=0.0,
                mode=0.0,
                right=self.shift * 0.98,
                size=np.sum(tensor_zero)
            )
            tensor[tensor_zero] = sampled_values

        # Apply a small epsilon to avoid log(0)
        epsilon = 1e-3

        # Apply log transform to stretch the distribution
        tensor = np.log(epsilon + tensor)

        return tensor

    def inverse_shift_and_sample(self, tensor, processed=False):

        # Ensure the tensor is on the correct device
        tensor = tensor

        # Reverse the log transform
        tensor = np.exp(tensor) - 1e-3

        # Masks to identify values after inverse log transform
        bigger_than_shift = tensor >= self.shift
        lower_than_shift = tensor < self.shift

        # For values less than the shift, set them to zero
        tensor[lower_than_shift] = 0.0

        # For values greater than or equal to the shift, reverse the shift operation
        tensor[bigger_than_shift] = tensor[bigger_than_shift] - self.shift + self.lowest_iso_value

        # Ensure the inverse transformation recovers the original tensor
        if not processed:
            difference = np.abs(self.before_transform - tensor)
            max_difference = difference.max()
            tolerance = 1e-5  # Adjust tolerance as needed
            assert max_difference < tolerance, f"Max difference {max_difference} exceeds tolerance {tolerance}"

        return tensor
    


def isolation_indices(variables_cms_names, isolation_variables):
    """
    Returns the indices of the isolation variables in the list of variable names.
    """

    indices = []
    shift = []
    for index, name in enumerate(variables_cms_names):
        if name in isolation_variables:
            indices.append(index)
            if name == "photon_hoe":
                shift.append(0.002)
            else:
                shift.append(0.05)
        else:
            continue
    
    return indices, shift



def undo_apply_scale_and_smooth(samples, vector_for_iso_constructors_mc, variables_cms_names, isolation_variables):

    indexes_for_iso_transform, Iso_transform_shift = isolation_indices(variables_cms_names, isolation_variables)
    
    #indexes_for_iso_transform = [7,8,9,10,11,12,13,14,15]
    #Iso_transform_shift = [ 0.002, 0.05 , 0.05 , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    
    # Now inverting the isolation transformation
    if len(indexes_for_iso_transform) > 0:
        
        counter = 0
        for index in indexes_for_iso_transform:
                
                # now transforming the 
                samples[:,index] = vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(samples[:,index], processed = True)

                counter = counter + 1

    return samples


def apply_scale_and_smooth(mc_training_inputs, data_training_inputs, variables_cms_names, isolation_variables):

    indexes_for_iso_transform, Iso_transform_shift = isolation_indices(variables_cms_names, isolation_variables)

    device = torch.device("cpu")
    
    #indexes_for_iso_transform = [7,8,9,10,11,12,13,14,15]
    #Iso_transform_shift = [ 0.002, 0.05 , 0.05 , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    vector_for_iso_constructors_mc   = []
    vector_for_iso_constructors_data = []

    # Performing the transformations of the non-continious variables
    if len(indexes_for_iso_transform) > 0:
            
            # Creating the constructors
            for index, shift_values in zip(indexes_for_iso_transform, Iso_transform_shift):
                
                vector_for_iso_constructors_data.append( Make_iso_continuous(data_training_inputs[:,index],device,  shift = shift_values) )
                vector_for_iso_constructors_mc.append( Make_iso_continuous(mc_training_inputs[:,index],device ,  shift= shift_values) )


            # Now really applying the transformations
            counter = 0
            for index in indexes_for_iso_transform:
                
                # transforming the training dataset 
                data_training_inputs[:,index] = vector_for_iso_constructors_data[counter].shift_and_sample(data_training_inputs[:,index])
                mc_training_inputs[:,index] = vector_for_iso_constructors_mc[counter].shift_and_sample(mc_training_inputs[:,index])

                counter = counter + 1

    return mc_training_inputs, data_training_inputs, vector_for_iso_constructors_data, vector_for_iso_constructors_mc



def perform_pre_processing( input_tensor: torch.tensor, conditions_tensor: torch.tensor , path = False ):
    
    # Fist we make the isolation variables transformations
    indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14] #[7,8,9,10,11,12,13,14,15] #[6,7,8,9,10,11,12,13,14] #this has to be changed once I take the energy raw out of the inputs
    vector_for_iso_constructors_mc   = []

    # creating the constructors
    for index in indexes_for_iso_transform:
            
        # since hoe has very low values, the shift value (value until traingular events are sampled) must be diferent here
        if( index == 6 ):
            vector_for_iso_constructors_mc.append( Make_iso_continuous(input_tensor[:,index] ,device = torch.device('cpu') , b= 0.001) )
        else:
            vector_for_iso_constructors_mc.append( Make_iso_continuous(input_tensor[:,index] , device = torch.device('cpu')) )

    # now really applying the transformations
    counter = 0
    for index in indexes_for_iso_transform:
            
        # transforming the training dataset 
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].shift_and_sample(input_tensor[:,index])
        counter = counter + 1

    # Now, the standartization -> The arrays are the same ones using during training
    input_mean_for_std      = torch.tensor(np.load( path +  'input_means.npy' ))
    input_std_for_std       = torch.tensor(np.load( path +  'input_std.npy'))
    condition_mean_for_std  = torch.tensor(np.load( path +  'conditions_means.npy'))
    condition_std_for_std   = torch.tensor(np.load( path +  'conditions_std.npy'))

    # Standardizing!
    input_tensor = ( input_tensor - input_mean_for_std  )/input_std_for_std
    conditions_tensor[:,:-1] = ( conditions_tensor[:,:-1] - condition_mean_for_std )/condition_std_for_std

    return input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc

# revert the corrected samples tranformation
def invert_pre_processing( input_tensor: torch.tensor, input_mean_for_std: torch.tensor, input_std_for_std: torch.tensor, vector_for_iso_constructors_mc) -> torch.tensor:
    
    indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14]

    # inverting the standartization
    input_tensor = ( input_tensor*input_std_for_std + input_mean_for_std )

    # Now inverting the isolation transformation
    counter = 0
    for index in indexes_for_iso_transform:
            
        # now transforming the 
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(input_tensor[:,index], processed = True)

        counter = counter + 1
    
    return input_tensor

def apply_flow( input_tensor: torch.tensor, conditions_tensor: torch.tensor, flow )-> torch.tensor:
    
    """
    This function is responsable for applying the normalizing flow to MC samples
    it takes as input
    """

    # making sure flow and input tensors have the same type
    flow = flow.type(   input_tensor.dtype )
    conditions_tensor = conditions_tensor.type( input_tensor.dtype  )

    # Use cuda if avaliable - maybe this is causing the meory problems?
    device = torch.device('cpu')
    flow = flow.to(device)
    input_tensor = input_tensor.to(device)
    conditions_tensor = conditions_tensor.to(device)

    # Disabling pytorch gradient calculation so operation uses less memory and is faster
    with torch.no_grad():

        # Now the flow transformation is done!
        trans      = flow( conditions_tensor ).transform
        sim_latent = trans( input_tensor )

        conditions_tensor = torch.tensor(np.concatenate( [ conditions_tensor[:,:-1].cpu() , np.ones_like( conditions_tensor[:,0].cpu() ).reshape(-1,1) ], axis =1 )).to(device)

        trans2 = flow(conditions_tensor).transform
        samples = trans2.inv( sim_latent)    
    
    return samples

def load_photonid_mva_run3(fname: str) -> Optional[xgboost.Booster]:

    """ Reads and returns both the EB and EE Xgboost run3 mvaID models """

    photonid_mva_EB = xgboost.Booster()
    photonid_mva_EB.load_model(fname + 'model.json')

    photonid_mva_EE = xgboost.Booster()
    photonid_mva_EE.load_model(fname + 'model_endcap.json')
    
    return photonid_mva_EB, photonid_mva_EE


def calculate_photonid_mva_run3(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: awkward.Array,
) -> awkward.Array:
   
    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll

    EE: + 
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_hcalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll    
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """
    photonid_mva, var_order = mva

    if photonid_mva is None:
        return awkward.ones_like(photon.pt)


    bdt_inputs = {}
    
    bdt_inputs = np.column_stack(
        [np.array(photon[name]) for name in var_order]
    )

    tempmatrix = xgboost.DMatrix(bdt_inputs)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    mvaID = 1.0 - 2.0 / (1.0 + np.exp(2.0 * mvaID))

    return mvaID


def add_corr_photonid_mva_run3( photons: awkward.Array, process) -> awkward.Array:

        preliminary_path = '/net/scratch_cms3a/daumann/HiggsDNA/higgs_dna/tools/'
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        # Now mvaID for the corrected variables

        inputs_EB = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "fixedGridRhoAll"]

        inputs_EE = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "hcalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "fixedGridRhoAll",
            "esEffSigmaRR_corr",
            "esEnergyOverRawE_corr"]

        photon_types_Zee = ["tag","probe"]
        corrected_mva_id = []

        for photon_type in photon_types_Zee:

            # Now calculating the corrected mvaID
            isEB = awkward.to_numpy(np.abs( np.array( photons[ photon_type + "_ScEta"])) < 1.442)

            inputs_EB_corr = [photon_type + "_" + s if "fixedGridRhoAll" not in s else s for s in inputs_EB]

            corr_mva_EB = calculate_photonid_mva_run3(
                [photonid_mva_EB,inputs_EB_corr], photons
            )
            
            inputs_EE_corr = [photon_type + "_" + s if "fixedGridRhoAll" not in s else s for s in inputs_EE]
            
            corr_mva_EE = calculate_photonid_mva_run3(
                [photonid_mva_EE, inputs_EE_corr], photons
            )
            corrected_mva_id.append( awkward.where(isEB, corr_mva_EB, corr_mva_EE) )

        return corrected_mva_id[0], corrected_mva_id[1]


def add_corr_photonid_mva_run3_zmmg( photons: awkward.Array, process) -> awkward.Array:

        preliminary_path = '/home/home1/institut_3a/tsommer/bsc-sommer-project/caios_dnf/0af6b4cee51715f0c187041f36ce7777/'
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        # Now mvaID for the corrected variables

        inputs_EB = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "Rho_fixedGridRhoAll"]

        inputs_EE = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "hcalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "Rho_fixedGridRhoAll",
            "esEffSigmaRR_corr",
            "esEnergyOverRawE_corr"]

        photon_types_Zee = ["photon"]

        for photon_type in photon_types_Zee:

            # Now calculating the corrected mvaID
            isEB = awkward.to_numpy(np.abs( np.array( photons[ photon_type + "_ScEta"])) < 1.442)

            inputs_EB_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EB]

            corr_mva_EB = calculate_photonid_mva_run3(
                [photonid_mva_EB,inputs_EB_corr], photons
            )
            
            inputs_EE_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EE]
            
            corr_mva_EE = calculate_photonid_mva_run3(
                [photonid_mva_EE, inputs_EE_corr], photons
            )
            corrected_mva_id = awkward.where(isEB, corr_mva_EB, corr_mva_EE) 

        return corrected_mva_id


def add_mc_photonid_mva_run3_zmmg( photons: awkward.Array, process) -> awkward.Array:

        preliminary_path = '/home/home1/institut_3a/tsommer/bsc-sommer-project/caios_dnf/0af6b4cee51715f0c187041f36ce7777/'
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        # Now mvaID for the corrected variables

        inputs_EB = ["energyRaw",
            "raw_r9", 
            "raw_sieie",
            "raw_etaWidth",
            "raw_phiWidth",
            "raw_sieip",
            "raw_s4",
            "raw_hoe",
            "raw_ecalPFClusterIso",
            "raw_trkSumPtHollowConeDR03",
            "raw_trkSumPtSolidConeDR04",
            "raw_pfChargedIso",
            "raw_pfChargedIsoWorstVtx",
            "ScEta",
            "Rho_fixedGridRhoAll"]

        inputs_EE = ["energyRaw",
            "raw_r9", 
            "raw_sieie",
            "raw_etaWidth",
            "raw_phiWidth",
            "raw_sieip",
            "raw_s4",
            "raw_hoe",
            "raw_ecalPFClusterIso",
            "raw_hcalPFClusterIso",
            "raw_trkSumPtHollowConeDR03",
            "raw_trkSumPtSolidConeDR04",
            "raw_pfChargedIso",
            "raw_pfChargedIsoWorstVtx",
            "ScEta",
            "Rho_fixedGridRhoAll",
            "raw_esEffSigmaRR",
            "raw_esEnergyOverRawE"]

        photon_types_Zee = ["photon"]

        for photon_type in photon_types_Zee:

            # Now calculating the corrected mvaID
            isEB = awkward.to_numpy(np.abs( np.array( photons[ photon_type + "_ScEta"])) < 1.442)

            inputs_EB_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EB]

            corr_mva_EB = calculate_photonid_mva_run3(
                [photonid_mva_EB,inputs_EB_corr], photons
            )
            
            inputs_EE_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EE]
            
            corr_mva_EE = calculate_photonid_mva_run3(
                [photonid_mva_EE, inputs_EE_corr], photons
            )
            corrected_mva_id = awkward.where(isEB, corr_mva_EB, corr_mva_EE) 

        return corrected_mva_id

def add_data_photonid_mva_run3_zmmg( photons: awkward.Array, process) -> awkward.Array:

        preliminary_path = '/home/home1/institut_3a/tsommer/bsc-sommer-project/caios_dnf/0af6b4cee51715f0c187041f36ce7777/'
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        # Now mvaID for the corrected variables

        inputs_EB = ["energyRaw",
            "r9", 
            "sieie",
            "etaWidth",
            "phiWidth",
            "sieip",
            "s4",
            "hoe",
            "ecalPFClusterIso",
            "trkSumPtHollowConeDR03",
            "trkSumPtSolidConeDR04",
            "pfChargedIso",
            "pfChargedIsoWorstVtx",
            "ScEta",
            "Rho_fixedGridRhoAll"]

        inputs_EE = ["energyRaw",
            "r9", 
            "sieie",
            "etaWidth",
            "phiWidth",
            "sieip",
            "s4",
            "hoe",
            "ecalPFClusterIso",
            "hcalPFClusterIso",
            "trkSumPtHollowConeDR03",
            "trkSumPtSolidConeDR04",
            "pfChargedIso",
            "pfChargedIsoWorstVtx",
            "ScEta",
            "Rho_fixedGridRhoAll",
            "esEffSigmaRR",
            "esEnergyOverRawE"]

        photon_types_Zee = ["photon"]

        for photon_type in photon_types_Zee:

            # Now calculating the corrected mvaID
            isEB = awkward.to_numpy(np.abs( np.array( photons[ photon_type + "_ScEta"])) < 1.442)

            inputs_EB_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EB]

            corr_mva_EB = calculate_photonid_mva_run3(
                [photonid_mva_EB,inputs_EB_corr], photons
            )
            
            inputs_EE_corr = [photon_type + "_" + s if "Rho_fixedGridRhoAll" not in s else s for s in inputs_EE]
            
            corr_mva_EE = calculate_photonid_mva_run3(
                [photonid_mva_EE, inputs_EE_corr], photons
            )
            corrected_mva_id = awkward.where(isEB, corr_mva_EB, corr_mva_EE) 

        return corrected_mva_id












###################################################################################################################################################################
#                                                                    Correlation Matrix Functions                                                                 #
###################################################################################################################################################################

def weighted_corr_matrix(df, var_list, w_col):
    """
    Compute the weighted correlation matrix for columns in var_list
    using weights in column w_col of df.
    """
    w = df[w_col].values
    n = len(var_list)
    
    # Initialize empty correlation matrix
    corr_mat = np.zeros((n, n))
    
    for i in range(n):
        x = np.nan_to_num(df[var_list[i]].values)
        for j in range(n):
            y = np.nan_to_num(df[var_list[j]].values)
            corr_mat[i, j] = weighted_corr(x, y, w)
    
    # Return as a Pandas DataFrame
    return pd.DataFrame(corr_mat, columns=var_list, index=var_list)

def weighted_corr(x, y, w):
    """
    Compute the weighted correlation between arrays x and y with weights w.
    """
    w_sum   = np.sum(w)
    x_bar   = np.sum(w * x) / w_sum
    y_bar   = np.sum(w * y) / w_sum
    
    # Weighted covariance
    cov_xy  = np.sum(w * (x - x_bar) * (y - y_bar)) / w_sum
    
    # Weighted variances
    var_x   = np.sum(w * (x - x_bar)**2) / w_sum
    var_y   = np.sum(w * (y - y_bar)**2) / w_sum
    
    #print( "var_x, var_y: ",  var_x, var_y )
    #print( "var_x * var_y: ",   var_x * var_y )
    
    # Weighted correlation
    corr_xy = cov_xy / np.sqrt(abs(var_x * var_y))
    #print( "corr_xy: ", corr_xy )
    return corr_xy

def weighted_covariance(x, w, unbiased=False):
    """
    x: tensor of shape (n_samples, n_features)
    w: tensor of shape (n_samples,) — may contain negatives
    unbiased: if True, applies the usual correction factor

    returns: (n_features x n_features) covariance matrix
    """
    # make sure w is float
    w = w.to(x.dtype)

    # 1) total weight
    w_sum = w.sum()
    if w_sum == 0:
        raise ValueError("Sum of weights must be non-zero")

    # 2) weighted mean
    mean = (w[:, None] * x).sum(dim=0) / w_sum

    # 3) center data
    xm = x - mean

    # 4) weighted outer products summed
    #    result is (n_features, n_features)
    cov = (w[:, None, None] * xm[:, :, None] * xm[:, None, :]).sum(dim=0) / w_sum

    if unbiased:
        # (optional) Bessel‐type correction for weighted samples
        # effective degrees of freedom:
        dof = w_sum - (w * w).sum() / w_sum
        cov = cov * (w_sum / dof)

    return cov

def plot_correlation_matrices(data_df, mc_df, var_list_barrel, var_list_endcap, region, path):
      
    # lets comments this for now
    mc_df["photon_trkSumPtSolidConeDR04"]     = np.log(mc_df["photon_trkSumPtSolidConeDR04"] + 1e-3) #- mc_df["photon_raw_trkSumPtHollowConeDR03"]
    data_df["photon_trkSumPtSolidConeDR04"]   = np.log(data_df["photon_trkSumPtSolidConeDR04"] + 1e-3)   #- data_df["photon_trkSumPtHollowConeDR03"]
    mc_df["photon_raw_trkSumPtSolidConeDR04"] = np.log(mc_df["photon_raw_trkSumPtSolidConeDR04"] + 1e-3) # - mc_df["photon_raw_trkSumPtHollowConeDR03"]

    data_df["photon_trkSumPtHollowConeDR03"] = np.log(data_df["photon_trkSumPtHollowConeDR03"] + 1e-3)
    mc_df["photon_raw_trkSumPtHollowConeDR03"] = np.log(mc_df["photon_raw_trkSumPtHollowConeDR03"] + 1e-3)
    mc_df["photon_trkSumPtHollowConeDR03"] = np.log(mc_df["photon_trkSumPtHollowConeDR03"] + 1e-3)

    data_df["photon_pfChargedIso"] = np.log(data_df["photon_pfChargedIso"] + 1e-3)
    mc_df["photon_raw_pfChargedIso"] = np.log(mc_df["photon_raw_pfChargedIso"] + 1e-3)
    mc_df["photon_pfChargedIso"] = np.log(mc_df["photon_pfChargedIso"] + 1e-3)
    
    data_df["photon_pfChargedIsoWorstVtx"] = np.log(data_df["photon_pfChargedIsoWorstVtx"] + 1e-3)
    mc_df["photon_raw_pfChargedIsoWorstVtx"] = np.log(mc_df["photon_raw_pfChargedIsoWorstVtx"] + 1e-3)
    mc_df["photon_pfChargedIsoWorstVtx"] = np.log(mc_df["photon_pfChargedIsoWorstVtx"] + 1e-3)
    
    data_df["photon_esEffSigmaRR"] = np.log(data_df["photon_esEffSigmaRR"] + 1e-3)
    mc_df["photon_raw_esEffSigmaRR"] = np.log(mc_df["photon_raw_esEffSigmaRR"] + 1e-3)
    mc_df["photon_esEffSigmaRR"] = np.log(mc_df["photon_esEffSigmaRR"] + 1e-3)
    
    data_df["photon_esEnergyOverRawE"] = np.log(data_df["photon_esEnergyOverRawE"] + 1e-3)
    mc_df["photon_raw_esEnergyOverRawE"] = np.log(mc_df["photon_raw_esEnergyOverRawE"] + 1e-3)
    mc_df["photon_esEnergyOverRawE"] = np.log(mc_df["photon_esEnergyOverRawE"] + 1e-3)
    
      
    eta_regions    = ['barrel', 'endcap']
    var_lists      = [var_list_barrel, var_list_endcap]
    eta_mc_masks   = [np.abs(mc_df["photon_ScEta"]) < 1.442,  np.abs(mc_df["photon_ScEta"]) > 1.566]
    eta_data_masks = [np.abs(data_df["photon_ScEta"]) < 1.442, np.abs(data_df["photon_ScEta"]) > 1.566]

    os.makedirs(path, exist_ok=True)

    for eta_region, var_list, mc_eta_mask, data_eta_mask in zip(eta_regions, var_lists, eta_mc_masks, eta_data_masks):
        
        if region != "all" and region != eta_region:
            continue

        corr_var_list = var_list
        
        #corr_var_list = [f"{var}_corr" for var in var_list]
        var_list_mc = [ var.replace("photon_", "photon_raw_") if var not in [  "photon_pt",
                     "photon_ScEta",
                     "photon_phi",
                     "Rho_fixedGridRhoAll", 
                     "photon_muon_near_dR"] else var for var in var_list ]
        
        #calculating the covariance matrix of the pytorch tensors
        #data_cov         = torch.cov( torch.tensor(data_df[var_list].loc[data_eta_mask].values).T  , aweights = torch.tensor( torch.tensor(np.abs(data_df["weights"].loc[data_eta_mask].values)) ) )  
        #mc_cov           = torch.cov( torch.tensor(mc_df[var_list_mc].loc[mc_eta_mask].values).T   , aweights = torch.tensor( torch.tensor(np.abs(mc_df["weights"].loc[mc_eta_mask].values)) ) )  
        #mc_corrected_cov = torch.cov( torch.tensor(mc_df[corr_var_list].loc[mc_eta_mask].values).T , aweights = torch.tensor( torch.tensor(np.abs(mc_df["weights"].loc[mc_eta_mask].values)) ) )  
        
        #### testing new weightd cov function
        # I had to write this weighted_covariance function as torch.cov does not accept negative weights
        data_cov         = weighted_covariance( torch.tensor(data_df[var_list].loc[data_eta_mask].values)  , w = torch.tensor( torch.tensor(np.abs(data_df["weights"].loc[data_eta_mask].values)) ) )  
        mc_cov           = weighted_covariance( torch.tensor(mc_df[var_list_mc].loc[mc_eta_mask].values)   , w = torch.tensor( torch.tensor(np.abs(mc_df["weights"].loc[mc_eta_mask].values)) ) )      
        mc_corrected_cov = weighted_covariance( torch.tensor(mc_df[corr_var_list].loc[mc_eta_mask].values) , w = torch.tensor( torch.tensor(np.abs(mc_df["weights"].loc[mc_eta_mask].values)) ) )  

        print(eta_region)
        ### now from cov to correlation matrices
        data_corr         = cov_to_corr(data_cov)
        mc_corr           = cov_to_corr(mc_cov)
        mc_corrected_corr = cov_to_corr(mc_corrected_cov)
        
        corrcoef_corrected = mc_corrected_corr.numpy()
        corrcoef_data      = data_corr.numpy()
        corrcoef_mc        = mc_corr.numpy()
        
        """ 
        # Weighted correlation matrix for data
        corrcoef_data = weighted_corr_matrix(
            df      = data_df.loc[data_eta_mask],
            var_list= var_list,
            w_col   = "weights"   # or whatever your weight column is called
        )
        
        # Weighted correlation matrix for MC (pre-correction)
        corrcoef_mc = weighted_corr_matrix(
            df      = mc_df.loc[mc_eta_mask],
            var_list= var_list_mc,
            w_col   = "weights"
        )
        
        # Weighted correlation matrix for MC (corrected)
        corrcoef_corrected = weighted_corr_matrix(
            df      = mc_df.loc[mc_eta_mask],
            var_list= corr_var_list,
            w_col   = "weights"
        )

        corrcoef_corrected = corrcoef_corrected.values
        corrcoef_data      = corrcoef_data.values
        corrcoef_mc        = corrcoef_mc.values
        """ 
        
        ##################################################
        #### Corrected vs data correlation matrix
        ##################################################

       # matrices setup ended! Now plotting part!
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow( 100*( corrcoef_data - corrcoef_corrected ), cmap = 'bwr', vmin = -35, vmax = 35)
        cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize = 70)
        cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110, labelpad=60)

        # ploting the cov matrix values
        factors_sum = 0
        mean,count = 0,0
        for (i, j), z in np.ndenumerate( 100*( corrcoef_data - corrcoef_corrected )):
            mean = mean + abs(z)
            count = count + 1
            factors_sum = factors_sum + abs(z)
            if( abs(z) < 1  ):
                pass
            else:
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 65)    
        
        ax.yaxis.labelpad = 20
        ax.xaxis.labelpad = 20
        mean = mean/count
        #ax.set_xlabel(r'$100 \cdot (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation^{Corr}}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
        plt.title( r'$\rho$(data) - $\rho$(corrected simulation)' + f' [{eta_region}]', fontweight='bold', fontsize = 130 , pad = 60 )
        
        ax.set_xticks(np.arange(len(var_list)))
        ax.set_yticks(np.arange(len(var_list)))
        
        # Apply the replace method to each element of the list
        cleaned_var_names = [name.replace("photon_", "").replace("raw_", "").replace("trkSum","").replace("ChargedIso","").replace("es","").replace("Cone","").replace("PF","").replace("Over","") for name in var_list]
         
        ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
        ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
        plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

        ax.tick_params(axis='both', which='major', pad=30)
        plt.tight_layout()

        plt.savefig(path + f'/new_code_correlation_matrix_corrected_{eta_region}.pdf')
        plt.close(fig)

        ##################################################
        #### Now nominal vs data correlation matrix
        ##################################################

        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow( 100*( corrcoef_data - corrcoef_mc ), cmap = 'bwr', vmin = -35, vmax = 35)
        cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize = 90)
        cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110,labelpad=60)
        
        #ploting the cov matrix values
        factors_sum = 0
        mean,count = 0,0
        for (i, j), z in np.ndenumerate( 100*( corrcoef_data - corrcoef_mc )):
            mean = mean + abs(z)
            count = count + 1
            factors_sum = factors_sum + abs(z)
            if( abs(z) < 1  ):
                pass
            else:
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 55)    
        
        mean = mean/count
        #ax.set_xlabel(r'$100 \cdot  (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
        plt.title( r'$\rho$(data) - $\rho$(nominal simulation)' + f' [{eta_region}]',fontweight='bold', fontsize = 140 , pad = 60 )
        
        ax.set_xticks(np.arange(len(var_list)))
        ax.set_yticks(np.arange(len(var_list)))
            
        ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
        ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
        plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

        ax.tick_params(axis='both', which='major', pad=30)
        plt.tight_layout()

        plt.savefig(path + f'/new_code_correlation_matrix_nominal_{eta_region}.pdf')
        plt.close(fig)

        ##################################################
        #### ADDED BELOW: Nominal correlation matrices themselves
        ##################################################
        # 1. Data correlation matrix
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow(corrcoef_data, cmap='bwr', vmin=-1, vmax=1)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=90)
        cbar.set_label(r'Correlation coefficient', rotation=90, loc='center', fontsize=110, labelpad=60)
        for (i, j), z in np.ndenumerate(corrcoef_data):
            if abs(z) < 0.01:
                continue
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=55)
        ax.set_xticks(np.arange(len(var_list)))
        ax.set_yticks(np.arange(len(var_list)))
        ax.set_xticklabels(cleaned_var_names, fontsize=50, rotation=90)
        ax.set_yticklabels(cleaned_var_names, fontsize=50, rotation=0)
        ax.tick_params(axis='both', which='major', pad=30)
        plt.title(r'$\rho$(data)' + f' [{eta_region}]', fontweight='bold', fontsize=140, pad=60)
        plt.tight_layout()
        plt.savefig(path + f'/new_code_correlation_matrix_dataonly_{eta_region}.pdf')
        plt.close(fig)

        # 2. Nominal MC correlation matrix
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow(corrcoef_mc, cmap='bwr', vmin=-1, vmax=1)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=90)
        cbar.set_label(r'Correlation coefficient', rotation=90, loc='center', fontsize=110, labelpad=60)
        for (i, j), z in np.ndenumerate(corrcoef_mc):
            if abs(z) < 0.01:
                continue
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=55)
        ax.set_xticks(np.arange(len(var_list)))
        ax.set_yticks(np.arange(len(var_list)))
        ax.set_xticklabels(cleaned_var_names, fontsize=50, rotation=90)
        ax.set_yticklabels(cleaned_var_names, fontsize=50, rotation=0)
        ax.tick_params(axis='both', which='major', pad=30)
        plt.title(r'$\rho$(nominal\, simulation)' + f' [{eta_region}]', fontweight='bold', fontsize=140, pad=60)
        plt.tight_layout()
        plt.savefig(path + f'/new_code_correlation_matrix_mc_nominal_only_{eta_region}.pdf')
        plt.close(fig)

        # 3. Corrected MC correlation matrix
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow(corrcoef_corrected, cmap='bwr', vmin=-1, vmax=1)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=90)
        cbar.set_label(r'Correlation coefficient', rotation=90, loc='center', fontsize=110, labelpad=60)
        for (i, j), z in np.ndenumerate(corrcoef_corrected):
            if abs(z) < 0.01:
                continue
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=55)
        ax.set_xticks(np.arange(len(var_list)))
        ax.set_yticks(np.arange(len(var_list)))
        ax.set_xticklabels(cleaned_var_names, fontsize=50, rotation=90)
        ax.set_yticklabels(cleaned_var_names, fontsize=50, rotation=0)
        ax.tick_params(axis='both', which='major', pad=30)
        plt.title(r'$\rho$(corrected\, simulation)' + f' [{eta_region}]', fontweight='bold', fontsize=140, pad=60)
        plt.tight_layout()
        plt.savefig(path + f'/new_code_correlation_matrix_mc_corrected_only_{eta_region}.pdf')
        plt.close(fig)

# Converting covariance to correlation matrices
def cov_to_corr(cov_matrix):
    stddev = torch.sqrt(torch.diag(cov_matrix))
    print(stddev)
    stddev_matrix = torch.diag_embed(stddev) # why is this here?
    corr_matrix = torch.inverse(stddev_matrix) @ cov_matrix @ torch.inverse(stddev_matrix)
    return corr_matrix