from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
PROACT_DATA_DIR = Path.joinpath(ROOT_DIR, "data/proact")
CALSNIC_DATA_DIR = Path.joinpath(ROOT_DIR, "data/calsnic")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
PLOTS_DIR = Path.joinpath(ROOT_DIR, 'plots')
HIERARCH_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'hierarch')
MENSA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mensa')
BAYESIAN_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'bayesian')
MCD_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mcd')

PATIENT_COLS = ['PSCID', 'Visit Label', 'Diagnosis', 'Age', 'Sex', 'Handedness',
                'MedicalExamination_Riluzole', 'YearsEd', 'SymptomOnset_Date',
                'Symptom_Duration', 'Visit_details', 'Visit_Date', 'Region_of_Onset']
SURVIVAL_COLS = ['Status', 'Date of death', 'Cause of death']
ALSFRS_COLS = ["ALSFRS_Date", "ALSFRS_1_Speech", "ALSFRS_2_Salivation", "ALSFRS_3_Swallowing",
                "ALSFRS_Bulbar_Subscore", "ALSFRS_4_Handwriting", "ALSFRS_GastrostomyPresent",
                "ALSFRS_5_Cuttingfood&handlingutensils", "ALSFRS_6_Dressing&hygiene", "ALSFRS_Fine Motor_subscore",
                "ALSFRS_7_Turninginbed", "ALSFRS_8_Walking", "ALSFRS_9_Climbingstairs", "ALSFRS_Gross Motor_subscore",
                "ALSFRS_10_Dyspnea", "ALSFRS_11_Orthopnea", "ALSFRS_12_RespiratoryInsufficiency",
                "ALSFRS_Breathing_Subscore", "ALSFRS_TotalScore"]
TAP_COLS = ["TAP_Trial1RightFinger", "TAP_Trial1LeftFinger", "TAP_Trial2RightFinger", "TAP_Trial2leftFinger",
            "TAP_Trial1RightFoot", "TAP_Trial1LeftFoot", "TAP_Trial2RightFoot", "TAP_Trial2LeftFoot",
            "TAP_Fingertapping_Right_avg", "TAP_Fingertapping_Left_avg", "TAP_Foottapping_Right_avg", "TAP_Foottapping_Left_avg"]
UMN_COLS = ['UMN_Right', 'UMN_Left', 'LMN_Right', 'LMN_Left']
ECAS_COLS = ['ECAS_ALSNonSpecific Total', 'ECAS_ALSSpecific Total']

COXPH_PARAMS = {
    'alpha': 0.001,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

RSF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'min_samples_split': 60,
    'min_samples_leaf': 30,
    'max_features': None,
    "random_state": 0
}

DEEPSURV_PARAMS = {
    'hidden_size': 32,
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

DEEPHIT_PARAMS = {
    'num_nodes_shared': [32],
    'num_nodes_indiv': [32],
    'batch_norm': True,
    'verbose': False,
    'dropout': 0.25,
    'alpha': 0.2,
    'sigma': 0.1,
    'batch_size': 128,
    'lr': 0.001,
    'weight_decay': 0.01,
    'eta_multiplier': 0.8,
    'epochs': 1000,
    'early_stop': True,
    'patience': 10,
}

MTLR_PARAMS = {
    'hidden_size': 32,
    'verbose': False,
    'lr': 0.001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}

HIERARCH_PARAMS = {
    'theta_layer_size': [100],
    'layer_size_fine_bins': [(50, 5), (50, 5)],
    'lr': 0.001,
    'reg_constant': 0.05,
    'n_batches': 10,
    'batch_size': 32,
    'backward_c_optim': False,
    'hierarchical_loss': True,
    'alpha': 0.0001,
    'sigma': 10,
    'use_theta': True,
    'use_deephit': False,
    'n_extra_bins': 1,
    'verbose': True
}

SYNTHETIC_SETTINGS = {
    "alpha_e1": 18,
    "alpha_e2": 17,
    "alpha_e3": 16,
    "gamma_e1": 4,
    "gamma_e2": 4,
    "gamma_e3": 4,
    "n_events": 3,
    "n_samples": 5000,
    "n_features": 10,
    "adm_censoring_time": 10
}