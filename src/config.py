from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

PATIENT_COLS = ['PSCID', 'Visit Label', 'Diagnosis', 'Age', 'Sex', 'Handedness',
                'YearsEd', 'SymptomOnset_Date', 'Symptom_Duration',
                'Visit_details', 'Visit_Date', 'Region_of_Onset',
                'CNSLS_Date', 'CNSLS_TotalScore']
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
UMN_COLS = ['UMN_Right', 'UMN_Left']

PARAMS_COX = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50}