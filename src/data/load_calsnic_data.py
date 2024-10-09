import re
import config as cfg
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_time_to_death(row):
    time = row['Date of death'] - row['Visit_Date']
    return time.total_seconds() / 86400

if __name__ == '__main__':
    # Load data
    patient_filename = "Final_Data_sheet_July2023_HenkJan.xlsx"
    survival_filename = "Survival_Jan2024.xlsx"
    patient_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, patient_filename), engine='openpyxl', date_format="%d/%m/%Y")
    survival_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, survival_filename), engine='openpyxl', date_format="%d/%m/%Y")
    
    patient_df = patient_df.loc[patient_df['Patient or Control'] == 'Patient'] # only use patients
    patient_df = patient_df[patient_df['Visit_Date'].notna()]
    patient_df['PSCID'] = patient_df['PSCID'].str.strip()
    survival_df['SUBJECT ID'] = survival_df['SUBJECT ID'].str.strip()
    survival_df = survival_df.rename({'SUBJECT ID': 'PSCID'}, axis=1)
    
    df = patient_df.merge(survival_df, on=['PSCID']) # Merge the two datasets
    df = df[cfg.PATIENT_COLS + cfg.SURVIVAL_COLS + cfg.ALSFRS_COLS
            + cfg.TAP_COLS + cfg.UMN_COLS] # Select relevant features
    
    # Convert empty strings to NaN
    df['UMN_Right'] = df['UMN_Right'].replace(to_replace=' ', value=np.nan, regex=True)
    df['UMN_Left'] = df['UMN_Left'].replace(to_replace=' ', value=np.nan, regex=True)
    df['LMN_Right'] = df['LMN_Right'].replace(to_replace=' ', value=np.nan, regex=True)
    df['LMN_Left'] = df['LMN_Left'].replace(to_replace=' ', value=np.nan, regex=True)
    
    # Drop rows without ALSFRS
    df = df[df['ALSFRS_Date'].notna()].copy(deep=True)
    
    # Sort by ID and Label
    df = df.sort_values(by=['PSCID', 'Visit Label']).reset_index(drop=True)
    
    # Repeat constant values
    constant_cols = ['Handedness', 'YearsEd', 'Diagnosis', 'Sex', 'Age',
                     'SymptomOnset_Date', 'Region_of_Onset', 'MedicalExamination_Riluzole']
    for col in constant_cols:
        new_col = df.groupby('PSCID')[col].apply(lambda x: x.bfill().ffill()).droplevel(level=1)
        df = df.join(new_col, on='PSCID', rsuffix='_r').drop_duplicates(subset=['PSCID', 'Visit Label'])
        df = df.drop(col, axis=1).rename({f'{col}_r': f'{col}'}, axis=1)
        
    # Use only observations from ALS patients
    df = df.loc[df['Diagnosis'] == 'ALS'].reset_index(drop=True)
    
    # Do some replacing
    df['Region_of_Onset'] = df['Region_of_Onset'].str.replace('{@}', '_')
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('not_available', None)
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_speech', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_speech_bulbar_swallowing', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_swallowing_upper_extremity', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_speech_upper_extremity', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_swallowing', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('bulbar_speech_bulbar_swallowing_lower_extremity', "bulbar")
    df['Region_of_Onset'] = df['Region_of_Onset'].replace('upper_extremity_ftd_cognitive', "upper_extremity")
    
    # Convert types
    df['Age'] = df['Age'].astype(int)
    df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], format="%Y-%m-%d")
    df['SymptomOnset_Date'] = pd.to_datetime(df['SymptomOnset_Date'], format="%Y-%m-%d")
    df['Date of death'] = pd.to_datetime(df['Date of death'], format="%Y-%m-%d", errors='coerce')
    df['ALSFRS_Date'] = pd.to_datetime(df['ALSFRS_Date'], format="%Y-%m-%d")
    df['UMN_Right'] = df['UMN_Right'].astype('Int64')
    df['UMN_Left'] = df['UMN_Left'].astype('Int64')
    df['LMN_Right'] = df['LMN_Right'].astype('Int64')
    df['LMN_Left'] = df['LMN_Left'].astype('Int64')
    
    # Record Riluzole use
    df = df.rename(columns={'MedicalExamination_Riluzole': 'Subject_used_Riluzole'})
    df['Subject_used_Riluzole'] = df['Subject_used_Riluzole'].replace('no', 'No')
    df['Subject_used_Riluzole'] = df['Subject_used_Riluzole'].replace('yes', 'Yes')
    df['Subject_used_Riluzole'] = df['Subject_used_Riluzole'].fillna('Unknown')
    
    # Calculate days between visitations
    df['Visit_Diff'] = df.groupby(['PSCID'])['Visit_Date'].diff().dt.days.fillna(0).astype(int)
    
    # Calculate number of days with symptoms
    df['SymptomDays'] = (df['Visit_Date'] - df['SymptomOnset_Date']).dt.days
    
    # Calculate disease progression rate
    df['DiseaseProgressionRate'] = (48 - df['ALSFRS_TotalScore']) / (df['SymptomDays']/30)
    
    # Calculate ALSFRS subscores
    df['ALSFRS_Bulbar_subscore'] = df['ALSFRS_1_Speech'] + df['ALSFRS_2_Salivation'] + df['ALSFRS_3_Swallowing']
    df['ALSFRS_FineMotor_subscore'] = df['ALSFRS_4_Handwriting'] + df['ALSFRS_5_Cuttingfood&handlingutensils'] + df['ALSFRS_6_Dressing&hygiene']
    df['ALSFRS_GrossMotor_subscore'] = df['ALSFRS_7_Turninginbed'] + df['ALSFRS_8_Walking'] + df['ALSFRS_9_Climbingstairs']
    df['ALSFRS_Breathing_subscore'] = df['ALSFRS_10_Dyspnea'] + df['ALSFRS_11_Orthopnea'] + df['ALSFRS_12_RespiratoryInsufficiency']

    # Annotate events
    event_cols = ['ALSFRS_Bulbar_subscore', 'ALSFRS_FineMotor_subscore',
                  'ALSFRS_GrossMotor_subscore', 'ALSFRS_Breathing_subscore']
    threshold = 6
    for event_col in event_cols:
        df[f'Event_{event_col}'] = (df[event_col] <= threshold).astype(bool)
        df[f'Event_{event_col}'] = df.groupby('PSCID')[f'Event_{event_col}'].shift(-1)
        df[f'TTE_{event_col}']  = df.groupby('PSCID')['Visit_Diff'].shift(-1)
        
    # Do some renaming
    df = df.rename(columns=lambda x: x.replace('Event_ALSFRS_', 'Event_') \
                   .replace('TTE_ALSFRS_', 'TTE_').replace('_subscore', ''))

    # Use only first visit
    #df = df.loc[df['Visit Label'] == 'Visit 1']
    
    # Rename "Visit Label" column to "Visit"
    df = df.rename(columns={'Visit Label': 'Visit'})
    
    # Extract visit number and replace the values with just "1", "2", or "3"
    df['Visit'] = df['Visit'].str.extract('(\d)').astype(int)
    
    # Remove patients that have the event on first visit # TODO: Maybe include this
    #left_censored = df.loc[(df['Visit Label'] == 'Visit 1') \
    #                        & (df[f'Event_{event_col}'] == 1)]['PSCID']
    #event_df = df.loc[~df['PSCID'].isin(left_censored)]
    
    # Drop NA and reset index
    event_cols = ['Event_Bulbar', 'Event_FineMotor', 'Event_GrossMotor', 'Event_Breathing']
    df = df.dropna(subset=event_cols).reset_index(drop=True)
    
    # Rename cols
    df = df.rename(columns=lambda x: re.sub(r'(Event|TTE)_ALSFRS_\d+_', r'\1_', x))
    
    # Record time to death
    df['Event_Death'] = df['Status'].apply(lambda x: True if x == 'Deceased' else False)
    tte_columns = [col for col in df.columns if col.startswith('TTE_')]
    df['TTE_Death'] = df[tte_columns].apply(lambda x: max(x), axis=1)
    df.loc[df['Event_Death'] == True, 'TTE_Death'] = df.loc[df['Event_Death'] == True].apply(calculate_time_to_death, axis=1)
    df.loc[df['TTE_Death'].isna(), 'TTE_Death'] = df.loc[df['TTE_Death'].isna(), tte_columns].apply(lambda x: max(x), axis=1)
        
    # Save data
    df.to_csv(f'{cfg.CALSNIC_DATA_DIR}/calsnic_processed.csv')
    