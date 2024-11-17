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

def convert_weight(row):
    if row['weight_scale'] == 'kg':
        return row['weight']  # Keep the value as is
    elif row['weight_scale'] == 'lbs':
        return row['weight'] * 0.453592  # Convert pounds to kg
    else:
        return None  # Handle any unexpected values

def convert_height(row):
    if row['height_scale'] == 'cm':
        return row['height']  # Keep the value as is
    elif row['height_scale'] == 'inches':
        return row['height'] * 2.54  # Convert inches to cm
    else:
        return None  # Handle any unexpected values

def annotate_left_censoring(row, event_name):
    if (row['Visit Label'] == 'Visit 1') and (row[f'Event_{event_name}'] == True):
        tte = min(row['SymptomDays'], 365) # lower bound on t
        event_censored = True
    else:
        tte = row[f'TTE_{event_name}'] # upper bound on t
        event_censored = False
    return pd.Series({f'TTE_{event_name}': tte,f'Event_{event_name}': event_censored})

if __name__ == '__main__':
    # Load data
    patient_filename = "Final_Data_sheet_July2023_HenkJan.xlsx"
    survival_filename = "Survival_Jan2024.xlsx"
    fvc1_filename = "FVC_CALSNIC1.xlsx"
    fvc2_filename = "FVC_CALSNIC2.xlsx"
    
    patient_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, patient_filename), engine='openpyxl', date_format="%d/%m/%Y")
    survival_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, survival_filename), engine='openpyxl', date_format="%d/%m/%Y")
    fvc1_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, fvc1_filename), engine='openpyxl', date_format="%d/%m/%Y")
    fvc2_df = pd.read_excel(Path.joinpath(cfg.CALSNIC_DATA_DIR, fvc2_filename), engine='openpyxl', date_format="%d/%m/%Y")
    
    patient_df = patient_df.loc[patient_df['Patient or Control'] == 'Patient'] # only use patients
    patient_df = patient_df[patient_df['Visit_Date'].notna()]
    patient_df['PSCID'] = patient_df['PSCID'].str.strip()
    survival_df['SUBJECT ID'] = survival_df['SUBJECT ID'].str.strip()
    survival_df = survival_df.rename({'SUBJECT ID': 'PSCID'}, axis=1)
    
    df = patient_df.merge(survival_df, on=['PSCID']) # Merge the two datasets
    df = df[cfg.PATIENT_COLS + cfg.SURVIVAL_COLS + cfg.ALSFRS_COLS
            + cfg.UMN_COLS + cfg.ECAS_COLS] # Select relevant features
    
    fvc1_df['PSCID'] = fvc1_df['Record ID'].str.strip()
    fvc2_df['PSCID'] = fvc2_df['PSCID'].str.strip()
    
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
    
    # Record ECAS
    df = df.rename(columns={'ECAS_ALSNonSpecific Total': 'ECAS_ALSNonSpecific_Total',
                            'ECAS_ALSSpecific Total': 'ECAS_ALSSpecific_Total'})
    
    # Record FVC mean
    fvc1_df_cols = ['FVC Trial1L', 'FVC Trial2L', 'FVC Trial3L', 'FVC Trial4L', 'FVC Trial5L']
    fvc2_df_cols = ['trial_one', 'trial_two', 'trial_three', 'trial_four', 'trial_five']
    fvc1_df[fvc1_df_cols] = fvc1_df[fvc1_df_cols].apply(pd.to_numeric, errors='coerce')
    fvc2_df[fvc2_df_cols] = fvc2_df[fvc2_df_cols].apply(pd.to_numeric, errors='coerce')
    fvc1_df['FVC_Mean'] = fvc1_df[fvc1_df_cols].mean(axis=1, skipna=True)
    fvc2_df['FVC_Mean'] = fvc2_df[fvc2_df_cols].mean(axis=1, skipna=True)
    fvc1_df = fvc1_df.rename({'Visit': 'Visit Label'}, axis=1)
    visit_mapping = {'V1': 'Visit 1', 'V2': 'Visit 2', 'V3': 'Visit 3', 'V4': 'Visit 4', 'V5': 'Visit 5',
                     'V6': 'Visit 6', 'V7': 'Visit 7', 'V8': 'Visit 8', 'V9': 'Visit 9', 'V10': 'Visit 10'}
    fvc2_df['Visit Label'] = fvc2_df['Visit Label'].replace(visit_mapping)
    df = pd.merge(df, fvc1_df[['PSCID', 'Visit Label', 'FVC_Mean']], on=['PSCID', 'Visit Label'], how='left')
    df = pd.merge(df, fvc2_df[['PSCID', 'Visit Label', 'FVC_Mean']], on=['PSCID', 'Visit Label'], how='left')
    df['FVC_Mean'] = df['FVC_Mean_x'].combine_first(df['FVC_Mean_y'])
    df = df.drop(columns=['FVC_Mean_x', 'FVC_Mean_y'])
    
    # Record race, height and weight
    df = pd.merge(df, fvc1_df[['PSCID', 'Visit Label', 'FVC Ethnicity',
                               'FVC Height (cm)', 'FVC Weight (kg)']], on=['PSCID', 'Visit Label'], how='left')
    fvc2_df['Weight'] = fvc2_df.apply(convert_weight, axis=1)
    fvc2_df['Height'] = fvc2_df.apply(convert_height, axis=1)
    df = pd.merge(df, fvc2_df[['PSCID', 'Visit Label', 'ethnicity',
                               'Weight', 'Height']], on=['PSCID', 'Visit Label'], how='left')
    df['Ethnicity'] = df['FVC Ethnicity'].combine_first(df['ethnicity'])
    df = df.drop(columns=['FVC Ethnicity', 'ethnicity'])
    df['Ethnicity'] = df['Ethnicity'].fillna('Unknown')
    df['Weight'] = df['FVC Weight (kg)'].combine_first(df['Weight'])
    df['Height'] = df['FVC Height (cm)'].combine_first(df['Height'])
    df = df.drop(columns=['FVC Height (cm)', 'FVC Weight (kg)'])
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    
    # Calculate days between visitations
    df['Visit_Diff'] = df.groupby(['PSCID'])['Visit_Date'].diff().dt.days.fillna(0).astype(int)
    
    # Calculate number of days with symptoms
    df['SymptomDays'] = (df['Visit_Date'] - df['SymptomOnset_Date']).dt.days
    
    # Calculate disease progression rate
    df['DiseaseProgressionRate'] = (48 - df['ALSFRS_TotalScore']) / (df['SymptomDays']/30)
    
    # Calculate ALSFRS subscores
    #df['ALSFRS_Bulbar_subscore'] = df['ALSFRS_1_Speech'] + df['ALSFRS_2_Salivation'] + df['ALSFRS_3_Swallowing']
    #df['ALSFRS_FineMotor_subscore'] = df['ALSFRS_4_Handwriting'] + df['ALSFRS_5_Cuttingfood&handlingutensils'] + df['ALSFRS_6_Dressing&hygiene']
    #df['ALSFRS_GrossMotor_subscore'] = df['ALSFRS_7_Turninginbed'] + df['ALSFRS_8_Walking'] + df['ALSFRS_9_Climbingstairs']
    #df['ALSFRS_Breathing_subscore'] = df['ALSFRS_10_Dyspnea'] + df['ALSFRS_11_Orthopnea'] + df['ALSFRS_12_RespiratoryInsufficiency']

    # Annotate events
    threshold = 2
    df[f'Event_Speech'] = (df['ALSFRS_1_Speech'] <= threshold)
    df[f'Event_Swallowing'] = (df['ALSFRS_3_Swallowing'] <= threshold)
    df[f'Event_Handwriting'] = (df['ALSFRS_4_Handwriting'] <= threshold)
    df[f'Event_Walking'] = (df['ALSFRS_8_Walking'] <= threshold)
    event_names = ["Speech", "Swallowing", "Handwriting", "Walking"]
    for event_col in event_names:
        df[f'Event_{event_col}'] = df.groupby('PSCID')[f'Event_{event_col}'].shift(-1)
        df[f'TTE_{event_col}']  = df.groupby('PSCID')['Visit_Diff'].shift(-1)
        
    # Do some renaming
    df = df.rename(columns=lambda x: x.replace('Event_ALSFRS_', 'Event_') \
                   .replace('TTE_ALSFRS_', 'TTE_').replace('_subscore', ''))
    
    # Handle left-censoring
    for event_col in event_names:
        df[[f'TTE_{event_col}', f'Event_{event_col}']] = df.apply(lambda x: annotate_left_censoring(x, event_col), axis=1)

    # Use only first visit
    #df = df.loc[df['Visit Label'] == 'Visit 1']
    
    # Rename "Visit Label" column to "Visit"
    df = df.rename(columns={'Visit Label': 'Visit'})
    
    # Extract visit number and replace the values with just "1", "2", or "3"
    df['Visit'] = df['Visit'].str.extract('(\d)').astype(int)
    
    # Remove patients that have the event on first visit # TODO
    #left_censored = df.loc[(df['Visit Label'] == 'Visit 1') \
    #                        & (df[f'Event_{event_col}'] == 1)]['PSCID']
    #event_df = df.loc[~df['PSCID'].isin(left_censored)]
    
    # Drop NA and reset index
    event_cols = ["Event_Speech", "Event_Swallowing", "Event_Handwriting", "Event_Walking"]
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
    