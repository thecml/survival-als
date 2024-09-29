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
    
    # Drop rows without ALSFRS
    df = df[df['ALSFRS_Date'].notna()].copy(deep=True)
    
    # Sort by ID and Label
    df = df.sort_values(by=['PSCID', 'Visit Label']).reset_index(drop=True)
    
    # Repeat constant values
    constant_cols = ['Handedness', 'YearsEd', 'Diagnosis', 'Sex', 'Age', 'SymptomOnset_Date', 'Region_of_Onset']
    for col in constant_cols:
        new_col = df.groupby('PSCID')[col].apply(lambda x: x.bfill().ffill()).droplevel(level=1)
        df = df.join(new_col, on='PSCID', rsuffix='_r').drop_duplicates(subset=['PSCID', 'Visit Label'])
        df = df.drop(col, axis=1).rename({f'{col}_r': f'{col}'}, axis=1)
        
    # Use only observations from ALS patients
    df = df.loc[df['Diagnosis'] == 'ALS']
    
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
    
    # Calculate days between visitations
    df['Visit_Diff'] = df.groupby(['PSCID'])['Visit_Date'].diff().dt.days.fillna(0).astype(int)
    
    # Calculate number of days with symptoms
    df['SymptomDays'] = (df['Visit_Date'] - df['SymptomOnset_Date']).dt.days
    
    # Calculate disease progression rate
    df['DiseaseProgressionRate'] = (48 - df['ALSFRS_TotalScore']) / (df['SymptomDays']/30)

    # Annotate events
    event_names = ['speech', 'swallowing', 'handwriting', 'walking']
    event_cols = ['ALSFRS_1_Speech', 'ALSFRS_3_Swallowing', 'ALSFRS_4_Handwriting', 'ALSFRS_8_Walking']
    threshold = 2
    for event_name, event_col in zip(event_names, event_cols):
        # Assess threshold
        df[f'Event_{event_col}'] = (df[event_col] <= threshold).astype(int)
        
        # Adjust event indicator and time
        df[f'Event_{event_col}'] = df.groupby('PSCID')[f'Event_{event_col}'].shift(-1)
        df[f'TTE_{event_col}']  = df.groupby('PSCID')['Visit_Diff'].shift(-1)
        
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
    nan_cols = [f"Event_{col}" for col in event_cols]
    df = df.dropna(subset=nan_cols).reset_index(drop=True)
    
    # Rename cols
    df = df.rename(columns={
        'Event_ALSFRS_1_Speech': 'Event_Speech',
        'Event_ALSFRS_3_Swallowing': 'Event_Swallowing',
        'Event_ALSFRS_4_Handwriting': 'Event_Handwriting',
        'Event_ALSFRS_8_Walking': 'Event_Walking',
        'TTE_ALSFRS_1_Speech': 'TTE_Speech',
        'TTE_ALSFRS_3_Swallowing': 'TTE_Swallowing',
        'TTE_ALSFRS_4_Handwriting': 'TTE_Handwriting',
        'TTE_ALSFRS_8_Walking': 'TTE_Walking'})
    
    # Record time to death
    df['Event_Death'] = df['Status'].apply(lambda x: True if x == 'Deceased' else False)
    df['TTE_Death'] = df.apply(lambda x: max(x['TTE_Speech'], x['TTE_Swallowing'],
                                             x['TTE_Handwriting'], x['TTE_Walking']), axis=1)
    df.loc[df['Event_Death'] == True, 'TTE_Death'] = df.loc[df['Event_Death'] == True].apply(calculate_time_to_death, axis=1)
    df.loc[df['TTE_Death'].isna(), 'TTE_Death'] = df.loc[df['TTE_Death'].isna()].apply(lambda x: max(x['TTE_Speech'], x['TTE_Swallowing'],
                                                                                                     x['TTE_Handwriting'], x['TTE_Walking']), axis=1)
        
    # Save data
    df.to_csv(f'{cfg.CALSNIC_DATA_DIR}/data.csv')
    