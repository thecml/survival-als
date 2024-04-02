import config as cfg
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    # Load data
    patient_filename = "Final_Data_sheet_July2023_HenkJan.xlsx"
    survival_filename = "Survival_Jan2024.xlsx"
    patient_df = pd.read_excel(Path.joinpath(cfg.DATA_DIR, patient_filename), engine='openpyxl', date_format="%d/%m/%Y")
    survival_df = pd.read_excel(Path.joinpath(cfg.DATA_DIR, survival_filename), engine='openpyxl', date_format="%d/%m/%Y")
    
    
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
    alsfrs_cols = ['ALSFRS_1_Speech', 'ALSFRS_3_Swallowing', 'ALSFRS_4_Handwriting', 'ALSFRS_8_Walking']
    threshold = 2
    for col in alsfrs_cols:
        df[f'Event_{col}'] = (df[col] <= threshold).astype(int)
        df[f'Event_{col}'] = df.groupby('PSCID')[f'Event_{col}'].shift(-1)
    df['TTE'] = df.groupby('PSCID')['Visit_Diff'].shift(-1)
    
    # Use only visit 1 and 2
    df = df[df['Visit Label'].isin(['Visit 1', 'Visit 2'])]
    
    # Drop NA TTE's
    df = df.dropna(subset=['TTE'])
    
    df = df.reset_index(drop=True)
    
    # Save data
    df.to_csv(f'{cfg.DATA_DIR}/data.csv')
    