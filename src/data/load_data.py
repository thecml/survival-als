import config as cfg
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    # Load data
    path = cfg.DATA_DIR
    patient_filename = "Final_Data_sheet_July2023_HenkJan.xlsx"
    survival_filename = "Survival_Jan2024.xlsx"
    patient_df = pd.read_excel(Path.joinpath(path, patient_filename), engine='openpyxl', date_format="%d/%m/%Y")
    survival_df = pd.read_excel(Path.joinpath(path, survival_filename), engine='openpyxl', date_format="%d/%m/%Y")
    
    # Clean data
    patient_df = patient_df.loc[patient_df['Patient or Control'] == 'Patient'] # only use patients
    patient_df = patient_df[patient_df['Visit_Date'].notna()]
    patient_df['PSCID'] = patient_df['PSCID'].str.strip()
    survival_df['SUBJECT ID'] = survival_df['SUBJECT ID'].str.strip()
    survival_df = survival_df.rename({'SUBJECT ID': 'PSCID'}, axis=1)
    
    df = patient_df.merge(survival_df, on=['PSCID']) # Merge the two datasets
    df = df[cfg.PATIENT_COLS + cfg.SURVIVAL_COLS] # Select relevant features
    
    # Repeat constant values
    diagnosis = df.groupby('PSCID')['Diagnosis'].apply(lambda x: x.bfill().ffill()).droplevel(level=1)
    sex = df.groupby('PSCID')['Sex'].apply(lambda x: x.bfill().ffill()).droplevel(level=1)
    age = df.groupby('PSCID')['Age'].apply(lambda x: x.bfill().ffill()).droplevel(level=1)
    df = df.join(pd.concat([diagnosis, sex, age], axis=1), on='PSCID', rsuffix='_r').drop_duplicates(subset=['PSCID', 'Visit Label'])
    df = df.drop(['Diagnosis', 'Age', 'Sex'], axis=1).rename({'Diagnosis_r':'Diagnosis', 'Sex_r': 'Sex', 'Age_r': 'Age'}, axis=1)
    
    # Convert types
    df['Age'] = df['Age'].astype(int)
    df['Visit_Date'] = pd.to_datetime(df['Visit_Date'], format="%d/%m/%Y")
    df['Date of death'] = pd.to_datetime(df['Date of death'], format="%d/%m/%Y")
    
    # Assign event (baseline = dead)
    df['Event'] = df['Status'].apply(lambda x: 1 if x == 'Deceased' else 0)
    df.loc[df['Event'] == 1, 'Time'] = (df['Date of death'] - df['Visit_Date']).dt.days
    
    #patient_df = patient_df[patient_df['PSCID'].isin(survival_df['SUBJECT ID'])] # selct only those we have survival data on
    
    
    print(0)
    
    # Clean the survival data
    
    
    # Select columns

    


    
    
    
    
    
    
    