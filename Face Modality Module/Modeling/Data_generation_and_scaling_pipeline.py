from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############################### THIS IS Data_preparation.py THE PATH OF THE RESULTING PROPERTY FILE  #####################
file_path = r"<project_root>\dataset\pyfeat_lokal_60_skipframes\parkinson_0_0.xlsx" # Use .xlsx or .xls extension
df = pd.read_excel(file_path)



###########   PİPELİNE ##############
df2 = df.copy()
def feature_engineering(df,df2):
    # Fixed parameters (For example, params[0] = 1, params[2] = 1)
    params= (1, 1, 1)
    params2= (1, 1, 2)
    params3=(1, 1, 1)
    params4=(1, 1, 1)
    params5=(2, 7, 1)
    params6=(1, 5, 2)
    params7=(1, 1, 1)
    params8=(1, 1, 2)
    params9=(1, 1, 1)
    params10=(1, 1, 1)
    params11=(1, 1, 1)
    params12=(1, 1, 3)
    params13=(1, 6, 1)
    params14=(1, 1, 2)
    params15=(1, 1, 1)
    params152=(1, 1, 1)
    params16=(1, 1, 1)
    paramstg1=(1, 1, 1)

    # new columns
    df['nf1'] = (df['AU01'] * params[0] + df['AU02'] * params[0] + df['AU04'] * params[0]) / params[2]

    df['nf2'] = (df['AU06']*params2[0] + df['AU09']*params2[0] + df['AU10']*params2[0]+df['AU11']*params2[0] +df['AU12']*params2[0] +df['AU17']*params2[0]) / (params2[2]*params2[1])

    df['nf3'] = (((df['AU01']+df['AU04'])*params3[0]) / (params3[2]))

    df['nf4'] =(params4[0]*(df['AU06']+df['AU12'])-params4[1]*(df['AU25']+df['AU26']))/params4[2]

    df['nf5'] = ((df['AU07'] * params5[0] + df['AU10'] * params5[1] + df['AU11'] * params5[2]) + (df['AU24'] * params5[2]))
    
    df['nf6'] = (df['AU12']*params6[0]+df['AU17']*params6[1]+df['AU24']*params6[2])/3

    df['nf7'] = 1-((df['AU06']+df['AU07']+df['AU43'])/params7[2])

    df['nf8'] = 1-((params8[2]*((df['AU17']+df['AU24'])*params8[0]))-params8[1]*df['AU28'])

    df['nf9'] =  1-((2*params9[0]*(df['AU10']+df['AU11']))+(params9[0]*(df['AU17']*df['AU24']))+(2*params9[0]*df['AU02']*params9[1]*params9[2]))
    
    df['nf10'] =  1-((params10[1]*(-1*df['AU01'])+df['AU02'])*params10[2]*df['AU04'])

    df['nf11'] =  (1/(1+df2['time']))*(df['AU06']+df['AU12']/params11[0])*(-1*(df['AU25']+df['AU26']/params11[0]))

    df['nf12'] =  ((df['AU12'])+(params12[1]*params12[2]*df['AU17'])+(params12[1]*params12[2]*df['AU24']))/3
    
    df['nf13'] =  1-((0.3*params13[1]*((df['AU04']+df['AU07'])/2))+(0.2*params13[1]*((df['AU10']+df['AU17'])/2)))
    
    df['nf14'] =  1-((params14[1]*params14[2]*df['AU43']))

    df['nf15'] = (df['AU06'] + df['AU07'] + df['AU09'] + df['AU10'] + df['AU11'] + df['AU12'] + df['AU14'] + df['AU15'] + df['AU17'] + df['AU23'] + df['AU24'] + df['AU28'] + df['AU43']) / (13 * df2['time'])

    df['nf152'] = (df['AU06']+ df['AU07']+ df['AU09']+ df['AU10']+ df['AU11']+ df['AU12']+ df['AU14']+ df['AU17']+ df['AU23']+ df['AU24']+ df['AU28']+ df['AU43']+ df['nf2']+ df['nf4']+df['nf5']+ df['nf6']+df['nf12']+ df['nf15']) / (13 * df2['time'])

    df['nf16'] =  ( df['AU05'] +df['AU25'] + df['AU26']  +df['nf7'] + df['nf8'] +df['nf11'] + df['nf13'] + df['nf14']) / (13 * df2['time'])

    df['tg1'] =  ( np.sin(df['AU05']) + np.sin(df['AU25']) + np.sin(df['AU26'])  + np.sin(df['nf7']) + np.sin(df['nf8']) + np.sin(df['nf11']) + np.sin(df['nf13']) + np.sin(df['nf14'])) / np.sin((13 * df2['time']))
    
    return df


df=feature_engineering(df,df2)
print(df.head())

# save path
save_path = r"<project_root>\Modeling\test.csv"

# Save DataFrame as CSV
df.to_csv(save_path, index=False)

print(f"DataFrame has been successfully saved at {save_path}.")


