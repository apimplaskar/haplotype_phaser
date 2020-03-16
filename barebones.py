# CS 124: Machine Learning in Genetics
# Project: Haplotype Phaser
# Contributors: Aditya Pimplaskar, Aditya Joglekar

# just to guarantee me a 50% on this Project

# Packages needed
import numpy as np
import pandas as pd

# Imputation functions
def fillna(col):
    if col.value_counts().index[0] == '1':
        col.fillna(col.value_counts().index[1], inplace=True) # ensure we don't fill heterozygous
    else:
        col.fillna(col.value_counts().index[0], inplace=True)
    return col

def imputeData(df):
    df = df.replace('*', np.NaN)
    #df = df.astype('int64')
    #imputer = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')
    #return pd.DataFrame(imputer.fit_transform(df))
    return df.apply(lambda col:fillna(col))

#does the work
def barebones(genotypes):
    genotypes = genotypes.astype('int64')
    output = [[] for i in range(len(genotypes) * 2)]
    #print("output is", len(output))
    for index in range(0,len(genotypes)):
        row = genotypes.iloc[index]
        index1 = index*2
        index2 = index*2 +1
        for i in row:
            if i == 2: #deterministic
                output[index1].append(1)
                output[index2].append(1)
            if i == 0: #determnistic
                output[index1].append(0)
                output[index2].append(0)
            if i == 1: #arbitarily assign 1 and 0
                output[index1].append(1)
                output[index2].append(0)
    return output

# wrapper function to do all the annoying stuff
print('provide the file path for the genotypes: ')
file = input()
file = str(file)
print('thank you. processing using barebones algorithm')
genotypes = pd.read_csv(file, sep = " ", header = None)
imputed = imputeData(genotypes)
sendin = imputed.T
test = barebones(sendin)
result = pd.DataFrame(test)
result = result.T
result.to_csv("test_data_sol.txt", sep = " ", header = False, index = False)
print('all done!')
