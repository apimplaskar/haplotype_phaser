# CS 124: Machine Learning in Genetics
# Project: Haplotype Phaser
# Contributors: Aditya Pimplaskar, Aditya Joglekar
import numpy as np
import pandas as pd
import itertools as it
# Packages needed

# some helpers
def fillna(col):
    # fills in NAs in a given column w most frequent value between 0 and 2
    if col.value_counts().index[0] == '1':
        col.fillna(col.value_counts().index[1], inplace=True) # ensure we don't fill heterozygous
    else:
        col.fillna(col.value_counts().index[0], inplace=True)
    return col

def imputeData(df):
    #simple lambda function applies fillna to each row
    df = df.replace('*', np.NaN)
    #df = df.astype('int64')
    #imputer = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')
    #return pd.DataFrame(imputer.fit_transform(df))
    return df.apply(lambda col:fillna(col))

def readAndImpute(file):
    # reads genotype file, imputes data, casts to integer and workable list file
    data = pd.read_csv(file, sep = " ", header = None)
    data = imputeData(data)
    data = data.astype('int64')
    return data.values.tolist()

def basic_phaser(genotype):
    # to phase deterministic genotypes (only 0s and 2s)
    h1, h2 = [], []
    for g in genotype:
        if g == 0:
            h1.append(0)
            h2.append(0)
        if g == 2:
            h1.append(1)
            h2.append(1)
        if g == 1:
            break # we don't want haplos that are non-deterministic
            #h1.append(0)
            #h2.append(1)
    return h1, h2

def add_haplos(h1, h2):
    # to get genotype given by h1+h2
    a = [h1[i] + h2[i] for i in range(len(h1))]
    return a # assume h1 and h2 same len

def difference(g,h):
    # find other haplotype given genotype and one haplotype
    d = [g[i]-h[i] for i in range(len(g))]
    return d

def valid_haplo(h):
    # is haplotype valid (only made up of 0s and 1s)
    for s in h:
        if s != 0 and s != 1:
            return False
    return True

def h2_from_g_and_h1(g, known):
    # applies difference to a given genotype using pool of known haplotypes
    for h1 in known:
        h2 = difference(g, h1)
        if valid_haplo(h2):
            return h1, h2
    return [],[]

def guessUnphased(df, haplotypes): # just guess for stragglers
    nSNPs = len(df)
    nIndiv = len(df[0])
    for i in range(nIndiv):
        for j in range(nSNPs):

            if haplotypes[j][2*i] == -1 and haplotypes[j][2*i+1] == -1:
                if df[j][i] == 0:
                    haplotypes[j][2*i], haplotypes[j][2*i+1] = 0,0
                elif df[j][i] == 1:
                    haplotypes[j][2*i], haplotypes[j][2*i+1] = 1,0
                elif df[j][i] == 2:
                    haplotypes[j][2*i], haplotypes[j][2*i+1] = 1,1

            elif haplotypes[j][2*i] == -1:
                if haplotypes[j][2*i+1] == 0:
                    if df[j][i] == 0:
                        haplotypes[j][2*i] = 0
                    elif df[j][i] == 1:
                        haplotypes[j][2*i] = 1
                    elif df[j][i] == 2:
                        # error
                        haplotypes[j][i] = 1
                elif haplotypes[j][2*i+1] == 1:
                    if df[j][i] == 0:
                        # error
                        haplotypes[j][2*i] = 1
                    elif df[j][i] == 1:
                        haplotypes[j][2*i] = 0
                    elif df[j][i] == 2:
                        haplotypes[j][i] = 1

            elif haplotypes[j][2*i+1] == -1:
                if haplotypes[j][2*i] == 0:
                    if df[j][i] == 0:
                        haplotypes[j][2*i+1] = 0
                    elif df[j][i] == 1:
                        haplotypes[j][2*i+1] = 1
                    elif df[j][i] == 2:
                        # error
                        haplotypes[j][2*i+1] = 1
                elif haplotypes[j][2*i] == 1:
                    if df[j][i] == 0:
                        # error
                        haplotypes[j][2*i+1] = 1
                    elif df[j][i] == 1:
                        haplotypes[j][2*i+1] = 0
                    elif df[j][i] == 2:
                        haplotypes[j][i+1] = 1

    return haplotypes

def clarks(genotypes):
    numSNPS = len(genotypes)
    numIndivs = len(genotypes[0])


    # need to get pool of known haplotypes
    def get_known_haps(df):
        nSNPs = len(df)
        nIndividuals = len(df[0])
        # make return df
        haplotypes = np.zeros((nSNPs, 2*nIndividuals), dtype = np.int)
        haplotypes.fill(-1)
        #start with -1s to say unphased -- got this idea from a haplotype phaser by Daniel park
        known = []
        for i in range(nIndividuals):
            genotype = [row[i] for row in df]
            h1, h2 = basic_phaser(genotype)
            if len(h1) == len(h2) == len(genotype): # this checks to make sure we have a full haplotype from deterministic genotype
                if h1 not in known:
                    known.append(h1)
                if h2 not in known:
                    known.append(h2)
                for SNP in range(nSNPs):
                    haplotypes[SNP][2*i] = h1[SNP]
                    haplotypes[SNP][2*i + 1] = h2[SNP]
        return haplotypes, known

    haplotypes, known = get_known_haps(genotypes)

    def hashing_combos(known):
        combos = {}
        for pair in list(it.combinations(known, 2)):
            combos[str(add_haplos(pair[0], pair[1]))] = [pair[0],pair[1]]
        return combos

    known_combos = hashing_combos(known)
    for i in range(7): #many interations
        for i in range(numIndivs):
            genotype = [row[i] for row in genotypes]
            genotype_str = str(genotype)
            if genotype_str in known_combos: # we have already phased this!
                h1, h2 = known_combos[genotype_str]
                for index in range(numSNPS):
                    haplotypes[index][2*i] = h1[index]
                    haplotypes[index][2*i+1] = h2[index]
            else: # we phase this
                h1, h2 = h2_from_g_and_h1(genotype, known)
                if len(h1) > 0 and len(h2) > 0:
                    for index in range(numSNPS):
                        haplotypes[index][2*i] = h1[index]
                        haplotypes[index][2*i+1] = h2[index]
                    known.append(h2)
                    known_combos = hashing_combos(known)

    haplotypes = guessUnphased(genotypes, haplotypes) #any stragglers
    return haplotypes

def chunking(genotypes, split_size): # to get splits
    if len(genotypes) % split_size == 0:
            numChunks = (int)(len(genotypes)/split_size)
    else:
        numChunks = (int)(len(genotypes)/split_size) + 1
    return numChunks

#main
print('provide the file path for the genotypes: ')
file = input()
file = str(file)
print('thank you. processing using Clark's algorithm')
genots = readAndImpute(file)
split_size = 5 # hyperparameter, selected by testing
numChunks = chunking(genots, split_size)
haplos = []
for i in range(numChunks):
    # we run clarks on our chunks and then combine them
    data = genots[i*split_size:(i+1)*split_size]
    if i == 0:
        haplos = clarks(data)
    elif i == numChunks - 1 and len(genots[i*split_size]) > 0:
        final = genots[i*split_size:]
        final_numSNPs = len(final)
        final_numIndivs = len(final[0])
        final_haplo_chunk = np.zeros((final_numSNPs, 2*final_numIndivs), dtype = np.int)
        final_haplo_chunk.fill(-1)
        final_haplo_chunk = guessUnphased(final, final_haplo_chunk)
        haplos = np.concatenate((haplos, final_haplo_chunk), axis = 0)
    else:
        haplos = np.concatenate((haplos, clarks(data)), axis = 0)

np.savetxt('../test_data_sol.txt', haplos, fmt='%i', delimiter = ' ')

print('all done!')
