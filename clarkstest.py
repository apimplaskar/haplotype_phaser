import numpy as np
import csv
import itertools as it
import pandas as pd
#https://github.com/danielpark95/haplotype-phasing/blob/master/src/clarks.py

# reading and imputing
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

def readAndImpute(file):
    data = pd.read_csv(file, sep = " ", header = None)
    data = imputeData(data)
    data = data.astype('int64')
    return data.values.tolist()

# phasing helpers
def basic_phaser(genotype):
    h1, h2 = [], []
    for g in genotype:
        if g == 0:
            h1.append(0)
            h2.append(0)
        if g == 2:
            h1.append(1)
            h2.append(1)
        if g == 1:
            h1.append(0)
            h2.append(1)
    return h1, h2

def add_haplos(h1, h2):
    #print(h1, h2)
    a = [h1[i] + h2[i] for i in range(len(h1))]
    return a # assume h1 and h2 same len

def difference(g,h):
    d = [g[i]-h[i] for i in range(len(g))]
    return d

def known_haplotypes(h, frame_size):
    nSNPs = len(h)
    nHaps = len(h[0])
    known = []
    for i in range(nHaps):
        haplotype = [row[i] for row in h]
        for j in range(0, nSNPs, frame_size):
            haplo_segment = haplotype[j:j+frame_size]
            if haplo_segment.count(-1) == 0 and haplo_segment not in known:
                known.append(haplo_segment)

    return known

def fill_known_haps(df, frame_size):
    nSNPs = len(df)
    nIndividuals = len(df[0])
    haplotypes = np.zeros((nSNPs, 2*nIndividuals), dtype = np.int)
    haplotypes.fill(-1)
    known = []
    for i in range(nIndividuals):
        genotype = [row[i] for row in df]
        for j in range(0, nSNPs, frame_size):
            genot_segment = genotype[j:j+frame_size]
            h1, h2 = basic_phaser(genot_segment)
            if h1 not in known:
                known.append(h1)
            if h2 not in known:
                known.append(h2)
            for SNP in range(frame_size):
                haplotypes[j+SNP][2*i] = h1[SNP]
                haplotypes[j+SNP][2*i + 1] = h2[SNP]
    return haplotypes, known

def hash_haplo_combos(known):
    table = {}
    combinations = list(it.combinations(known, 2))
    for pair in combinations:
        table[str(add_haplos(pair[0], pair[1]))] = [pair[0],pair[1]]
    return table

def validHaplotype(h):
    for s in h:
        if s is not 0 and s is not 1:
            return False
    return True

def h2_from_g_and_h1(g, known):
    for h1 in known:
        h2 = difference(g, h1)
        if validHaplotype(h2):
            return h1, h2
    return [],[]

def guessUnphased(df, haplotypes):
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

def clarks(df, frame_size):
    nSNPs = len(df)
    nIndiv = len(df[0])
    haplotypes, known = fill_known_haps(df, frame_size)

    haplotype_hash = hash_haplo_combos(known)

    for n_iter in range(10):
        print("iteration", n_iter)
        print("frame_size", frame_size)
        current_known_size = len(known)
        for i in range(nIndiv):
            genotype = [row[i] for row in df]
            for j in range(0, nSNPs, frame_size):
                genot_segment = genotype[j:j+frame_size]
                genot_segment_string = str(genot_segment)

                if genot_segment_string in haplotype_hash:
                    h1, h2 = haplotype_hash[genot_segment_string]
                    for index in range(frame_size):
                        haplotypes[j+index][2*i] = h1[index]
                        haplotypes[j+index][2*i+1] = h2[index]

                else:
                    h1, h2 = h2_from_g_and_h1(genot_segment, known)
                    if len(h1) > 0 and len(h2) > 0:
                        for index in range(frame_size):
                            haplotypes[j+index][2*i] = h1[index]
                            haplotypes[j+index][2*i+1] = h2[index]
                        known.append(h2)
                        haplotype_hash = hash_haplo_combos(known)

        if len(known) - current_known_size == 0:
            if frame_size == 30:
                frame_size -= 10
            elif frame_size == 15 or frame_size == 20:
                frame_size -= 5
            else:
                 break
            known = known_haplotypes(haplotypes,frame_size)

    to_return = guessUnphased(df, haplotypes)
    return to_return


ex1 = readAndImpute("assignment/example_data_1_masked.txt")
nSNPs = len(ex1)
nIndiv = len(ex1[0])
block = 180
frame = 30
haplos = []
if len(ex1) % block == 0:
    n_blocks = (int)(len(ex1)/block)
else:
    n_blocks = (int)(len(ex1)/block) + 1

for i in range(n_blocks):
    print("on block", i)
    data_block = ex1[i*block:i*block+block]
    if i == 0:
        haplos_block = clarks(data_block, frame)
        haplos = haplos_block
    elif i == n_blocks -1 and len(ex1[i*block]>0):
        final_block = ex1[i*block]
        final_nSNP = len(final_block)
        final_nIndiv = len(final_block[0])
        final_haplo_block = np.zeros((final_nSNP, 2*final_nIndiv), dtype = np.int)
        final_haplo_block.fill(-1)
        final_haplo_block = guessUnphased(final_block, final_haplo_block)
        haplos = np.concatenate((haplos, final_haplo_block), axis = 0)
    else:
        haplos_block = clarks(data_block, frame)
        haplos = np.concatenate((haplos, haplos_block), axis = 0)

np.savetxt('test_data_1_my_sol.txt', haplos, fmt='%i', delimiter = ' ')
