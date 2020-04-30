import spacy
import pysubs2
import numpy as np 
import pandas as pd
import lap
import os
from scipy.optimize import linear_sum_assignment, minimize

np.set_printoptions(suppress = True)

def naive_assignment(cost_mat):
    grid = np.indices(cost_mat.shape)
    sort_ind = np.argsort(cost_mat, axis = None)
    row_grid = grid[0][sort_ind]
    col_grid = grid[1][sort_ind]
    cost = cost_mat[sort_ind]

    row_ind, col_ind = [], []
    for i in range(cost.shape[0]):
        row = row_grid[i]
        col = col_grid[i]
        if row in row_ind or col in col_ind: continue
        row_ind.append(row)
        col_ind.append(col)

    return np.asarray(row_ind, dtype=np.uint8), np.asarray(col_ind, dtype=np.uint8)
    
def cost_mat(x, y):
    cost_mat = create_cost_mat(x, y)
    try:
        ans = lap.lapjv(cost_mat)
        row_ind, col_ind = np.arange(len(ans[1])), ans[1]
    except:
        row_ind, col_ind = naive_assignment(cost_mat)

    return row_ind, col_ind

def create_cost_mat(x, y):
    size = max(len(x), len(y))
#     cost_mat = np.ones((len(x), len(y)))
    cost_mat = np.ones((size, size))*100
    
    COST_W = np.array([7, 1])
    for i in range(len(x)):
        for j in range(len(y)):
            #type cost
            #ind cost
            type_cost = x[i][1] != y[j][1]
            ind_cost = abs(i-j)        
            costs = np.array([type_cost, ind_cost])
            cost_mat[i, j] = COST_W.dot(costs)
    return cost_mat

def chop_tokens(tokens):
    start_times = {}
    groupings = {}
    for key in tokens.keys():
        start_times[key] = np.asarray(tokens[key])[:, -1].astype(np.int32)
        groupings[key] = -np.ones_like(start_times[key])
    
    timestamps = np.unique(start_times[key])
        
    for i in range(timestamps.shape[0]):
        for key in tokens.keys():
            ind = start_times[key] < timestamps[i]
            groupings[key][ind] = i
            start_times[key][ind] = 1000000

    for key in tokens.keys():
        ind = groupings[key] == -1
        groupings[key][ind] = i
    
    return groupings, i+1

def align_tokens(tokens):
    aligned_tokens = {}
    for key in tokens.keys(): aligned_tokens[key] = []

    groupings, max_val = chop_tokens(tokens)
    for i in range(max_val):
        invalid = False
        sentences = []
        for key in tokens.keys():
            ind = np.where(groupings[key] == i)[0]
            if len(ind) == 0: 
                invalid = True
                continue
            start, end = np.amin(ind), np.amax(ind)
            sentence = tokens[key][start:end+1]
            sentences.append(sentence)
        if invalid: continue
        keys = list(tokens.keys())
    
        if len(sentences[1]) > len(sentences[0]):
            sentences[0], sentences[1] = sentences[1], sentences[0]
            keys = keys[::-1]
            
        row_ind, col_ind = cost_mat(sentences[0], sentences[1])
    
        for j in range(len(row_ind)):
            row, col = row_ind[j], col_ind[j]
            try:
                aligned_tokens[keys[0]].append(sentences[0][row])
            except:
                aligned_tokens[keys[0]].append([0, 0, 0, 0])
            try:
                aligned_tokens[keys[1]].append(sentences[1][col])
            except:
                aligned_tokens[keys[1]].append([0, 0, 0, 0])
                
    return aligned_tokens
    
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    try:
        sec =  int(h) * 3600 + int(m) * 60 + float(s)
    except:
        sec =  int(h) * 3600 + int(m) * 60

    return int(sec*1000)



if __name__ == '__main__':

    modalities = ['en', 'es']
    encodings = {
                'en': 'ISO-8859-1', 
                'es': 'latin-1'
    }           

    data_base_dir = '../raw_data/ted/'
    destination = '../datasets/subtitles/'
    path_file = data_base_dir + 'vid_'+modalities[0]+'_'+modalities[1]+'_lang.txt'

    nlp = {}
    for modal in modalities:
        nlp[modal] = spacy.load(modal)

    with open(path_file) as f:
        for line in f:
            try:
                vid_name = line.rstrip()
                output_csv = destination + vid_name +'_'+modalities[0]+'_'+modalities[1]+'.csv'
                if os.path.exists(output_csv): continue
                print(vid_name)

                tokens = {}
                for modal in modalities: tokens[modal] = []

                for modal in modalities:
                    lang_path = data_base_dir + 'transcripts/'+ vid_name +'_'+modal+'.srt'
                    subs = pysubs2.load(lang_path, encoding= encodings[modal], format_= "srt")
                    print('loading sub', modal)    
                    for sub in subs:
                        split = sub.text.split('\\N')
                        if len(split) > 1:
                            doc_0 = nlp[modal](split[0])
                            doc_1 = nlp[modal](split[-1])
                            # print(split)
                            sub_start_1 = get_sec(split[3].split(',0')[0])

                            for token in doc_0:
                                tokens[modal].append([token.text, token.pos_, token.vector_norm, sub.start])

                            for token in doc_1:
                                tokens[modal].append([token.text, token.pos_, token.vector_norm, sub_start_1])

                        else:
                            doc = nlp[modal](sub.text)
                        
                            for token in doc:
                                tokens[modal].append([token.text, token.pos_, token.vector_norm, sub.start])

                print('aligning tokens')
                aligned_tokens = align_tokens(tokens)
                
                print('converting to df')
                for key in aligned_tokens.keys():
                    aligned_tokens[key] = pd.DataFrame(aligned_tokens[key])
                df = pd.concat(aligned_tokens, axis=1)
                df.to_csv(output_csv)     
                print(df)
            except:
                continue
            # break