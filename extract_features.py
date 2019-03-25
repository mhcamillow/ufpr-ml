import time
import sys
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath, header = 0, sep = ',')

def encode(labels):
    lblEncoder = LabelEncoder()
    return lblEncoder.fit_transform(labels)

def transform(messages, max_features):
    vect = TfidfVectorizer(
        decode_error='ignore', 
        stop_words='english', 
        max_features=int(max_features),
        min_df=5
        )
    matrix = vect.fit_transform(messages)
    return matrix

def export(labels, matrix):
    fout = open("features.txt", "w")
    cx = scipy.sparse.coo_matrix(matrix)
    controller = -1
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if controller <> i:
            controller = i
            fout.write("\n")
            fout.write(str(labels[i]))

        fout.write(" " + str(j) + ":" + str(v) + " ")
    fout.close

def export2(labels, matrix):
    fout = open("features.txt", "w")
    for i, comment in enumerate(matrix):
        fout.write(str(labels[i]) + " ")
        for j, attribute in enumerate(comment):
            fout.write(str(j) + ":" + str(attribute) + " ")
        fout.write("\n")
            
    fout.close

def main(data, max_features):

    print "Loading data"
    data = load_data(data)
    
    print "Encoding data"
    labels = encode(data['label'])

    print "Running tf-idf code"
    matrix = transform(data['review'], max_features)

    t1_export = time.time()
    export2(labels, matrix.toarray())
    print "Time to export (seconds): " + str(time.time() - t1_export)

if __name__ == "__main__":
    if len(sys.argv) != 3:
            sys.exit("Use: extract_features.py <data> <max_features>")
    
    t1_start = time.time()
    main(sys.argv[1], sys.argv[2])
    print "Finished. Total time (seconds): " + str(time.time() - t1_start)