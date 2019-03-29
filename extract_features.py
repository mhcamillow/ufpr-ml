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

def extract_features(train_data, test_data, max_features):
    vect = TfidfVectorizer(
        decode_error='ignore', 
        stop_words='english', 
        max_features=int(max_features),
        min_df=5
        )
    train = vect.fit_transform(train_data)
    test = vect.transform(test_data)
    return train, test

def export(labels, matrix, filename):
    fout = open(filename, "w")
    for i, comment in enumerate(matrix):
        fout.write(str(labels[i]) + " ")
        for j, attribute in enumerate(comment):
            fout.write(str(j) + ":" + str(attribute) + " ")
        fout.write("\n")

    fout.close

def main(train_data, test_data, max_features):

    print "Loading data"
    train_data = load_data(train_data)
    test_data = load_data(test_data)
    
    print "Encoding data"
    train_labels = encode(train_data['label'])
    test_labels = encode(test_data['label'])

    print "Running tf-idf code"
    train_features, test_features = extract_features(train_data['review'], test_data['review'], max_features)

    t1_export = time.time()
    export(test_labels, test_features.toarray(), 'test-features.txt')
    export(train_labels, train_features.toarray(), 'train-features.txt')
    print "Time to export (seconds): " + str(time.time() - t1_export)

if __name__ == "__main__":
    if len(sys.argv) != 4:
            sys.exit("Use: extract_features.py <train_data> <test_data> <max_features>")
    
    t1_start = time.time()
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    print "Finished. Total time (seconds): " + str(time.time() - t1_start)