import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

def main(data, test_size):
    # loads data
    print "Loading data..."
    features, classes = load_svmlight_file(data)

    # splits data
    print "Spliting data..."
    train_features, test_features, train_classes, test_classes = train_test_split(features, classes, test_size=float(test_size), random_state = 5)

    export_data(train_features, train_classes, "train.txt")
    export_data(test_features, test_classes, "test.txt")

def export_data(feature_data, class_data, filename):
    print "Exporting " + str(len(class_data)) + " lines to " + filename

    fout = open(filename, "w")
    for x, features in enumerate(feature_data.toarray()):
        fout.write(str(int(class_data[x])) + " ")
        for y, feature in enumerate(features):
            fout.write(str(y) + ":" + str(feature) + " ")
        fout.write("\n")
    fout.close

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: generate-test-files.py <data> <test_size>")

    main(sys.argv[1], sys.argv[2])
