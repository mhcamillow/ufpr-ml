import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

def main(data, test_size, file_prefix):
    features, classes = load_svmlight_file(data)
    train_features, test_features, train_classes, test_classes = train_test_split(features, classes, test_size=float(test_size), random_state = 5)
    export_data(train_features, train_classes, file_prefix + "-train.txt")
    export_data(test_features, test_classes, file_prefix + "-test.txt")

def export_data(feature_data, class_data, filename):
    print("Exporting " + str(len(class_data)) + " lines to " + filename)

    fout = open(filename, "w")
    for x, features in enumerate(feature_data.toarray()):
        fout.write(str(int(class_data[x])) + " ")
        for y, feature in enumerate(features):
            fout.write(str(y) + ":" + str(feature) + " ")
        fout.write("\n")
    fout.close

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Use: generate-test-files.py <data> <test_size> <file_prefix>")

    main(sys.argv[1], sys.argv[2], sys.argv[3])
