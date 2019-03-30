
partition_data_in_x_lines = 1000
data_size = len(data) #25000
number_of_batches = data_size / partition_data_in_x_lines + 1

print("Number of examples: " + str(data_size))
print("Batches size" + str(partition_data_in_x_lines))
print("Batch number: " + str(number_of_batches))

for i in range(number_of_batches):
    batch_start = i * partition_data_in_x_lines
    batch_end = (i + 1) * partition_data_in_x_lines - 1
    batch_labels = labels[batch_start:batch_end]
    batch_features = matrix[batch_start:batch_end].toarray()

    print("Exporting batch " + str(i+1) + "/" + str(number_of_batches))
    t1_export = time.time()
    export2(batch_labels, batch_features)
    # export(labels, matrix)
    print("Time to export (seconds): " + str(time.time() - t1_export))