

import csv
import Constants


def save(data):

    with open(Constants._RESULTS_FILENAME, 'w', newline = '') as file:
        writer = csv.writer(file)
        for i in range(len(data)):
            writer.writerow(data[i])

    print("Results saved as : " + Constants._RESULTS_FILENAME)









