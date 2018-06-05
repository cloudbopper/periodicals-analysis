"""
Loads documents for topic modeling
"""

from pandas import read_csv

METADATA_FILENAME = "Publication_Timestamps.csv"
PUBLICATION = "Publication"
FILENAME = "File Name"
YEAR = "Year"
DATA = "Data"

def load_documents(input_dir):
    """Loads document data and meta-data"""
    documents = read_csv("%s/%s" % (input_dir, METADATA_FILENAME))
    data = []
    for filename in documents[FILENAME]:
        with open("%s/%s.txt" % (input_dir, filename)) as data_file:
            data.append(data_file.read())
    documents[DATA] = data
    return documents
