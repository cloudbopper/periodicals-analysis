"""
Loads documents for topic modeling
"""

import csv

METADATA_FILENAME = "Publication_Timestamps.csv"
PUBLICATION = "Publication"
FILENAME = "File Name"
YEAR = "Year"

class Document:
    """Document data class"""
    # pylint: disable = too-few-public-methods
    name: str
    data: str
    publication: str
    year: int

    def __init__(self, name, publication, year):
        self.name = name
        self.publication = publication
        self.year = year


def load_documents(input_dir):
    """Loads document data and meta-data"""
    documents = []
    with open("%s/%s" % (input_dir, METADATA_FILENAME)) as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            document = Document(row[FILENAME], row[PUBLICATION], int(row[YEAR]))
            with open("%s/%s.txt" % (input_dir, document.name)) as data_file:
                document.data = data_file.read()
            documents.append(document)
    return documents
