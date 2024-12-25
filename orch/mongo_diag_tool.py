import os
from pymongo import MongoClient
from pprint import pprint


def diagnose_mongodb():
    # Connect to MongoDB
    mongo_conn_string = "mongodb://192.168.88.27:27018/foode"
    client = MongoClient(mongo_conn_string)

    # List all databases
    print("\nAvailable databases:")
    dbs = client.list_database_names()
    print(dbs)

    # Connect to foode database
    db = client.foode

    # List all collections
    print("\nCollections in 'foode' database:")
    collections = db.list_collection_names()
    print(collections)

    # Try to access the results collection directly
    results_collection = db['results']

    # Count documents
    doc_count = results_collection.count_documents({})
    print(f"\nTotal documents in results collection: {doc_count}")

    # Get a sample document
    if doc_count > 0:
        print("\nSample document from results collection:")
        sample_doc = results_collection.find_one()
        pprint(sample_doc)

    # Try to find specific user's documents
    user_docs = results_collection.find({'username': 'GaryTheGrillMaster'})
    print("\nDocuments for GaryTheGrillMaster:")
    for doc in user_docs:
        pprint(doc)


if __name__ == "__main__":
    diagnose_mongodb()