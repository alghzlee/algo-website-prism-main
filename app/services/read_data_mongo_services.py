def read_data_mongo(collection, icustayid):
    """
    Ambil seluruh data monitoring pasien berdasarkan icustayid
    dan diurutkan berdasarkan charttime.
    """
    cursor = collection.find(
        {"icustayid": icustayid},
        {"_id": 0}
    ).sort("charttime", 1)

    return list(cursor)
