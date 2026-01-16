def read_data_mongo(collection, icustayid):
    """
    Ambil seluruh data monitoring pasien berdasarkan icustayid
    dan diurutkan berdasarkan charttime.
    Supports both string and numeric icustayid.
    """
    # Try to convert icustayid to int for matching
    try:
        icustayid_int = int(float(icustayid))
    except (ValueError, TypeError):
        icustayid_int = icustayid
    
    # Try searching with integer first, then string if no results
    cursor = collection.find(
        {"icustayid": icustayid_int},
        {"_id": 0}
    ).sort("charttime", 1)
    
    results = list(cursor)
    
    # If no results with int, try with original value (string)
    if len(results) == 0 and str(icustayid) != str(icustayid_int):
        cursor = collection.find(
            {"icustayid": str(icustayid)},
            {"_id": 0}
        ).sort("charttime", 1)
        results = list(cursor)
    
    return results
