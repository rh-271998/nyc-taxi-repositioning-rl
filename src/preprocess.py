import pandas as pd
import geopandas as gpd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def bucketize_time(df, datetimeCol, timeInterval):

    seconds_from_midnight = (
    df[datetimeCol].dt.hour * 3600 + 
    df[datetimeCol].dt.minute * 60 + 
    df[datetimeCol].dt.second
)
    df[f'{datetimeCol}_bucket_id'] = seconds_from_midnight // (timeInterval * 60)

    df = df[df['fare_amount'] > 0]

    return df

def get_zoneName_zoneID(df, columnName, zoneName):

    df = df[df[columnName] == zoneName]

    return df['LocationID']

def prepare_data(tripData):

    # get trip duration in seconds

    df = tripData.copy()
    df['duration_seconds'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

    # get demand count

    demand = (df.groupby(['PULocationID', 'tpep_pickup_datetime_bucket_id'])
            .size()
            .rename('demand'))
    
    # get destination distribution

    destCounts = (df.groupby(['PULocationID', 'tpep_pickup_datetime_bucket_id', 'DOLocationID'])
                .size())
    destDist = (destCounts / destCounts.groupby(level=[0, 1]).transform('sum')).rename('prob')

    # get fare and duration stats (median)

    odStats = (df[df['fare_amount'] > 0]
             .groupby(['PULocationID', 'DOLocationID'])
             .agg(fare_median     = ('fare_amount',      'median'),
                  duration_median = ('duration_seconds', 'median'),
                  n_trips         = ('fare_amount',       'size')))

    return demand, destDist, odStats

def filter_zone_data(df, zoneSeries):

    filtered_df = df[df['PULocationID'].isin(zoneSeries) & df['DOLocationID'].isin(zoneSeries)]
    return filtered_df

def conversion_dicts(zoneDf):

    id2idx = {}
    idx2id = {}

    for i in range(len(zoneDf)):
        id2idx[int(zoneDf.iloc[i])] = i
        idx2id[i] = int(zoneDf.iloc[i])
    
    return id2idx, idx2id

def get_adjacent_zones(gdf, zoneDf):

    target_gdf = gdf[gdf['LocationID'].isin(zoneDf)].copy()
    
    target_gdf['geometry'] = target_gdf.geometry.buffer(10)

    # internal_touches = gpd.sjoin(target_gdf, target_gdf, how='inner', predicate='touches')

    internal_touches = gpd.sjoin(target_gdf, target_gdf, predicate='intersects')

    pairs = internal_touches[internal_touches['LocationID_left'] != internal_touches['LocationID_right']][['LocationID_left', 'LocationID_right']]

    # adjacency_dict = pairs.groupby('LocationID_left')['LocationID_right'].apply(list).to_dict()

    adjacency_dict = {
    loc_id: pairs.loc[pairs['LocationID_left'] == int(loc_id), 'LocationID_right'].tolist()
    for loc_id in zoneDf
    }

    return adjacency_dict

def get_demand_lookup_table(demandDf, nLocations, nBuckets, mappingDict):

    demandArr = np.zeros((nLocations, nBuckets))

    for i in range(nLocations):
        rawId = mappingDict.get(i)
        for j in range(nBuckets):
            demandArr[i,j] = demandDf.get((rawId, j), 0)
    
    return demandArr

def get_destination_lookup_table(destinationDf, nLocations, nBuckets, idx2id, id2idx):

    destArr = np.zeros((nLocations, nBuckets), dtype=object)

    for i in range(nLocations):
        rawId = idx2id.get(i)
        for j in range(nBuckets):

            try:
                subSeries = destinationDf.loc[(rawId, j)]
                doLoc = [id2idx[d] for d in subSeries.index]
                probValues = subSeries.values.tolist()

            except KeyError:
                doLoc = []
                probValues = []

            destArr[i, j] = (doLoc, probValues)            
            
    return destArr

def get_median_fare_duration_from_adj_zones(src, dest, adj_dict, threshold_df):

    neighbors = adj_dict[dest]

    fareMedian = 0
    durationMedian = 0
    nTrips = 0

    for n in neighbors:
        if n != src:
            try:
                row = threshold_df.loc[(src, n)]
                fareMedian += row['fare_median'] * row['n_trips']
                durationMedian += row['duration_median'] * row['n_trips']
                nTrips += row['n_trips']

            except KeyError:
                fareMedian += 0
                durationMedian += 0
                nTrips += 0
    
    if nTrips == 0:
        return None, None

    fareMedian = fareMedian / nTrips
    durationMedian = durationMedian / nTrips

    return fareMedian, durationMedian    

def get_fare_and_duration_lookup_table(odStats, nLocations, idx2id, threshold, adjDict):

    odArr = np.zeros((nLocations, nLocations), dtype=object)

    threshold_df = odStats[odStats['n_trips'] >= threshold]

    assert len(threshold_df) > 0, f"odStats df thresholded with {threshold} minimum n-trips returned 0 rows. please check."

    globalFareMedian = threshold_df['fare_median'].median()
    globalDurationMedian = threshold_df['duration_median'].median()

    for i in range(nLocations):
        PUloc = idx2id[i]
        for j in range(nLocations):
            DOloc = idx2id[j]

            try:
                row = odStats.loc[(PUloc, DOloc)]

                if row['n_trips'] >= threshold:
                    odArr[i, j] = (float(row['fare_median']), float(row['duration_median'])/60.0)
                else:
                    fareMed, durationMed = get_median_fare_duration_from_adj_zones(PUloc, DOloc, adjDict, threshold_df)
                    if fareMed is not None:
                        odArr[i,j] = (float(fareMed), float(durationMed)/60.0)
                    else:
                        odArr[i,j] = (float(globalFareMedian), float(globalDurationMedian)/60.0)
            except KeyError:
                fareMed, durationMed = get_median_fare_duration_from_adj_zones(PUloc, DOloc, adjDict, threshold_df)
                if fareMed is not None:
                    odArr[i,j] = (float(fareMed), float(durationMed)/60.0)
                else:
                    odArr[i,j] = (float(globalFareMedian), float(globalDurationMedian)/60.0)

    return odArr

def get_centroids(gdf, zoneDf, id2idx):

    target_gdf = gdf[gdf['LocationID'].isin(zoneDf)].copy()

    centroids = target_gdf.geometry.centroid

    coords = np.array([[point.x, point.y] for point in centroids])

    mins   = coords.min(axis=0)           
    ranges = coords.max(axis=0) - mins    
    scale  = ranges.max()                 
    normalized_coords = (coords - mins) / scale

    centroidArr = [0]*len(normalized_coords)
    for i in range(len(target_gdf)):
        centroidArr[id2idx[target_gdf.iloc[i, 4]]] = [float(normalized_coords[i][0]), float(normalized_coords[i][1])]

    return centroidArr


if __name__ == '__main__':

    tripData = pd.read_parquet('../data/yellow_tripdata_2019-06.parquet')
    zoneData = pd.read_csv('../data/taxi_zone_lookup.csv')
    gdf = gpd.read_file('../data/taxi_zones/taxi_zones.shp')

    processedTripData = bucketize_time(tripData, 'tpep_pickup_datetime', 5)
    processedTripData = bucketize_time(processedTripData, 'tpep_dropoff_datetime', 5)

    manhattanZoneIds = get_zoneName_zoneID(zoneData, 'Borough', 'Manhattan')

    processedTripData = filter_zone_data(processedTripData, manhattanZoneIds)

    demand, destDist, odStats = prepare_data(processedTripData)

    id2idx, idx2id = conversion_dicts(manhattanZoneIds)

    adjacency_dict = get_adjacent_zones(gdf, manhattanZoneIds)

    assert len(adjacency_dict) == len(manhattanZoneIds), "zone count in zone series donot match with adjecency dict count. please check."

    demandArray = get_demand_lookup_table(demand, len(manhattanZoneIds), 24*60//5, idx2id)

    destinationProbabArray = get_destination_lookup_table(destDist, len(manhattanZoneIds), 24*60//5, idx2id, id2idx)

    fareDurationArray = get_fare_and_duration_lookup_table(odStats, len(manhattanZoneIds), idx2id, 10, adjacency_dict)
    
    centroids = get_centroids(gdf, manhattanZoneIds, id2idx)

    np.save('../data/artifacts/demand.npy', demandArray, allow_pickle=True)
    np.save('../data/artifacts/destinationProbab.npy', destinationProbabArray, allow_pickle=True)
    np.save('../data/artifacts/fare_duration.npy', fareDurationArray, allow_pickle=True)
    np.save('../data/artifacts/centroids.npy', centroids, allow_pickle=True)
    
    with open('../data/artifacts/id2idx.json', 'w') as f:
        json.dump(id2idx, f, indent=4)

    with open('../data/artifacts/idx2id.json', 'w') as f:
        json.dump(idx2id, f, indent=4)
    
    with open('../data/artifacts/adjacent_zones.json', 'w') as f:
        json.dump(adjacency_dict, f, indent=4)
    
