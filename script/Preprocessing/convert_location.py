# 使用するライブラリのインポート

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiLineString

from tqdm import tqdm
tqdm.pandas()

# Pandarallelの準備
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import seaborn as sns
import matplotlib.pyplot as plt

TRAINING_PATH = '../data/npy/train/'
VAL_PATH = '../data/npy/validate/'
TEST_PATH = '../data/npy/test/'

TRAINING_OUTPUT_PATH = '../output/train/'
VAL_OUTPUT_PATH = '../output/validate/'
TEST_OUTPUT_PATH = '../output/test/'

# ロケーションデータの読み込み
training_location = np.load(TRAINING_PATH + 'Hand/Location.npy')
training_location = pd.DataFrame(training_location[:, [0,3,4,5,6]], columns=['EpochTime', 'Accuracy', 'Latitude', 'Longitude', 'Altitude'])

training_location['EpochTime'] = pd.to_datetime(training_location['EpochTime'], unit='ms')
training_location = training_location.set_index('EpochTime')

# ## 位置のラグ特徴量をとる
training_location_lag = training_location.copy()
training_location_lag.index = training_location.index - pd.Timedelta(seconds=10)

training_location = pd.merge_asof(training_location, training_location_lag, left_index=True, right_index=True, suffixes=('', '_lag'), tolerance=pd.Timedelta(seconds=11))

# ## バス
bus_routes = gpd.read_file('../data/bus_stop.json')
bus_routes = MultiLineString(bus_routes.loc[:, 'geometry'].to_list())
def min_distance_bus_routes(location, bus_routes):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return point.distance(bus_routes)

bus_stops = gpd.read_file('../data/bus_stops.json')
def min_distance_bus_stops(location, bus_stops):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(bus_stops['geometry']))

# ## 電車
railways = gpd.read_file('../data/railways_by_overpass.json')
def min_distance_railway_routes(location, railways):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(railways['geometry']))

def min_distance_railway_station(location, railways):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(railways[railways.geometry.type == 'Point']['geometry']))

# ## 地下鉄
subways = gpd.read_file('../data/subway_by_overpass.json')
def min_distance_subway_routes(location, subways):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(subways['geometry']))

def min_distance_subway_station(location, subways):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(subways[subways.geometry.type == 'Point']['geometry']))

# ## 車
car_road = gpd.read_file('../data/car_road.json')
def min_distance_car_routes(location, car_road):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(car_road['geometry']))

# ## 最寄りの信号
traffic_signals = gpd.read_file('../data/traffic_signals.json')
def min_distance_traffic_signals(location, traffic_signals):
    from shapely.geometry import Point
    point = Point(location['Longitude'], location['Latitude'])
    return min(point.distance(traffic_signals['geometry']))

# ## Speed、Direction
import sys
sys.path.append('../script/')
from surveryPackage import Surverycalc
surveryCalc = Surverycalc.Surverycalc()

def calc_speed(location, surveryCalc):
    return surveryCalc.distance2P([location['Latitude_lag'], location['Longitude_lag']], [location['Latitude'], location['Longitude'],]) * 0.36

def calc_direction(location, surveryCalc):
    return surveryCalc.direction2P([location['Latitude_lag'], location['Longitude_lag']], [location['Latitude'], location['Longitude'],])

def exec_convert_location():
    training_location['distance_bus_routes'] = training_location.parallel_apply(min_distance_bus_routes, bus_routes=bus_routes, axis=1)
    training_location['distance_bus_stops'] = training_location.parallel_apply(min_distance_bus_stops, bus_stops=bus_stops, axis=1)
    training_location['distance_railways'] = training_location.parallel_apply(min_distance_railway_routes, railways=railways, axis=1)
    training_location['distance_railways_station'] = training_location.parallel_apply(min_distance_railway_station, railways=railways, axis=1)
    training_location['distance_subways'] = training_location.parallel_apply(min_distance_subway_routes, subways=subways, axis=1)
    training_location['distance_subways_station'] = training_location.parallel_apply(min_distance_subway_station, subways=subways, axis=1)
    training_location['distance_car_roads'] = training_location.parallel_apply(min_distance_car_routes, car_road=car_road, axis=1)
    training_location['distance_traffic_signals'] = training_location.parallel_apply(min_distance_traffic_signals, traffic_signals=traffic_signals, axis=1)
    training_location['speed'] = training_location.parallel_apply(calc_speed, surveryCalc=surveryCalc, axis=1)
    training_location['direction'] = training_location.parallel_apply(calc_direction, surveryCalc=surveryCalc, axis=1)
    training_location.to_csv(TRAINING_OUTPUT_PATH + 'Hand/location_feature.csv')

    # ## Validationデータにも処理する
    # ロケーションデータの読み込み
    val_location = np.load(VAL_PATH + 'Hand/Location.npy')
    val_location = pd.DataFrame(val_location[:, [0,3,4,5,6]], columns=['EpochTime', 'Accuracy', 'Latitude', 'Longitude', 'Altitude'])

    thresh = 1.5e+8
    np.concatenate([[0], np.where(np.abs(np.diff(val_location['EpochTime'])) > thresh)[0] + 1])

    val_location['index'] = val_location.index
    val_location['EpochTime_timestamp'] = pd.to_datetime(val_location['EpochTime'], unit='ms')
    val_location.loc[120178:, 'EpochTime_timestamp'] += pd.Timedelta(days=10000)
    val_location = val_location.set_index('EpochTime_timestamp')
    val_location = val_location.sort_values('EpochTime_timestamp')

    val_location_lag = val_location.copy()
    val_location_lag.index = val_location.index - pd.Timedelta(seconds=10)
    val_location = pd.merge_asof(val_location, val_location_lag, left_index=True, right_index=True, suffixes=('', '_lag'), tolerance=pd.Timedelta(seconds=11))

    val_location['distance_bus_routes'] = val_location.parallel_apply(min_distance_bus_routes, bus_routes=bus_routes, axis=1)
    val_location['distance_bus_stops'] = val_location.parallel_apply(min_distance_bus_stops, bus_stops=bus_stops, axis=1)
    val_location['distance_railways'] = val_location.parallel_apply(min_distance_railway_routes, railways=railways, axis=1)
    val_location['distance_railways_station'] = val_location.parallel_apply(min_distance_railway_station, railways=railways, axis=1)
    val_location['distance_subways'] = val_location.parallel_apply(min_distance_subway_routes, subways=subways, axis=1)
    val_location['distance_subways_station'] = val_location.parallel_apply(min_distance_subway_station, subways=subways, axis=1)
    val_location['distance_car_roads'] = val_location.parallel_apply(min_distance_car_routes, car_road=car_road, axis=1)
    val_location['distance_traffic_signals'] = val_location.parallel_apply(min_distance_traffic_signals, traffic_signals=traffic_signals, axis=1)
    val_location['speed'] = val_location.parallel_apply(calc_speed, surveryCalc=surveryCalc, axis=1)
    val_location['direction'] = val_location.parallel_apply(calc_direction, surveryCalc=surveryCalc, axis=1)

    val_location = val_location.sort_values('index') # ソートし直す
    val_location = val_location.set_index('index')

    val_location['EpochTime_timestamp'] = pd.to_datetime(val_location['EpochTime'], unit='ms')

    val_location.to_csv(VAL_OUTPUT_PATH + 'Hand/location_feature.csv')


    # ## Testデータも処理する
    # TestデータはTrainingデータと同じようにできて楽ですね

    # ロケーションデータの読み込み
    test_location = np.load(TEST_PATH + 'Location.npy')
    test_location = pd.DataFrame(test_location[:, [0,3,4,5,6]], columns=['EpochTime', 'Accuracy', 'Latitude', 'Longitude', 'Altitude'])

    test_location['EpochTime'] = pd.to_datetime(test_location['EpochTime'], unit='ms')
    test_location = test_location.set_index('EpochTime')

    # ## ラグ特徴量

    test_location_lag = test_location.copy()
    test_location_lag.index = test_location.index - pd.Timedelta(seconds=10)
    test_location = pd.merge_asof(test_location, test_location_lag, left_index=True, right_index=True, suffixes=('', '_lag'), tolerance=pd.Timedelta(seconds=11))

    test_location['distance_bus_routes'] = test_location.parallel_apply(min_distance_bus_routes, bus_routes=bus_routes, axis=1)
    test_location['distance_bus_stops'] = test_location.parallel_apply(min_distance_bus_stops, bus_stops=bus_stops, axis=1)
    test_location['distance_railways'] = test_location.parallel_apply(min_distance_railway_routes, railways=railways, axis=1)
    test_location['distance_railways_station'] = test_location.parallel_apply(min_distance_railway_station, railways=railways, axis=1)
    test_location['distance_subways'] = test_location.parallel_apply(min_distance_subway_routes, subways=subways, axis=1)
    test_location['distance_subways_station'] = test_location.parallel_apply(min_distance_subway_station, subways=subways, axis=1)
    test_location['distance_car_roads'] = test_location.parallel_apply(min_distance_car_routes, car_road=car_road, axis=1)
    test_location['distance_traffic_signals'] = test_location.parallel_apply(min_distance_traffic_signals, traffic_signals=traffic_signals, axis=1)

    test_location.to_csv(TEST_OUTPUT_PATH + 'location_feature_distance_traffic_signals.csv')

    test_location['speed'] = test_location.parallel_apply(calc_speed, surveryCalc=surveryCalc, axis=1)
    test_location['direction'] = test_location.parallel_apply(calc_direction, surveryCalc=surveryCalc, axis=1)

    test_location.to_csv(TEST_OUTPUT_PATH + 'location_feature.csv')
