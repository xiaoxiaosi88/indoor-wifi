# data_utils.py
import os
import numpy as np
import pandas as pd
import traceback

def load_and_process_data(filename, building_id, floor_id, radgt, no_signal_value=-110.0, rssi_min_value=-100.0):
    """
    Loads, filters, processes fingerprint data and calculates normalization parameters.

    Args:
        filename (str): Path to the CSV data file.
        building_id (int): ID of the building to filter.
        floor_id (int): ID of the floor to filter.
        radgt (float): Target radius used for estimating radius normalization params.
        no_signal_value (float): Value to replace '100' (no signal).
        rssi_min_value (float): Minimum RSSI considered 'detected' for calculating stats.

    Returns:
        tuple: (rssi_data, location_data, map_bounds, norm_params) or (None, None, None, None) on error.
    """
    print(f"Data Utils: Loading data from {filename}...")
    if not os.path.exists(filename):
        print(f"Data Utils Error: Data file '{filename}' not found.")
        return None, None, None, None

    try:
        df = pd.read_csv(filename)
        print(f"Data Utils: Loaded data shape: {df.shape}")
        wap_columns = [col for col in df.columns if col.startswith('WAP')]
        if not wap_columns: raise ValueError("No WAP columns found.")
        num_aps_found = len(wap_columns)
        print(f"Data Utils: Found {num_aps_found} WAP columns.")

        required_meta_cols = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
        if not all(col in df.columns for col in required_meta_cols):
            raise ValueError(f"Missing required columns: {[col for col in required_meta_cols if col not in df.columns]}")

        print(f"Data Utils: Filtering for BUILDINGID {building_id} and FLOOR {floor_id}...")
        filtered_df = df[(df['BUILDINGID'] == building_id) & (df['FLOOR'] == floor_id)].copy()###过滤出想要楼层的列和行
        print(f"Data Utils: Filtered data shape: {filtered_df.shape}")
        if filtered_df.empty: raise ValueError(f"No data found for BUILDINGID {building_id}, FLOOR {floor_id}.")

        print(f"Data Utils: Processing RSSI values...")
        rssi_data = filtered_df[wap_columns].replace(100, no_signal_value).astype(np.float32)##将筛选出来的列中的100替换为-110
        rssi_data[rssi_data > 0] = rssi_min_value # Clamp potential positive RSSI
        rssi_data = np.maximum(rssi_data, no_signal_value) # Ensure no value below no_signal_value
        ##返回一个完整的RSSI数据时设置信号底线，。防止极端异常值

        location_data = filtered_df[['LONGITUDE', 'LATITUDE']].values.astype(np.float32)##将维度和经度提取出来

        print("Data Utils: Calculating map bounds...")
        if location_data.shape[0] > 0:
            xmin, ymin = np.min(location_data, axis=0)
            xmax, ymax = np.max(location_data, axis=0)
            map_bounds = {'xmin': float(xmin), 'xmax': float(xmax), 'ymin': float(ymin), 'ymax': float(ymax)}
            ###将边界范围写入字典
            print(f"Data Utils: Calculated Map Bounds: {map_bounds}")
        else: raise ValueError("Cannot determine map bounds from filtered data.")

        print("Data Utils: Calculating normalization parameters...")
        # Calculate stats only on detected signals in the filtered dataset
        detected_mask = rssi_data.values >= rssi_min_value
        detected_values = rssi_data.values[detected_mask]###将rssi_data的值大于0的赋值给detected_values

        if detected_values.size > 1: # Need at least 2 points for std dev
            rssi_mean = np.mean(detected_values)##计算RSSI的均值和标准差
            rssi_std = np.std(detected_values)
            if rssi_std < 1e-6: rssi_std = 1.0
        else:
            print("Warning: Not enough detected signals to compute robust RSSI stats. Using defaults.")
            rssi_mean = rssi_min_value
            rssi_std = 15.0 # Use a reasonable default std dev like 15dBm

        coords_mean = np.mean(location_data, axis=0).astype(np.float32)##中心的均值和标准差
        coords_std = np.std(location_data, axis=0).astype(np.float32)
        coords_std[coords_std < 1e-6] = 1.0

        # Estimate radius stats (based on config's radgt)
        max_span_calc = max(map_bounds['xmax'] - map_bounds['xmin'], map_bounds['ymax'] - map_bounds['ymin'])
        initial_rad_est = (max_span_calc / 2.0) + radgt
        radius_mean_est = np.exp((np.log(initial_rad_est) + np.log(radgt)) / 2.0) if radgt > 1e-6 and initial_rad_est > 1e-6 else (initial_rad_est + radgt) / 2.0
        ###半径的均值和标准差
        radius_std_est = max((initial_rad_est - radgt) / 4.0, 1e-6)

        norm_params = {
            'rssi_mean': float(rssi_mean), 'rssi_std': float(rssi_std),
            'coords_mean': coords_mean, 'coords_std': coords_std,
            'radius_mean': float(radius_mean_est), 'radius_std': float(radius_std_est)
        }
        print(f"Data Utils: Normalization params calculated: {norm_params}")

        print("Data Utils: Data loading and processing complete.")
        # Return raw RSSI data (agent handles normalization using params) and locations
        return rssi_data.values, location_data, map_bounds, norm_params
    ###返回RSSI有信号的数据，坐标中心位置，房间边界，归一化参数

    except Exception as e:
        print(f"An error occurred during data loading/processing: {e}")
        traceback.print_exc()
        return None, None, None, None