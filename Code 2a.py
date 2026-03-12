
# Code 2

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import tkinter as tk
from tkinter import filedialog

# ==========================================
# GLOBAL BEST MATCH TRACKER (ALL 5 SETS)
# ==========================================

import os
import pandas as pd

# 🔴 UPDATE these bounds from Code 1

RUN_MODE = "final"   # "scan" or "final"

REF_LAT_MIN = 60.50
REF_LAT_MAX = 61.05
REF_LON_MIN = 14.50
REF_LON_MAX = 15.40

data2_path = r"D:\Project Folder\Data 2"  # 
results_file = r"D:\Results\global_best_results.txt"

best_folder_current_set = None
best_score_current_set = 0

print("\nScanning current set...\n")

for folder in os.listdir(data2_path):
    folder_path = os.path.join(data2_path, folder)

    if os.path.isdir(folder_path):

        lat_file = os.path.join(folder_path, "GPS.latitude.csv")
        lon_file = os.path.join(folder_path, "GPS.longitude.csv")

        if os.path.exists(lat_file) and os.path.exists(lon_file):

            try:
                lat = pd.read_csv(lat_file, header=None, usecols=[0], nrows=500000)
                lon = pd.read_csv(lon_file, header=None, usecols=[0], nrows=500000)

                lat_min, lat_max = lat[0].min(), lat[0].max()
                lon_min, lon_max = lon[0].min(), lon[0].max()

                # Overlap logic
                lat_overlap = (lat_max >= REF_LAT_MIN) and (lat_min <= REF_LAT_MAX)
                lon_overlap = (lon_max >= REF_LON_MIN) and (lon_min <= REF_LON_MAX)

                if lat_overlap and lon_overlap:

                    lat_coverage = min(lat_max, REF_LAT_MAX) - max(lat_min, REF_LAT_MIN)
                    lon_coverage = min(lon_max, REF_LON_MAX) - max(lon_min, REF_LON_MIN)

                    score = lat_coverage * lon_coverage

                    print(f"{folder} → Score: {score:.6f}")

                    if score > best_score_current_set:
                        best_score_current_set = score
                        best_folder_current_set = folder

            except Exception as e:
                print(f"Error reading {folder}: {e}")

print("\nBest Folder in This Set:", best_folder_current_set)
print("Best Score in This Set:", best_score_current_set)

# ==========================================
# SAVE & COMPARE WITH GLOBAL BEST
# ==========================================

global_best_score = 0
global_best_folder = None

# If file already exists, read previous best
if os.path.exists(results_file):

    with open(results_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(" | ")
        score = float(parts[2])
        if score > global_best_score:
            global_best_score = score
            global_best_folder = parts[1]

# Append current result
with open(results_file, "a") as f:
    f.write(f"{data2_path} | {best_folder_current_set} | {best_score_current_set}\n")

# Compare current with global
if best_score_current_set > global_best_score:
    global_best_score = best_score_current_set
    global_best_folder = best_folder_current_set

print("\n======================================")
print("GLOBAL BEST ACROSS ALL TESTED SETS:")
print("Folder:", global_best_folder)
print("Score :", global_best_score)
print("======================================\n")

if RUN_MODE == "scan":
    print("\nScan complete. Load next set and run again.\n")
    exit()

print("Now use Tkinter to select files from:")
print(best_folder_current_set)

# =====================================================
# Tkinter File Selection
# =====================================================
if RUN_MODE == "final":

 root = tk.Tk()
 root.withdraw()

 files = {
    "latitude": None,
    "longitude": None,
    "vibration1": None,
    "vibration2": None,
    "speed": None
 }

 def load_file(key):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    if file_path:
        files[key] = file_path
        print(f"{key} file loaded:", file_path)

 print("Select Latitude File")
 load_file("latitude")
 print("Select Longitude File")
 load_file("longitude")
 print("Select Vibration 1 File")
 load_file("vibration1")
 print("Select Vibration 2 File")
 load_file("vibration2")
 print("Select Speed File")
 load_file("speed")

 # =====================================================
 # Load Data
 # =====================================================
 dataframes = {}

 for key, path in files.items():
    if path:
        df = pd.read_csv(path, header=None, names=[key])
        df["index"] = df.index
        dataframes[key] = df
    else:
        print(f"{key} file missing")


 # =====================================================
 # Merge GPS
 # =====================================================
 if "latitude" in dataframes and "longitude" in dataframes:
    df_gps = pd.merge(
        dataframes["latitude"],
        dataframes["longitude"],
        on="index"
    )

    df_gps.rename(columns={
        "latitude": "Latitude",
        "longitude": "Longitude"
    }, inplace=True)

    df_gps["PointIndex"] = df_gps.index
    #Create GPS timestamps
    gps_dt = 0.05  
    t0 = 0  # starting time
    df_gps["Timestamp"] = t0 + df_gps["PointIndex"] * gps_dt

    print("\nGPS timestamps created")
    print(df_gps[["PointIndex", "Timestamp"]].head())

  
    # =====================================================
    # GPS NOISE FILTERING
    # =====================================================

    # 1️⃣ Remove invalid coordinates
    df_gps = df_gps[
     (df_gps["Latitude"].between(-90, 90)) &
     (df_gps["Longitude"].between(-180, 180))
    ]

    df_gps = df_gps.dropna(subset=["Latitude","Longitude"])

    # Sort by time
    df_gps = df_gps.sort_values("Timestamp")

    # =====================================================
    # Haversine distance function
    # =====================================================

    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
      R = 6371000
    
      lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
      dlat = lat2 - lat1
      dlon = lon2 - lon1
    
      a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
      c = 2 * atan2(sqrt(a), sqrt(1-a))
    
      return R * c

    # =====================================================
    # Detect GPS jumps
    # =====================================================

    distances = [0]

    for i in range(1, len(df_gps)):

      d = haversine(
        df_gps.iloc[i-1]["Latitude"],
        df_gps.iloc[i-1]["Longitude"],
        df_gps.iloc[i]["Latitude"],
        df_gps.iloc[i]["Longitude"]
      )

      distances.append(d)

    df_gps["gps_distance"] = distances

    # =====================================================
    # Remove abnormal GPS jumps
    # =====================================================

    df_gps = df_gps[df_gps["gps_distance"] < 100]

    # =====================================================
    # Smooth GPS coordinates
    # =====================================================

    df_gps["Latitude"] = df_gps["Latitude"].rolling(5, center=True).mean()
    df_gps["Longitude"] = df_gps["Longitude"].rolling(5, center=True).mean()

    df_gps = df_gps.dropna()
    # Reset index after removing rows
    df_gps = df_gps.reset_index(drop=True)

    df_gps["PointIndex"] = df_gps.index

    print("GPS noise filtering applied")
    print("Remaining GPS points:", len(df_gps))

 else:
    df_gps = pd.DataFrame()
    print("GPS data missing")

 # =====================================================
 # Merge Vibration
 # =====================================================
 if "vibration1" in dataframes and "vibration2" in dataframes:
    df_vibration = pd.merge(
        dataframes["vibration1"],
        dataframes["vibration2"],
        on="index"
    )
    # Create vibration timestamps
    vib_dt = 0.002 
    df_vibration["Timestamp"] = t0 + df_vibration["index"] * vib_dt
    print("\nVibration timestamps created")
    print(df_vibration[["index", "Timestamp"]].head())
   
 else:
    df_vibration = pd.DataFrame()
    print("Vibration data missing")
 # =====================================
 # CHECK TIME RANGE ALIGNMENT
 # =====================================

 print("\nGPS time range:")
 print(df_gps["Timestamp"].min(), "to", df_gps["Timestamp"].max())

 print("\nVibration time range:")
 print(df_vibration["Timestamp"].min(), "to", df_vibration["Timestamp"].max())
 print("\nChecking vibration samples per GPS interval...")

 for i in range(5):

    gps_time = df_gps.loc[i, "Timestamp"]

    vib_window = df_vibration[
        (df_vibration["Timestamp"] >= gps_time) &
        (df_vibration["Timestamp"] < gps_time + gps_dt)
    ]

    print(f"GPS index {i} → vibration samples:", len(vib_window))

 # =====================================================
 # Sampling Rates (Instructor Method)
 # =====================================================
 gps_dt = 0.05      # 20 Hz
 vib_dt = 0.002     # 500 Hz

 samples_per_gps = int(gps_dt / vib_dt)  # Should be 25
 print("Vibration samples per GPS point:", samples_per_gps)
 # Create vibration index mapping
 df_gps["vibration_index"] = (df_gps["PointIndex"] * samples_per_gps).astype(int)



 # =====================================================
 # Create Map
 # =====================================================
 if not df_gps.empty:
    map_fig = px.scatter_mapbox(
        df_gps,
        lat="Latitude",
        lon="Longitude",
        custom_data=["PointIndex"],
        zoom=10, 
        title="GPS Points (Click to View Vibration Window)"
    )
    map_fig.update_layout(mapbox_style="open-street-map", height=600)
 else:
    map_fig = go.Figure()
    map_fig.update_layout(title="No GPS Data")

 # Empty vibration plot
 vib_empty_fig = go.Figure()
 vib_empty_fig.update_layout(
    title="Vibration Signal",
    xaxis_title="Time (s)",
    yaxis_title="Acceleration"
 )
 
 # Filtering Infrastructure 
 # ===============================
 # Load Infrastructure Dataset (from Code 1)
 # ===============================
 infra_path = r"D:\Project Folder\Data 1\infrastructure_combined.csv"
 df_infra = pd.read_csv(infra_path)
 print("Infrastructure dataset loaded:", len(df_infra), "rows")
 print(df_infra.head())

 from sklearn.neighbors import BallTree
 import numpy as np

 # Convert to radians
 gps_coords = np.radians(df_gps[["Latitude", "Longitude"]])
 infra_coords = np.radians(df_infra[["Latitude", "Longitude"]])

 # Build tree
 tree = BallTree(gps_coords, metric='haversine')

 # Query nearest GPS point
 distances, indices = tree.query(infra_coords, k=1)

 # Convert to meters
 earth_radius = 6371000
 distances_m = distances.flatten() * earth_radius

 df_infra["DistanceToTrack"] = distances_m

 # Keep only close points
 threshold = 10  # meters
 df_infra_filtered = df_infra[df_infra["DistanceToTrack"] < threshold]

 print("Original infra points:", len(df_infra))
 print("Filtered infra points:", len(df_infra_filtered))
 
 
 # =====================================================
 # Labelling GPS Data
 # =====================================================
 infra_coords_filtered = np.radians(df_infra_filtered[["Latitude", "Longitude"]])
 infra_tree = BallTree(infra_coords_filtered, metric='haversine')
 gps_coords = np.radians(df_gps[["Latitude", "Longitude"]])

 distances, indices = infra_tree.query(gps_coords, k=1)

 distances_m = distances.flatten() * earth_radius

 df_gps["Label"] = "Other"

 event_threshold = 5  # meters

 for i, dist in enumerate(distances_m):
    if dist < event_threshold:
        df_gps.loc[i, "Label"] = df_infra_filtered.iloc[indices[i][0]]["Category"]

 print("\n===== GPS LABEL SUMMARY =====")
 print(df_gps["Label"].value_counts())

 print("\nSample GPS labeled rows:")
 print(df_gps[df_gps["Label"] != "Other"].head())   

 # =====================================================
 # EVENT-CENTERED VIBRATION SEGMENTATION - after augmentation
 # =====================================================

 print("\nCreating event-centered vibration segments...")
 window = 2
 samples_window = int(window / vib_dt)

 step = int(0.5 / vib_dt)   # 0.5 sec shift between windows

 event_segments = []
 event_labels = []

 for i, row in df_gps.iterrows():

    label = row["Label"]

    vib_center = int(row["Timestamp"] / vib_dt)

    start_base = vib_center - samples_window
    end_base = vib_center + samples_window

    if start_base < 0 or end_base > len(df_vibration):
        continue

    # Sliding windows around event
    for shift in range(-samples_window, samples_window, step):

        start = start_base + shift
        end = end_base + shift

        if start < 0 or end > len(df_vibration):
            continue

        segment = df_vibration.iloc[start:end]

        event_segments.append(segment)
        event_labels.append(label)

 print("\n===== EVENT SEGMENT SUMMARY =====")
 print(pd.Series(event_labels).value_counts())
 
 # =====================================================
 #  UNDER SAMPLING "OTHERS"
 # =====================================================
 import random

 max_other = 50

 other_indices = [i for i,l in enumerate(event_labels) if l == "Other"]

 if len(other_indices) > max_other:

    keep_other = random.sample(other_indices, max_other)

    new_segments = []
    new_labels = []

    for i,(seg,lab) in enumerate(zip(event_segments,event_labels)):

        if lab != "Other" or i in keep_other:
            new_segments.append(seg)
            new_labels.append(lab)

    event_segments = new_segments
    event_labels = new_labels
 print("Total event segments after balancing:", len(event_segments))

 print("\n===== EVENT SEGMENT SUMMARY AFTER BALANCING =====")
 print(pd.Series(event_labels).value_counts())
 
 # =====================================================
 #  FEATURE EXTRACTION CODE
 # =====================================================
 
 print("\nExtracting vibration features...")
 from scipy.stats import kurtosis, skew, entropy
 from scipy.signal import welch
 from scipy.signal import correlate

 features = []
 labels = []
 sampling_rate=500

 # =====================================================
 # Hjorth Parameters
 # =====================================================
 def hjorth_parameters(signal):

    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var0 = np.var(signal)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)

    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility

    return mobility, complexity


 # =====================================================
 # Teager Energy
 # =====================================================
 def teager_energy(x):
    return np.mean(x[1:-1]**2 - x[:-2]*x[2:])


 for segment, label in zip(event_segments, event_labels):

    ch1 = (segment["vibration1"] - segment["vibration1"].mean()).values
    ch2 = (segment["vibration2"] - segment["vibration2"].mean()).values

    feat = {}

    # Channel 1 features
    feat["ch1_mean"] = np.mean(ch1)
    feat["ch1_std"] = np.std(ch1)
    feat["ch1_rms"] = np.sqrt(np.mean(ch1**2))
    feat["ch1_max"] = np.max(ch1)
    feat["ch1_min"] = np.min(ch1)
    feat["ch1_ptp"] = np.ptp(ch1)
    feat["ch1_kurtosis"] = kurtosis(ch1)
    feat["ch1_skewness"] = skew(ch1)
    feat["ch1_crest_factor"] = np.max(np.abs(ch1)) / feat["ch1_rms"]

   

    # Channel 2 features
    feat["ch2_mean"] = np.mean(ch2)
    feat["ch2_std"] = np.std(ch2)
    feat["ch2_rms"] = np.sqrt(np.mean(ch2**2))
    feat["ch2_max"] = np.max(ch2)
    feat["ch2_min"] = np.min(ch2)
    feat["ch2_ptp"] = np.ptp(ch2)
    feat["ch2_kurtosis"] = kurtosis(ch2)
    feat["ch2_skewness"] = skew(ch2)
    feat["ch2_crest_factor"] = np.max(np.abs(ch2)) / feat["ch2_rms"]

    # =============================
    # Hjorth Parameters
    # =============================
    mob1, comp1 = hjorth_parameters(ch1)
    mob2, comp2 = hjorth_parameters(ch2)

    feat["ch1_hjorth_mobility"] = mob1
    feat["ch1_hjorth_complexity"] = comp1
    feat["ch2_hjorth_mobility"] = mob2
    feat["ch2_hjorth_complexity"] = comp2


    # =============================
    # Entropy
    # =============================
    hist1, _ = np.histogram(ch1, bins=50, density=True)
    hist2, _ = np.histogram(ch2, bins=50, density=True)

    feat["ch1_entropy"] = entropy(hist1 + 1e-12)
    feat["ch2_entropy"] = entropy(hist2 + 1e-12)


  # =============================
  # Teager Energy
  # =============================
    feat["ch1_teager_energy"] = teager_energy(ch1)
    feat["ch2_teager_energy"] = teager_energy(ch2)


    features.append(feat)
    labels.append(label)

    # =====================================================
    # Cross Channel Features
    # =====================================================

    # Pearson Correlation
    feat["ch1_ch2_corr"] = np.corrcoef(ch1, ch2)[0,1]

 
    # Cross Correlation Peak
    xcorr = correlate(ch1, ch2, mode='full')
    feat["ch1_ch2_xcorr_max"] = np.max(np.abs(xcorr))


   # Energy Ratio
    energy1 = np.sum(ch1**2)
    energy2 = np.sum(ch2**2)

    feat["energy_ratio_ch1_ch2"] = energy1 / (energy2 + 1e-12)

 df_features = pd.DataFrame(features)
 df_features["Label"] = labels

 print(df_features.head())

 # =====================================================
 #  PREPARING ML Dataset
 #  =====================================================

 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import LabelEncoder


 X = df_features.drop("Label", axis=1)
 y = df_features["Label"]

 encoder = LabelEncoder()
 y_encoded = encoder.fit_transform(y)

 X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42,stratify = y)

 from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler() 

 X_train = scaler.fit_transform(X_train)
 X_test = scaler.transform(X_test)
 
 # =====================================================
 #  Extra Trees Model
 #  =====================================================
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.metrics import classification_report
 extra_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
 )

 extra_model.fit(X_train, y_train)
 y_pred_extra = extra_model.predict(X_test)

 print("\nExtra Trees Results")
 print(classification_report(y_test, y_pred_extra))

 # =====================================================
 # FEATURE SELECTION METHODS
 # =====================================================

 from sklearn.feature_selection import SelectKBest
 from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
 from sklearn.preprocessing import LabelEncoder, MinMaxScaler

 X = df_features.drop("Label", axis=1)
 y = df_features["Label"]

 encoder = LabelEncoder()
 y_encoded = encoder.fit_transform(y)

 # Chi-square requires positive values
 scaler = MinMaxScaler()
 X_scaled = scaler.fit_transform(X)

 # ----------------------------
 # Chi-square
 # ----------------------------
 chi_selector = SelectKBest(score_func=chi2, k=20)
 chi_selector.fit(X_scaled, y_encoded)
 chi_features = X.columns[chi_selector.get_support()]

 print("\nChi-Square Selected Features:")
 print(chi_features)

 # ----------------------------
 # ANOVA F-test
 # ----------------------------
 anova_selector = SelectKBest(score_func=f_classif, k=20)
 anova_selector.fit(X, y_encoded)
 anova_features = X.columns[anova_selector.get_support()]

 print("\nANOVA Selected Features:")
 print(anova_features)

 # ----------------------------
 # Mutual Information
 # ----------------------------
 mi_selector = SelectKBest(score_func=mutual_info_classif, k=20)
 mi_selector.fit(X, y_encoded)
 mi_features = X.columns[mi_selector.get_support()]

 print("\nMutual Information Selected Features:")
 print(mi_features)

 # =====================================================
 # FUNCTION TO TRAIN EXTRA TREES
 # =====================================================

 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.metrics import classification_report
 import seaborn as sns
 import matplotlib.pyplot as plt
 from sklearn.metrics import confusion_matrix
 from sklearn.metrics import accuracy_score

 def run_extra_trees(selected_features, method_name):

    print("\n===============================")
    print("Extra Trees using:", method_name)
    print("===============================")

    X = df_features[selected_features]
    y = df_features["Label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    extra_accuracy = accuracy_score(y_test,y_pred)

    # ================================
    # Confusion Matrix
    # ================================
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title(f"Confusion Matrix - Extra Trees ({method_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.show()

    # ================================
    # Feature Importance
    # ================================

    feature_importance = pd.Series(
        model.feature_importances_,
        index=selected_features
    )

    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 15 Important Features:")
    print(feature_importance.head(15))

    plt.figure(figsize=(12,8))
    top_features = feature_importance.head(15).sort_values()
    top_features.plot(kind="barh")

    #feature_importance.head(15).plot(kind="bar")

    #plt.title(f"Top 15 Features - Extra Trees ({method_name})")
    #plt.xlabel("Features")
    #plt.ylabel("Importance Score")
    plt.title(f"Top 15 Features - Extra Trees ({method_name})", fontsize=16)
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Features", fontsize=14)

    #plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)

    plt.tight_layout()
   
    plt.show()
    plt.savefig("top_features_chi_square.png", dpi=300)
    return extra_accuracy
 # =====================================================
 # RUN MODELS
 # =====================================================

 chi_extra_accuracy=run_extra_trees(chi_features, "Chi-Square Features")

 run_extra_trees(anova_features, "ANOVA Features")

 run_extra_trees(mi_features, "Mutual Information Features")
 

 #671 -781 Higher accuracy code for Extra classifiers commented
 # =====================================================
 #  RANDOM FOREST
 #  =====================================================
 
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import classification_report
 np.random.seed(42)
 random.seed(42)

 rf = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state =42)

 rf.fit(X_train, y_train)

 y_pred = rf.predict(X_test)

 print("\nRandom Forest Results")
 print(classification_report(y_test, y_pred, target_names=encoder.classes_))
 rf_accuracy = accuracy_score(y_test,y_pred)
 
 
 # =====================================================
 #  SVM
 #  =====================================================
 
 from sklearn.svm import SVC

 svm = SVC(kernel='rbf', C=10, gamma='scale',class_weight='balanced')

 svm.fit(X_train, y_train)

 y_pred = svm.predict(X_test)

 print("\nSVM Results-Normal")
 print(classification_report(y_test, y_pred, target_names=encoder.classes_))
 svm_accuracy = accuracy_score(y_test, y_pred)


 # =====================================================
 #  KNN
 #  =====================================================
 from sklearn.neighbors import KNeighborsClassifier

 knn = KNeighborsClassifier(n_neighbors=5,weights='distance')

 knn.fit(X_train, y_train)

 y_pred = knn.predict(X_test)

 print("\nKNN Results-Normal")
 print(classification_report(y_test, y_pred, target_names=encoder.classes_))
 knn_accuracy = accuracy_score(y_test,y_pred)
 
 # =====================================================
 # CNN MODEL FOR VIBRATION CLASSIFICATION
 # =====================================================

 import tensorflow as tf

 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler, LabelEncoder
 Sequential = tf.keras.models.Sequential
 Conv1D = tf.keras.layers.Conv1D
 MaxPooling1D = tf.keras.layers.MaxPooling1D
 Flatten = tf.keras.layers.Flatten
 Dense = tf.keras.layers.Dense
 Dropout = tf.keras.layers.Dropout
 to_categorical = tf.keras.utils.to_categorical

 # =====================================================
 # PREPARE DATA
 # =====================================================

 X = df_features.drop("Label", axis=1)
 y = df_features["Label"]

 encoder = LabelEncoder()
 y_encoded = encoder.fit_transform(y)

 # Convert labels to categorical (for CNN)
 y_categorical = to_categorical(y_encoded)

 # Train/Test split
 X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y
 )

 # Normalize features
 scaler = StandardScaler()

 X_train = scaler.fit_transform(X_train)
 X_test = scaler.transform(X_test)

 # CNN requires 3D input: (samples, features, channels)

 X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
 X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

 print("Training shape:", X_train.shape)

 # =====================================================
 # BUILD CNN MODEL
 # =====================================================

 model = Sequential()

 model.add(Conv1D(filters=32, kernel_size=3, activation='relu',
                 input_shape=(X_train.shape[1],1)))

 model.add(MaxPooling1D(pool_size=2))

 model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

 model.add(MaxPooling1D(pool_size=2))

 model.add(Flatten())

 model.add(Dense(128, activation='relu'))

 model.add(Dropout(0.3))

 model.add(Dense(y_categorical.shape[1], activation='softmax'))

 model.summary()

 # =====================================================
 # COMPILE MODEL
 # =====================================================

 model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
 )

 # =====================================================
 # TRAIN MODEL
 # =====================================================
 from tensorflow.keras.callbacks import EarlyStopping

 early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
 )
 history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
 )

 # =====================================================
 # CLASSIFICATION REPORT
 # =====================================================

 from sklearn.metrics import classification_report
 import numpy as np
 from sklearn.metrics import accuracy_score

 y_pred = model.predict(X_test)

 y_pred_classes = np.argmax(y_pred, axis=1)
 y_test_classes = np.argmax(y_test, axis=1)
 cnn_accuracy = accuracy_score(y_test_classes, y_pred_classes)

 print(classification_report(y_test_classes, y_pred_classes,
                            target_names=encoder.classes_))
 
 # =====================================================
 # TRAINING CURVES
 # =====================================================

 import matplotlib.pyplot as plt

 plt.plot(history.history['accuracy'])
 plt.plot(history.history['val_accuracy'])

 plt.title("CNN Accuracy")
 plt.ylabel("Accuracy")
 plt.xlabel("Epoch")
 plt.legend(['Train','Validation'])
 plt.show()

 # =====================================================
 # Accuracy Comparison Chart (ML + CNN)
 # =====================================================

 import matplotlib.pyplot as plt

 models = [
    "Random Forest",
    "SVM",
    "KNN",
    "Extra Trees",
    "CNN"
 ]

 accuracies = [
    rf_accuracy,
    svm_accuracy,
    knn_accuracy,
    chi_extra_accuracy,
    cnn_accuracy
 ]

 plt.figure(figsize=(9,6))

 colors = ["steelblue","steelblue","steelblue","steelblue","darkorange"]
 bars = plt.bar(models, accuracies,color=colors)

 plt.title("Accuracy Comparison of ML and Deep Learning Models", fontsize=16)
 plt.xlabel("Models", fontsize=12)
 plt.ylabel("Accuracy", fontsize=12)
 
 plt.ylim(0,1)

 # Show accuracy values on top
 for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha='center',
        fontsize=11
    )

 plt.tight_layout()
 plt.show()

 # =====================================================
 #  VISUALIZE EVENT VIBRATION SEGMENTS
 #  =====================================================

 import matplotlib.pyplot as plt
 def analyze_segment(segment, label):

    ch1 = segment["vibration1"] - segment["vibration1"].mean()

    print(f"\n--- {label} ---")
    print("RMS:", np.sqrt(np.mean(ch1**2)))
    print("STD:", np.std(ch1))
    print("Max:", np.max(ch1))
    print("Min:", np.min(ch1))
    print("\nVisualizing event segments...")

 event_types = ["Bridge", "Turnout", "RailJoint"]

 for event in event_types:

     event_indices = [i for i, label in enumerate(event_labels) if label == event]

     if len(event_indices) == 0:
         print(f"No segments found for {event}")
         continue

     print(f"\nShowing first segment for {event}")

     # Take first segment of that event
     idx = event_indices[0]
     segment = event_segments[idx]

     ch1 = segment["vibration1"] - segment["vibration1"].mean()
     ch2 = segment["vibration2"] - segment["vibration2"].mean()
     time_axis = np.arange(len(segment)) * vib_dt

     #plt.figure(figsize=(12, 4))
     #plt.plot(time_axis, segment["vibration1"], label="Channel 1")
     #plt.plot(time_axis, segment["vibration2"], label="Channel 2")
  
    
     plt.figure(figsize=(12, 4))
     plt.plot(time_axis, ch1)
     plt.title(f"{event} - Channel 1 (Segment {idx})")
     plt.xlabel("Time (seconds)")
     plt.ylabel("Acceleration")
     plt.show()

     plt.figure(figsize=(12, 4))
     plt.plot(time_axis, ch2)
     plt.title(f"{event} - Channel 2 (Segment {idx})")
     plt.xlabel("Time (seconds)")
     plt.ylabel("Acceleration")
     plt.show()
     analyze_segment(segment, event)
     
 # =====================================================
 # Dash App
 # =====================================================
 app = dash.Dash(__name__)

 app.layout = html.Div([
    html.Div([
        dcc.Graph(id="gps-map", figure=map_fig)
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id="vibration-plot", figure=vib_empty_fig)
    ], style={'width': '48%', 'display': 'inline-block'})
 ])

 # =====================================================
 # Correct Time-Based Linking Callback
 # =====================================================
 @app.callback(
    Output("vibration-plot", "figure"),
    Input("gps-map", "clickData")
 )
 def update_vibration_plot(clickData):

    if clickData is None:
        return vib_empty_fig

    try:
        gps_index = clickData["points"][0]["pointIndex"]
    except:
        return vib_empty_fig

    if df_vibration.empty:
        return vib_empty_fig

    # GPS timestamp
    gps_time = gps_index * gps_dt

    # Corresponding vibration indices
    vib_start = int(gps_time / vib_dt)
    vib_end = vib_start + samples_per_gps

    if vib_end > len(df_vibration):
        vib_end = len(df_vibration)

    segment = df_vibration.iloc[vib_start:vib_end]

    if segment.empty:
        return vib_empty_fig

    time_axis = np.arange(len(segment)) * vib_dt

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=segment["vibration1"],
        mode="lines",
        name="Vibration Channel 1"
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=segment["vibration2"],
        mode="lines",
        name="Vibration Channel 2"
    ))

    fig.update_layout(
        title=f"Vibration Window for GPS Index {gps_index} (0.05s)",
        xaxis_title="Time within GPS window (s)",
        yaxis_title="Acceleration"
    )

    return fig

 # =====================================================
 # Run App
 # =====================================================
 if __name__ == "__main__":
    app.run(debug=True, port=8060,use_reloader=False)

