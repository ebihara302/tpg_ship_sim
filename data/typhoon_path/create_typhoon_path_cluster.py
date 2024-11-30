import csv
import os

import matplotlib.cm as cm
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

# 初期入力
# 読み込み関連
start_year = 2012
end_year = 2023
# folder_path = "typhoon_path"
# 出力関連
output_folder = "cluster_center_trajectories"
os.makedirs(output_folder, exist_ok=True)  # 保存するフォルダを作成


### メイン処理 ###


def load_and_filter_typhoon_data(start_year, end_year):
    """
    指定された年の範囲の台風データを読み込み、条件を満たす経路を持つ台風のデータのみを収集する。

    Parameters:
        start_year (int): 読み込み開始年
        end_year (int): 読み込み終了年
        folder_path (str): データファイルが保存されているフォルダのパス

    Returns:
        pl.DataFrame: フィルタリングされた台風データ
    """
    filtered_data = []
    valid_typhoon_numbers = set()

    # 指定された範囲のファイルを処理
    for year in range(start_year, end_year + 1):
        file_path = f"typhoon_data_{year}_6_interval.csv"

        try:
            df = pl.read_csv(file_path)

            # 条件を満たすTYPHOON NUMBERを取得
            condition = (df["LAT"] >= 30) & (df["LON"] <= 146)
            typhoon_numbers = df.filter(condition)["TYPHOON NUMBER"].unique()
            valid_typhoon_numbers.update(typhoon_numbers)  # 条件を満たす台風番号を収集

        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping.")

    # 年平均の台風数を計算
    num_years = end_year - start_year + 1
    avg_typhoon_count = len(valid_typhoon_numbers) / num_years
    print(f"Average number of typhoons per year: {avg_typhoon_count:.2f}")

    # 収集したTYPHOON NUMBERでデータを追記
    for year in range(start_year, end_year + 1):
        file_path = f"typhoon_data_{year}_6_interval.csv"

        for ty_num in valid_typhoon_numbers:
            try:
                df = pl.read_csv(file_path)
                # 必要な列だけを選択
                df = df.select(["TYPHOON NUMBER", "LAT", "LON", "YEAR", "MONTH"])
                filtered_data.append(df.filter(df["TYPHOON NUMBER"] == ty_num))

            except FileNotFoundError:
                continue

    # フィルタリングされたデータを結合
    result_df = pl.concat(filtered_data) if filtered_data else pl.DataFrame()
    return result_df


# 軌跡を等長にリサンプリングする関数
def resample_trajectory(traj, num_points=100):
    """軌跡を等間隔でリサンプリング"""
    traj = np.array(traj)
    distances = np.cumsum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # 距離の累積
    uniform_distances = np.linspace(0, distances[-1], num_points)
    interp_func = interp1d(distances, traj, axis=0)
    return interp_func(uniform_distances)


# クラスタリング後の平均軌跡を計算する関数
def compute_cluster_centroids(trajectories, labels, num_points=100):
    """各クラスタの平均軌跡を計算"""
    centroids = []
    for cluster_id in np.unique(labels):
        cluster_trajectories = [
            trajectories[i] for i in range(len(labels)) if labels[i] == cluster_id
        ]
        resampled_trajectories = np.array(
            [resample_trajectory(traj, num_points) for traj in cluster_trajectories]
        )
        centroid = np.mean(resampled_trajectories, axis=0)  # 平均
        centroids.append(centroid)
    return centroids


# クラスターごとの台風活動時間の平均を計算する関数
def compute_average_activity_duration(df, labels):
    """
    クラスターごとの台風活動時間の平均を計算

    Parameters:
        df (pl.DataFrame): フィルタリングされた台風データ
        labels (list): クラスタリングの結果ラベル

    Returns:
        dict: クラスターIDをキー、平均活動時間を値とする辞書
    """
    activity_durations = {cluster_id: [] for cluster_id in np.unique(labels)}

    typhoon_numbers = df["TYPHOON NUMBER"].unique()

    for i, typhoon_number in enumerate(typhoon_numbers):
        # クラスタラベルを取得
        cluster_id = labels[i]

        # 該当するTYPHOON NUMBERのデータをフィルタリング
        filtered_df = df.filter(df["TYPHOON NUMBER"] == typhoon_number)

        # データ数を元に活動時間を計算
        n = len(filtered_df)
        activity_duration = 6 * (n - 1)  # 活動時間 [hours]

        # 該当クラスタに追加
        activity_durations[cluster_id].append(activity_duration)

    # 各クラスタの平均活動時間を計算
    average_durations = {
        cluster_id: np.mean(durations) if durations else 0
        for cluster_id, durations in activity_durations.items()
    }

    return average_durations


def compute_months_per_cluster_with_counts(df, labels):
    """
    クラスターごとのMONTH列の値と該当軌跡の個数を取得

    Parameters:
        df (pl.DataFrame): フィルタリングされた台風データ
        labels (list): クラスタリングの結果ラベル

    Returns:
        dict: クラスターIDをキー、該当するMONTHとそのカウントを含む辞書
    """
    months_per_cluster = {cluster_id: {} for cluster_id in np.unique(labels)}

    typhoon_numbers = df["TYPHOON NUMBER"].unique()

    for i, typhoon_number in enumerate(typhoon_numbers):
        # クラスタラベルを取得
        cluster_id = labels[i]

        # 該当するTYPHOON NUMBERのデータをフィルタリング
        filtered_df = df.filter(df["TYPHOON NUMBER"] == typhoon_number)

        # MONTH列の値を取得してカウントを更新
        for month in filtered_df["MONTH"].unique().to_list():
            if month not in months_per_cluster[cluster_id]:
                months_per_cluster[cluster_id][month] = 0
            months_per_cluster[cluster_id][month] += 1

    # 辞書のキーをソートして整形
    sorted_months_per_cluster = {
        cluster_id: {
            month: months_per_cluster[cluster_id][month]
            for month in sorted(months_per_cluster[cluster_id])
        }
        for cluster_id in months_per_cluster
    }

    return sorted_months_per_cluster


# クラスターごとのYEARの集計を行う関数
def compute_years_per_cluster(df, labels):
    """
    クラスターごとのYEAR列の値を取得

    Parameters:
        df (pl.DataFrame): フィルタリングされた台風データ
        labels (list): クラスタリングの結果ラベル

    Returns:
        dict: クラスターIDをキー、該当するYEAR値のリストを値とする辞書
    """
    years_per_cluster = {cluster_id: set() for cluster_id in np.unique(labels)}

    typhoon_numbers = df["TYPHOON NUMBER"].unique()

    for i, typhoon_number in enumerate(typhoon_numbers):
        # クラスタラベルを取得
        cluster_id = labels[i]

        # 該当するTYPHOON NUMBERのデータをフィルタリング
        filtered_df = df.filter(df["TYPHOON NUMBER"] == typhoon_number)

        # YEAR列の値を取得してクラスタに追加
        years = filtered_df["YEAR"].unique().to_list()
        years_per_cluster[cluster_id].update(years)

    # セットをリストに変換してソート
    sorted_years_per_cluster = {
        cluster_id: sorted(list(years))
        for cluster_id, years in years_per_cluster.items()
    }

    return sorted_years_per_cluster


# CSVファイルを読み込む＋フィルタリング＋1つに集約
df = load_and_filter_typhoon_data(start_year, end_year)

# 軌跡データを作成
trajectories = []
typhoon_numbers = df["TYPHOON NUMBER"].unique()  # 一意のTYPHOON NUMBERを取得

for number in typhoon_numbers:
    # TYPHOON NUMBERでフィルタリング
    filtered_df = df.filter(df["TYPHOON NUMBER"] == number)

    # 緯度と経度をペアにしてリスト化
    trajectory = list(zip(filtered_df["LAT"].to_list(), filtered_df["LON"].to_list()))
    trajectories.append(trajectory)

# 軌跡データを出力
# print("Trajectories:")
# for i, trajectory in enumerate(trajectories, start=1):
#     print(f"TYPHOON NUMBER={i}: {trajectory}")

# リサンプリングして等長化
num_points = 100
resampled_trajectories = [
    resample_trajectory(traj, num_points) for traj in trajectories
]

# 距離行列の計算（ユークリッド距離）
distance_matrix = np.zeros((len(resampled_trajectories), len(resampled_trajectories)))
for i in range(len(resampled_trajectories)):
    for j in range(len(resampled_trajectories)):
        distance_matrix[i, j] = np.mean(
            cdist(resampled_trajectories[i], resampled_trajectories[j])
        )

# クラスタリング
n_clusters = 30  # クラスタ数 2012-2023年のデータで30クラスタ程度がちょうど良さそうであった　データ数が増えた場合適宜調整してね
clustering = AgglomerativeClustering(
    n_clusters=n_clusters, metric="precomputed", linkage="average"
)
labels = clustering.fit_predict(distance_matrix)

# 各クラスタの中心軌跡を計算
centroids = compute_cluster_centroids(resampled_trajectories, labels, num_points)

# クラスターごとの平均活動時間を計算
average_activity_durations = compute_average_activity_duration(df, labels)

# クラスターIDを制限して出力
max_cluster_id = 2  # 出力するクラスタIDの上限

# 結果を出力
print("Cluster-wise Average Activity Durations (hours):")
for cluster_id, avg_duration in average_activity_durations.items():
    if cluster_id <= max_cluster_id:  # クラスタIDを制限
        print(f"Cluster {cluster_id}: {avg_duration:.2f} hours")

# クラスターごとのMONTH値を計算
months_per_cluster_counts = compute_months_per_cluster_with_counts(df, labels)

# 結果を出力
print("Cluster-wise Months with Counts:")
for cluster_id, month_counts in months_per_cluster_counts.items():
    if cluster_id <= max_cluster_id:  # クラスタIDを制限
        month_str = ", ".join(
            f"{month}({count})" for month, count in month_counts.items()
        )
        print(f"Cluster {cluster_id}: {month_str}")


# クラスターごとのYEAR値を計算
years_per_cluster = compute_years_per_cluster(df, labels)

# 結果を出力
print("Cluster-wise Years:")
for cluster_id, years in years_per_cluster.items():
    if cluster_id <= max_cluster_id:  # クラスタIDを制限
        year_str = ", ".join(map(str, years))
        print(f"Cluster {cluster_id}: {year_str}")


# クラスタ中心軌跡をCSVに保存 平面座標での分割になっているが、距離のばらけ方がそれらしいのでとりあえずは放置 必要になったらgeopy等を使って等距離分割を行う
for cluster_id, centroid in enumerate(centroids):
    if cluster_id > max_cluster_id:
        continue

    # 平均活動時間を取得
    avg_duration_hours = average_activity_durations[
        cluster_id
    ]  # compute_average_activity_duration から得た結果
    num_steps = int(avg_duration_hours // 6) + 1  # 6時間ごとのステップ数を計算

    # クラスタ中心の軌跡に沿って等間隔で座標をサンプリング
    # centroidはすでにリサンプリングされた軌跡で、曲線に沿ってポイントが配置されている
    sampled_lat_lon = []

    # centroidの長さに合わせて等間隔のインデックスを選択
    for i in range(num_steps):
        t = i / (num_steps - 1) if num_steps > 1 else 0
        # 曲線上での位置tに対応する座標を取得
        lat, lon = centroid[
            int(t * (num_points - 1))
        ]  # 曲線上の位置tに対応する座標を選択
        sampled_lat_lon.append((lat, lon))

    # CSVファイルに保存
    csv_file_path = os.path.join(output_folder, f"cluster_{cluster_id}.csv")
    with open(csv_file_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["LAT", "LON"])  # ヘッダー
        writer.writerows(sampled_lat_lon)

    print(f"Cluster {cluster_id} trajectory saved to {csv_file_path}")

### 可視化 ###

# 描画対象のクラスタラベル
target_labels = [0, 1, 2]  # , 3, 4]  # 描画したいクラスタラベルを指定

# クラスタラベルに対応する色を指定（3色を明確に設定）
cluster_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
}  # クラスタラベルに色を割り当て

# 各クラスタのデータ数をカウント
unique_labels, counts = np.unique(labels, return_counts=True)
cluster_sizes = dict(zip(unique_labels, counts))

# 描画開始
plt.figure(figsize=(10, 6))

# 軌跡データをプロット（指定したクラスタラベルのみ）
for i, traj in enumerate(resampled_trajectories):
    if labels[i] in target_labels:  # 対象ラベルのみ
        traj = np.array(traj)
        plt.plot(
            traj[:, 1],
            traj[:, 0],
            color=cluster_colors[labels[i]],  # 指定色を使用
            alpha=0.3,
            label=(
                f"Typhoon {i + 1}" if labels[i] == target_labels[0] else None
            ),  # 凡例を重複させない
        )

# クラスタ中心の線をプロット（指定したクラスタラベルのみ）
for i, centroid in enumerate(centroids):
    if i in target_labels:  # 対象ラベルのみ
        centroid = np.array(centroid)
        cluster_size = cluster_sizes.get(i, 0)

        # 線の太さと透明度の計算
        linewidth = max(
            1.5, min(1.5, 0.5 + (cluster_size - 1) * 0.2)
        )  # 数が多いほど太く
        alpha = min(
            1.0, max(0.5, 0.5 + (cluster_size - 1) * 0.05)
        )  # 数が多いほど濃く（不透明に）

        # クラスタ中心をプロット
        plt.plot(
            centroid[:, 1],
            centroid[:, 0],
            color=cluster_colors[i],  # 指定色を使用
            linewidth=linewidth,
            alpha=alpha,
            label=f"Cluster {i} Center",
        )

# クラスタ中心のCSVデータを重ねてプロット
for cluster_id in target_labels:
    csv_file_path = os.path.join(output_folder, f"cluster_{cluster_id}.csv")
    if os.path.exists(csv_file_path):
        # CSVファイルを読み込む
        csv_data = pl.read_csv(csv_file_path)
        latitudes = csv_data["LAT"].to_numpy()
        longitudes = csv_data["LON"].to_numpy()

        # CSVのデータポイントをプロット
        plt.scatter(
            longitudes,
            latitudes,
            color=cluster_colors[cluster_id],
            edgecolor="black",
            s=10,  # 点のサイズ
            label=(
                f"Cluster {cluster_id} Points"
                if f"Cluster {cluster_id} Points"
                not in plt.gca().get_legend_handles_labels()[1]
                else None
            ),
        )
    else:
        print(f"CSV file for Cluster {cluster_id} not found.")

# 軸ラベルとタイトル
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Typhoon Trajectories for Clusters {target_labels}")
plt.grid(True)  # グリッドを追加
# Cluster n Center と Cluster n Points の凡例のみを表示し、他の凡例は非表示
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
filtered_labels = {k: v for k, v in by_label.items() if "Cluster" in k}
plt.legend(filtered_labels.values(), filtered_labels.keys())


# PNGファイルとして保存
output_image_path = output_folder + "/typhoon_trajectories_clusters.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")  # 高解像度で保存
print(f"画像が {output_image_path} に保存されました。")

plt.close()  # 描画を閉じる
