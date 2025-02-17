import csv
from datetime import datetime, timedelta, timezone

### 初期入力データ ###
year = 2023
start_month = 6
end_month = 10
num_typhoons = 13

# 出力ファイル名
output_file = (
    "generated_typhoon_data_"
    + str(year)
    + "_6_interval_tynum"
    + str(num_typhoons)
    + ".csv"
)

# create_typhoon_path_cluster.py の出力を反映してください。
# クラスターごとの出現月とその出現回数
cluster_data = {
    0: {6: 4, 7: 7, 8: 16, 9: 10, 10: 6, 11: 1},
    1: {8: 5, 9: 8, 10: 3},
    2: {5: 2, 6: 7, 7: 19, 8: 20, 9: 16, 10: 7, 11: 1},
}
# クラスター番号をキー、クラスタ中心経路ファイルを値とする辞書
cluster_files = {
    0: "cluster_center_trajectories/cluster_0.csv",
    1: "cluster_center_trajectories/cluster_1.csv",
    2: "cluster_center_trajectories/cluster_2.csv",
}


### メイン処理 ###


# UTCタイムゾーンの設定
UTC = timezone(timedelta(hours=+0), "UTC")


def generate_typhoon_trajectories(
    year, start_month, end_month, num_typhoons, cluster_data, cluster_files, output_file
):
    """
    擬似的な台風進路データを生成する。

    Parameters:
        year (int): 発生年度
        start_month (int): 発生期間の開始月
        end_month (int): 発生期間の終了月
        num_typhoons (int): 発生数
        cluster_data (dict): クラスターごとの出現月とその出現回数
        cluster_files (dict): クラスター番号をキー、クラスタ中心経路ファイルを値とする辞書
        output_file (str): 出力CSVファイルのパス
    """
    # 発生月のリスト
    active_months = list(range(start_month, end_month + 1))

    # 指定された発生期間に基づき、発生のない月やクラスターを除外
    filtered_clusters = {}
    for cluster_id, data in cluster_data.items():
        filtered_data = {
            month: count for month, count in data.items() if month in active_months
        }
        if filtered_data:  # 発生月が一つ以上ある場合に追加
            filtered_clusters[cluster_id] = filtered_data

    # 発生がない場合は終了
    if not filtered_clusters:
        print("指定された期間に対応する台風発生データがありません。")
        return

    # 発生日時の間隔を計算
    total_hours = (
        (end_month - start_month + 1) * 30 * 24
    )  # 月ごとの平均日数を30日と仮定
    interval_hours = total_hours / num_typhoons
    # 6時間刻みに修正
    interval_hours = round(interval_hours / 6) * 6

    # 発生時刻を計算 (UTCタイムゾーンを適用)
    typhoon_times = [
        datetime(year, start_month, 1, tzinfo=UTC) + timedelta(hours=i * interval_hours)
        for i in range(num_typhoons)
    ]

    # 出力データを格納するリスト
    output_data = []

    # 各クラスターの中心経路データを読み込む
    cluster_trajectories = {}
    for cluster_id, file_path in cluster_files.items():
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            cluster_trajectories[cluster_id] = [
                (float(row["LAT"]), float(row["LON"])) for row in reader
            ]

    # 台風データを生成
    typhoon_number = 1
    valid_clusters = list(filtered_clusters.keys())  # 使用可能なクラスター番号
    for i, typhoon_time in enumerate(typhoon_times):
        # 現在の月を取得
        current_month = typhoon_time.month

        # 使用可能なクラスターから、現在の月に該当するクラスターを選択
        available_clusters = [
            cluster_id
            for cluster_id in valid_clusters
            if current_month in filtered_clusters[cluster_id]
        ]

        # 発生がない場合はスキップ
        if not available_clusters:
            continue

        # クラスターを順に選択
        cluster_id = available_clusters[i % len(available_clusters)]
        trajectory = cluster_trajectories[cluster_id]  # クラスターの中心経路

        for step, (lat, lon) in enumerate(trajectory):
            current_time = typhoon_time + timedelta(hours=6 * step)
            unix_time = int(current_time.timestamp())
            output_data.append(
                {
                    "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "unixtime": unix_time,
                    "YEAR": current_time.year,
                    "MONTH": current_time.month,
                    "DAY": current_time.day,
                    "HOUR": current_time.hour,
                    "TYPHOON NUMBER": f"{year % 100:02d}{typhoon_number:02d}",
                    "CLUSTER NUMBER": cluster_id,
                    "LAT": lat,
                    "LON": lon,
                }
            )

        typhoon_number += 1

    # CSVファイルに保存
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
        writer.writeheader()
        writer.writerows(output_data)

    print(f"台風データが {output_file} に保存されました。")


# 関数の呼び出し例
generate_typhoon_trajectories(
    year,
    start_month,
    end_month,
    num_typhoons,
    cluster_data,
    cluster_files,
    output_file,
)
