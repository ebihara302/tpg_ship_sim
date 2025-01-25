import os

import polars as pl

# 初期入力
# 読み込み関連
start_year = 2020
end_year = 2020
start_month = 6
end_month = 10
# folder_path = "typhoon_path"
# 出力関連
output_folder = "filtered_typhoon_path"
os.makedirs(output_folder, exist_ok=True)  # 保存するフォルダを作成


### メイン処理 ###


def load_and_filter_typhoon_data(year, start_month, end_month):
    """
    指定された年の範囲の台風データを読み込み、ソート開始月からソート終了月までに発生した台風のデータをフィルタリング

    初めて台風番号が登場した時点をその台風の発生時刻とし、発生時刻がソート開始月以降かつソート終了月以前の台風のデータを抽出

    Parameters:
        year (int): 読み込み年
        start_month (int): ソート開始月
        end_month (int): ソート終了月

    Returns:
        pl.DataFrame: フィルタリングされた台風データ
    """
    filtered_data = []

    # polarsによるファイルの読み込みとデータフレーム化
    file_path = f"typhoon_data_{year}_6_interval.csv"
    df = pl.read_csv(file_path)

    # 台風番号のリスト
    typhoon_numbers = df["TYPHOON NUMBER"].unique().to_numpy()

    for typhoon_number in typhoon_numbers:
        # 台風番号ごとにデータをフィルタリング
        typhoon_data = df.filter(pl.col("TYPHOON NUMBER") == typhoon_number)

        # 台風の発生時刻を取得
        first_occurrence = typhoon_data.sort("unixtime").head(1)
        first_month = first_occurrence["MONTH"][0]

        # 発生時刻がソート開始月以降かつソート終了月以前の場合にデータを追加
        if start_month <= first_month <= end_month:
            # 台風が北緯30度以上東経146度以下に侵入したか確認
            if (typhoon_data["LAT"] >= 30).any() and (typhoon_data["LON"] <= 146).any():
                filtered_data.append(typhoon_data)

    # フィルタリングされたデータを結合
    if filtered_data:
        filtered_df = pl.concat(filtered_data)
    else:
        filtered_df = pl.DataFrame()

    # TYPHOON NUMBERの振り直し
    typhoon_numbers = filtered_df["TYPHOON NUMBER"].unique().to_numpy()
    typhoon_number_map = {typhoon_number: f"{year % 100:02d}{i+1:02d}" for i, typhoon_number in enumerate(typhoon_numbers)}
    filtered_df = filtered_df.with_columns(
        pl.col("TYPHOON NUMBER").apply(lambda x: typhoon_number_map.get(x, x)).alias("TYPHOON NUMBER")
    )

    return filtered_df


def main():
    for year in range(start_year, end_year + 1):
        filtered_df = load_and_filter_typhoon_data(year, start_month, end_month)
        typhoon_count = (
            len(filtered_df["TYPHOON NUMBER"].unique())
            if not filtered_df.is_empty()
            else 0
        )
        if not filtered_df.is_empty():
            output_file = os.path.join(
                output_folder, f"filtered_typhoon_data_{year}.csv"
            )
            filtered_df.write_csv(output_file)
            print(f"{year}年のフィルタリングされたデータを保存しました: {output_file}")
            print(f"{year}年のフィルタリングされた台風の個数: {typhoon_count}")
        else:
            print(f"{year}年のフィルタリングされたデータはありませんでした")
            print(f"{year}年のフィルタリングされた台風の個数: {typhoon_count}")


if __name__ == "__main__":
    main()
