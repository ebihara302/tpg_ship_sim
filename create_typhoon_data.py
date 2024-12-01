# 2019~2023年の台風データ「data/typhoon_path/combined_typhoon_data_2019-2023.csv」を用いて、2021〜2023年の台風データを作成する
# 2021~2023年の台風データ「data/typhoon_path/typhoon_data_2021-2023.csv」を作成する

import polars as pl

# 2019~2023年の台風データを読み込む
typhoon_data = pl.read_csv('data/typhoon_path/combined_typhoon_data_2019-2023.csv')

# 2021~2023年の台風データを作成する
typhoon_data_2021_2023 = typhoon_data.filter(typhoon_data['YEAR'] >= 2021)
typhoon_data_2021_2023.write_csv('data/typhoon_path/typhoon_data_2021-2023.csv')

print('台風データの作成が完了しました。')