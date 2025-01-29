import csv
import math
import os
import traceback
from datetime import datetime, timedelta, timezone

import hydra
import optuna
import polars as pl
from geopy.distance import geodesic
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.optimize import fsolve, newton
from tqdm import tqdm

from tpg_ship_sim import simulator_optimize
from tpg_ship_sim.model import base, forecaster, support_ship, tpg_ship

# 起動時の出力フォルダ名を取得するためグローバル変数に設定
output_folder_path = None
save_dataframe = None
sim_year = int(2021)


# 進捗バーを更新するコールバック関数を定義
class TqdmCallback(object):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Optuna Trials")

    def __call__(self, study, trial):
        self.pbar.update(1)


# 硬翼帆本数を硬翼帆密度と硬翼帆面積の従属変数にした場合の必要関数
def cal_dwt(storage_method, storage):
    """
    ############################ def cal_dwt ############################

    [ 説明 ]

    載貨重量トンを算出する関数です。

    ##############################################################################

    引数 :
        storage_method (int) : 貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵
        storage (float) : 貯蔵容量[Wh]

    戻り値 :
        dwt (float) : 載貨重量トン

    #############################################################################
    """
    # 載貨重量トンを算出する。単位はt。
    # storageの容量は中の物質を完全燃焼させた場合のエネルギー量として考える

    if storage_method == 1:  # 電気貯蔵
        # 重量エネルギー密度1000Wh/kgの電池を使うこととする。
        dwt = storage / 1000 / 1000

    elif storage_method == 2:  # MCH貯蔵
        # 有機ハイドライドで水素を貯蔵することとする。
        dwt = storage / 5000 * 0.0898 / 47.4

    elif storage_method == 3:  # メタン貯蔵
        # 物性より計算　メタン1molの完全燃焼で802kJ=802/3600kWh
        # mol数の計算
        mol = storage / ((802 / 3600) * 1000)
        # メタンの分子量16.04g/molを用いてtに変換
        dwt = mol * 16.04 / 10**6

    elif storage_method == 4:  # メタノール貯蔵
        # 物性より計算　メタノール1molの完全燃焼で726.2kJ=726.2/3600kWh
        # mol数の計算
        mol = storage / ((726.2 / 3600) * 1000)
        # メタノールの分子量32.04g/molを用いてtに変換
        dwt = mol * 32.04 / 10**6

    elif storage_method == 5:  # e-ガソリン貯蔵
        # 代表の分子としてC8H18（オクタン）を用いる
        # オクタン1molの完全燃焼で5500kJ=5500/3600kWh
        # mol数の計算
        mol = storage / ((5500 / 3600) * 1000)
        # オクタンの分子量114.23g/molを用いてtに変換
        dwt = mol * 114.23 / 10**6

    else:
        print("cannot cal")

    return dwt


# コンテナ型の船体寸法計算
def calculate_LB_container(total_ship_weight_per_body):
    """
    ############################ def calculate_LB_container ############################

    [ 説明 ]

    コンテナ型の船体の寸法を算出する関数です。

    船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

    ##############################################################################

    引数 :
        total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

    戻り値 :
        L_oa (float) : 船体の全長
        B (float) : 船体の幅

    #############################################################################
    """

    if total_ship_weight_per_body < 35000:
        L_oa = 6.0564 * (total_ship_weight_per_body**0.3398)
        B = 1.4257 * (total_ship_weight_per_body**0.2883)

    elif 35000 <= total_ship_weight_per_body < 45000:
        L_oa = 228.3
        B = 31.8

    elif 45000 <= total_ship_weight_per_body < 55000:
        L_oa = 268.8
        B = 33.7

    elif 55000 <= total_ship_weight_per_body < 65000:
        L_oa = 284.5
        B = 35.5

    elif 65000 <= total_ship_weight_per_body < 75000:
        L_oa = 291.0
        B = 39.2

    elif 75000 <= total_ship_weight_per_body < 85000:
        L_oa = 304.8
        B = 42.0

    elif 85000 <= total_ship_weight_per_body < 95000:
        L_oa = 310.9
        B = 44.1

    elif 95000 <= total_ship_weight_per_body < 105000:
        L_oa = 338.0
        B = 45.3

    elif 105000 <= total_ship_weight_per_body < 135000:
        L_oa = 343.1
        if total_ship_weight_per_body < 115000:
            B = 47.3
        elif 115000 <= total_ship_weight_per_body < 125000:
            B = 48.0
        else:
            B = 48.5

    elif 135000 <= total_ship_weight_per_body < 155000:
        L_oa = 367.5
        if total_ship_weight_per_body < 145000:
            B = 48.5
        else:
            B = 52.0

    elif 155000 <= total_ship_weight_per_body < 175000:
        L_oa = 378.3
        B = 52.0

    else:
        L_oa = 399.7
        B = 59.4

    return L_oa, B


# タンカー型の船体寸法計算
def calculate_LB_tanker(total_ship_weight_per_body):
    """
    ############################ def calculate_LB_tanker ############################

    [ 説明 ]

    タンカー型の船体の寸法を算出する関数です。

    船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

    ##############################################################################

    引数 :
        total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

    戻り値 :
        L_oa (float) : 船体の全長
        B (float) : 船体の幅

    #############################################################################
    """

    if total_ship_weight_per_body < 20000:
        L_oa = 5.4061 * (total_ship_weight_per_body**0.3500)
        B = 1.4070 * (total_ship_weight_per_body**0.2864)

    elif 20000 <= total_ship_weight_per_body < 280000:
        L_oa = 10.8063 * (total_ship_weight_per_body**0.2713)
        if total_ship_weight_per_body < 40000:
            B = 1.4070 * (total_ship_weight_per_body**0.2864)
        elif 40000 <= total_ship_weight_per_body < 80000:
            B = 32.9
        elif 80000 <= total_ship_weight_per_body < 120000:
            B = 43.5
        elif 120000 <= total_ship_weight_per_body < 200000:
            B = 48.9
        else:
            B = 60.2

    else:
        L_oa = 333.7
        B = 60.2

    return L_oa, B


# LNG船型の船体寸法計算
def calculate_LB_lng(total_ship_weight_per_body):
    """
    ############################ def calculate_LB_lng ############################

    [ 説明 ]

    LNG船型の船体の寸法を算出する関数です。

    船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

    ##############################################################################

    引数 :
        total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

    戻り値 :
        L_oa (float) : 船体の全長
        B (float) : 船体の幅

    #############################################################################
    """
    # 横軸がGT（総トン数）で記載されていたので、DWTをGTに変換する。一般にDWTがGTの0.6〜0.8倍程度とのことを利用する。
    GT = total_ship_weight_per_body / 0.7

    # GTに対する船体の全長、船体の幅を算出
    if GT < 150000:
        L_oa = 6.1272 * (GT**0.3343)
        B = 1.1239 * (GT**0.3204)
    else:
        L_oa = 345.2
        B = 54.6

    return L_oa, B


def calculate_max_sail_num(
    storage_method,
    max_storage,
    electric_propulsion_max_storage_wh,
    hull_num,
    sail_area,
    sail_space,
):
    """
    ############################ def calculate_max_sail_num ############################

    [ 説明 ]

    台風発電船が搭載可能な帆の本数を算出する関数です。

    ##############################################################################

    戻り値 :
        max_sail_num (int) : 台風発電船が搭載可能な帆の本数

    #############################################################################
    """

    # ウインドチャレンジャーの帆を基準とする
    base_sail_area = 880  # 基準帆面積 [m^2]
    base_sail_width = 15  # 基準帆幅 [m]
    assumed_num_sails = 100  # 帆の仮想本数

    # 船体の載貨重量トンを計算
    hull_dwt = cal_dwt(storage_method, max_storage)
    # バッテリーの重量トンを計算
    battery_weight_ton = cal_dwt(1, electric_propulsion_max_storage_wh)

    # 1. 帆の本数を仮定して、重量から船の寸法を計算する
    # 2. 計算した船の寸法から、甲板面積を算出
    # 3. 甲板面積と帆の幅から搭載可能な最大帆数を算出
    # 4. 仮の帆の本数と搭載可能な最大帆数を比較する
    # 5. 仮の帆の本数を更新し、帆の本数が等しくなるまで繰り返す

    while True:

        # 1. 帆の本数を仮定して、重量から船の寸法を計算する
        sail_weight = 120 * (sail_area / base_sail_area)  # 帆の重量 [t]

        # 船の総重量(DWT[t])を計算
        total_ship_weight = (
            hull_dwt + battery_weight_ton + (assumed_num_sails * sail_weight)
        )
        total_ship_weight_per_body = total_ship_weight / hull_num

        # 甲板面積を計算
        # 「統計解析による船舶諸元に関する研究」よりDWTとL_oa,Bの値を算出する
        # 船体の寸法を計算
        if storage_method == 1:  # 電気貯蔵 = コンテナ型
            L_oa, B = calculate_LB_container(total_ship_weight_per_body)

        elif (
            (storage_method == 2) or (storage_method == 4) or (storage_method == 5)
        ):  # MCH・メタノール・e-ガソリン貯蔵 = タンカー型
            L_oa, B = calculate_LB_tanker(total_ship_weight_per_body)

        elif storage_method == 3:  # メタン貯蔵 = LNG船型
            L_oa, B = calculate_LB_lng(total_ship_weight_per_body)

        # 2. 計算した船の寸法から、甲板面積を算出

        # L_oa,Bの記録
        hull_L_oa = L_oa
        hull_B = B

        # 甲板面積を算出
        if hull_num == 2:
            # 船体が2つの場合、Bは3.5倍とする
            B = B * 3.5
            hull_B = B

        deck_area = L_oa * B  # 簡易甲板面積 [m^2]

        # 3. 甲板面積と帆の幅から搭載可能な最大帆数を算出

        # 帆の寸法を基準帆から算出
        scale_factor = (sail_area / base_sail_area) ** 0.5
        sail_width = base_sail_width * scale_factor

        # 帆の搭載間隔を指定
        sail_space_per_sail = sail_width * sail_space

        if B < sail_space_per_sail:
            # 甲板幅が帆幅より狭い場合、船長に合わせて帆の本数を算出
            max_sails_by_deck_area = L_oa / sail_space_per_sail
            # 本数を四捨五入
            max_sails_by_deck_area = round(max_sails_by_deck_area)
        else:
            # 甲板面積から搭載できる最大帆数を算出
            max_sails_by_deck_area_L = L_oa / sail_space_per_sail
            max_sails_by_deck_area_B = B / sail_space_per_sail
            max_sails_by_deck_area = round(max_sails_by_deck_area_L) * round(
                max_sails_by_deck_area_B
            )

        # 4. 仮の帆の本数と搭載可能な最大帆数を比較する
        # 5. 仮の帆の本数を更新し、帆の本数が等しくなるまで繰り返す

        if assumed_num_sails == max_sails_by_deck_area:
            break
        else:
            assumed_num_sails = max_sails_by_deck_area

    max_sail_num = max_sails_by_deck_area

    return max_sail_num


def sp_ship_EP_storage_cal(
    storage_method,
    max_storage_wh,
    support_ship_speed_kt,
    elect_trust_efficiency,
    st_base_locate,
    sp_base_locate,
):
    """
    ############################ def sp_ship_EP_storage_cal ############################

    [ 説明 ]

    輸送船の電動機バッテリー容量を計算する関数です。

    ##############################################################################

    引数 :
        storage_method (int) : 貯蔵方法の種類。1=電気貯蔵,2=MCH貯蔵,3=メタン貯蔵,4=メタノール貯蔵,5=e-ガソリン貯蔵
        max_storage_wh (float) : 輸送船の最大電気貯蔵量[Wh]
        support_ship_speed_kt (float) : 輸送船の最大速度[kt]
        elect_trust_efficiency (float) : サポート船の電気推進効率
        st_base_locate (list) : 貯蔵拠点の緯度経度
        sp_base_locate (list) : 供給拠点の緯度経度

    戻り値 :
        sp_ship_EP_storage (float) : サポート船の電気貯蔵量[Wh]

    #############################################################################
    """
    # 船型で決まる定数k 以下はタンカーでk=2.2（船速がktの時）における処理
    if storage_method == 1:  # 電気貯蔵
        k = 1.7 / (1.852**3)
    elif storage_method >= 2:  # 電気貯蔵以外
        k = 2.2 / (1.852**3)

    # 輸送船の電気貯蔵量を計算
    # geopyで貯蔵拠点から供給拠点までの距離を大圏距離で計算
    distance = geodesic(st_base_locate, sp_base_locate).kilometers
    # support_ship_speed_ktをkm/hに変換
    max_speed_kmh = support_ship_speed_kt * 1.852

    # max_storage_whをDWTに変換
    max_storage_ton = cal_dwt(storage_method, max_storage_wh)

    # 反復計算時の初期値
    initial_guess = max_storage_ton

    # バッテリー容量のマージン倍率
    margin = 1.2  # 20%のマージン

    # 輸送船のバッテリー容量xの計算の方程式定義　Xがバッテリー容量[t]であることに注意
    def equation(X):
        # value = max_storage_ton + X
        # print(f"x: {X}, max_storage_ton + x: {value}")
        return (X * 1000 * 1000) - (
            (
                (k * 2 * margin * max_speed_kmh**3 * (distance / max_speed_kmh))
                / elect_trust_efficiency
            )
            * ((max_storage_ton + X) ** (2 / 3))
        )

    # 以下の処理でバッテリー容量[t]が求まる
    EP_storage_solution = fsolve(equation, initial_guess)

    # 結果をチェック（負の値の場合エラーを出す）
    if EP_storage_solution[0] < 0:
        print(EP_storage_solution)
        raise ValueError("計算結果が負の値です。入力値を確認してください。")

    # バッテリー容量をWhに変換する　重量エネルギー密度1000Wh/kgの電池を使うこととする。
    sp_ship_EP_storage = min(EP_storage_solution) * 1000 * 1000

    # バッテリー容量をもとに航続能力を計算してチェック equation(x)の時のdistanceを求めることになる
    # 単位時間の消費エネルギー[W]
    consumption_elect_per_hour = (
        k
        * ((max_storage_ton + min(EP_storage_solution)) ** (2 / 3))
        * (max_speed_kmh**3)
    ) / elect_trust_efficiency
    # 往復で消費するエネルギー[Wh]
    total_consumption_elect = (
        consumption_elect_per_hour * (2 * distance) / max_speed_kmh
    )
    # 消費エネルギーの見積もりとバッテリー容量の見積もりの差分
    if (total_consumption_elect - sp_ship_EP_storage) > 0:
        raise ValueError("バッテリー容量が足りません。入力値を確認してください。")
    # else:
    # print(
    #     "バッテリー容量チェックOK",
    #     total_consumption_elect * 1.2,
    #     " and ",
    #     sp_ship_EP_storage,
    # )

    # sp_ship_EP_storageの値をMWhにした時に整数になるように切り上げ
    sp_ship_EP_storage = int(sp_ship_EP_storage / 10**6) * 10**6

    return sp_ship_EP_storage


def tank_capacity_ton_to_wh(tank_capacity_ton, tpgship_storage_method):
    """
    ############################ def tank_capacity_ton_to_wh ############################

    [ 説明 ]

    タンク容量[t]をWhに変換する関数です。

    ##############################################################################

    引数 :
        tank_capacity_ton (float) : タンク容量[t]
        tpgship_storage_method (int) : 台風発電船の貯蔵方法

    戻り値 :
        tank_capacity_wh (float) : タンク容量[Wh]

    #############################################################################
    """
    if tpgship_storage_method == 1:  # 電気貯蔵
        tank_capacity_wh = tank_capacity_ton * 1000 * 1000

    elif tpgship_storage_method == 2:  # MCH貯蔵
        # MCHのエネルギー密度は1kgあたり12000Whとする
        tank_capacity_wh = tank_capacity_ton * 47.4 / 0.0898 * 5000

    elif tpgship_storage_method == 3:  # メタン貯蔵
        # メタンのエネルギー密度は1molあたり802kJ=802/3600kWhとする
        # mol数の計算
        mol = (tank_capacity_ton * 10**6) / 16.04
        tank_capacity_wh = mol * (802 / 3600 * 1000)

    elif tpgship_storage_method == 4:  # メタノール貯蔵
        # メタノールのエネルギー密度は1molあたり726.2kJ=726.2/3600kWhとする
        # mol数の計算
        mol = (tank_capacity_ton * 10**6) / 32.04
        tank_capacity_wh = mol * (726.2 / 3600 * 1000)

    elif tpgship_storage_method == 5:  # e-ガソリン貯蔵
        # オクタンのエネルギー密度は1molあたり5500kJ=5500/3600kWhとする
        # mol数の計
        mol = (tank_capacity_ton * 10**6) / 114.23
        tank_capacity_wh = mol * (5500 / 3600 * 1000)

    else:
        print("cannot cal")

    return tank_capacity_wh


def objective_value_calculation(
    tpg_ship,
    st_base,
    sp_base,
    support_ship_1,
    support_ship_2,
    simulation_start_time,
    simulation_end_time,
):
    """
    ############################ def objective_value_calculation ############################

    [ 説明 ]

    目的関数の値を算出する関数です。

    適宜設定し直してください。

    ##############################################################################

    引数 :
        tpg_ship (TPG_ship) : TPG ship
        st_base (Base) : Storage base
        sp_base (Base) : Supply base
        support_ship_1 (Support_ship) : Support ship 1
        support_ship_2 (Support_ship) : Support ship 2

    戻り値 :
        objective_value (float) : 目的関数の値

    #############################################################################
    """
    # コスト計算
    # 台風発電船関連[億円]
    tpg_ship.cost_calculate()
    # 供給拠点関連[億円]
    sp_base.cost_calculate(tpg_ship)

    # 帆の大きさによるペナルティ
    sail_length_penalty = 0
    max_sail_length = 180.0  # 今までの検証結果でそれらしい値となるものを設定した[m]
    allowable_sail_length = (
        tpg_ship.hull_B * 1.3
    )  # 許容される帆の大きさ[m] 船体の幅の1.8倍とする
    # ペナルティが生じる帆の長さを決める
    if allowable_sail_length > max_sail_length:
        penalty_sail_length = max_sail_length
    else:
        penalty_sail_length = allowable_sail_length

    # 帆の大きさによるペナルティの計算
    if tpg_ship.sail_height > penalty_sail_length:
        sail_length_penalty = 100 * (tpg_ship.sail_height - penalty_sail_length)
    else:
        sail_length_penalty = 0

    # 供給拠点への輸送が行われなかった時のペナルティ
    supply_zero_penalty = 0
    if sp_base.total_supply_list[-1] == 0:
        supply_zero_penalty = 500
    else:
        supply_zero_penalty = 0

    # EC供給量[GWh]
    total_profit = sp_base.total_supply_list[-1] / 10**9

    total_pure_profit_peryear = total_profit

    # ペナルティの合計を計算
    total_penalty = (
        sail_length_penalty
        + tpg_ship.minus_storage_penalty_list[-1]
        + supply_zero_penalty
    )

    # 目的関数の値を計算
    objective_value = total_pure_profit_peryear - total_penalty

    return objective_value


def simulation_result_to_df(
    tpg_ship,
    st_base,
    sp_base,
    support_ship_1,
    support_ship_2,
    simulation_start_time,
    simulation_end_time,
):
    """
    ############################ def simulation_result_to_df ############################

        [ 説明 ]

        シミュレーション結果をデータフレームに出力する関数です。

        各モデルのハイパーパラメータと目的関数の指標たり得る値を記録します。

        一列分（試行1回分）のデータをまとめるものであり、それを繰り返し集積することで、別の処理で全体のデータをまとめます。

    ##############################################################################

    引数 :
        tpg_ship (TPG_ship) : TPG ship
        st_base (Base) : Storage base
        sp_base (Base) : Supply base
        support_ship_1 (Support_ship) : Support ship 1
        support_ship_2 (Support_ship) : Support ship 2

    #############################################################################
    """
    # コスト計算(損失)
    # 運用年数　simulation_start_time、simulation_end_time (ex."2023-01-01 00:00:00")から年数を計算 365で割って端数切り上げ
    operating_years = math.ceil(
        (
            datetime.strptime(simulation_end_time, "%Y-%m-%d %H:%M:%S")
            - datetime.strptime(simulation_start_time, "%Y-%m-%d %H:%M:%S")
        ).days
        / 365
    )

    # 台風発電船関連[億円]
    tpg_ship.cost_calculate()
    tpg_ship_total_cost = (
        tpg_ship.building_cost
        + tpg_ship.carrier_cost
        + tpg_ship.maintenance_cost * operating_years
    )
    # サポート船1関連[億円]
    support_ship_1.cost_calculate()
    support_ship_1_total_cost = (
        support_ship_1.building_cost
        + support_ship_1.maintenance_cost * operating_years
        + support_ship_1.transportation_cost
    )
    # サポート船2関連[億円]
    support_ship_2.cost_calculate()
    support_ship_2_total_cost = (
        support_ship_2.building_cost
        + support_ship_2.maintenance_cost * operating_years
        + support_ship_2.transportation_cost
    )
    # 貯蔵拠点関連[億円]
    st_base.cost_calculate(tpg_ship)
    st_base_total_cost = (
        st_base.building_cost + st_base.maintenance_cost * operating_years
    )
    # 供給拠点関連[億円]
    sp_base.cost_calculate(tpg_ship)
    sp_base_total_cost = (
        sp_base.building_cost + sp_base.maintenance_cost * operating_years
    )

    # 総コスト[億円]
    total_cost = (
        tpg_ship_total_cost
        + support_ship_1_total_cost
        + support_ship_2_total_cost
        + st_base_total_cost
        + sp_base_total_cost
    )

    # 総利益[億円]
    total_profit = sp_base.profit

    # 減価償却費 耐用年数について、船は一律20年、拠点はタンク部分は20年、その他は50年とする
    tpg_ship_depreciation_expense = tpg_ship.building_cost / 20

    # 総減価償却費[億円]
    total_depreciation_expense = tpg_ship_depreciation_expense

    total_maintainance_cost = tpg_ship.maintenance_cost

    total_operation_cost = tpg_ship.carrier_cost

    total_pure_profit_peryear = (
        total_profit
        - total_depreciation_expense
        - total_maintainance_cost
        - total_operation_cost
    )

    unit_price = sp_base.unit_price  # 供給拠点売却時の単価[円]
    income = total_pure_profit_peryear
    # 営業利益が0円(利益はないが操業が続けられる)の時の単価を計算
    if total_profit == 0:
        appropriate_unit_price = (total_profit - income) * unit_price
    else:
        appropriate_unit_price = ((total_profit - income) / total_profit) * unit_price

    data = pl.DataFrame(
        {
            # TPG ship (列名の先頭にT_を付与。探索しないものはコメントアウト)
            ## 装置パラメータ関連
            "T_hull_num": [int(tpg_ship.hull_num)],
            "T_storage_method": [int(tpg_ship.storage_method)],
            "T_max_storage[GWh]": [float(tpg_ship.max_storage / 10**9)],
            "T_EP_max_storage_wh[GWh]": [
                float(tpg_ship.electric_propulsion_max_storage_wh / 10**9)
            ],
            "T_sail_num": [int(tpg_ship.sail_num)],  # Int64 型の例
            "T_sail_area[m2]": [int(tpg_ship.sail_area)],
            "T_sail_width[m]": [float(tpg_ship.sail_width)],
            "T_sail_height[m]": [float(tpg_ship.sail_height)],
            "T_sail_space": [float(tpg_ship.sail_space)],
            "T_sail_steps": [int(tpg_ship.sail_steps)],  # Int64 型の例
            "T_sail_weight": [float(tpg_ship.sail_weight)],
            "T_num_sails_per_row": [int(tpg_ship.num_sails_per_row)],  # Int64 型の例
            "T_num_sails_rows": [int(tpg_ship.num_sails_rows)],  # Int64 型の例
            "T_sail_penalty": [float(tpg_ship.sail_penalty)],
            "T_dwt[t]": [float(tpg_ship.ship_dwt)],
            "T_hull_L_oa[m]": [float(tpg_ship.hull_L_oa)],
            "T_hull_B[m]": [float(tpg_ship.hull_B)],
            "T_trust_efficiency": [float(tpg_ship.trust_efficiency)],
            "T_carrier_to_elect_efficiency": [
                float(tpg_ship.carrier_to_elect_efficiency)
            ],
            "T_elect_to_carrier_efficiency": [
                float(tpg_ship.elect_to_carrier_efficiency)
            ],
            "T_generator_num": [int(tpg_ship.generator_num)],  # Int64 型の例
            "T_generator_turbine_radius[m]": [float(tpg_ship.generator_turbine_radius)],
            "T_generator_pillar_width": [float(tpg_ship.generator_pillar_width)],
            "T_generator_rated_output[GW]": [
                float(tpg_ship.generator_rated_output_w / 10**9)
            ],
            "T_generator_efficiency": [float(tpg_ship.generator_efficiency)],
            "T_generator_drag_coefficient": [
                float(tpg_ship.generator_drag_coefficient)
            ],
            "T_generator_pillar_chord": [float(tpg_ship.generator_pillar_chord)],
            "T_generator_pillar_max_tickness": [
                float(tpg_ship.generator_pillar_max_tickness)
            ],
            "T_generating_speed[kt]": [float(tpg_ship.generating_speed_kt)],
            "T_tpgship_return_speed[kt]": [float(tpg_ship.nomal_ave_speed)],
            "T_forecast_weight": [float(tpg_ship.forecast_weight)],
            "T_govia_base_judge_energy_storage_per": [
                float(tpg_ship.govia_base_judge_energy_storage_per)
            ],
            "T_judge_time_times": [float(tpg_ship.judge_time_times)],
            "T_operational_reserve_percentage": [
                int(tpg_ship.operational_reserve_percentage)
            ],
            "T_standby_lat": [float(tpg_ship.standby_lat)],
            "T_standby_lon": [float(tpg_ship.standby_lon)],
            "T_typhoon_effective_range[km]": [int(tpg_ship.typhoon_effective_range)],
            "T_total_gene_elect[GWh]": [
                float(tpg_ship.total_gene_elect_list[-1] / 10**9)
            ],
            "T_total_gene_carrier[GWh]": [
                float(tpg_ship.total_gene_carrier_list[-1] / 10**9)
            ],
            "T_total_loss_elect[GWh]": [
                float(tpg_ship.total_loss_elect_list[-1] / 10**9)
            ],
            "T_sum_supply_elect[GWh]": [
                float(tpg_ship.sum_supply_elect_list[-1] / 10**9)
            ],
            "T_minus_storage_penalty": [int(tpg_ship.minus_storage_penalty_list[-1])],
            # 貯蔵拠点 (列名の先頭にSt_を付与。探索しないものはコメントアウト)
            "St_base_type": [int(st_base.base_type)],  # Int64 型の例
            "St_lat": [float(st_base.locate[0])],
            "St_lon": [float(st_base.locate[1])],
            "St_max_storage[GWh]": [float(st_base.max_storage / 10**9)],
            "St_call_per": [float(st_base.call_per)],
            "St_total_supply[GWh]": [
                float(st_base.total_quantity_received_list[-1] / 10**9)
            ],
            # 供給拠点 (列名の先頭にSp_を付与。探索しないものはコメントアウト)
            "Sp_base_type": [int(sp_base.base_type)],  # Int64 型の例
            "Sp_lat": [float(sp_base.locate[0])],
            "Sp_lon": [float(sp_base.locate[1])],
            "Sp_max_storage[GWh]": [float(sp_base.max_storage / 10**9)],
            "Sp_total_supply[GWh]": [float(sp_base.total_supply_list[-1] / 10**9)],
            # 輸送船1 (列名の先頭にSs1_を付与。探索しないものはコメントアウト)
            "Ss1_max_storage[GWh]": [float(support_ship_1.max_storage / 10**9)],
            "Ss1_ship_speed[kt]": [int(support_ship_1.support_ship_speed)],
            "Ss1_EP_max_storage[GWh]": [float(support_ship_1.EP_max_storage / 10**9)],
            "Ss1_Total_consumption_elect[GWh]": [
                float(support_ship_1.sp_total_consumption_elect_list[-1] / 10**9)
            ],
            "Ss1_Total_received_elect[GWh]": [
                float(support_ship_1.sp_total_received_elect_list[-1] / 10**9)
            ],
            # 輸送船2 (列名の先頭にSs2_を付与。探索しないものはコメントアウト)
            "Ss2_max_storage[GWh]": [float(support_ship_2.max_storage / 10**9)],
            "Ss2_ship_speed[kt]": [int(support_ship_2.support_ship_speed)],
            "Ss2_EP_max_storage[GWh]": [float(support_ship_2.EP_max_storage / 10**9)],
            "Ss2_Total_consumption_elect[GWh]": [
                float(support_ship_2.sp_total_consumption_elect_list[-1] / 10**9)
            ],
            "Ss2_Total_received_elect[GWh]": [
                float(support_ship_2.sp_total_received_elect_list[-1] / 10**9)
            ],
            # コスト関連
            "T_hull_cost[100M JPY]": [float(tpg_ship.hull_cost / 10**8)],
            "T_wing_sail_cost[100M JPY]": [float(tpg_ship.wing_sail_cost / 10**8)],
            "T_underwater_turbine_cost[100M JPY]": [
                float(tpg_ship.underwater_turbine_cost / 10**8)
            ],
            "T_battery_cost[100M JPY]": [float(tpg_ship.battery_cost / 10**8)],
            "T_building_cost[100M JPY]": [float(tpg_ship.building_cost)],
            "T_carrier_cost[100M JPY]": [float(tpg_ship.carrier_cost)],
            "T_maintenance_cost[100M JPY]": [
                float(tpg_ship.maintenance_cost * operating_years)
            ],
            "T_depreciation_expense[100M JPY]": [float(tpg_ship.building_cost / 20)],
            "T_total_cost[100M JPY]": [float(tpg_ship_total_cost)],
            "St_building_cost[100M JPY]": [float(st_base.building_cost)],
            "St_maintenance_cost[100M JPY]": [
                float(st_base.maintenance_cost * operating_years)
            ],
            "St_depreciation_expense[100M JPY]": [
                st_base.tank_total_cost / 20
                + (st_base.building_cost - st_base.tank_total_cost) / 50
            ],
            "St_total_cost[100M JPY]": [float(st_base_total_cost)],
            "Sp_building_cost[100M JPY]": [float(sp_base.building_cost)],
            "Sp_maintenance_cost[100M JPY]": [
                float(sp_base.maintenance_cost * operating_years)
            ],
            "Sp_depreciation_expense[100M JPY]": [
                sp_base.tank_total_cost / 20
                + (sp_base.building_cost - sp_base.tank_total_cost) / 50
            ],
            "Sp_total_cost[100M JPY]": [float(sp_base_total_cost)],
            "Ss1_building_cost[100M JPY]": [float(support_ship_1.building_cost)],
            "Ss1_maintenance_cost[100M JPY]": [
                float(support_ship_1.maintenance_cost * operating_years)
            ],
            "Ss1_transportation_cost[100M JPY]": [
                float(support_ship_1.transportation_cost)
            ],
            "Ss1_depreciation_expense[100M JPY]": [
                float(support_ship_1.building_cost / 20)
            ],
            "Ss1_total_cost[100M JPY]": [float(support_ship_1_total_cost)],
            "Ss2_building_cost[100M JPY]": [float(support_ship_2.building_cost)],
            "Ss2_maintenance_cost[100M JPY]": [
                float(support_ship_2.maintenance_cost * operating_years)
            ],
            "Ss2_transportation_cost[100M JPY]": [
                float(support_ship_2.transportation_cost)
            ],
            "Ss2_depreciation_expense[100M JPY]": [
                float(support_ship_2.building_cost / 20)
            ],
            "Ss2_total_cost[100M JPY]": [float(support_ship_2_total_cost)],
            "Total_maintain_cost_per_Year[100M JPY]": [
                float(
                    tpg_ship.maintenance_cost
                    + st_base.maintenance_cost
                    + sp_base.maintenance_cost
                    + support_ship_1.maintenance_cost
                    + support_ship_2.maintenance_cost
                )
            ],
            "Total_operating_cost_per_Year[100M JPY]": [
                float(
                    tpg_ship.carrier_cost
                    + support_ship_1.transportation_cost
                    + support_ship_2.transportation_cost
                )
            ],
            "Total_depreciation_expense_per_Year[100M JPY]": [
                float(
                    tpg_ship.building_cost / 20
                    + st_base.tank_total_cost / 20
                    + (st_base.building_cost - st_base.tank_total_cost) / 50
                    + sp_base.tank_total_cost / 20
                    + (sp_base.building_cost - sp_base.tank_total_cost) / 50
                    + support_ship_1.building_cost / 20
                    + support_ship_2.building_cost / 20
                )
            ],
            "Total_fixed_cost[100M JPY]": [
                float(
                    tpg_ship.building_cost
                    + st_base.building_cost
                    + sp_base.building_cost
                    + support_ship_1.building_cost
                    + support_ship_2.building_cost
                )
            ],
            "Total_cost[100M JPY]": [float(total_cost)],
            "Total_profit[100M JPY]": [float(total_profit)],
            "Unit_price[JPY]": [float(sp_base.unit_price)],
            "Ideal_unit_price[JPY]": [float(appropriate_unit_price)],
            "Objective_value": [
                float(
                    objective_value_calculation(
                        tpg_ship,
                        st_base,
                        sp_base,
                        support_ship_1,
                        support_ship_2,
                        simulation_start_time,
                        simulation_end_time,
                    )
                )
            ],
        }
    )

    # 念の為、データ型をcast
    data = data.with_columns(
        [
            pl.col("T_hull_num").cast(pl.Int64),
            pl.col("T_storage_method").cast(pl.Int64),
            pl.col("T_max_storage[GWh]").cast(pl.Float64),
            pl.col("T_EP_max_storage_wh[GWh]").cast(pl.Float64),
            pl.col("T_sail_num").cast(pl.Int64),
            pl.col("T_sail_area[m2]").cast(pl.Int64),
            pl.col("T_sail_width[m]").cast(pl.Float64),
            pl.col("T_sail_height[m]").cast(pl.Float64),
            pl.col("T_sail_space").cast(pl.Float64),
            pl.col("T_sail_steps").cast(pl.Int64),
            pl.col("T_sail_weight").cast(pl.Float64),
            pl.col("T_num_sails_per_row").cast(pl.Int64),
            pl.col("T_num_sails_rows").cast(pl.Int64),
            pl.col("T_sail_penalty").cast(pl.Float64),
            pl.col("T_dwt[t]").cast(pl.Float64),
            pl.col("T_hull_L_oa[m]").cast(pl.Float64),
            pl.col("T_hull_B[m]").cast(pl.Float64),
            pl.col("T_trust_efficiency").cast(pl.Float64),
            pl.col("T_carrier_to_elect_efficiency").cast(pl.Float64),
            pl.col("T_elect_to_carrier_efficiency").cast(pl.Float64),
            pl.col("T_generator_num").cast(pl.Int64),
            pl.col("T_generator_turbine_radius[m]").cast(pl.Float64),
            pl.col("T_generator_pillar_width").cast(pl.Float64),
            pl.col("T_generator_rated_output[GW]").cast(pl.Float64),
            pl.col("T_generator_efficiency").cast(pl.Float64),
            pl.col("T_generator_drag_coefficient").cast(pl.Float64),
            pl.col("T_generator_pillar_chord").cast(pl.Float64),
            pl.col("T_generator_pillar_max_tickness").cast(pl.Float64),
            pl.col("T_generating_speed[kt]").cast(pl.Float64),
            pl.col("T_tpgship_return_speed[kt]").cast(pl.Float64),
            pl.col("T_forecast_weight").cast(pl.Float64),
            pl.col("T_govia_base_judge_energy_storage_per").cast(pl.Float64),
            pl.col("T_judge_time_times").cast(pl.Float64),
            pl.col("T_operational_reserve_percentage").cast(pl.Int64),
            pl.col("T_standby_lat").cast(pl.Float64),
            pl.col("T_standby_lon").cast(pl.Float64),
            pl.col("T_typhoon_effective_range[km]").cast(pl.Int64),
            pl.col("T_total_gene_elect[GWh]").cast(pl.Float64),
            pl.col("T_total_gene_carrier[GWh]").cast(pl.Float64),
            pl.col("T_total_loss_elect[GWh]").cast(pl.Float64),
            pl.col("T_sum_supply_elect[GWh]").cast(pl.Float64),
            pl.col("T_minus_storage_penalty").cast(pl.Int64),
            pl.col("St_base_type").cast(pl.Int64),
            pl.col("St_lat").cast(pl.Float64),
            pl.col("St_lon").cast(pl.Float64),
            pl.col("St_max_storage[GWh]").cast(pl.Float64),
            pl.col("St_call_per").cast(pl.Float64),
            pl.col("St_total_supply[GWh]").cast(pl.Float64),
            pl.col("Sp_base_type").cast(pl.Int64),
            pl.col("Sp_lat").cast(pl.Float64),
            pl.col("Sp_lon").cast(pl.Float64),
            pl.col("Sp_max_storage[GWh]").cast(pl.Float64),
            pl.col("Sp_total_supply[GWh]").cast(pl.Float64),
            pl.col("Ss1_max_storage[GWh]").cast(pl.Float64),
            pl.col("Ss1_ship_speed[kt]").cast(pl.Int64),
            pl.col("Ss1_EP_max_storage[GWh]").cast(pl.Float64),
            pl.col("Ss1_Total_consumption_elect[GWh]").cast(pl.Float64),
            pl.col("Ss1_Total_received_elect[GWh]").cast(pl.Float64),
            pl.col("Ss2_max_storage[GWh]").cast(pl.Float64),
            pl.col("Ss2_ship_speed[kt]").cast(pl.Int64),
            pl.col("Ss2_EP_max_storage[GWh]").cast(pl.Float64),
            pl.col("Ss2_Total_consumption_elect[GWh]").cast(pl.Float64),
            pl.col("Ss2_Total_received_elect[GWh]").cast(pl.Float64),
            pl.col("T_hull_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_wing_sail_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_underwater_turbine_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_battery_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_building_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_carrier_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_maintenance_cost[100M JPY]").cast(pl.Float64),
            pl.col("T_depreciation_expense[100M JPY]").cast(pl.Float64),
            pl.col("T_total_cost[100M JPY]").cast(pl.Float64),
            pl.col("St_building_cost[100M JPY]").cast(pl.Float64),
            pl.col("St_maintenance_cost[100M JPY]").cast(pl.Float64),
            pl.col("St_depreciation_expense[100M JPY]").cast(pl.Float64),
            pl.col("St_total_cost[100M JPY]").cast(pl.Float64),
            pl.col("Sp_building_cost[100M JPY]").cast(pl.Float64),
            pl.col("Sp_maintenance_cost[100M JPY]").cast(pl.Float64),
            pl.col("Sp_depreciation_expense[100M JPY]").cast(pl.Float64),
            pl.col("Sp_total_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss1_building_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss1_maintenance_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss1_transportation_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss1_depreciation_expense[100M JPY]").cast(pl.Float64),
            pl.col("Ss1_total_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss2_building_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss2_maintenance_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss2_transportation_cost[100M JPY]").cast(pl.Float64),
            pl.col("Ss2_depreciation_expense[100M JPY]").cast(pl.Float64),
            pl.col("Ss2_total_cost[100M JPY]").cast(pl.Float64),
            pl.col("Total_maintain_cost_per_Year[100M JPY]").cast(pl.Float64),
            pl.col("Total_operating_cost_per_Year[100M JPY]").cast(pl.Float64),
            pl.col("Total_depreciation_expense_per_Year[100M JPY]").cast(pl.Float64),
            pl.col("Total_fixed_cost[100M JPY]").cast(pl.Float64),
            pl.col("Total_cost[100M JPY]").cast(pl.Float64),
            pl.col("Total_profit[100M JPY]").cast(pl.Float64),
            pl.col("Unit_price[JPY]").cast(pl.Float64),
            pl.col("Ideal_unit_price[JPY]").cast(pl.Float64),
            pl.col("Objective_value").cast(pl.Float64),
        ]
    )

    return data


def save_to_csv_on_error_or_completion(file_path):
    """
    save_dataframeをCSVとして保存する関数
    """
    global save_dataframe
    if save_dataframe.height > 0:  # データが存在する場合のみ保存
        save_dataframe.write_csv(file_path)
        print(f"データを保存しました: {file_path}")
    else:
        print("データが存在しないため保存はスキップされました。")


def run_simulation(cfg):
    global output_folder_path, save_dataframe, final_csv_path

    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    models_param_log_file_name = cfg.output_env.models_param_log_file_name

    final_csv_path = output_folder_path + "/" + models_param_log_file_name

    # TPG ship
    initial_position = cfg.tpg_ship.initial_position
    hull_num = cfg.tpg_ship.hull_num
    storage_method = cfg.tpg_ship.storage_method
    max_storage_wh = cfg.tpg_ship.max_storage_wh
    electric_propulsion_max_storage_wh = cfg.tpg_ship.electric_propulsion_max_storage_wh
    trust_efficiency = cfg.tpg_ship.trust_efficiency
    carrier_to_elect_efficiency = cfg.tpg_ship.carrier_to_elect_efficiency
    elect_to_carrier_efficiency = cfg.tpg_ship.elect_to_carrier_efficiency
    generator_turbine_radius = cfg.tpg_ship.generator_turbine_radius
    generator_efficiency = cfg.tpg_ship.generator_efficiency
    generator_drag_coefficient = cfg.tpg_ship.generator_drag_coefficient
    generator_pillar_chord = cfg.tpg_ship.generator_pillar_chord
    generator_pillar_max_tickness = cfg.tpg_ship.generator_pillar_max_tickness
    generator_pillar_width = generator_turbine_radius + 1
    generator_num = cfg.tpg_ship.generator_num
    sail_area = cfg.tpg_ship.sail_area
    sail_space = cfg.tpg_ship.sail_space

    sail_num = calculate_max_sail_num(
        storage_method,
        max_storage_wh,
        electric_propulsion_max_storage_wh,
        hull_num,
        sail_area,
        sail_space,
    )
    sail_num = int(sail_num)  # 整数でない出力になぜかなることがあったので予防策

    sail_steps = cfg.tpg_ship.sail_steps
    ship_return_speed_kt = cfg.tpg_ship.ship_return_speed_kt
    ship_max_speed_kt = cfg.tpg_ship.ship_max_speed_kt
    forecast_weight = cfg.tpg_ship.forecast_weight
    typhoon_effective_range = cfg.tpg_ship.typhoon_effective_range
    govia_base_judge_energy_storage_per = (
        cfg.tpg_ship.govia_base_judge_energy_storage_per
    )
    judge_time_times = cfg.tpg_ship.judge_time_times
    operational_reserve_percentage = cfg.tpg_ship.operational_reserve_percentage
    standby_position = cfg.tpg_ship.standby_position

    tpg_ship_1 = tpg_ship.TPG_ship(
        initial_position,
        hull_num,
        storage_method,
        max_storage_wh,
        electric_propulsion_max_storage_wh,
        trust_efficiency,
        carrier_to_elect_efficiency,
        elect_to_carrier_efficiency,
        generator_turbine_radius,
        generator_efficiency,
        generator_drag_coefficient,
        generator_pillar_chord,
        generator_pillar_max_tickness,
        generator_pillar_width,
        generator_num,
        sail_area,
        sail_space,
        sail_num,
        sail_steps,
        ship_return_speed_kt,
        ship_max_speed_kt,
        forecast_weight,
        typhoon_effective_range,
        govia_base_judge_energy_storage_per,
        judge_time_times,
        operational_reserve_percentage,
        standby_position,
    )

    # Forecaster
    forecast_time = cfg.forecaster.forecast_time
    forecast_error_slope = cfg.forecaster.forecast_error_slope
    typhoon_path_forecaster = forecaster.Forecaster(forecast_time, forecast_error_slope)

    # Storage base
    st_base_type = cfg.storage_base.base_type
    st_base_locate = cfg.storage_base.locate
    st_base_max_storage_wh = cfg.storage_base.max_storage_wh
    st_base_call_per = cfg.storage_base.call_per
    st_base = base.Base(
        st_base_type, st_base_locate, st_base_max_storage_wh, st_base_call_per
    )

    # Supply base
    sp_base_type = cfg.supply_base.base_type
    sp_base_locate = cfg.supply_base.locate
    sp_base_max_storage_wh = cfg.supply_base.max_storage_wh
    sp_base_call_per = cfg.supply_base.call_per
    sp_base = base.Base(
        sp_base_type, sp_base_locate, sp_base_max_storage_wh, sp_base_call_per
    )

    # Support ship 1
    support_ship_1_supply_base_locate = cfg.supply_base.locate
    support_ship_1_storage_method = cfg.tpg_ship.storage_method
    support_ship_1_max_storage_wh = cfg.support_ship_1.max_storage_wh
    support_ship_1_support_ship_speed = cfg.support_ship_1.ship_speed_kt
    support_ship_1_elect_trust_efficiency = cfg.support_ship_1.elect_trust_efficiency
    if support_ship_1_max_storage_wh == 0:
        support_ship_1_EP_max_storage_wh = 0
    else:
        support_ship_1_EP_max_storage_wh = sp_ship_EP_storage_cal(
            support_ship_1_storage_method,
            support_ship_1_max_storage_wh,
            support_ship_1_support_ship_speed,
            support_ship_1_elect_trust_efficiency,
            st_base_locate,
            sp_base_locate,
        )
    support_ship_1 = support_ship.Support_ship(
        support_ship_1_supply_base_locate,
        support_ship_1_storage_method,
        support_ship_1_max_storage_wh,
        support_ship_1_support_ship_speed,
        support_ship_1_EP_max_storage_wh,
        support_ship_1_elect_trust_efficiency,
    )

    # Support ship 2
    support_ship_2_supply_base_locate = cfg.supply_base.locate
    support_ship_2_storage_method = cfg.tpg_ship.storage_method
    support_ship_2_max_storage_wh = cfg.support_ship_2.max_storage_wh
    support_ship_2_support_ship_speed = cfg.support_ship_2.ship_speed_kt
    support_ship_2_elect_trust_efficiency = cfg.support_ship_2.elect_trust_efficiency
    if support_ship_2_max_storage_wh == 0:
        support_ship_2_EP_max_storage_wh = 0
    else:
        support_ship_2_EP_max_storage_wh = sp_ship_EP_storage_cal(
            support_ship_2_storage_method,
            support_ship_2_max_storage_wh,
            support_ship_2_support_ship_speed,
            support_ship_2_elect_trust_efficiency,
            st_base_locate,
            sp_base_locate,
        )
    support_ship_2 = support_ship.Support_ship(
        support_ship_2_supply_base_locate,
        support_ship_2_storage_method,
        support_ship_2_max_storage_wh,
        support_ship_2_support_ship_speed,
        support_ship_2_EP_max_storage_wh,
        support_ship_2_elect_trust_efficiency,
    )

    # Run simulation
    simulator_optimize.simulate(
        simulation_start_time,
        simulation_end_time,
        tpg_ship_1,
        typhoon_path_forecaster,
        st_base,
        sp_base,
        support_ship_1,
        support_ship_2,
        typhoon_data_path,
        output_folder_path,
    )

    # 目的関数の値を算出
    objective_value = objective_value_calculation(
        tpg_ship_1,
        st_base,
        sp_base,
        support_ship_1,
        support_ship_2,
        simulation_start_time,
        simulation_end_time,
    )

    # 結果をデータフレームに出力
    data = simulation_result_to_df(
        tpg_ship_1,
        st_base,
        sp_base,
        support_ship_1,
        support_ship_2,
        simulation_start_time,
        simulation_end_time,
    )

    # グローバルのデータフレームに追記
    save_dataframe = save_dataframe.vstack(data)

    return objective_value


# 探索範囲の指定用関数
def objective(trial):
    global sim_year

    config = hydra.compose(config_name="config")

    config.env.typhoon_data_path = (
        f"data/typhoon_path/filtered_typhoon_path/filtered_typhoon_data_{sim_year}.csv"
    )
    config.env.simulation_start_time = f"{sim_year}-05-15 00:00:00"
    config.env.simulation_end_time = f"{sim_year}-11-15 00:00:00"

    ############ TPG shipのパラメータを指定 ############

    # config.tpg_ship.hull_num = 1

    # config.tpg_ship.hull_num = trial.suggest_int("hull_num", 1, 2)
    # 1: 電気(コンテナ型), 2: MCH(タンカー型), 3: メタン(LNG船型), 4: メタノール(ケミカルタンカー型), 5: e-ガソリン(タンカー型)
    config.tpg_ship.storage_method = 3  # trial.suggest_int("storage_method", 1, 5)

    max_storage_GWh = trial.suggest_int(
        "tpgship_max_storage_GWh", 50, 1000
    )  # max_storage_whの刻み幅は10^9とする
    config.tpg_ship.max_storage_wh = max_storage_GWh * 1000000000

    # EP_max_storage_GWh_10 = trial.suggest_int(
    #     "tpgship_EP_max_storage_GWh_10", 5, 200
    # )  # electric_propulsion_max_storage_whの刻み幅は10^8とする
    # config.tpg_ship.electric_propulsion_max_storage_wh = (
    #     EP_max_storage_GWh_10 * 100000000
    # )
    config.tpg_ship.electric_propulsion_max_storage_wh = 0.0

    config.tpg_ship.trust_efficiency = (
        0.68  # trial.suggest_float("tpgship_elect_trust_efficiency", 0.7, 0.9)
    )
    config.tpg_ship.carrier_to_elect_efficiency = (
        1.0  # trial.suggest_float("tpgship_MCH_to_elect_efficiency", 0.4, 0.6)
    )
    config.tpg_ship.elect_to_carrier_efficiency = (
        0.80  # trial.suggest_float("tpgship_elect_to_MCH_efficiency", 0.7, 0.9)
    )
    # config.tpg_ship.sail_num = trial.suggest_int("tpgship_sail_num", 10, 60)
    sail_area_100m2 = trial.suggest_int("tpgship_sail_area_every_100m2", 50, 100)
    config.tpg_ship.sail_area = sail_area_100m2 * 100
    # config.tpg_ship.sail_space = trial.suggest_float("sail_space", 2, 4)
    config.tpg_ship.sail_steps = trial.suggest_int("tpgship_sail_steps", 1, 7)
    config.tpg_ship.ship_return_speed_kt = trial.suggest_int(
        "tpgship_return_speed_kt", 4, 20
    )
    config.tpg_ship.generator_turbine_radius = trial.suggest_int(
        "tpgship_generator_turbine_radius", 5, 25
    )
    config.tpg_ship.forecast_weight = trial.suggest_int(
        "tpgship_forecast_weight", 10, 90
    )
    # config.tpg_ship.typhoon_effective_range = trial.suggest_int("typhoon_effective_range", 50, 150)
    config.tpg_ship.govia_base_judge_energy_storage_per = trial.suggest_int(
        "tpgship_govia_base_judge_energy_storage_per", 10, 90
    )
    config.tpg_ship.judge_time_times = trial.suggest_float(
        "tpgship_judge_time_times", 1.0, 2.0
    )

    config.tpg_ship.operational_reserve_percentage = trial.suggest_int(
        "tpgship_operational_reserve_percentage", 0, 50
    )

    tpgship_standby_lat = trial.suggest_int("tpgship_standby_lat", 0, 30)
    tpgship_standby_lon = trial.suggest_int("tpgship_standby_lon", 134, 180)
    config.tpg_ship.standby_position = [tpgship_standby_lat, tpgship_standby_lon]

    ############ Storage Baseのパラメータを指定 ############

    # 拠点位置に関する変更
    # stbase_lat = trial.suggest_int("stbase_lat", 0, 30)
    # stbase_lon = trial.suggest_int("stbase_lon", 134, 180)
    # config.storage_base.locate = [stbase_lat, stbase_lon]
    # config.tpg_ship.initial_position = config.storage_base.locate
    stbase_list = [
        [24.47, 122.98],  # 与那国島
        [25.83, 131.23],  # 南大東島
        [24.78, 141.32],  # 硫黄島
        [20.42, 136.08],  # 沖ノ鳥島
        [24.29, 153.98],  # 南鳥島
    ]
    stbase_locate = 4
    config.storage_base.locate = stbase_list[stbase_locate]
    config.tpg_ship.initial_position = config.storage_base.locate
    # 貯蔵量に関する変更 (先に10万トン単位で決めてから1GWhあたり379トンとしてWhに変換)
    config.storage_base.max_storage_wh = 300 * 10**9

    # 輸送船呼び出しタイミングに関する変更
    config.storage_base.call_per = 1

    ############ Supply Baseのパラメータを指定 ############

    # 拠点位置に関する変更
    # 候補となる場所のリストから選択する
    spbase_list = [
        [35.48, 139.66],  # ENEOS横浜製造所
        [38.27, 141.04],  # ENEOS仙台製油所
        [34.11, 135.11],  # ENEOS和歌山製造所
        [33.28, 131.69],  # ENEOS大分製油所
    ]
    spbase_locate = 0
    config.supply_base.locate = spbase_list[spbase_locate]
    # 貯蔵量に関する変更 (先に10万トン単位で決めてから1GWhあたり379トンとしてWhに変換)
    config.supply_base.max_storage_wh = 300 * 10**9

    ############ Support Ship 1のパラメータを指定 ############

    # 貯蔵量に関する変更
    support_ship_1_max_storage_GWh = 100
    config.support_ship_1.max_storage_wh = support_ship_1_max_storage_GWh * 1000000000
    # 船速に関する変更
    support_ship_1_ship_speed_kt = 8
    config.support_ship_1.ship_speed_kt = support_ship_1_ship_speed_kt

    ############ Support Ship 2のパラメータを指定 ############

    # 貯蔵量に関する変更
    support_ship_2_max_storage_GWh = 100
    config.support_ship_2.max_storage_wh = support_ship_2_max_storage_GWh * 1000000000
    # 船速に関する変更
    support_ship_2_ship_speed_kt = 8
    config.support_ship_2.ship_speed_kt = support_ship_2_ship_speed_kt

    # シミュレーションを実行
    objective_value = run_simulation(config)

    return objective_value


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    global output_folder_path, save_dataframe, final_csv_path
    output_folder_path = HydraConfig.get().run.dir
    save_dataframe = pl.DataFrame()
    models_param_log_file_name = cfg.output_env.models_param_log_file_name
    final_csv_path = output_folder_path + "/" + models_param_log_file_name

    # ローカルフォルダに保存するためのストレージURLを指定します。
    storage = f"sqlite:///experiences/MAN_{sim_year}s.db"
    # スタディの作成または既存のスタディのロード
    study = optuna.create_study(
        study_name="example-study",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # ログ出力を無効化　ターミナルが落ちることがあったため予防措置
    optuna.logging.disable_default_handler()

    n_jobs = int((os.cpu_count()))
    print(f"Number of CPUs: {n_jobs}")

    # 進捗バーのコールバックを使用してoptimizeを実行
    trial_num = 500
    try:
        # 進捗バーのコールバックを使用してoptimizeを実行
        study.optimize(
            objective,
            n_trials=trial_num,
            callbacks=[TqdmCallback(total=trial_num)],
            n_jobs=n_jobs,
        )

    except Exception as e:
        # エラー時の処理
        print(f"エラーが発生しました: {e}")
        traceback.print_exc()
        save_to_csv_on_error_or_completion(final_csv_path)
        raise

    finally:
        # トライアル終了時の保存
        save_to_csv_on_error_or_completion(final_csv_path)

    # 最良の試行を出力
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
