import csv
import math
import os
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

    if storage_method == 1:  # 電気貯蔵
        # 重量エネルギー密度1000Wh/kgの電池を使うこととする。
        dwt = storage / 1000 / 1000

    elif storage_method == 2:  # 水素貯蔵
        # 有機ハイドライドで水素を貯蔵することとする。
        dwt = storage / 5000 * 0.0898 / 47.4

    else:
        print("cannot cal")

    return dwt


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
        if storage_method == 1:  # 電気貯蔵 = バルカー型
            if total_ship_weight_per_body < 220000:
                L_oa = 7.9387 * (total_ship_weight_per_body**0.2996)
                B = 1.4257 * (total_ship_weight_per_body**0.2883)

            elif 220000 <= total_ship_weight_per_body < 330000:
                L_oa = 139.3148 * (total_ship_weight_per_body**0.069)
                B = 13.8365 * (total_ship_weight_per_body**0.1127)

            else:
                L_oa = 361.2
                B = 65.0

        elif storage_method == 2:  # 水素貯蔵 = タンカー型
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
    k = 2.2 / (1.852**3)

    # 輸送船の電気貯蔵量を計算
    # geopyで貯蔵拠点から供給拠点までの距離を大圏距離で計算
    distance = geodesic(st_base_locate, sp_base_locate).kilometers
    # support_ship_speed_ktをkm/hに変換
    max_speed_kmh = support_ship_speed_kt * 1.852

    # max_storage_whをDWTに変換　MCHのタンクとして2を指定してcal_dwt関数を使う
    max_storage_ton = cal_dwt(2, max_storage_wh)

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
        + tpg_ship.toluene_cost
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
    st_base.cost_calculate()
    st_base_total_cost = (
        st_base.building_cost + st_base.maintenance_cost * operating_years
    )
    # 供給拠点関連[億円]
    sp_base.cost_calculate()
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

    data = pl.DataFrame(
        {
            # TPG ship (列名の先頭にT_を付与。探索しないものはコメントアウト)
            ## 装置パラメータ関連
            # "hull_num": [tpg_ship.hull_num],
            # "storage_method": [tpg_ship.storage_method],
            "T_max_storage[GWh]": [tpg_ship.max_storage / 10**9],
            "T_EP_max_storage_wh[GWh]": [
                tpg_ship.electric_propulsion_max_storage_wh / 10**8
            ],
            "T_sail_num": [tpg_ship.sail_num],
            "T_sail_area[m2]": [tpg_ship.sail_area],
            "T_sail_width[m]": [tpg_ship.sail_width],
            "T_sail_height[m]": [tpg_ship.sail_height],
            "T_sail_space": [tpg_ship.sail_space],
            "T_sail_steps": [tpg_ship.sail_steps],
            "T_sail_weight": [tpg_ship.sail_weight],
            "T_num_sails_per_row": [tpg_ship.num_sails_per_row],
            "T_num_sails_rows": [tpg_ship.num_sails_rows],
            "T_sail_penalty": [tpg_ship.sail_penalty],
            "T_dwt[t]": [tpg_ship.ship_dwt],
            "T_hull_L_oa[m]": [tpg_ship.hull_L_oa],
            "T_hull_B[m]": [tpg_ship.hull_B],
            # "elect_trust_efficiency": [tpg_ship.elect_trust_efficiency],
            # "MCH_to_elect_efficiency": [tpg_ship.MCH_to_elect_efficiency],
            # "elect_to_MCH_efficiency": [tpg_ship.elect_to_MCH_efficiency],
            "T_generator_num": [tpg_ship.generator_num],
            "T_generator_turbine_radius[m]": [tpg_ship.generator_turbine_radius],
            "T_generator_pillar_width": [tpg_ship.generator_pillar_width],
            "T_generator_rated_output[GW]": [tpg_ship.generator_rated_output_w / 10**9],
            # "generator_efficiency": [tpg_ship.generator_efficiency],
            # "generator_drag_coefficient": [tpg_ship.generator_drag_coefficient],
            # "generator_pillar_chord": [tpg_ship.generator_pillar_chord],
            # "generator_pillar_max_tickness": [tpg_ship.generator_pillar_max_tickness],
            "T_generating_speed[kt]": [tpg_ship.generating_speed_kt],
            ## 航行・判断パラメータ関連
            "T_tpgship_return_speed[kt]": [tpg_ship.nomal_ave_speed],
            # "max_speed": [tpg_ship.max_speed],
            "T_forecast_weight": [tpg_ship.forecast_weight],
            "govia_base_judge_energy_storage_per": [
                tpg_ship.govia_base_judge_energy_storage_per
            ],
            "T_judge_time_times": [tpg_ship.judge_time_times],
            "T_operational_reserve_percentage": tpg_ship.operational_reserve_percentage,
            "T_standby_lat": [tpg_ship.standby_lat],
            "T_standby_lon": [tpg_ship.standby_lon],
            # "typhoon_effective_range": [tpg_ship.typhoon_effective_range],
            ## シミュレーション結果関連
            "T_total_gene_elect(mch)[GWh]": tpg_ship.total_gene_elect_list[-1] / 10**9,
            "T_total_loss_elect[GWh]": tpg_ship.total_loss_elect_list[-1] / 10**9,
            "T_sum_supply_elect[GWh]": tpg_ship.sum_supply_elect_list[-1] / 10**9,
            "T_minus_storage_penalty": tpg_ship.minus_storage_penalty_list[-1],
            # Storage base (列名の先頭にSt_を付与。探索しないものはコメントアウト)
            "St_base_type": [st_base.base_type],
            "St_lat": [st_base.locate[0]],
            "St_lon": [st_base.locate[1]],
            "St_max_storage[GWh]": [st_base.max_storage / 10**9],
            "St_call_per": [st_base.call_per],
            "St_total_supply[GWh]": [st_base.total_quantity_received_list[-1] / 10**9],
            # Supply base (列名の先頭にSp_を付与。探索しないものはコメントアウト)
            "Sp_base_type": [sp_base.base_type],
            "Sp_lat": [sp_base.locate[0]],
            "Sp_lon": [sp_base.locate[1]],
            "Sp_max_storage[GWh]": [sp_base.max_storage / 10**9],
            # "Sp_call_per": [sp_base.call_per],
            "Sp_total_supply[GWh]": [sp_base.total_supply_list[-1] / 10**9],
            # Support ship 1 (列名の先頭にSs1_を付与。探索しないものはコメントアウト)
            "Ss1_max_storage[GWh]": [support_ship_1.max_storage / 10**9],
            "Ss1_ship_speed[kt]": [support_ship_1.support_ship_speed],
            "Ss1_EP_max_storage[GWh]": [support_ship_1.EP_max_storage / 10**9],
            "Ss1_Total_consumption_elect[GWh]": [
                support_ship_1.sp_total_consumption_elect_list[-1] / 10**9
            ],
            "Ss1_Total_received_elect[GWh]": [
                support_ship_1.sp_total_received_elect_list[-1] / 10**9
            ],
            # Support ship 2 (列名の先頭にSs2_を付与。探索しないものはコメントアウト)
            "Ss2_max_storage[GWh]": [support_ship_2.max_storage / 10**9],
            "Ss2_ship_speed[kt]": [support_ship_2.support_ship_speed],
            "Ss2_EP_max_storage[GWh]": [support_ship_2.EP_max_storage / 10**9],
            "Ss2_Total_consumption_elect[GWh]": [
                support_ship_2.sp_total_consumption_elect_list[-1] / 10**9
            ],
            "Ss2_Total_received_elect[GWh]": [
                support_ship_2.sp_total_received_elect_list[-1] / 10**9
            ],
            # コスト関連
            "T_total_cost[100M JPY]": [tpg_ship_total_cost],
            "St_total_cost[100M JPY]": [st_base_total_cost],
            "Sp_total_cost[100M JPY]": [sp_base_total_cost],
            "Ss1_total_cost[100M JPY]": [support_ship_1_total_cost],
            "Ss2_total_cost[100M JPY]": [support_ship_2_total_cost],
            "Total_cost[100M JPY]": [total_cost],
            "Total_profit[100M JPY]": [total_profit],
        }
    )

    return data


def run_simulation(cfg):
    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    output_folder_path = HydraConfig.get().run.dir

    models_param_log_file_name = cfg.output_env.models_param_log_file_name

    final_csv_path = output_folder_path + "/" + models_param_log_file_name

    # TPG ship
    initial_position = cfg.tpg_ship.initial_position
    hull_num = cfg.tpg_ship.hull_num
    storage_method = cfg.tpg_ship.storage_method
    max_storage_wh = cfg.tpg_ship.max_storage_wh
    electric_propulsion_max_storage_wh = cfg.tpg_ship.electric_propulsion_max_storage_wh
    elect_trust_efficiency = cfg.tpg_ship.elect_trust_efficiency
    MCH_to_elect_efficiency = cfg.tpg_ship.MCH_to_elect_efficiency
    elect_to_MCH_efficiency = cfg.tpg_ship.elect_to_MCH_efficiency
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
        elect_trust_efficiency,
        MCH_to_elect_efficiency,
        elect_to_MCH_efficiency,
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
    support_ship_1_max_storage_wh = cfg.support_ship_1.max_storage_wh
    support_ship_1_support_ship_speed = cfg.support_ship_1.ship_speed_kt
    support_ship_1_elect_trust_efficiency = cfg.support_ship_1.elect_trust_efficiency
    if support_ship_1_max_storage_wh == 0:
        support_ship_1_EP_max_storage_wh = 0
    else:
        support_ship_1_EP_max_storage_wh = sp_ship_EP_storage_cal(
            support_ship_1_max_storage_wh,
            support_ship_1_support_ship_speed,
            support_ship_1_elect_trust_efficiency,
            st_base_locate,
            sp_base_locate,
        )
    support_ship_1 = support_ship.Support_ship(
        support_ship_1_supply_base_locate,
        support_ship_1_max_storage_wh,
        support_ship_1_support_ship_speed,
        support_ship_1_EP_max_storage_wh,
        support_ship_1_elect_trust_efficiency,
    )

    # Support ship 2
    support_ship_2_supply_base_locate = cfg.supply_base.locate
    support_ship_2_max_storage_wh = cfg.support_ship_2.max_storage_wh
    support_ship_2_support_ship_speed = cfg.support_ship_2.ship_speed_kt
    support_ship_2_elect_trust_efficiency = cfg.support_ship_2.elect_trust_efficiency
    if support_ship_2_max_storage_wh == 0:
        support_ship_2_EP_max_storage_wh = 0
    else:
        support_ship_2_EP_max_storage_wh = sp_ship_EP_storage_cal(
            support_ship_2_max_storage_wh,
            support_ship_2_support_ship_speed,
            support_ship_2_elect_trust_efficiency,
            st_base_locate,
            sp_base_locate,
        )
    support_ship_2 = support_ship.Support_ship(
        support_ship_2_supply_base_locate,
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

    # # 供給拠点に輸送された電力量を取得
    # print(sp_base.total_supply)
    # objective_value = (
    #     sp_base.total_supply / (10**9) - tpg_ship_1.minus_storage_penalty_list[-1]
    # )

    # コスト計算(損失)
    # 運用年数　simulation_start_time、simulation_end_time (ex."2023-01-01 00:00:00")から年数を計算 365で割って端数切り上げ
    operating_years = math.ceil(
        (
            datetime.strptime(simulation_end_time, "%Y-%m-%d %H:%M:%S")
            - datetime.strptime(simulation_start_time, "%Y-%m-%d %H:%M:%S")
        ).days
        / 365
    )
    # print(f"運用年数: {operating_years}年")

    # 台風発電船関連[億円]
    tpg_ship_1.cost_calculate()
    tpg_ship_total_cost = (
        tpg_ship_1.building_cost
        + tpg_ship_1.toluene_cost
        + tpg_ship_1.maintenance_cost * operating_years
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
    st_base.cost_calculate()
    st_base_total_cost = (
        st_base.building_cost + st_base.maintenance_cost * operating_years
    )
    # 供給拠点関連[億円]
    sp_base.cost_calculate()
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

    # 利益強め
    objective_value = (
        total_profit - total_cost - tpg_ship_1.minus_storage_penalty_list[-1]
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

    # final_csv_pathの既存ファイルにdfを追記
    if os.path.exists(final_csv_path):
        # ファイルが存在する場合、既存のデータを読み込む
        existing_data = pl.read_csv(final_csv_path)
        # 新しいデータを既存のデータに追加
        updated_data = existing_data.vstack(data)
        # 更新されたデータをCSVファイルに書き込む
        updated_data.write_csv(final_csv_path)
    else:
        # ファイルが存在しない場合、ヘッダーを出力する
        data.write_csv(final_csv_path)

    return objective_value


# 探索範囲の指定用関数
def objective(trial):
    config = hydra.compose(config_name="config")

    ############ TPG shipのパラメータを指定 ############

    # config.tpg_ship.hull_num = 1

    # config.tpg_ship.hull_num = trial.suggest_int("hull_num", 1, 2)
    # config.tpg_ship.storage_method = trial.suggest_int("storage_method", 1, 2)

    max_storage_GWh = trial.suggest_int(
        "tpgship_max_storage_GWh", 50, 1500
    )  # max_storage_whの刻み幅は10^9とする
    config.tpg_ship.max_storage_wh = max_storage_GWh * 1000000000

    EP_max_storage_GWh_10 = trial.suggest_int(
        "tpgship_EP_max_storage_GWh_10", 5, 200
    )  # electric_propulsion_max_storage_whの刻み幅は10^8とする
    config.tpg_ship.electric_propulsion_max_storage_wh = (
        EP_max_storage_GWh_10 * 100000000
    )

    # config.tpg_ship.elect_trust_efficiency = trial.suggest_float("tpgship_elect_trust_efficiency", 0.7, 0.9)
    # config.tpg_ship.MCH_to_elect_efficiency = trial.suggest_float("tpgship_MCH_to_elect_efficiency", 0.4, 0.6)
    # config.tpg_ship.elect_to_MCH_efficiency = trial.suggest_float("tpgship_elect_to_MCH_efficiency", 0.7, 0.9)
    # config.tpg_ship.sail_num = trial.suggest_int("tpgship_sail_num", 10, 60)
    sail_area_100m2 = trial.suggest_int("tpgship_sail_area_every_100m2", 50, 200)
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
        [24.47, 122.98],
        [25.83, 131.23],
        [24.78, 141.32],
        [20.42, 136.08],  # 与那国島  # 南大東島  # 硫黄島
        [24.29, 153.98],  # 沖ノ鳥島  # 南鳥島
    ]
    stbase_locate = trial.suggest_int("stbase_locate", 0, 4)
    config.storage_base.locate = stbase_list[stbase_locate]
    # 貯蔵量に関する変更 (先に10万トン単位で決めてから1GWhあたり379トンとしてWhに変換)
    stbase_max_storage_ton_100k = trial.suggest_int(
        "stbase_max_storage_ton_100k", 1, 15
    )
    stbase_max_storage_ton = stbase_max_storage_ton_100k * 100000
    config.storage_base.max_storage_wh = (stbase_max_storage_ton / 379) * 10**9

    # 輸送船呼び出しタイミングに関する変更
    # config.storage_base.call_per = trial.suggest_int("stbase_call_per", 10, 100)

    ############ Supply Baseのパラメータを指定 ############

    # 拠点位置に関する変更
    # 候補となる場所のリストから選択する
    spbase_list = [
        [34.74, 134.78],  # 高砂水素パーク
        [35.48, 139.66],  # ENEOS横浜製造所
        [38.27, 141.04],  # ENEOS仙台製油所
        [34.11, 135.11],  # ENEOS和歌山製造所
        [33.28, 131.69],  # ENEOS大分製油所
    ]
    spbase_locate = trial.suggest_int("spbase_locate", 0, 4)
    config.supply_base.locate = spbase_list[spbase_locate]
    # 貯蔵量に関する変更 (先に10万トン単位で決めてから1GWhあたり379トンとしてWhに変換)
    spbase_max_storage_ton_100k = trial.suggest_int(
        "spbase_max_storage_ton_100k", 1, 15
    )
    spbase_max_storage_ton = spbase_max_storage_ton_100k * 100000
    config.supply_base.max_storage_wh = (spbase_max_storage_ton / 379) * 10**9
    # 輸送船呼び出しタイミングに関する変更(多分使うことはない)
    # config.supply_base.call_per = trial.suggest_int("spbase_call_per", 10, 100)

    ############ Support Ship 1のパラメータを指定 ############

    # 貯蔵量に関する変更
    support_ship_1_max_storage_GWh = trial.suggest_int(
        "support_ship_1_max_storage_GWh", 10, 1500
    )
    config.support_ship_1.max_storage_wh = support_ship_1_max_storage_GWh * 1000000000
    # 船速に関する変更
    support_ship_1_ship_speed_kt = trial.suggest_int(
        "support_ship_1_ship_speed_kt", 1, 20
    )
    config.support_ship_1.ship_speed_kt = support_ship_1_ship_speed_kt
    # # 電気推進効率に関する変更
    # config.support_ship_1.elect_trust_efficiency = trial.suggest_float(
    #     "support_ship_1_elect_trust_efficiency", 0.7, 0.9
    # )
    # # バッテリー容量に関する変更
    # support_ship_1_EP_max_storage = trial.suggest_int(
    #     "support_ship_1_EP_max_storage_GWh_10", 10, 1500
    # )
    # config.support_ship_1.EP_max_storage = support_ship_1_EP_max_storage * 10**8

    ############ Support Ship 2のパラメータを指定 ############

    # 貯蔵量に関する変更
    support_ship_2_max_storage_GWh = trial.suggest_int(
        "support_ship_2_max_storage_GWh", 0, 1500
    )
    config.support_ship_2.max_storage_wh = support_ship_2_max_storage_GWh * 1000000000
    # 船速に関する変更
    support_ship_2_ship_speed_kt = trial.suggest_int(
        "support_ship_2_ship_speed_kt", 1, 20
    )
    config.support_ship_2.ship_speed_kt = support_ship_2_ship_speed_kt
    # # 電気推進効率に関する変更
    # config.support_ship_2.elect_trust_efficiency = trial.suggest_float(
    #     "support_ship_2_elect_trust_efficiency", 0.7, 0.9
    # )
    # # バッテリー容量に関する変更
    # support_ship_2_EP_max_storage = trial.suggest_int(
    #     "support_ship_2_EP_max_storage_GWh_10", 10, 1500
    # )
    # config.support_ship_2.EP_max_storage = support_ship_2_EP_max_storage * 10**8

    # シミュレーションを実行
    objective_value = run_simulation(config)

    return objective_value


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    # ローカルフォルダに保存するためのストレージURLを指定します。
    # storage = "sqlite:///experiences/catmaran_journal_first_casestudy_neo.db"  # または storage = "sqlite:///path/to/your/folder/example.db"
    storage = "sqlite:///experiences/catamaran_cost_optimize.db"
    # スタディの作成または既存のスタディのロード
    study = optuna.create_study(
        study_name="example-study",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # 進捗バーのコールバックを使用してoptimizeを実行
    trial_num = 1000
    study.optimize(
        objective, n_trials=trial_num, callbacks=[TqdmCallback(total=trial_num)]
    )

    # 最良の試行を出力
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()