import math
from datetime import datetime, timedelta, timezone

import hydra
import optuna
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from tpg_ship_sim import simulator, utils
from tpg_ship_sim.model import base, forecaster, support_ship, tpg_ship


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

    # 総利益[億円]
    total_profit = sp_base.profit

    # 減価償却費 耐用年数について、船は一律20年、拠点はタンク部分は20年、その他は50年とする
    tpg_ship_depreciation_expense = tpg_ship.building_cost / 20
    support_ship_1_depreciation_expense = support_ship_1.building_cost / 20
    support_ship_2_depreciation_expense = support_ship_2.building_cost / 20
    st_base_depreciation_expense = (
        st_base.tank_total_cost / 20
        + (st_base.building_cost - st_base.tank_total_cost) / 50
    )
    sp_base_depreciation_expense = (
        sp_base.tank_total_cost / 20
        + (sp_base.building_cost - sp_base.tank_total_cost) / 50
    )

    # 総減価償却費[億円]
    total_depreciation_expense = (
        tpg_ship_depreciation_expense
        + support_ship_1_depreciation_expense
        + support_ship_2_depreciation_expense
        + st_base_depreciation_expense
        + sp_base_depreciation_expense
    )

    total_maintainance_cost = (
        tpg_ship.maintenance_cost
        + support_ship_1.maintenance_cost
        + support_ship_2.maintenance_cost
        + st_base.maintenance_cost
        + sp_base.maintenance_cost
    )

    total_operation_cost = (
        tpg_ship.carrier_cost
        + support_ship_1.transportation_cost
        + support_ship_2.transportation_cost
    )

    total_pure_profit_peryear = (
        total_profit
        - total_depreciation_expense
        - total_maintainance_cost
        - total_operation_cost
    )

    # ペナルティの合計を計算
    total_penalty = (
        sail_length_penalty
        + tpg_ship.minus_storage_penalty_list[-1]
        + supply_zero_penalty
    )

    # 目的関数の値を計算
    # ECの単価を最小化する場合
    unit_price = sp_base.unit_price  # 供給拠点売却時の単価[円]
    income = total_pure_profit_peryear - total_penalty
    # 営業利益が0円(利益はないが操業が続けられる)の時の単価を計算
    if total_profit == 0:
        appropriate_unit_price = (total_profit - income) * unit_price
    else:
        appropriate_unit_price = ((total_profit - income) / total_profit) * unit_price

    # 目的関数の値を計算
    objective_value = appropriate_unit_price

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
            pl.col("Objective_value").cast(pl.Float64),
        ]
    )

    return data


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    output_folder_path = HydraConfig.get().run.dir

    tpg_ship_log_file_name = cfg.output_env.tpg_ship_log_file_name
    storage_base_log_file_name = cfg.output_env.storage_base_log_file_name
    supply_base_log_file_name = cfg.output_env.supply_base_log_file_name
    support_ship_1_log_file_name = cfg.output_env.support_ship_1_log_file_name
    support_ship_2_log_file_name = cfg.output_env.support_ship_2_log_file_name
    png_map_folder_name = cfg.output_env.png_map_folder_name
    png_graph_folder_name = cfg.output_env.png_graph_folder_name
    png_map_graph_folder_name = cfg.output_env.png_map_graph_folder_name

    progress_bar = tqdm(total=6, desc=output_folder_path)

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
    sail_num = cfg.tpg_ship.sail_num
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
    support_ship_1_max_speed_kt = cfg.support_ship_1.ship_speed_kt
    support_ship_1_EP_max_storage_wh = cfg.support_ship_1.EP_max_storage_wh
    support_ship_1_elect_trust_efficiency = cfg.support_ship_1.elect_trust_efficiency
    support_ship_1 = support_ship.Support_ship(
        support_ship_1_supply_base_locate,
        support_ship_1_storage_method,
        support_ship_1_max_storage_wh,
        support_ship_1_max_speed_kt,
        support_ship_1_EP_max_storage_wh,
        support_ship_1_elect_trust_efficiency,
    )

    # Support ship 2
    support_ship_2_supply_base_locate = cfg.supply_base.locate
    support_ship_2_storage_method = cfg.tpg_ship.storage_method
    support_ship_2_max_storage_wh = cfg.support_ship_2.max_storage_wh
    support_ship_2_max_speed_kt = cfg.support_ship_2.ship_speed_kt
    support_ship_2_EP_max_storage_wh = cfg.support_ship_2.EP_max_storage_wh
    support_ship_2_elect_trust_efficiency = cfg.support_ship_2.elect_trust_efficiency
    support_ship_2 = support_ship.Support_ship(
        support_ship_2_supply_base_locate,
        support_ship_2_storage_method,
        support_ship_2_max_storage_wh,
        support_ship_2_max_speed_kt,
        support_ship_2_EP_max_storage_wh,
        support_ship_2_elect_trust_efficiency,
    )

    simulator.simulate(
        simulation_start_time,
        simulation_end_time,
        tpg_ship_1,  # TPG ship
        typhoon_path_forecaster,  # Forecaster
        st_base,  # Storage base
        sp_base,  # Supply base
        support_ship_1,  # Support ship 1
        support_ship_2,  # Support ship 2
        typhoon_data_path,
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + supply_base_log_file_name,
        output_folder_path + "/" + support_ship_1_log_file_name,
        output_folder_path + "/" + support_ship_2_log_file_name,
    )
    progress_bar.update(1)

    # Tpg ship cost
    print("tpg_ship_cost")
    tpg_ship_1.cost_calculate()
    print("DWT", tpg_ship_1.ship_dwt)
    print(
        tpg_ship_1.hull_cost,
        tpg_ship_1.underwater_turbine_cost,
        tpg_ship_1.wing_sail_cost,
        tpg_ship_1.battery_cost,
    )
    print(tpg_ship_1.building_cost)
    print(tpg_ship_1.maintenance_cost, tpg_ship_1.carrier_cost)
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
    print("objective_value", objective_value)

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
    data.write_csv(output_folder_path + "/result.csv")

    utils.draw_map(
        typhoon_data_path,
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + support_ship_1_log_file_name,
        output_folder_path + "/" + support_ship_2_log_file_name,
        output_folder_path + "/" + png_map_folder_name,
        st_base_locate,
        sp_base_locate,
    )
    progress_bar.update(1)

    utils.draw_graph(
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + supply_base_log_file_name,
        output_folder_path + "/" + png_graph_folder_name,
    )
    progress_bar.update(1)

    # TODO : Just for getting the length of simulation data.
    sim_data_length = len(
        pl.read_csv(output_folder_path + "/" + tpg_ship_log_file_name)
    )

    utils.merge_map_graph(
        sim_data_length,
        output_folder_path + "/" + png_map_folder_name,
        output_folder_path + "/" + png_graph_folder_name,
        output_folder_path + "/" + png_map_graph_folder_name,
    )
    progress_bar.update(1)

    # create_movie
    utils.create_movie(
        output_folder_path + "/" + png_map_graph_folder_name,
        output_folder_path,
    )
    progress_bar.update(1)

    # finish
    progress_bar.update(1)


if __name__ == "__main__":
    main()
