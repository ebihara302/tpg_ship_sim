import hydra
import optuna
import polars as pl
import math
from datetime import datetime, timedelta, timezone
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
