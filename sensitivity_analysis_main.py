import csv

import hydra
import matplotlib.pyplot as plt
import optuna
import polars as pl
import scienceplots
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from tpg_ship_sim import optuna_simulator, utils
from tpg_ship_sim.model import forecaster, storage_base, support_ship, tpg_ship


def run_simulation(cfg):
    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    output_folder_path = HydraConfig.get().run.dir

    tpg_ship_sensitivity_analysis_param_log_file_name = (
        cfg.output_env.tpg_ship_sensitivity_analysis_param_log_file_name
    )
    temp_tpg_ship_param_log_file_name = (
        "temp_" + cfg.output_env.tpg_ship_sensitivity_analysis_param_log_file_name
    )

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
    )

    # Forecaster
    forecast_time = cfg.forecaster.forecast_time
    forecast_error_slope = cfg.forecaster.forecast_error_slope
    typhoon_path_forecaster = forecaster.Forecaster(forecast_time, forecast_error_slope)

    # Storage base
    storage_base_locate = cfg.storage_base.locate
    storage_base_max_storage_wh = cfg.storage_base.max_storage_wh
    st_base = storage_base.Storage_base(
        storage_base_locate, storage_base_max_storage_wh
    )

    # Support ship 1
    support_ship_1_supply_base_locate = cfg.support_ship_1.supply_base_locate
    support_ship_1_max_storage_wh = cfg.support_ship_1.max_storage_wh
    support_ship_1_max_speed_kt = cfg.support_ship_1.ship_speed_kt
    support_ship_1 = support_ship.Support_ship(
        support_ship_1_supply_base_locate,
        support_ship_1_max_storage_wh,
        support_ship_1_max_speed_kt,
    )

    # Support ship 2
    support_ship_2_supply_base_locate = cfg.support_ship_2.supply_base_locate
    support_ship_2_max_storage_wh = cfg.support_ship_2.max_storage_wh
    support_ship_2_max_speed_kt = cfg.support_ship_2.ship_speed_kt
    support_ship_2 = support_ship.Support_ship(
        support_ship_2_supply_base_locate,
        support_ship_2_max_storage_wh,
        support_ship_2_max_speed_kt,
    )

    # Run simulation
    optuna_simulator.simulate(
        simulation_start_time,
        simulation_end_time,
        tpg_ship_1,
        typhoon_path_forecaster,
        st_base,
        support_ship_1,
        support_ship_2,
        typhoon_data_path,
        output_folder_path,
        tpg_ship_sensitivity_analysis_param_log_file_name,
        temp_tpg_ship_param_log_file_name,
    )

    print(tpg_ship_1.total_gene_elect)

    return tpg_ship_1.total_gene_elect


# 指定パラメータの下限値、上限値、刻み幅を与えたときに繰り返し run_simulation を行う関数
def run_simulations_for_sensitivity_analysis(cfg, param_min, param_max, step, x):
    # 最適パラメータの取得
    config = hydra.compose(config_name="config")
    opti_param = config.tpg_ship.generator_turbine_radius  # 水中タービン半径
    # opti_param = config.tpg_ship.max_storage_wh  # 船の貯蔵容量

    # パラメータ調整
    analysis_param_name = "generator_turbine_radius"  # 水中タービン半径
    # analysis_param_name = "max_storage"  # 船の最大蓄電量
    unit_name = "[m]"
    for param in tqdm(
        range(param_min * x, param_max * x + step, step), desc="total_sim..."
    ):
        cfg.tpg_ship.generator_turbine_radius = param / x  # 水中タービン半径
        # cfg.tpg_ship.max_storage_wh = param / x  # 船の最大蓄電量
        run_simulation(cfg)

        next_param = param + step
        # 次のループ時のパラメータが最適パラメータよりも大きく、今のパラメータが最適パラメータよりも小さい場合を判別
        if next_param / x > opti_param and param / x < opti_param:
            run_simulation(config)

    return analysis_param_name, unit_name


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    x = 10  # stepが整数のときは1、0.1にするときは10、0.01にするときは100にする
    param_min = 14  # 720 * 10**9
    param_max = 18  # 740 * 10**9
    step = 2  # 10**9  # 小数点の刻み幅にしたいときはxを調整する

    # 結果保存用のCSVファイルを初期化
    output_folder_path = HydraConfig.get().run.dir
    tpg_ship_S_A_param_log_file_name = (
        cfg.output_env.tpg_ship_sensitivity_analysis_param_log_file_name
    )

    # 結果保存用のCSVファイルを初期化
    final_csv = output_folder_path + "/" + tpg_ship_S_A_param_log_file_name

    columns = [
        ("Base_lat", pl.Float64),
        ("Base_lon", pl.Float64),
        ("hull_num", pl.Int64),
        ("hull_L_oa", pl.Float64),
        ("hull_B", pl.Float64),
        ("storage_method", pl.Int64),
        ("max_storage", pl.Float64),
        ("electric_propulsion_max_storage_wh", pl.Float64),
        ("elect_trust_efficiency", pl.Float64),
        ("MCH_to_elect_efficiency", pl.Float64),
        ("elect_to_MCH_efficiency", pl.Float64),
        ("generator_turbine_radius", pl.Float64),
        ("generator_efficiency", pl.Float64),
        ("generator_drag_coefficient", pl.Float64),
        ("generator_pillar_chord", pl.Float64),
        ("generator_pillar_max_tickness", pl.Float64),
        ("generator_pillar_width", pl.Float64),
        ("generator_num", pl.Int64),
        ("generator_rated_output_w", pl.Float64),
        ("sail_num", pl.Int64),
        ("sail_width", pl.Float64),
        ("sail_area", pl.Float64),
        ("sail_space", pl.Float64),
        ("sail_steps", pl.Int64),
        ("sail_weight", pl.Float64),
        ("num_sails_per_row", pl.Int64),
        ("num_sails_rows", pl.Int64),
        ("nomal_ave_speed", pl.Float64),
        ("max_speed", pl.Float64),
        ("generating_speed_kt", pl.Float64),
        ("forecast_weight", pl.Float64),
        ("typhoon_effective_range", pl.Float64),
        ("govia_base_judge_energy_storage_per", pl.Float64),
        ("judge_time_times", pl.Float64),
        ("sail_penalty", pl.Float64),
        ("total_gene_elect", pl.Float64),
    ]

    # Create an empty DataFrame with the specified schema
    df = pl.DataFrame(schema=columns)

    df.write_csv(final_csv)

    # 結果のプロット
    analysis_param_name, unit_name = run_simulations_for_sensitivity_analysis(
        cfg, param_min, param_max, step, x
    )

    # dfのグラフ化
    plt.style.use(["science", "no-latex", "high-vis", "grid"])
    Title_name = "Sensitivity Analysis"
    x_label = analysis_param_name + unit_name
    y_label = "Total MCH acquired in operations[GWh]"
    df1 = pl.read_csv(final_csv)
    # polarsで指定列要素の取得
    x = df1[analysis_param_name].to_numpy()
    y = df1["total_gene_elect"].to_numpy()
    # xがmax_storageの場合GWhに変換
    if analysis_param_name == "max_storage":
        x = x / 10**9

    # yはWhなのでGWhに変換
    y = y / 10**9
    # グラフ化 点ありの折れ線グラフ
    plt.plot(x, y, marker="o", markersize=3)
    plt.ylim(0, None)
    plt.title(Title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(output_folder_path + "/sensitivity_analysis.png")


if __name__ == "__main__":
    main()
