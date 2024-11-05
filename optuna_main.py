import csv

import hydra
import optuna
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from tpg_ship_sim import optuna_simulator, utils
from tpg_ship_sim.model import forecaster, storage_base, support_ship, tpg_ship


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


def run_simulation(cfg):
    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    output_folder_path = HydraConfig.get().run.dir

    tpg_ship_param_log_file_name = cfg.output_env.tpg_ship_param_log_file_name
    temp_tpg_ship_param_log_file_name = (
        "temp_" + cfg.output_env.tpg_ship_param_log_file_name
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
        tpg_ship_param_log_file_name,
        temp_tpg_ship_param_log_file_name,
    )

    print(tpg_ship_1.total_gene_elect)

    return tpg_ship_1.total_gene_elect


# 探索範囲の指定用関数
def objective(trial):
    config = hydra.compose(config_name="config")

    config.tpg_ship.hull_num = 1

    # config.tpg_ship.hull_num = trial.suggest_int("hull_num", 1, 2)
    # config.tpg_ship.storage_method = trial.suggest_int("storage_method", 1, 2)

    max_storage_GWh = trial.suggest_int(
        "max_storage_GWh", 50, 1000
    )  # max_storage_whの刻み幅は10^9とする
    config.tpg_ship.max_storage_wh = max_storage_GWh * 1000000000

    EP_max_storage_GWh_10 = trial.suggest_int(
        "EP_max_storage_GWh_10", 5, 200
    )  # electric_propulsion_max_storage_whの刻み幅は10^8とする
    config.tpg_ship.electric_propulsion_max_storage_wh = (
        EP_max_storage_GWh_10 * 100000000
    )

    # config.tpg_ship.elect_trust_efficiency = trial.suggest_float("elect_trust_efficiency", 0.7, 0.9)
    # config.tpg_ship.MCH_to_elect_efficiency = trial.suggest_float("MCH_to_elect_efficiency", 0.4, 0.6)
    # config.tpg_ship.elect_to_MCH_efficiency = trial.suggest_float("elect_to_MCH_efficiency", 0.7, 0.9)
    # config.tpg_ship.sail_num = trial.suggest_int("sail_num", 10, 60)
    sail_area_100m2 = trial.suggest_int("sail_area_every_100m2", 50, 200)
    config.tpg_ship.sail_area = sail_area_100m2 * 100
    # config.tpg_ship.sail_space = trial.suggest_float("sail_space", 2, 4)
    config.tpg_ship.sail_steps = trial.suggest_int("sail_steps", 1, 7)
    config.tpg_ship.ship_return_speed_kt = trial.suggest_int(
        "ship_return_speed_kt", 4, 20
    )
    config.tpg_ship.generator_turbine_radius = trial.suggest_int(
        "generator_turbine_radius", 5, 25
    )
    config.tpg_ship.forecast_weight = trial.suggest_int("forecast_weight", 10, 90)
    # config.tpg_ship.typhoon_effective_range = trial.suggest_int("typhoon_effective_range", 50, 150)
    config.tpg_ship.govia_base_judge_energy_storage_per = trial.suggest_int(
        "govia_base_judge_energy_storage_per", 10, 90
    )
    config.tpg_ship.judge_time_times = trial.suggest_float("judge_time_times", 1.0, 2.0)

    # 拠点位置に関する変更
    # base_lat = trial.suggest_int("Base_lat", 0, 30)
    # base_lon = trial.suggest_int("Base_lon", 134, 180)
    # config.storage_base.locate = [base_lat, base_lon]
    # config.tpg_ship.initial_position = config.storage_base.locate

    # シミュレーションを実行
    total_generation = run_simulation(config)

    return total_generation


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # 結果保存用のCSVファイルを初期化
    output_folder_path = HydraConfig.get().run.dir
    tpg_ship_param_log_file_name = cfg.output_env.tpg_ship_param_log_file_name

    # ローカルフォルダに保存するためのストレージURLを指定します。
    # storage = "sqlite:///experiences/catmaran_journal_first_casestudy_neo.db"  # または storage = "sqlite:///path/to/your/folder/example.db"
    storage = "sqlite:///experiences/catmaran_journal_discuss_monohull.db"
    # スタディの作成または既存のスタディのロード
    study = optuna.create_study(
        study_name="example-study",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # 結果保存用のCSVファイルを初期化
    final_csv = output_folder_path + "/" + tpg_ship_param_log_file_name

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

    # 進捗バーのコールバックを使用してoptimizeを実行
    trial_num = 70
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
