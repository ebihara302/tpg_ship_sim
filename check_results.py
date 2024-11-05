import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tpg_ship_sim.model import storage_base, tpg_ship


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

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

    # Storage base
    storage_base_locate = cfg.storage_base.locate
    storage_base_max_storage_wh = cfg.storage_base.max_storage_wh
    st_base = storage_base.Storage_base(
        storage_base_locate, storage_base_max_storage_wh
    )

    #
    # 発電船拠点位置
    tpg_ship_1.base_lat = st_base.locate[0]
    tpg_ship_1.base_lon = st_base.locate[1]
    tpg_ship_1.set_initial_states()
    # tpg_ship_1.sail_num = tpg_ship_1.calculate_max_sail_num()
    print("最大帆数")
    print(tpg_ship_1.sail_num, tpg_ship_1.calculate_max_sail_num())
    tpg_ship_1.cal_generating_ship_speed(tpg_ship_1.sail_num)

    # 台風下での台風発電船の推力と抵抗をprint
    print("台風下での推力と抵抗")
    print(tpg_ship_1.tpgship_generating_lift, tpg_ship_1.tpgship_generating_drag)
    print(tpg_ship_1.sails_lift, tpg_ship_1.sails_drag)
    print(tpg_ship_1.tpgship_turbine_drag, tpg_ship_1.tpgship_hull_drag)
    # 最大推力を得た時の風向をプリント
    print("最大推力を得た時の風向")
    print(tpg_ship_1.max_wind_force_direction)
    # 載貨重量トン数をプリント
    print("載貨重量トン数")
    print(
        tpg_ship_1.main_storage_weight,
        tpg_ship_1.ep_storage_weight,
        tpg_ship_1.sails_weight,
    )
    # 船の発電時の速度をプリント kt,m/s
    print("船の発電時の速度")
    print(tpg_ship_1.generating_speed_kt, "kt", tpg_ship_1.generating_speed_mps, "m/s")


if __name__ == "__main__":
    main()
