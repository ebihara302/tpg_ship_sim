import os

import hydra
import kaleido
import optuna
import optuna.visualization as vis
import plotly
import plotly.io as pio
import sklearn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # 結果保存用のCSVファイルを初期化
    output_folder_path = HydraConfig.get().run.dir

    # SQLiteデータベースからStudyを読み込む
    study = optuna.load_study(
        study_name="example-study",
        # storage="sqlite:///experiences/catmaran_journal_discuss_baseposition.db",
        storage="sqlite:///experiences/catmaran_journal_discuss_monohull.db",
        # storage="sqlite:///experiences/catmaran_journal_first_casestudy_neo.db",
    )

    output_folder_path = HydraConfig.get().run.dir

    # 出力フォルダの指定
    output_png_folder_path = os.path.join(output_folder_path, "param_results")

    # フォルダが存在しない場合は作成
    os.makedirs(output_png_folder_path, exist_ok=True)

    # パラレルコーディネートプロットの生成と保存
    # fig_parallel = vis.plot_parallel_coordinate(study)
    # pio.write_image(
    #     fig_parallel, os.path.join(output_png_folder_path, "parallel_coordinate.png")
    # )

    # 重要度プロットの生成と保存
    fig_importance = vis.plot_param_importances(study)
    pio.write_image(
        fig_importance, os.path.join(output_png_folder_path, "param_importances.png")
    )
    # パラメータ名の変更と絞り込み
    param_list = [
        "max_storage_GWh",
        "sail_area_every_100m2",
        "generator_turbine_radius",
        "EP_max_storage_GWh_10",
        "govia_base_judge_energy_storage_per",
        "ship_return_speed_kt",
    ]
    fig_importance = vis.plot_param_importances(study, params=param_list)
    # プロットで使用するラベルのマッピングを設定
    label_mapping = {
        "max_storage_GWh": "Ship Capacity",
        "sail_area_every_100m2": "Sail Area of Rigid Wing Sail",
        "generator_turbine_radius": "Underwater Turbine Radius",
        "EP_max_storage_GWh_10": "Electric Motor Battery Capacity",
        "govia_base_judge_energy_storage_per": "Percentage of Capacity of port call trigger",
        "ship_return_speed_kt": "Minimum Ship Speed at port of call",
    }
    # 縦軸のパラメータ名を更新して、重要な4つのパラメータに絞り込む
    fig_importance.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(label_mapping.keys()),
            ticktext=list(label_mapping.values()),
        )
    )
    pio.write_image(
        fig_importance, os.path.join(output_png_folder_path, "param_importances2.png")
    )

    label_mapping = {
        "max_storage_GWh": "C_ship",
        "sail_area_every_100m2": "S_sail",
        "generator_turbine_radius": "r_PG",
        "EP_max_storage_GWh_10": "C_battery",
        "govia_base_judge_energy_storage_per": "C_trigger",
        "ship_return_speed_kt": "U_return(min)",
    }
    # 縦軸のパラメータ名を更新して、重要な4つのパラメータに絞り込む
    fig_importance.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(label_mapping.keys()),
            ticktext=list(label_mapping.values()),
        )
    )
    pio.write_image(
        fig_importance, os.path.join(output_png_folder_path, "param_importances3.png")
    )

    # Contourプロットの生成と保存
    # fig_contour = vis.plot_contour(study)
    # fig_contour.update_layout(
    #     autosize=False, width=3000, height=2400, font=dict(size=10)
    # )
    # pio.write_image(fig_contour, os.path.join(output_png_folder_path, "contour.png"))

    # Sliceプロットの生成と保存
    # fig_slice = vis.plot_slice(study)
    # pio.write_image(fig_slice, os.path.join(output_png_folder_path, "slice.png"))

    # パラメータ履歴プロットの生成と保存
    # fig_history = vis.plot_optimization_history(study)
    # pio.write_image(
    #     fig_history, os.path.join(output_png_folder_path, "optimization_history.png")
    # )

    print(f"プロットが{output_png_folder_path}にPNGファイルとして保存されました。")


if __name__ == "__main__":
    main()
