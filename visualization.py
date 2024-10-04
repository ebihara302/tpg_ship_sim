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
        storage="sqlite:///experiences/catmaran_limit_speed_32kt_sail_space_2.0_version_journal.db",
    )

    output_folder_path = HydraConfig.get().run.dir

    # 出力フォルダの指定
    output_png_folder_path = os.path.join(output_folder_path, "param_results")

    # フォルダが存在しない場合は作成
    os.makedirs(output_png_folder_path, exist_ok=True)

    # パラレルコーディネートプロットの生成と保存
    fig_parallel = vis.plot_parallel_coordinate(study)
    pio.write_image(
        fig_parallel, os.path.join(output_png_folder_path, "parallel_coordinate.png")
    )

    # 重要度プロットの生成と保存
    fig_importance = vis.plot_param_importances(study)
    pio.write_image(
        fig_importance, os.path.join(output_png_folder_path, "param_importances.png")
    )

    # Contourプロットの生成と保存
    fig_contour = vis.plot_contour(study)
    fig_contour.update_layout(
        autosize=False, width=3000, height=2400, font=dict(size=10)
    )
    pio.write_image(fig_contour, os.path.join(output_png_folder_path, "contour.png"))

    # Sliceプロットの生成と保存
    fig_slice = vis.plot_slice(study)
    pio.write_image(fig_slice, os.path.join(output_png_folder_path, "slice.png"))

    # パラメータ履歴プロットの生成と保存
    fig_history = vis.plot_optimization_history(study)
    pio.write_image(
        fig_history, os.path.join(output_png_folder_path, "optimization_history.png")
    )

    print(f"プロットが{output_png_folder_path}にPNGファイルとして保存されました。")


if __name__ == "__main__":
    main()
