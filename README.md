# tpg-ship-sim

This is a TPG ship simulator.

TPG ship is typhoon power generation ship which is a novel and future movable device for getting energy from typhoon. 

![concept_tpg_ship](https://github.com/mitsuyukiLab/tpg-ship-sim/assets/12507469/6e6a75da-0e18-4e98-9d2b-80659b883408)

## How to run

```shell
$ python main.py
```
- This simulator is developed based on [Hydra](https://hydra.cc/).
- Typhoon track history data is stored in the [data](data) folder.

For normal use
- You can change the setting of this simulator by editing [conf/config.yaml](conf/config.yaml)

For use in optimisation
```shell
$ python main_optimize.py
```
- You can change the objective function and design variables by editing [main_optimize.py](main_optimize.py)

There is another program with different settings in [customisation_examples](customisation_examples)

## Citation

- Mitsuyuki, T., Ebihara, H., & Kado, S. (2024). Concept design of typhoon power generation ship using system simulation. Proc. of the 15th International Marine Design Conference. https://doi.org/10.59490/imdc.2024.839

    > @article{Mitsuyuki_Ebihara_Kado_2024,
        title = {Concept Design of Typhoon Power Generation Ship Using System Simulation},
        author = {Mitsuyuki, Taiga and Ebihara, Haruki and Kado, Shunsuke},
        date = {2024-05},
        journaltitle = {Proc. of the 15th International Marine Design Conference},
        doi = {10.59490/imdc.2024.839}
    }