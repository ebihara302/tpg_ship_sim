import math

import numpy as np
import polars as pl
from geopy.distance import geodesic


class Support_ship:
    """
    ############################### class support_ship ###############################

    [ 説明 ]

    このクラスは補助船を作成するクラスです。

    主にTPGshipが生成した電力や水素を貯蔵した中間貯蔵拠点から供給地点に輸送します。

    補助船の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)
        storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)
        supply_base_locate (taple) : 供給拠点の座標(緯度,経度)

    属性 :
        supplybase_lat (float) : 供給拠点の緯度
        supplybase_lon (float) : 供給拠点の経度
        ship_lat (float) : 補助船のその時刻での緯度
        ship_lon (float) : 補助船のその時刻での経度
        max_storage (float) : 中継貯蔵拠点の蓄電容量の上限値
        support_ship_speed (float) : 補助船の最大船速[kt]
        storage (float) : 中継貯蔵拠点のその時刻での蓄電量
        ship_gene (int) : 補助船が発生したかどうかのフラグ
        arrived_supplybase (int) : 供給拠点に到着したかどうかのフラグ
        arrived_storagebase (int) : 中継貯蔵拠点に到着したかどうかのフラグ
        target_lat (float) : 補助船の目標地点の緯度
        target_lon (float) : 補助船の目標地点の経度
        brance_condition (str) : 補助船の行動の記録

    """

    ship_gene = 0
    storage = 0
    supply_elect = 0
    arrived_supplybase = 1
    arrived_storagebase = 0
    stay_time = 0

    batttery_method = 1  # ELECT

    target_lat = np.nan
    target_lon = np.nan
    brance_condition = "no action"

    total_consumption_elect = 0
    total_received_elect = 0

    # コスト関係　単位は億円
    building_cost = 0
    maintenance_cost = 0
    transportation_cost = 0

    def __init__(
        self,
        supply_base_locate,
        storage_method,
        max_storage_wh,
        support_ship_speed_kt,
        EP_max_storage,
        elect_trust_efficiency,
    ) -> None:
        self.supplybase_lat = supply_base_locate[0]
        self.supplybase_lon = supply_base_locate[1]
        self.ship_lat = supply_base_locate[0]
        self.ship_lon = supply_base_locate[1]
        self.storage_method = storage_method
        self.max_storage = max_storage_wh
        self.support_ship_speed = support_ship_speed_kt
        self.EP_max_storage = EP_max_storage
        self.EP_storage = self.EP_max_storage
        self.elect_trust_efficiency = elect_trust_efficiency

    def set_outputs(self):
        """
        ############################ def set_outputs ############################

        [ 説明 ]

        補助船の出力を記録するリストを作成する関数です。

        ##############################################################################

        """

        self.sp_target_lat_list = []
        self.sp_target_lon_list = []
        self.sp_storage_list = []
        self.sp_st_per_list = []
        self.sp_ep_storage_list = []
        self.sp_ship_lat_list = []
        self.sp_ship_lon_list = []
        self.sp_brance_condition_list = []
        self.sp_total_consumption_elect_list = []
        self.sp_total_received_elect_list = []

    def outputs_append(self):
        """
        ############################ def outputs_append ############################

        [ 説明 ]

        set_outputs関数で作成したリストに出力を記録する関数です。

        ##############################################################################

        """

        self.sp_target_lat_list.append(float(self.target_lat))
        self.sp_target_lon_list.append(float(self.target_lon))
        self.sp_storage_list.append(float(self.storage))
        if self.max_storage == 0:
            self.sp_st_per_list.append(0)
        else:
            self.sp_st_per_list.append(float(self.storage / self.max_storage * 100))
        self.sp_ep_storage_list.append(float(self.EP_storage))
        self.sp_ship_lat_list.append(float(self.ship_lat))
        self.sp_ship_lon_list.append(float(self.ship_lon))
        self.sp_brance_condition_list.append(self.brance_condition)
        self.sp_total_consumption_elect_list.append(float(self.total_consumption_elect))
        self.sp_total_received_elect_list.append(float(self.total_received_elect))

    def get_outputs(self, unix_list, date_list):
        data = pl.DataFrame(
            {
                "unixtime": unix_list,
                "datetime": date_list,
                "targetLAT": self.sp_target_lat_list,
                "targetLON": self.sp_target_lon_list,
                "LAT": self.sp_ship_lat_list,
                "LON": self.sp_ship_lon_list,
                "STORAGE[Wh]": self.sp_storage_list,
                "STORAGE PER[%]": self.sp_st_per_list,
                "EP STORAGE[Wh]": self.sp_ep_storage_list,
                "BRANCH CONDITION": self.sp_brance_condition_list,
                "TOTAL CONSUMPTION ELECT[Wh]": self.sp_total_consumption_elect_list,
                "TOTAL RECEIVED ELECT[Wh]": self.sp_total_received_elect_list,
            }
        )

        return data

    def cal_dwt(self, storage_method, storage):
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

    def cal_maxspeedpower(
        self,
        max_speed,
        storage1,
        storage1_method,
        storage2,
        storage2_method,
    ):
        """
        ############################ def cal_maxspeedpower ############################

        [ 説明 ]

        最大船速時に船体を進めるのに必要な出力を算出する関数です。

        ##############################################################################

        引数 :
            max_speed (float) : 最大船速(kt)
            storage1 (float) : メインストレージの貯蔵容量1[Wh]
            storage1_method (int) : メインストレージの貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵
            storage2 (float) : 電気推進用の貯蔵容量[Wh]
            storage2_method (int) : 電気推進用の貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵

        戻り値 :
            power (float) : 船体を進めるのに必要な出力

        #############################################################################
        """

        main_storage_dwt = self.cal_dwt(storage1_method, storage1)
        electric_propulsion_storage_dwt = self.cal_dwt(storage2_method, storage2)

        self.main_storage_weight = main_storage_dwt
        self.ep_storage_weight = electric_propulsion_storage_dwt

        sum_dwt_t = main_storage_dwt + electric_propulsion_storage_dwt
        self.ship_dwt = sum_dwt_t

        if storage1_method == 1:  # 電気貯蔵
            # バルカー型
            k = 1.7
            power = k * ((sum_dwt_t) ** (2 / 3)) * (max_speed**3)

        elif storage1_method >= 2:  # 水素貯蔵
            # タンカー型
            k = 2.2
            power = k * ((sum_dwt_t) ** (2 / 3)) * (max_speed**3)

        else:
            print("cannot cal")

        return power

    # 状態量計算
    def get_distance(self, storage_base_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        補助船(または拠点位置or観測地点)からUUVへの距離を計算する関数です。

        ##############################################################################

        引数 :
            storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)


        戻り値 :
            distance (float) : 補助船から上記拠点への距離(km)

        #############################################################################
        """

        A_position = (self.ship_lat, self.ship_lon)

        # AーB間距離
        distance = geodesic(A_position, storage_base_position).km

        return distance

    # 状態量計算
    def change_kt_kmh(self, ship_speed_kt):
        """
        ############################ def change_kt_kmh ############################

        [ 説明 ]

        ktをkm/hに変換する関数です

        ##############################################################################


        戻り値 :
            speed_kmh (float) : km/hに変換された船速

        #############################################################################
        """

        speed_kmh = ship_speed_kt * 1.852

        return speed_kmh

    # 拠点または観測地点からの距離を計算
    def get_base_dis_data(self, storage_base_position):
        """
        ############################ def get_base_dis_data ############################

        [ 説明 ]

        get_target_dataで選ばれ、追従対象となった台風のcurrent_time + time_stepの時刻での座標を取得する関数です。

        存在しない場合は空のデータフレームが返ります。

        本関数は複数の供給拠点が本土にある場合に初めて有効となる。今後の拡張を見越し、事前実装する。

        課題点としては、各拠点で運用される補助船がおそらく共通化してしまうことである。

        コスト計算をするなら拠点ごとに設定する必要がある。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]
            storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)


        戻り値 :
            self.route_plus_dis_data (dataflame) : 供給拠点または中継貯蔵拠点からの距離が追加されたデータ

        #############################################################################
        """
        distance_list = []

        if len(self.base_data) != 0:
            for i in range(len(self.base_data)):

                A_position = (self.base_data[i, "LAT"], self.base_data[i, "LON"])
                distance = geodesic(A_position, storage_base_position).km
                distance_list.append(distance)

            # 台風の距離を一応書いておく
            base_plus_dis_data = self.base_data.with_columns(
                pl.Series(distance_list).alias("distance")
            )

            # 距離が近い順番に並び替え
            base_plus_dis_data = base_plus_dis_data.select(
                pl.col("*").sort_by("distance", descending=False)
            )

        return base_plus_dis_data

    # 状態量計算
    # 次の時刻での船の座標
    def get_next_position(self, time_step):
        """
        ############################ def get_next_position ############################

        [ 説明 ]

        補助船の次の時刻での座標を計算するための関数です。

        現在地から目標地点まで直線に進んだ場合にいる座標を計算して返します。

        補助船が次の時刻で目的地に到着できる場合は座標は目的地のものになります。

        状態量が更新されるのみなのでreturnでの戻り値はありません。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        target_position = (self.target_lat, self.target_lon)

        # 目的地と現在地の距離
        Goal_now_distance = self.get_distance(target_position)  # [km]

        # 船がtime_step時間で進める距離
        advance_distance = self.change_kt_kmh(self.speed_kt) * time_step

        # 緯度の差
        g_lat = self.target_lat
        n_lat = self.ship_lat

        lat_difference = g_lat - n_lat

        # 経度の差
        g_lon = self.target_lon
        n_lon = self.ship_lon

        lon_difference = g_lon - n_lon

        # 進める距離と目的地までの距離の比を出す
        if Goal_now_distance != 0:
            distance_ratio = advance_distance / Goal_now_distance
        else:
            distance_ratio = 0

        # 念の為の分岐
        # 距離の比が1を超える場合目的地に到着できることになるので座標を目的地へ、そうでないなら当該距離進める

        if distance_ratio < 1 and distance_ratio > 0:

            # 次の時間にいるであろう緯度
            next_lat = lat_difference * distance_ratio + n_lat

            # 次の時間にいるであろう経度
            next_lon = lon_difference * distance_ratio + n_lon

        else:

            # 次の時間にいるであろう緯度
            next_lat = g_lat

            # 次の時間にいるであろう経度
            next_lon = g_lon

        next_position = [next_lat, next_lon]

        self.ship_lat = next_lat
        self.ship_lon = next_lon

    # まだ使わないver5では拠点は1つ
    def set_start_position(self, storage_base_position):

        self.change_ship = 0
        self.base_plus_dis_data = self.get_base_dis_data(storage_base_position)
        if not np.isnan(self.ship_lat):
            uuv_ship_dis = self.get_distance(storage_base_position)
            if uuv_ship_dis > self.base_plus_dis_data[0, "distance"]:
                self.change_ship = 1
                # self.ship_lat = self.route_plus_dis_data[0,"LAT"]
                # self.ship_lon = self.route_plus_dis_data[0,"LON"]
                # self.base_lat = self.route_plus_dis_data[0,"LAT"]
                # self.base_lon = self.route_plus_dis_data[0,"LON"]
            else:
                # 更新なし、その場からuuvへ向かう
                self.ship_gene = 1
                self.arrived_supplybase = 0
                self.arrived_storagebase = 0

                self.target_lat = storage_base_position[0]
                self.target_lon = storage_base_position[1]

        else:
            self.ship_lat = self.base_plus_dis_data[0, "LAT"]
            self.ship_lon = self.base_plus_dis_data[0, "LON"]
            self.base_lat = self.base_plus_dis_data[0, "LAT"]
            self.base_lon = self.base_plus_dis_data[0, "LON"]
            self.arrived_supplybase = 0
            self.arrived_storagebase = 0
            self.ship_gene = 1

            self.target_lat = storage_base_position[0]
            self.target_lon = storage_base_position[1]

    def go_storagebase_action(self, storage_base_position, time_step):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        補助船が拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        # 帰港での船速の入力
        self.speed_kt = self.support_ship_speed

        self.target_lat = storage_base_position[0]
        self.target_lon = storage_base_position[1]

        distance = self.get_distance(storage_base_position)
        storagebase_ship_dis_time = distance / self.change_kt_kmh(self.speed_kt)
        self.arrived_storagebase = 0
        self.arrived_supplybase = 0

        # 消費電力の計算
        consumption_elect = (
            self.cal_maxspeedpower(
                self.speed_kt,
                self.max_storage,
                self.storage_method,
                self.EP_max_storage,
                self.batttery_method,
            )
            / self.elect_trust_efficiency
        ) * (
            distance / self.change_kt_kmh(self.support_ship_speed)
        )  # 移動距離に対する消費電力の計算
        self.total_consumption_elect = (
            self.total_consumption_elect + consumption_elect
        )  # Totalの消費電力の記録
        self.EP_storage = (
            self.EP_storage - consumption_elect
        )  # バッテリー容量から消費電力を引く

        # timestep後にUUVに船がついている場合
        if (storagebase_ship_dis_time <= time_step) and (self.stay_time > 0):
            self.brance_condition = "arrival at storage Base"
            self.arrived_storagebase = 1  # 到着のフラグ
            self.arrived_supplybase = 0
            self.ship_lat = storage_base_position[0]
            self.ship_lon = storage_base_position[1]

            self.speed_kt = 0

            self.stay_time = 0

        elif (storagebase_ship_dis_time <= time_step) and (self.stay_time == 0):
            self.brance_condition = "go to storage Base(stay)"
            self.arrived_storagebase = 0
            self.stay_time = self.stay_time + 1

        else:
            self.brance_condition = "go to storage Base"
            self.arrived_storagebase = 0

    def go_supplybase_action(self, time_step):
        """
        ############################ def go_supplybase_action ############################

        [ 説明 ]

        補助船が供給拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        # 帰港での船速の入力
        self.speed_kt = self.support_ship_speed
        self.target_lat = self.supplybase_lat
        self.target_lon = self.supplybase_lon
        supplybase_position = (self.supplybase_lat, self.supplybase_lon)
        distance = self.get_distance(supplybase_position)
        uuv_ship_dis_time = distance / self.change_kt_kmh(self.speed_kt)
        self.arrived_supplybase = 0

        # 消費電力の計算
        consumption_elect = (
            self.cal_maxspeedpower(
                self.speed_kt,
                self.storage,
                self.storage_method,
                self.EP_max_storage,
                self.batttery_method,
            )
            / self.elect_trust_efficiency
        ) * (
            distance / self.change_kt_kmh(self.support_ship_speed)
        )  # 移動距離に対する消費電力の計算
        self.total_consumption_elect = (
            self.total_consumption_elect + consumption_elect
        )  # Totalの消費電力の記録
        self.EP_storage = (
            self.EP_storage - consumption_elect
        )  # バッテリー容量から消費電力を引く

        # timestep後にBaseに船がついている場合
        if (uuv_ship_dis_time <= time_step) and (self.stay_time > 0):
            self.brance_condition = "arrival at supply Base"
            self.ship_gene = 0
            self.arrived_supplybase = 1  # 到着のフラグ
            self.arrived_storagebase = 0  # 履歴削除
            self.ship_lat = self.supplybase_lat
            self.ship_lon = self.supplybase_lon
            self.target_lat = np.nan
            self.target_lon = np.nan
            self.target_distance = np.nan
            self.supply_elect = self.storage
            self.total_received_elect = self.total_received_elect + (
                self.EP_max_storage - self.EP_storage
            )
            self.EP_storage = self.EP_max_storage
            # self.storage = 0

            self.speed_kt = 0
            self.stay_time = 0

        elif (uuv_ship_dis_time <= time_step) and (self.stay_time == 0):
            self.brance_condition = "go to supply Base(stay)"
            self.arrived_supplybase = 0
            self.supply_elect = 0
            self.stay_time = self.stay_time + 1

        else:
            self.brance_condition = "go to supply Base"
            self.arrived_supplybase = 0
            self.supply_elect = 0

    def get_next_ship_state(self, storage_base_position, year, current_time, time_step):

        # 移動まえの船の座標を取得
        position_before = [self.ship_lat, self.ship_lon]

        if (self.arrived_storagebase == 0) and (self.arrived_supplybase == 1):
            self.go_storagebase_action(storage_base_position, time_step)
        elif (self.arrived_storagebase == 1) and (self.arrived_supplybase == 0):
            self.go_supplybase_action(time_step)
        elif (self.arrived_storagebase == 0) and (self.arrived_supplybase == 0):
            self.go_storagebase_action(storage_base_position, time_step)
        else:
            self.arrived_supplybase = 0
            self.arrived_storagebase = 0

        # 拠点に帰った場合を弾く
        if not np.isnan(self.target_lat):
            self.get_next_position(time_step)

            # 目標地点との距離
            target_position = (self.target_lat, self.target_lon)
            self.target_distance = self.get_distance(target_position)

        # 移動後の船の座標を取得
        position_after = [self.ship_lat, self.ship_lon]

        # 船の座標が変わった場合のみ消費電力を計算する
        if position_after != position_before:
            # 現在位置と次の位置の距離を計算して、船速を用いて航行時間を計算して消費電力を計算する
            # distance = geodesic(position_before, position_after).km
            # consumption_elect = (
            #     self.cal_maxspeedpower(
            #         self.speed_kt,
            #         self.storage,
            #         self.storage_method,
            #         self.EP_max_storage,
            #         self.batttery_method,
            #     )
            #     / self.elect_trust_efficiency
            # ) * (distance / self.change_kt_kmh(self.support_ship_speed))
            # self.total_consumption_elect = self.total_consumption_elect + consumption_elect
            # self.EP_storage = self.EP_storage - consumption_elect
            self.total_received_elect = self.total_received_elect

        else:
            if self.EP_storage < self.EP_max_storage:
                self.total_received_elect += self.EP_max_storage - self.EP_storage
                self.total_consumption_elect += 0
                self.EP_storage = self.EP_max_storage

            else:
                self.total_received_elect += 0
                self.total_consumption_elect += 0

    # コンテナ型の船体寸法計算
    def calculate_LB_container(self, total_ship_weight_per_body):
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
    def calculate_LB_tanker(self, total_ship_weight_per_body):
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
    def calculate_LB_lng(self, total_ship_weight_per_body):
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

    def calculate_hull_size(self):
        """
        ############################ def calculate_hull_size ############################

        [ 説明 ]

        船体の寸法を算出する関数です。

        ##############################################################################

        戻り値 :
            hull_L_oa (float) : 船体の全長
            hull_B (float) : 船体の幅

        #############################################################################
        """
        # 甲板面積を計算
        # 「統計解析による船舶諸元に関する研究」よりDWTとL_oa,Bの値を算出する

        # DWTの記録
        main_storage_dwt = self.cal_dwt(self.storage_method, self.max_storage)
        electric_propulsion_storage_dwt = self.cal_dwt(
            self.batttery_method, self.EP_max_storage
        )

        self.main_storage_weight = main_storage_dwt
        self.ep_storage_weight = electric_propulsion_storage_dwt

        sum_dwt_t = main_storage_dwt + electric_propulsion_storage_dwt
        self.ship_dwt = sum_dwt_t

        total_ship_weight = sum_dwt_t

        # 船体の寸法を計算
        if self.storage_method == 1:  # 電気貯蔵 = コンテナ型
            L_oa, B = self.calculate_LB_container(total_ship_weight)

        elif (
            (self.storage_method == 2)
            or (self.storage_method == 4)
            or (self.storage_method == 5)
        ):  # MCH・メタノール・e-ガソリン貯蔵 = タンカー型
            L_oa, B = self.calculate_LB_tanker(total_ship_weight)

        elif self.storage_method == 3:  # メタン貯蔵 = LNG船型
            L_oa, B = self.calculate_LB_lng(total_ship_weight)

        # L_oa,Bの記録
        self.hull_L_oa = L_oa
        self.hull_B = B

    def cost_calculate(self):
        """
        ############################ def cost_calculate ############################

        [ 説明 ]

        補助船のコストを計算する関数です。

        修論(2025)に沿った設定となっています。

        ##############################################################################
        """

        # 船の建造費用
        # 船長と船幅を計算
        self.calculate_hull_size()
        # storage_methodで異なるコストを計算
        if self.storage_method == 1:  # 電気貯蔵(コンテナ型)
            # 船体価格+甲板補強価格[円]
            self.hull_cost = (
                0.00483 * (self.ship_dwt**0.878) * 10**6 * 160
                + 500000 * self.hull_L_oa * self.hull_B
            )

        elif self.storage_method == 2:  # MCH貯蔵
            self.hull_cost = (
                0.212 * (self.ship_dwt**0.5065) * 10**6 * 160
                + 500000 * self.hull_L_oa * self.hull_B
            )

        elif self.storage_method == 3:  # メタン貯蔵
            self.hull_cost = (
                4.41 * 0.212 * (self.ship_dwt**0.5065) * 10**6 * 160
                + 500000 * self.hull_L_oa * self.hull_B
            )  # LNG船の補正あり

        elif self.storage_method == 4:  # メタノール貯蔵
            self.hull_cost = (
                0.212 * (self.ship_dwt**0.5065) * 10**6 * 160
                + 500000 * self.hull_L_oa * self.hull_B
            )  # タンカーと同価格

        elif self.storage_method == 5:  # e-ガソリン貯蔵
            self.hull_cost = (
                0.212 * (self.ship_dwt**0.5065) * 10**6 * 160
                + 500000 * self.hull_L_oa * self.hull_B
            )  # タンカーと同価格

        # 電動機モーターの価格[円]　船体価格の10%
        motor_cost = 0.1 * self.hull_cost
        # バッテリーの価格[円] 75ドル/kWhで1ドル=160円 240MWhの電池を必要分搭載するとする。
        n_battery = math.ceil(
            (self.EP_max_storage / 10**6) / 240
        )  # バッテリーの個数を端数切り上げで求める
        battery_cost = (240 * 10**3 * 75) * n_battery * 160

        # 船の建造費用[億円]
        self.building_cost = (self.hull_cost + motor_cost + battery_cost) / 10**8

        # 船の維持費用[億円] 年間で建造コストの3％とする
        self.maintenance_cost = self.building_cost * 0.03

        # MCHの輸送コスト
        # 輸送船の合計消費電力を用いて計算。25円/kWhとする。
        self.transportation_cost = (
            (self.sp_total_consumption_elect_list[-1] / 1000) * 25 / 10**8
        )
