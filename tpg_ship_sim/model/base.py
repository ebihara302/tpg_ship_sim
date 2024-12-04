import math

import polars as pl


class Base:
    """
    ############################### class Base ###############################

    [ 説明 ]

    このクラスは供給拠点(spbase)、貯蔵拠点(stbase)、兼用拠点(cbbase)を作成するクラスです。

    貯蔵拠点は主にTPGshipが生成した電力や水素を貯蔵し、補助船に渡します。

    供給拠点は補助船が寄港し、貯蔵拠点から電力や水素を受け取り、利用地に供給(売買)します。

    兼用拠点は貯蔵拠点と供給拠点の機能を持ち、TPGshipからの発電成果物を受け取り、そのまま利用地に供給します。

    各種拠点の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        base_type (str) : 拠点の種類
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)
        support_ship_1 (class) : Support_shipクラスのインスタンスその1
        support_ship_2 (class) : Support_shipクラスのインスタンスその2
        TPGship1 (class) : TPGshipクラスのインスタンスその1

    属性 :
        max_storage (float) : 拠点の蓄電容量の上限値
        storage (float) : 拠点のその時刻での蓄電量
        call_num (int) : supportSHIPを読んだ回数
        call_ship1 (int) : support_ship_1を呼ぶフラグ
        call_ship2 (int) : support_ship_1を呼ぶフラグ
        call_per (int) : supprotSHIPを呼ぶ貯蔵パーセンテージ

    """

    ####################################  パラメータ  ######################################

    storage = 0
    call_num = 0
    call_ship1 = 0
    call_ship2 = 0

    supply_time_count = 0
    brance_condition = "Ready"
    total_quantity_received = 0
    total_supply = 0

    # コスト関連　単位は億円
    building_cost = 0
    maintenance_cost = 0
    profit = 0

    def __init__(self, base_type, locate, max_storage, call_per) -> None:
        self.base_type = base_type
        self.locate = locate
        self.max_storage = max_storage
        self.call_per = call_per

    def set_outputs(self):
        """
        ############################ def set_outputs ############################

        [ 説明 ]

        中継貯蔵拠点の出力を記録するリストを作成する関数です。

        ##############################################################################

        """
        self.storage_list = []
        self.total_quantity_received_list = []
        self.total_supply_list = []
        self.st_per_list = []
        self.condition_list = []

    def outputs_append(self):
        """
        ############################ def outputs_append ############################

        [ 説明 ]

        set_outputs関数で作成したリストに出力を記録する関数です。

        ##############################################################################

        """
        self.storage_list.append(float(self.storage))
        self.total_quantity_received_list.append(float(self.total_quantity_received))
        self.total_supply_list.append(float(self.total_supply))
        self.st_per_list.append(float(self.storage / self.max_storage * 100))
        self.condition_list.append(self.brance_condition)

    def get_outputs(self, unix, date):
        """
        ############################ def get_outputs ############################

        [ 説明 ]

        set_outputs関数で作成したリストを出力する関数です。

        ##############################################################################

        """
        data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "STORAGE[Wh]": self.storage_list,
                "STORAGE PER[%]": self.st_per_list,
                "BRANCH CONDITION": self.condition_list,
                "TOTAL QUANTITY RECEIVED[Wh]": self.total_quantity_received_list,
                "TOTAL SUPPLY[Wh]": self.total_supply_list,
            }
        )

        return data

    ####################################  メソッド  ######################################

    def stbase_storage_elect(self, TPGship1):
        """
        ############################ def stbase_storage_elect ############################

        [ 説明 ]

        中継貯蔵拠点がTPGshipから発電成果物を受け取り、蓄電容量を更新する関数です。

        TPGshipが拠点に帰港した時にのみ増加する。それ以外は出力の都合で、常に0を受け取っている。

        """

        self.storage = self.storage + TPGship1.supply_elect
        self.total_quantity_received = (
            self.total_quantity_received + TPGship1.supply_elect
        )

        if TPGship1.supply_elect > 0:
            TPGship1.supply_elect = 0

    def stbase_supply_elect(
        self, support_ship_1, support_ship_2, year, current_time, time_step
    ):
        """
        ############################ def stbase_supply_elect ############################

        [ 説明 ]

        中継貯蔵拠点がsupportSHIPを呼び出し、貯蔵しているエネルギーを渡す関数。

        呼ぶまでの関数なので、呼んだ後のsupportSHIPが帰るフェーズは別で記載している。

        """
        if (
            support_ship_1.arrived_supplybase == 1 or self.call_ship1 == 1
        ) and support_ship_1.max_storage != 0:  # support_ship_1が活動可能な場合
            self.brance_condition = "call ship1"
            self.call_ship1 = 1
            support_ship_1.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

            if support_ship_1.arrived_storagebase == 1:
                self.call_ship1 = 0
                if self.storage <= support_ship_1.max_storage:
                    support_ship_1.storage = support_ship_1.storage + self.storage
                    self.total_supply = self.total_supply + self.storage
                    self.storage = 0
                else:
                    support_ship_1.storage = support_ship_1.max_storage
                    self.total_supply = self.total_supply + support_ship_1.max_storage
                    self.storage = self.storage - support_ship_1.max_storage

                self.call_num = self.call_num + 1

        elif (
            support_ship_2.arrived_supplybase == 1 or self.call_ship2 == 1
        ) and support_ship_2.max_storage != 0:  # support_ship_2が活動可能な場合
            self.brance_condition = "call ship2"
            self.call_ship2 = 1
            support_ship_2.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

            if support_ship_2.arrived_storagebase == 1:
                self.call_ship2 = 0
                if self.storage <= support_ship_2.max_storage:
                    support_ship_2.storage = support_ship_2.storage + self.storage
                    self.total_supply = self.total_supply + self.storage
                    self.storage = 0
                else:
                    support_ship_2.storage = support_ship_2.max_storage
                    self.total_supply = self.total_supply + support_ship_2.max_storage
                    self.storage = self.storage - support_ship_2.max_storage

                self.call_num = self.call_num + 1
        else:  # 両方ダメな場合
            self.brance_condition = "can't call anyone"

    def spbase_storage_elect(self, support_ship_1, support_ship_2):
        """
        ############################ def spbase_storage_elect ############################

        [ 説明 ]

        供給拠点がsupportSHIPから発電成果物を受け取り、蓄電容量を更新する関数です。

        supportSHIPが拠点に帰港した時にのみ増加する。それ以外は出力の都合で、常に0を受け取っている。

        """

        # 補助船の位置
        sp_ship_1_position = [support_ship_1.ship_lat, support_ship_1.ship_lon]
        sp_ship_2_position = [support_ship_2.ship_lat, support_ship_2.ship_lon]

        # 補助船の位置が拠点に到着したら、補助船の貯蔵量を0にする
        # 拠点位置と補助船の位置が等しく、補助船のストレージが0より大きい場合、補助船のストレージを0にする
        if self.locate == sp_ship_1_position and support_ship_1.storage > 0:
            self.storage = self.storage + support_ship_1.storage
            self.total_quantity_received = (
                self.total_quantity_received + support_ship_1.storage
            )

            support_ship_1.storage = 0

        if self.locate == sp_ship_2_position and support_ship_2.storage > 0:
            self.storage = self.storage + support_ship_2.storage
            self.total_quantity_received = (
                self.total_quantity_received + support_ship_2.storage
            )

            support_ship_2.storage = 0

    def spbase_supply_elect(self):
        """
        ############################ def spbase_supply_elect ############################

        [ 説明 ]

        供給拠点が利用地に発電成果物を供給する関数です。

        供給拠点に貯蔵され次第、速やかに供給され供給拠点の容量が確保される。それ以外は出力の都合で、常に0を供給している。

        一応、関数にしてあるが、現状の機能ではそこまでの処理はない。

        """

        if self.storage > 0:
            self.total_supply = self.total_supply + self.storage
            self.storage = 0
        else:
            self.total_supply = self.total_supply

    def operation_base(
        self, TPGship1, support_ship_1, support_ship_2, year, current_time, time_step
    ):
        """
        ############################ def operation_base ############################

        [ 説明 ]

        貯蔵・供給・兼用拠点の運用を行う関数。

        """

        if self.base_type == 1:
            ###############################  貯蔵拠点  ###############################
            # 貯蔵量の更新
            self.stbase_storage_elect(TPGship1)
            self.brance_condition = "while in storage"

            # supportSHIPの寄港動作完遂までは動かす。呼び出しもキャンセル。
            if support_ship_1.arrived_supplybase == 0:
                support_ship_1.get_next_ship_state(
                    self.locate, year, current_time, time_step
                )

            if support_ship_2.arrived_supplybase == 0:
                support_ship_2.get_next_ship_state(
                    self.locate, year, current_time, time_step
                )

            judge = support_ship_1.max_storage * (self.call_per / 100)
            if self.storage >= judge:
                self.stbase_supply_elect(
                    support_ship_1, support_ship_2, year, current_time, time_step
                )

        elif self.base_type == 2:
            ###############################  供給拠点  ###############################
            # 補助船から供給される
            # 貯蔵量の更新
            self.spbase_storage_elect(support_ship_1, support_ship_2)
            self.brance_condition = "spbase Standby"

            if self.storage > 0:

                if self.supply_time_count >= 1:
                    self.spbase_supply_elect()
                    self.supply_time_count = 0
                    self.brance_condition = "spbase Supply"
                else:
                    self.supply_time_count = self.supply_time_count + 1
                    self.brance_condition = "spbase Storage"

        elif self.base_type == 3:
            ###############################  兼用拠点  ###############################
            # 台風発電船から供給される
            # 貯蔵量の更新
            self.stbase_storage_elect(TPGship1)
            self.brance_condition = "cbbase Standby"

            if self.storage > 0:

                if self.supply_time_count >= 1:
                    self.spbase_supply_elect()
                    self.supply_time_count = 0
                    self.brance_condition = "cbbase Supply"
                else:
                    self.supply_time_count = self.supply_time_count + 1
                    self.brance_condition = "cbbase Storage"

    def cost_calculate(self):
        """
        ############################ def cost_calculate ############################

        [ 説明 ]

        拠点のコストを計算する関数です。

        修論(2025)に沿った設定となっています。

        """

        # 10万トン貯蔵できるタンクのコストを10億円とする
        tank_cost = 10**9
        tank_capacity = 10**5
        # MCHを1GWh分で379tとして、10万トンタンクがいくつ必要か計算する、端数は切り上げ
        need_capacity = (self.max_storage / 10**9) * 379
        tank_num = math.ceil(need_capacity / tank_capacity)
        # タンクのコストを計算
        tank_total_cost = tank_cost * tank_num
        # 入港できるようにするための拡張工事コスト(ドックも含む)を50億円とする
        extension_cost = 5 * 10**9
        # 建設コストを計算
        self.building_cost = (tank_total_cost + extension_cost) / 10**8

        # メンテナンスコストを計算。年間で建設コストの3％とする
        self.maintenance_cost = self.building_cost * 0.03

        # total_supply_list[-1]を基に利益を計算（供給拠点・兼用拠点の場合のみ計算）
        if self.base_type == 2 or self.base_type == 3:
            # MCHをWhからtに変換　1GWh = 379t
            total_supply_t = self.total_supply_list[-1] / 10**9 * 379
            # MCHの価格を1tあたり水素を679[Nm3]生成するとして、量を計算
            total_supply_hydrogen = total_supply_t * 679
            # 水素の価格を1[Nm3]あたり20円として、利益を計算
            self.profit = total_supply_hydrogen * 20
        else:
            self.profit = 0
