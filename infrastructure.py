from leaf.infrastructure import Link, Node
from leaf.power import PowerModelNode

from settings import CLOUD_CU, CLOUD_WATT_PER_CU


class Cloud(Node):
    def __init__(self):
        super().__init__("cloud", cu=CLOUD_CU, power_model=PowerModelNode(power_per_cu=CLOUD_WATT_PER_CU))
