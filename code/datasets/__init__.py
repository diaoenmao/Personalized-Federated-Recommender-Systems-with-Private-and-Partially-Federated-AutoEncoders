# from .movielens import ML100K, ML1M, ML10M, ML20M
from .movielens import ML100K, ML1M, ML10M, ML20M
from .taobaoclick import taobaoclicksmall
from .nfp import NFP
from .douban import Douban
from .amazon import Amazon
from .anime import Anime
from .netflix import Netflix
from .datasets_utils import *

__all__ = ('ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon', 'Anime', 'Netflix')
