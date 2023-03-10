"""Top-level package for MSM_a7_nachrs."""

__author__ = """Yuxuan Zhuang"""
__email__ = 'yuxuan.zhuang@dbb.su.se'
__version__ = '0.1.0'

from .vampnet import (
            VAMPNETInitializer,
            MultimerNet,
            VAMPNet_Multimer,

            VAMPNet_Multimer_NOSYM
            )

from .vampnet_rev import (
            VAMPNet_Multimer_Rev
)

from .vampnet_sum import (
            VAMPNet_Multimer_SYM,
            VAMPNet_Multimer_SYM_REV
)

#from .vampnet_rev import (
#            VAMPNet_Multimer_Rev_Model,
#            VAMPNet_Multimer_Rev,
#            )

#from .srv import (
#            VAMPNet_Multimer_Sym,
#            VAMPNet_Multimer_Sym_NOSYM
#            )
