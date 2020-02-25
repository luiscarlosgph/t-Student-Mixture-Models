import sys

__version__ = '0.1.5'

if (sys.version_info < (3, 3)):
    from smm import SMM
else:
    from smm.smm import SMM
