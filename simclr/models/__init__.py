from simclr.models import encoder
from simclr.models import resnet
from simclr.models import losses
from simclr.models import ssl

REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
    'eval': ssl.SSLEval,
    'semi-supervised-eval': ssl.SemiSupervisedEval,
}