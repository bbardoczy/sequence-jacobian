import numpy as np
from sequence_jacobian.blocks.support.law_of_motion import (Markov, ConditionalMarkov)


def test_conditional_markov():
    shape = (5, 6, 7)
    np.random.seed(12345)

    Pi = np.random.rand(shape[0], shape[0])
    lom = Markov(Pi, 0) 

    Pi2 = np.broadcast_to(Pi[..., np.newaxis, np.newaxis], Pi.shape + shape[1:])
    lom2 = ConditionalMarkov(Pi2, 0)

    D = np.random.rand(*shape)
    Dout = lom @ D
    Dout2 = lom2 @ D

    assert np.allclose(Dout, Dout2)

    