"""Generated EML surrogate. Inputs outside the documented range are unvalidated."""
import math

def E(a, b):
    return math.exp(max(-80., min(80., a))) - math.log(max(1e-6, min(1e30, b)))

EXP = math.exp

MEAN = [1.0537595748901367, 0.15797123312950134]
STD = [0.5558863878250122, 0.0806470736861229]
LOW = [0.10004105418920517, 0.020044656470417976]
HIGH = [1.9999860525131226, 0.2999803423881531]

def predict_one(features):
    if len(features) != 2:
        raise ValueError('Wrong feature count')
    if not all(math.isfinite(v) for v in features):
        raise ValueError('Inputs must be finite')
    if any(v < lo or v > hi for v,lo,hi in zip(features, LOW, HIGH)):
        raise ValueError('Outside the evaluated input range')
    x = [(v-m)/s for v,m,s in zip(features, MEAN, STD)]
    value = -1.1641532182693481e-07
    value += (-15.127342224121094 + (0.9982279539108276 * E(E(1.0000090599060059, 0.9999966621398926), E(x[1], 0.9996875524520874))))
    value += (0.035489924252033234 + (0.033691421151161194 * E(x[0], E(E(0.6319540143013, 1.135402798652649), E((x[0] * x[1]), E(1.036826491355896, (x[0] - x[1])))))))
    value += (-0.011369839310646057 + (0.002058399608358741 * E(E(x[0], E(E(1.000001072883606, (x[0] + x[1])), E(x[1], (x[0] * x[1])))), E(1.0000635385513306, E((x[1] - x[0]), (x[0] + x[1]))))))
    value += (0.017102017998695374 + (-0.009280896745622158 * E(x[1], E(x[1], 1.1368780136108398))))
    value = value * 0.5046855211257935 + -0.9803708791732788
    return (1-math.cos(features[0])) * math.exp(value)
