"""Generated EML surrogate. Inputs outside the documented range are unvalidated."""
import math

def E(a, b):
    return math.exp(max(-80., min(80., a))) - math.log(max(1e-6, min(1e30, b)))

EXP = math.exp

MEAN = [2710.01611328125, 6.131596088409424, 0.14971107244491577, 42.768402099609375, 0.010897611267864704]
STD = [2788.998779296875, 6.301061153411865, 0.10072671622037888, 8.646236419677734, 0.014433151111006737]
LOW = [200.0, 0.0, 0.02539999969303608, 31.700000762939453, 0.0004122900136280805]
HIGH = [20000.0, 22.200000762939453, 0.30480000376701355, 55.5, 0.05841130018234253]

def predict_one(features):
    if len(features) != 5:
        raise ValueError('Wrong feature count')
    if not all(math.isfinite(v) for v in features):
        raise ValueError('Inputs must be finite')
    if any(v < lo or v > hi for v,lo,hi in zip(features, LOW, HIGH)):
        raise ValueError('Outside the evaluated input range')
    x = [(v-m)/s for v,m,s in zip(features, MEAN, STD)]
    value = 0.006805382203310728
    value += (-0.8175646662712097 + (0.07908730208873749 * E((x[3] - x[2]), (x[0] + x[1]))))
    value += (-0.07235784828662872 + (-0.23466037213802338 * E((x[4] - x[1]), E((x[4] - x[0]), (x[1] + x[2])))))
    value += (0.3812495172023773 + (-0.029396282508969307 * E(E(0.9450835585594177, E((x[2] + x[4]), (x[0] * x[2]))), (x[2] * x[3]))))
    value += (-0.07456394284963608 + (-0.13354143500328064 * E((x[0] * x[1]), E((x[2] - x[3]), (x[2] - x[4])))))
    value = value * 6.890585899353027 + 123.99857330322266
    return value
