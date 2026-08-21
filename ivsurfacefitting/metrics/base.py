import pandas as pd

class IVMetric:
    """
    Abstract data class for a mtric for iv surface fitting.

    Currently forced to take in the predicted and true surfaces, even though some metrics,
    like no aribitrage violation may only require the predicted surface.

    It is only being used for type managing at the moment, if no functionality is rquired, which it probably wont,
    it could simply be defined as a union of types. This is just a quick fix.
    """

    def __init__(self, name) -> None:
        self.name = name
        pass

    def __call__(self, *args, **kwds):
        return ""
