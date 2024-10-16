class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def load_tf_color():
    return TFColor

def map_model_name(model_name):
    if model_name == "coxph":
        return "CoxPH"
    elif model_name == "dgp":
        model_name = "DGP"
    elif model_name == "rsf":
        model_name = "RSF"
    elif model_name == "dsm":
        model_name = "DSM"
    elif model_name == "deephit":
        model_name = "DeepHit"
    elif model_name == "deepsurv":
        model_name = "DeepSurv"
    elif model_name == "hierarch":
        model_name = "Hierarch."
    elif model_name == "mtlrcr":
        model_name = "MTLR-CR"
    elif model_name == "mtlr":
        model_name = "MTLR"
    elif model_name == "mensa":
        model_name = "MENSA"
    else:
        pass
    return model_name
    