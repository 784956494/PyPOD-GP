import argparse

def add_flags_from_config(parser, config_dict):

    def OrNone(default):
        def func(x):
            if x.lower() == "none":
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

config_args = {
    'training_config': {
        'x': (0.014, 'space dimension for x-axis in mesh'),
        'y': (0.012, 'space dimension for y-axis in mesh'),
        'z': (0.00065, 'space dimension for z-axis in mesh'),
        'x-dim': (129, 'number of cells in x-axis'),
        'y-dim': (129, 'number of cells in y-axis'),
        'z-dim': (14, 'number of cells in z-axis'),
        'time-steps': (240, 'time steps for discretization'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'tol': (1e-14, 'padding for float point comparisons'),
        'num-modes': (8, 'the number of modes to use, could be integer or list'),
        'Nu': (11, 'the number of functional units'),
        'surfaces': (5, 'subdomain of interests for G matrix'),
        'sampling-interval':(4.347826086956521e-6, 'the last time step'),
        'degree': (2, 'the degree of polynomials to integrate'),
        'steps': (20, 'the number of steps to take to solve the ODE'),
        'save': (0, 'whether or not to save training results'),
        'save-dir':('log', 'paths to the direcotry to save training results'),
        'task': ('both', 'which task to perform, can be any of [train, predict, both]'),
        'save-format': ('txt', 'which format to save/read the trained constants, can be any of [txt, csv]')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)