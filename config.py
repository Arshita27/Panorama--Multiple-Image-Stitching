class ConfigNode(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
               setattr(self, key, [ConfigNode(x) if isinstance(x, dict) else x for x in value])
            else:
               setattr(self, key, ConfigNode(value) if isinstance(value, dict) else value)
