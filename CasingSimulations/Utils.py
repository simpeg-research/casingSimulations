import json
import CasingSimulations


def load_properties(filename, targetClass):
    """
    Open a json file and load the properties into the target class
    :param str filename: name of file to read in
    :param str targetClass: name of the target class to recreate
    """
    with open(filename, 'r') as outfile:
        data = getattr(
            CasingSimulations, targetClass
        ).deserialize(json.load(outfile))
    return data

