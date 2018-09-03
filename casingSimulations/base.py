import properties
import os
import json
import warnings

from .info import __version__
from .utils import load_properties


class LoadableInstance(properties.Instance):

    class_info = "an instance of a class or the name of a file from which the "
    "instance can be created"

    def validate(self, instance, value):
        if isinstance(value, str):
            value = load_properties(value)
        return super(LoadableInstance, self).validate(instance, value)


class BaseCasing(properties.HasProperties):
    """
    Base class that contains working directories, code version and can be saved
    """

    # Properties
    filename = properties.String(
        "Filename to which the properties are serialized and written to",
    )

    directory = properties.String(
        "Working directory",
        default="."
    )

    version = properties.String(
        "version of the software",
        default = __version__
    )

    # Observers and validators
    @properties.validator('version')
    def _validate_version(self, change):
        if change['value'] != __version__:
            warnings.warn(
                "Instance was created on version {}, but the current version "
                "is {}".format(
                    change['value'], __version__
                )
            )

    @properties.validator('directory')
    def _ensure_abspath(self, change):
        val = change['value']
        fullpath = os.path.abspath(os.path.expanduser(val))

        # if not os.path.isdir(fullpath):
        #     os.mkdir(fullpath)

        # change['value'] = fullpath

    # methods
    def save(self, filename=None, directory=None):
        """
        Save the casing properties to json
        :param str file: filename for saving the casing properties
        :param str directory: working directory for saving the file
        """

        # make sure properties are all valid prior to saving
        self.validate()

        # if no filename provided, grab it from the class
        if filename is None:
            filename = self.filename

        if directory is None:
            directory = self.directory

        # check if the directory exists, if not, create it
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # filename to save
        f = '/'.join([directory, filename])
        with open(f, 'w') as outfile:
            cp = json.dump(self.serialize(), outfile)

        print('Saved {}'.format(f))

    def copy(self):
        """
        Make a copy of the current casing object
        """
        return properties.copy(self)

