import pkg.helpers as h
from pkg import models


def main():
    h.greet()
    return models.build_widget()


def outer():
    def inner():
        return 1

    return inner()


def other(inner):
    return inner()
