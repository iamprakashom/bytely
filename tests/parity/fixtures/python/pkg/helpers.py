MAX_ITEMS = 8
default_name = "widget"


def greet():
    return "hello"


def _private():
    return greet()


class Base:
    def describe(self):
        return self.label()

    def label(self):
        return "base"
