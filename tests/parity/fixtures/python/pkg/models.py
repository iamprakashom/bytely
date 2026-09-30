from typing import TYPE_CHECKING

from .helpers import Base, greet

if TYPE_CHECKING:
    from .cli import main


class Widget(Base):
    @staticmethod
    def make():
        return Widget()

    @property
    def name(self):
        return greet()

    async def fetch(self):
        return await self.load()

    async def load(self):
        return None

    class Inner:
        def method(self):
            return build_widget()


def build_widget():
    return Widget.make()
