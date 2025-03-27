from collections import defaultdict


class NamedInstanceMixin:
    _instance_count = defaultdict(int)

    def instance_name(self):
        if self._instance_count[self.__class__] == 0:
            return self.__class__.__name__
        else:
            self._instance_count[self.__class__] += 1
            return (
                f"{self.__class__.__name__}_{self._instance_count[self.__class__] - 1}"
            )
