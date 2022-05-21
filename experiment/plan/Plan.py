from abc import ABC, abstractmethod


class Plan(ABC):
    def __init__(self, *variables):
        self.variables = variables

    def pre_task(self, variable):
        pass

    def post_task(self, variable):
        pass

    @abstractmethod
    def task(self, variable):
        pass

    def execute(self, verbose: bool = False):
        if len(self.variables) == 0:
            return self.task(None)
        else:
            result = []
            for i, var in enumerate(self.variables):
                if verbose:
                    print(f'{i + 1}/{len(self.variables)}\t:', var)
                self.pre_task(var)
                result.append(self.task(var))
                self.post_task(var)

            return result
