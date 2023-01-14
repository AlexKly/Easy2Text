import os
from Easy2Text import Configurations


class Init(Configurations):
    def init(self):
        if not os.path.exists(self.dir_models):
            os.mkdir(self.dir_models)
