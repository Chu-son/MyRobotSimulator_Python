import time

# 処理時間を計測するクラス
class ProcessingTimer(object):

    def __init__(self, label = "processing time"):
        self.start_time = 0  # 開始時間
        self._label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print( '{} : {:.3f} [s]'.format(self._label, time.time() - self.start_time))