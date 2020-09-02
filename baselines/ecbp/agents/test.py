import numpy as np
from multiprocessing import Process, Pipe
import time


class Test(Process):
    def __init__(self, conn):
        super(Test, self).__init__()
        self.list = []
        self.conn = conn

    def run(self) -> None:
        for i in range(10):
            self.list.append(i)
            print("here", self.list)
            self._empty_pipe()
            time.sleep(1)

    def _empty_pipe(self):
        while self.conn.poll():
            msg = self.conn.recv()
            code, obj = msg
            assert code == 0
            self.modify(obj)

    def modify(self, x):
        self.list.append(x)


parent_conn, child_conn = Pipe()
a = Test(child_conn)
a.start()
for i in range(10):
    print(a.list)
    parent_conn.send((0, i + 20))
    time.sleep(0.8)

a.join()
print(a.list)
