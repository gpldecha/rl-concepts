import numpy as np
from enum import Enum


class ExperienceBuffer:

    """ Inspired by http://www.boost.org/doc/libs/master/doc/html/circular_buffer.html """

    class EntryType(Enum):
        s = 0,
        a = 1,
        r = 2,
        sp = 3,
        done = 4

    def __init__(self, state_size, action_size, num_samples):
        self._buffer = np.empty(shape=[num_samples, 2*state_size + action_size + 1 + 1])
        self._num_samples = num_samples
        self._s_size = state_size
        self._a_size = action_size
        self._head = 0
        self._tail = -1
        self._count = 0

    def append(self, s, a, r, sp, done):

        self._tail = (self._tail + 1) % self._num_samples

        self._buffer[self._tail, 0:self._s_size] = s
        self._buffer[self._tail,  self._s_size:(self._s_size + self._a_size)] = a
        self._buffer[self._tail, (self._s_size + self._a_size):(self._s_size + self._a_size+1)] = r
        self._buffer[self._tail, (self._s_size + self._a_size+1):-1] = sp
        self._buffer[self._tail, -1] = done

        if self._count < self._num_samples:
            self._count += 1
        else:
            self._head = (self._head + 1) % self._num_samples

    def begin(self):
        if self._tail == -1:
            return None
        else:
            return self._buffer[self._head, :]

    def end(self):
        if self._tail == -1:
            return None
        else:
            return self._buffer[self._tail, :]

    def __getitem__(self, item):
        if isinstance(item, int):
            if (self._tail == -1) or (item > self._count-1):
                return None
            else:
                return self._buffer[(self._head+item)%self._num_samples, :]
        else:
            return self._buffer[(self._head+item)%self._num_samples, :]

    def __len__(self):
        return self._count

    def get_item(self, sub_buffer, entry_type):
        if entry_type == self.EntryType.s:
            return sub_buffer[:, 0:self._s_size]
        elif entry_type == self.EntryType.a:
            return sub_buffer[:, self._s_size:(self._s_size + self._a_size)]
        elif entry_type == self.EntryType.r:
            return sub_buffer[:, (self._s_size + self._a_size):(self._s_size + self._a_size+1)]
        elif entry_type == self.EntryType.sp:
            return sub_buffer[:, (self._s_size + self._a_size + 1):-1]
        elif entry_type == self.EntryType.done:
            return sub_buffer[:, -1]
