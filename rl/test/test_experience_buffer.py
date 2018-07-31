import unittest

import numpy as np
from rl.utils.math import distance
from utils.experience_buffer import ExperienceBuffer
from utils.math import uniform_sample


class TestExperienceBuffer(unittest.TestCase):

    def test_length_3(self):

        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        buffer.append(s=1, a=0, r=0, sp=1, done=0)
        self.assertEqual(distance(buffer.end(), np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[0], np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(len(buffer), 1)

        buffer.append(s=2, a=0, r=0, sp=1, done=0)

        self.assertEqual(distance(buffer.begin(), np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer.end(), np.array([2, 0, 0, 1, 0])), 0.0)

        self.assertEqual(distance(buffer[0], np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[1], np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(len(buffer), 2)

        buffer.append(s=3, a=0, r=0, sp=1, done=0)

        self.assertEqual(distance(buffer.begin(), np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer.end(), np.array([3, 0, 0, 1, 0])), 0.0)

        self.assertEqual(distance(buffer[0], np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[1], np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[2], np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(len(buffer), 3)

        buffer.append(s=4, a=0, r=0, sp=1, done=0)

        self.assertEqual(distance(buffer.begin(), np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer.end(), np.array([4, 0, 0, 1, 0])), 0.0)

        self.assertEqual(distance(buffer[0], np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[1], np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[2], np.array([4, 0, 0, 1, 0])), 0.0)
        self.assertEqual(len(buffer), 3)

        buffer.append(s=5, a=0, r=0, sp=1, done=0)
        self.assertEqual(distance(buffer.begin(), np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer.end(), np.array([5, 0, 0, 1, 0])), 0.0)

        self.assertEqual(distance(buffer[0], np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[1], np.array([4, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(buffer[2], np.array([5, 0, 0, 1, 0])), 0.0)
        self.assertEqual(len(buffer), 3)

    def test_idx_outside1(self):
        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        self.assertIsNone(buffer[0])
        self.assertIsNone(buffer[-1])
        self.assertIsNone(buffer[100])

    def test_idx_outside2(self):
        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        buffer.append(s=1, a=0, r=0, sp=1, done=0)
        self.assertIsNone(buffer[1])

    def test_multiple_idxes(self):
        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        buffer.append(s=1, a=0, r=0, sp=1, done=0)
        buffer.append(s=2, a=0, r=0, sp=1, done=0)
        buffer.append(s=3, a=0, r=0, sp=1, done=0)

        output = buffer[np.array([0, 1, 2], dtype=int)]

        self.assertEqual(distance(output[0, :], np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[1, :], np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[2, :], np.array([3, 0, 0, 1, 0])), 0.0)

        output = buffer[np.array([2, 1, 0], dtype=int)]

        self.assertEqual(distance(output[0, :], np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[1, :], np.array([2, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[2, :], np.array([1, 0, 0, 1, 0])), 0.0)

        output = buffer[np.array([2, 0, 0], dtype=int)]

        self.assertEqual(distance(output[0, :], np.array([3, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[1, :], np.array([1, 0, 0, 1, 0])), 0.0)
        self.assertEqual(distance(output[2, :], np.array([1, 0, 0, 1, 0])), 0.0)

    def test_uniform_sample(self):

        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        buffer.append(s=1, a=0, r=0, sp=1, done=0)
        buffer.append(s=2, a=0, r=0, sp=1, done=0)
        buffer.append(s=3, a=0, r=0, sp=1, done=0)

        mini_batch = uniform_sample(buffer, 10)

        self.assertEqual(len(mini_batch), 3)

    def test_get_item(self):
        buffer = ExperienceBuffer(state_size=1, action_size=1, num_samples=3)
        buffer.append(s=1, a=1, r=0, sp=1, done=0)
        buffer.append(s=2, a=0, r=0, sp=1, done=0)
        buffer.append(s=3, a=1, r=1, sp=1, done=0)

        mini_batch = buffer[np.array([0, 1, 2], dtype=int)]
        entry_type = ExperienceBuffer.EntryType

        s = buffer.get_item(mini_batch, entry_type.s)
        self.assertEqual(distance(s.flatten(), np.array([1., 2., 3.])), 0.0)

        a = buffer.get_item(mini_batch, entry_type.a)
        self.assertEqual(distance(a.flatten(), np.array([1., 0., 1.])), 0.0)

        r = buffer.get_item(mini_batch, entry_type.r)
        self.assertEqual(distance(r.flatten(), np.array([0., 0., 1.])), 0.0)

        sp = buffer.get_item(mini_batch, entry_type.sp)
        self.assertEqual(distance(sp.flatten(), np.array([1., 1., 1.])), 0.0)

        done = buffer.get_item(mini_batch, entry_type.done)
        self.assertEqual(distance(done.flatten(), np.array([0., 0., 0.])), 0.0)