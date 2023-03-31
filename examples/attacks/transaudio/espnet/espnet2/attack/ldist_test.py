from ldist import recursive_ldist, iterative_ldist, ncells
import unittest

class BaseTestLDist(unittest.TestCase):

    def _test_by_data(self, data):
        ldist = self.ldist
        for d in data:
            self.assertEqual(ldist(d[0], d[1]), d[2], '"%s" "%s"' % (d[0], d[1]))
            self.assertTrue(ncells <= max(len(d[0]), len(d[1])) * (2 * d[2] + 1))
            self.assertEqual(ldist(d[1], d[0]), d[2], '"%s" "%s"' % (d[1], d[0]))
            self.assertTrue(ncells <= max(len(d[0]), len(d[1])) * (2 * d[2] + 1))

    def test_same(self):
        data_x = [ 'foo', '', 'baba qga' ]
        self._test_by_data(map(lambda s: (s, s, 0), data_x))
        # self.assertEqual(ldist("foo", "foo"), 0)
        # self.assertEqual(ldist("", ""), 0)
        # self.assertEqual(ldist("baba qga", "baba qga"), 0)

    def test_insertions(self):
        data = [ ('foo', 'fooo', 1)
            , ('foo', 'foio', 1)
            , ('foo', 'falaobalao', 7)
            ]
        self._test_by_data(data)

    def test_deletions(self):
        data = [ ('foo', 'oo', 1)
            , ('foo', '', 3)
            , ('foo', 'f', 2)
            ]
        self._test_by_data(data)

    def test_changes(self):
        data = [ ('foo', 'foz', 1)
            , ('sitten', 'sittin', 1)
            , ('baba qga', 'mama qga', 2)
            ]
        self._test_by_data(data)

    def test_all(self):
        data = [ ('acgtacgtacgt', 'acatacttgtact', 4)
            ]
        self._test_by_data(data)

class TestRecursiveLDist(BaseTestLDist):
    def __init__(self, *args):
        self.ldist = recursive_ldist
        super().__init__(*args)

class TestIterativeLDist(BaseTestLDist):
    def __init__(self, *args):
        self.ldist = iterative_ldist
        super().__init__(*args)

    def test_huge(self):
        data = [ ('a' * 10000 + 'c' * 1, 'a' * 10000 + 'b' * 1, 1)
            , ('a' * 10000 + 'c' * 100, 'a' * 10000 + 'b' * 100, 100)
            ]
        self._test_by_data(data)

if __name__ == '__main__':
    for testClass in [TestRecursiveLDist, TestIterativeLDist]:
        suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
        unittest.TextTestRunner(verbosity=2).run(suite)
