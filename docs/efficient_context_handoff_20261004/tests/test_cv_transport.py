import math
import unittest
from reference.context_plan import transport_ttc_constant_velocity

class TransportContracts(unittest.TestCase):
    def test_approach(self):
        self.assertAlmostEqual(transport_ttc_constant_velocity(2., .2), 1.8)
    def test_receding(self):
        self.assertAlmostEqual(transport_ttc_constant_velocity(-2., .2), -2.2)
    def test_infinite(self):
        self.assertTrue(math.isinf(transport_ttc_constant_velocity(float('inf'), .2)))
    def test_no_post_contact_sign_flip(self):
        self.assertIsNone(transport_ttc_constant_velocity(.2, .3))
    def test_domain_guard(self):
        self.assertIsNone(transport_ttc_constant_velocity(.15, .1))
    def test_current(self):
        self.assertEqual(transport_ttc_constant_velocity(2., 0.), 2.)
    def test_future(self):
        with self.assertRaises(ValueError):
            transport_ttc_constant_velocity(2., -.1)
    def test_invalid_value(self):
        self.assertIsNone(transport_ttc_constant_velocity(float('nan'), .1))

if __name__ == '__main__':
    unittest.main()
