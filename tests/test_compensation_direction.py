import unittest
from convert_to_recoiltrainer import normalize_coordinates, validate_compensation_coordinates, ConversionError

class CompensationDirectionTests(unittest.TestCase):
    def test_right_up_recoil_requires_left_down_mouse(self):
        rows = [{'x': 7., 'y': -4.}, {'x': 17., 'y': 16.}, {'x': 2., 'y': 36.}]
        points, scale, _ = normalize_coordinates(rows, 240, 20)
        self.assertEqual(scale, 6)
        self.assertEqual(points, [(80., 20.), (20., 140.), (110., 260.)])
        validate_compensation_coordinates(rows, points, scale)

    def test_old_unmirrored_export_is_rejected(self):
        rows = [{'x': 0., 'y': 0.}, {'x': 10., 'y': 20.}]
        with self.assertRaises(ConversionError):
            validate_compensation_coordinates(rows, [(0., 0.), (10., 20.)], 1)

    def test_downward_recoil_segment_requires_upward_mouse(self):
        rows = [{'x': 0., 'y': 0.}, {'x': -3., 'y': 20.}, {'x': 2., 'y': 18.}]
        points, scale, _ = normalize_coordinates(rows, 0, 0)
        self.assertLess(points[2][1], points[1][1])
        self.assertLess(points[2][0], points[1][0])
        validate_compensation_coordinates(rows, points, scale)

if __name__ == '__main__':
    unittest.main()
