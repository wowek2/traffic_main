"""
Tests for BoundingBox value object.
"""
from math import isclose

import pytest
from src.shared_kernel.value_objects import BoundingBox, InvalidBoundingBoxError, Point


class TestBoundingBoxCreation:
    """Tests for BoundingBox creation and validation."""

    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 30.0
        assert bbox.y2 == 40.0

    def test_invalid_bounding_box_x1_greater_equal_x2(self):
        """Test creating a bounding box with x1 >= x2 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=30.0, y1=20.0, x2=10.0, y2=40.0)
        assert "x1 (30.0) must be less than x2 (10.0)" in str(exc_info.value)

    def test_invalid_bounding_box_y1_greater_equal_y2(self):
        """Test creating a bounding box with y1 >= y2 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=10.0, y1=40.0, x2=30.0, y2=20.0)
        assert "y1 (40.0) must be less than y2 (20.0)" in str(exc_info.value)

    def test_invalid_bounding_box_negative_x1(self):
        """Test creating a bounding box with negative x1 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=-5.0, y1=20.0, x2=30.0, y2=40.0)
        assert "x1 (-5.0) must be non-negative" in str(exc_info.value)

    def test_invalid_bounding_box_negative_y1(self):
        """Test creating a bounding box with negative y1 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=10.0, y1=-10.0, x2=30.0, y2=40.0)
        assert "y1 (-10.0) must be non-negative" in str(exc_info.value)

    def test_x1_equals_x2_raises_error(self):
        """Test creating a bounding box with x1 == x2 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=15.0, y1=20.0, x2=15.0, y2=40.0)
        assert "x1 (15.0) must be less than x2 (15.0)" in str(exc_info.value)

    def test_y1_equals_y2_raises_error(self):
        """Test creating a bounding box with y1 == y2 raises error."""
        with pytest.raises(InvalidBoundingBoxError) as exc_info:
            BoundingBox(x1=10.0, y1=25.0, x2=30.0, y2=25.0)
        assert "y1 (25.0) must be less than y2 (25.0)" in str(exc_info.value)

class TestBoundingBoxFactoryMethods:
    """Tests for BoundingBox factory methods."""

    def test_from_coordinates(self):
        """Test creating BoundingBox from coordinates tuple."""
        coords = (5.0, 10.0, 15.0, 20.0)
        bbox = BoundingBox.from_coordinates(coords)
        assert bbox.x1 == 5.0
        assert bbox.y1 == 10.0
        assert bbox.x2 == 15.0
        assert bbox.y2 == 20.0

class TestBoundingBoxProperties:
    """Tests for BoundingBox properties."""

    def test_width_property(self):
        """Test width property calculation."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        assert isclose(bbox.width, 20.0)

    def test_height_property(self):
        """Test height property calculation."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        assert isclose(bbox.height, 20.0)

    def test_area_property(self):
        """Test area property calculation."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        assert isclose(bbox.area, 400.0)

class TestBoundingBoxIoU:
    """Tests for Intersection over Union (IoU) calculation."""

    def test_iou_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        bbox1 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        bbox2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox1.iou(bbox2) == 1.0

    def test_iou_no_overlap(self):
        """Test IoU calculation when boxes have no overlap."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=15.0, y1=15.0, x2=25.0, y2=25.0)
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_touching_boxes(self):
        """Test IoU calculation when boxes are touching."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_full_overlap(self):
        """Test IoU calculation when boxes fully overlap."""
        bbox1 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        bbox2 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        assert isclose(bbox1.iou(bbox2), 1.0)

    def test_iou_partial_overlap(self):
        """Test IoU calculation when boxes partially overlap."""
        bbox1 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        bbox2 = BoundingBox(x1=8.0, y1=8.0, x2=18.0, y2=18.0)
        # Intersection area = 7x7 = 49, Union area = 100 + 100 - 49 = 151
        assert isclose(bbox1.iou(bbox2), 49 / 151, rel_tol=1e-6)

    def test_iou_one_inside_another(self):
        """Test IoU calculation when one box is inside another."""
        outer = BoundingBox(x1=0.0, y1=0.0, x2=20.0, y2=20.0)
        inner = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)

        # Inner area = 100
        # Union = 400
        # IoU = 100 / 400 = 0.25
        assert isclose(outer.iou(inner), 0.25, rel_tol=1e-6)
        assert isclose(inner.iou(outer), 0.25, rel_tol=1e-6)

    def test_iou_symmetry(self):
        """IoU should be symmetric: IoU(A, B) == IoU(B, A)."""
        bbox1 = BoundingBox(x1=2.0, y1=2.0, x2=12.0, y2=12.0)
        bbox2 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)

        assert bbox1.iou(bbox2) == bbox2.iou(bbox1)

class TestBoundingBoxContainment:
    """Tests for containment checks."""

    def test_contains_point_inside(self):
        """Test if bounding box contains a point inside."""
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox.contains_point(15.0, 15.0) is True

    def test_contains_point_outside(self):
        """Test if bounding box does not contain a point outside."""
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox.contains_point(25.0, 25.0) is False

    def test_contains_point_on_edge(self):
        """Points on the edge should be contained."""
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox.contains_point(10.0, 15.0) is True
        assert bbox.contains_point(20.0, 15.0) is True
        assert bbox.contains_point(15.0, 10.0) is True
        assert bbox.contains_point(15.0, 20.0) is True

    def test_contains_point_corner(self):
        """Points at the corner should be contained."""
        bbox = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert bbox.contains_point(10.0, 10.0) is True
        assert bbox.contains_point(20.0, 20.0) is True
        assert bbox.contains_point(10.0, 20.0) is True
        assert bbox.contains_point(20.0, 10.0) is True

    def test_contains_box_inner(self):
        """Smaller box inside should be contained."""
        outer = BoundingBox(x1=0.0, y1=0.0, x2=20.0, y2=20.0)
        inner = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        assert outer.contains_box(inner) is True

    def test_contains_box_identical(self):
        """Identical boxes should be contained."""
        box1 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        box2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert box1.contains_box(box2) is True

    def test_contains_box_partial_overlap(self):
        """Partially overlapping boxes should not be contained."""
        box1 = BoundingBox(x1=0.0, y1=0.0, x2=15.0, y2=15.0)
        box2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert box1.contains_box(box2) is False
        assert box2.contains_box(box1) is False

    def test_overlaps_true(self):
        """Boxes that overlap should return True."""
        box1 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        box2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        assert box1.overlaps(box2) is True
        assert box2.overlaps(box1) is True

    def test_overlaps_false(self):
        """Boxes that do not overlap should return False."""
        box1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        box2 = BoundingBox(x1=15.0, y1=15.0, x2=25.0, y2=25.0)
        assert box1.overlaps(box2) is False
        assert box2.overlaps(box1) is False

class TestBoundingBoxTransformation:
    """Tests for bounding box transformations methods."""

    def test_translate(self):
        """Translate should move bounding box by (dx, dy)."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        translated_bbox = bbox.translate(dx=5.0, dy=-10.0)
        assert translated_bbox.x1 == 15.0
        assert translated_bbox.y1 == 10.0
        assert translated_bbox.x2 == 35.0
        assert translated_bbox.y2 == 30.0

    def test_translate_preserves_size(self):
        """Translate should not change the size of the bounding box."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        translated  = bbox.translate(dx=-5.0, dy=15.0)

        assert translated.width == bbox.width
        assert translated.height == bbox.height

    def translate_negative(self):
        """Translate with negative offsets should work correctly."""
        bbox = BoundingBox(x1=20.0, y1=30.0, x2=40.0, y2=50.0)
        translated_bbox = bbox.translate(dx=-10.0, dy=-20.0)
        assert translated_bbox.x1 == 10.0
        assert translated_bbox.y1 == 10.0
        assert translated_bbox.x2 == 30.0
        assert translated_bbox.y2 == 30.0

    def test_scale_up(self):
        """Scaling up should enlarge the bounding box."""
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        scaled_bbox = bbox.scale(factor=2.0)

        assert scaled_bbox.width == 200.0
        assert scaled_bbox.height == 200.0
        assert scaled_bbox.x1 == 0.0
        assert scaled_bbox.y1 == 0.0
        assert scaled_bbox.x2 == 200.0
        assert scaled_bbox.y2 == 200.0

    def test_scale_down(self):
        """Scaling down should shrink the bounding box."""
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        scaled_bbox = bbox.scale(factor=0.5)

        assert scaled_bbox.width == 50.0
        assert scaled_bbox.height == 50.0
        assert scaled_bbox.x1 == 75.0
        assert scaled_bbox.y1 == 75.0
        assert scaled_bbox.x2 == 125.0
        assert scaled_bbox.y2 == 125.0

    def test_scale_preserves_center(self):
        """Scaling should preserve the center of the bounding box."""
        bbox = BoundingBox(x1=20.0, y1=30.0, x2=60.0, y2=70.0)
        scaled = bbox.scale(factor=1.5)
        expected_center = Point(40.0, 50.0)
        assert scaled.center == expected_center
        assert bbox.center == expected_center

class TestBoundingBoxDistance:
    """Tests for bounding box distance calculations."""
    def test_center_distance_same_center(self):
        """Center distance between identical boxes should be zero."""
        bbox1 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)
        bbox2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)

        assert bbox1.center_distance(bbox2) == 0.0

    def test_center_distance_horizontal(self):
        """Center distance for boxes aligned horizontally."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=20.0, y1=0.0, x2=30.0, y2=10.0)

        assert bbox1.center_distance(bbox2) == 20.0

    def test_center_distance_vertical(self):
        """Center distance for boxes aligned vertically."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=0.0, y1=20.0, x2=10.0, y2=30.0)

        assert bbox1.center_distance(bbox2) == 20.0

    def test_center_distance_diagonal(self):
        """Center distance for boxes aligned diagonally."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0)

        expected_distance = ((15.0 - 5.0) ** 2 + (15.0 - 5.0) ** 2) ** 0.5
        assert bbox1.center_distance(bbox2) == expected_distance

    def test_edge_distance_overlapping(self):
        """Edge distance for overlapping boxes should be zero."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)

        assert bbox1.edge_distance(bbox2) == 0.0

    def test_edge_distance_touching(self):
        """Edge distance for touching boxes should be zero."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=10.0, y1=0.0, x2=20.0, y2=10.0)

        assert bbox1.edge_distance(bbox2) == 0.0

    def test_edge_distance_separated_horizontal(self):
        """Edge distance for horizontally separated boxes."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=20.0, y1=0.0, x2=30.0, y2=10.0)

        assert bbox1.edge_distance(bbox2) == 10.0

    def test_edge_distance_separated_vertical(self):
        """Edge distance for vertically separated boxes."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=0.0, y1=20.0, x2=10.0, y2=30.0)

        assert bbox1.edge_distance(bbox2) == 10.0

    def test_edge_distance_separated_diagonal(self):
        """Edge distance for diagonally separated boxes."""
        bbox1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        bbox2 = BoundingBox(x1=15.0, y1=15.0, x2=25.0, y2=25.0)

        expected_distance = ((15.0 - 10.0) ** 2 + (15.0 - 10.0) ** 2) ** 0.5
        assert bbox1.edge_distance(bbox2) == expected_distance

class TestBoundingBoxImmutability:
    """Tests for immutability guarantees."""

    def test_frozen_dataclass(self):
        """BoundingBox should be frozen."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        with pytest.raises(AttributeError):
            bbox.x1 = 50.0 # type: ignore
        with pytest.raises(AttributeError):
            bbox.y2 = 150.0 # type: ignore

    def test_translate_returns_new(self) -> None:
        """Translate should return new object, not modify original."""
        original = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        translated = original.translate(dx=5.0, dy=5.0)

        assert original.x1 == 0.0
        assert translated.x1 == 5.0
        assert original is not translated

    def test_scale_returns_new(self) -> None:
        """Scale should return new object, not modify original."""
        original = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        scaled = original.scale(factor=2.0)
        print(scaled)
        assert original.width == 10.0
        assert scaled.width == 15.0
        assert original is not scaled

class TestBoundingBoxEquality:
    """Tests for equality comparisons."""

    def test_equal_boxes(self):
        """Boxes with same coordinates should be equal."""
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        bbox2 = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)

        assert bbox1 == bbox2

    def test_unequal_boxes(self):
        """Boxes with different coordinates should not be equal."""
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        bbox2 = BoundingBox(x1=15.0, y1=25.0, x2=35.0, y2=45.0)

        assert bbox1 != bbox2

    def test_hashable(self) -> None:
        """BoundingBox should be hasable."""
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
        bbox2 = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)

        # Should be usable in a set
        bbox_set = {bbox1, bbox2}
        assert len(bbox_set) == 1 # Same boxes should only appear once

        # Should be usable as dict keys
        bbox_dict = {bbox1: "test"}
        assert bbox_dict[bbox2] == "test"

class TestBoundingBoxConversions:
    """Tests for conversion and intersection helpers."""

    def test_to_tuple(self):
        """`to_tuple` should return the original coordinate tuple."""
        bbox = BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert bbox.to_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_intersection_overlap_returns_bbox(self):
        """`intersection` should return a BoundingBox for overlapping boxes."""
        b1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        b2 = BoundingBox(x1=5.0, y1=5.0, x2=15.0, y2=15.0)
        inter = b1.intersection(b2)
        assert isinstance(inter, BoundingBox)
        assert inter.to_tuple() == (5.0, 5.0, 10.0, 10.0)

    def test_intersection_none_when_no_overlap(self):
        """`intersection` should return None when boxes do not overlap."""
        b1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        b2 = BoundingBox(x1=20.0, y1=20.0, x2=30.0, y2=30.0)
        assert b1.intersection(b2) is None

    def test_corner_properties(self):
        """Top-left, top-right and bottom-left should match coordinates."""
        bbox = BoundingBox(x1=1.0, y1=2.0, x2=4.0, y2=8.0)
        assert bbox.top_left == (1.0, 2.0)
        assert bbox.top_right == (4.0, 2.0)
        assert bbox.bottom_left == (1.0, 8.0)
