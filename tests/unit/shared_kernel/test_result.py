import pytest
from unittest.mock import Mock

from src.shared_kernel import Ok, Err, Result

class TestOkState:
    """Test veryfing monad behaviour in succes state."""

    def test_basic_properties(self):
        res: Result[int, str] = Ok(10)
        assert res.is_ok() is True
        assert res.is_err() is False
        assert repr(res) == "Ok(10)"

    def test_unwrap_methods(self):
        """Method unwraping values."""
        res: Result[int, str] = Ok(5)
        assert res.unwrap() == 5
        assert res.expect("Panic") == 5
        assert res.unwrap_or(0) == 5
        assert res.unwrap_or_else(lambda e: 0) == 5

    def test_unwrap_err_raises_error(self):
        """Trying to retrive error from Ok should throw exception."""
        with pytest.raises(ValueError, match="Called unwrap_err on Ok"):
            Ok(5).unwrap_err()
    
    def test_mapping_transform_value(self):
        """Mapping should change value inside of Ok."""
        assert Ok(2).map(lambda x: x * 2) == Ok(4)

    def test_map_err_is_ignored(self):
        """Maping error shouldn't influence Ok."""
        assert Ok(2).map_err(lambda e: "New error") == Ok(2)

    def test_inspect_executes_side_effect(self):
        """Inspect should execute function."""
        spy = Mock()
        res: Result[str, str] = Ok("data").inspect(spy)
        spy.assert_called_once_with("data")
        assert res == Ok("data")

    def test_inspect_err_is_ignored(self):
        """Inspect error should be ignored."""
        spy = Mock()
        Ok("data").inspect_err(spy)
        spy.assert_not_called()

class TestErrState:
    """Tests veryfing monad behaviour in error state."""

    def test_basic_properties(self):
        res: Result[int, str] = Err("Error")
        assert res.is_ok() is False
        assert res.is_err() is True
        assert repr(res) == "Err(Error)"

    def test_unwrap_raises_error(self):
        """Unwrap on error should throw exception."""
        with pytest.raises(ValueError, match="Called unwrap on Err"):
            Err("Fail").unwrap()

    def test_expect_raises_with_message(self):
        with pytest.raises(ValueError, match="Critical: Fail"):
            Err("Fail").expect("Critical")

    def test_unwrap_methods(self):
        """Recovery methods."""
        res: Result[int, str] = Err("Fail")
        assert res.unwrap_err() == "Fail"
        assert res.unwrap_or(100) == 100
        assert res.unwrap_or_else(lambda e: len(e)) == 4    # len("Fail")

    def test_map_is_ignored(self):
        """Mapping values shouldn't influence Err."""
        assert Err("E").map(lambda x: x * 2) == Err("E")

    def test_inspect_is_ignored(self):
        spy = Mock()
        Err("E").inspect(spy)
        spy.assert_not_called()

    def test_inspect_err_execute_side_effect(self):
        spy = Mock()
        res: Result[int, str] = Err("E").inspect_err(spy)
        spy.assert_called_once_with("E")
        assert res == Err("E")

class TestMonadicChaining:
    """
    Tests verifying flow between states Ok <-> Err.
    """

    def test_flat_map_ok_to_ok(self):
        """Path: Ok -> flat_map -> Ok"""
        res: Result[int, str] = Ok(10).flat_map(lambda x: Ok(x+1))
        assert res == Ok(11)

    def test_flat_map_ok_to_err(self):
        """Path: Ok -> flat_map -> (skip) -> Err."""
        start: Result[int, str] = Ok(10)

        res: Result[int, str] = start.flat_map(lambda x: Err("Too big"))

        assert res == Err("Too big")

    def test_flat_map_on_err_is_skipped(self):
        """Path: Err -> flat_map -> (skip) -> Err."""
        spy = Mock()
        res = Err("Original").flat_map(spy)
        spy.assert_not_called()
        assert res == Err("Original")

    def test_or_else_recovery(self):
        """Path: Err -> or_else -> Ok (error fixed)"""
        start: Result[str, str] = Err("Glitch")
        res: Result[str, str] = start.or_else(lambda e: Ok("Fixed"))
        assert res == Ok("Fixed")

    def test_or_else_failure(self):
        """Path: Err -> or_else -> Err (recovery failure)."""
        res: Result[int, str] = Err("Glitch").or_else(lambda e: Err("Critical failure"))
        assert res == Err("Critical failure")

    def test_or_else_on_ok_is_skipped(self):
        """Path: Ok -> or_else -> (skip) -> Ok."""
        spy = Mock()
        res = Ok(1).or_else(spy)
        spy.assert_not_called()
        assert res == Ok(1)

    def test_complex_pipeline(self):
        """Testing longer chain of events."""
        def validation(x: int) -> Result[int, str]:
            return Ok(x) if x > 0 else Err("Negative")
        
        start_node: Result[int, str] = Ok(10)

        # 1. Succes of a whole chain
        chain_sucess: Result[int, str] = (
            start_node
            .map(lambda x: x * 2)     # -> Ok(20)
            .flat_map(validation)     # -> Ok(20)
            .map_err(lambda e: "Err") # Ignored
        )
        assert chain_sucess == Ok(20)

        start_node = Ok(-5)

        # 2. Break in the middle
        chain_fail: Result[int, str] = (
            start_node
            .map(lambda x: x * 2)       # -> Ok(-10)
            .flat_map(validation)       # -> Err("Negative")
            .map(lambda x: x + 1)     # -> Ignored
        )
        assert chain_fail == Err("Negative")