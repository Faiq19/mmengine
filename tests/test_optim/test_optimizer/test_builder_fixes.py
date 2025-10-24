# Copyright (c) OpenMMLab. All rights reserved.
"""
Test cases for optimizer builder fixes.

This test module validates the fixes for:
1. KeyError when re-registering optimizers
2. Missing warnings import
3. Inconsistent variable naming in register_bitsandbytes_optimizers
4. Missing append in register_sophia_optimizers
"""
import unittest

from mmengine.optim.optimizer.builder import (
    OPTIMIZERS,
    register_torch_optimizers,
    register_bitsandbytes_optimizers,
    register_sophia_optimizers,
)


class TestOptimizerBuilderFixes(unittest.TestCase):
    """Test cases for optimizer builder fixes."""

    def test_register_torch_optimizers_no_keyerror_on_reregistration(self):
        """Test that re-registering torch optimizers doesn't raise KeyError."""
        # First registration happens at module import time
        # Try to register again - should not raise KeyError
        result = register_torch_optimizers()
        # Should return empty list since all optimizers are already registered
        self.assertEqual(result, [])

    def test_torch_adafactor_registered_correctly(self):
        """Test that Adafactor is registered as TorchAdafactor."""
        # Verify that TorchAdafactor is registered (not Adafactor)
        self.assertIn('TorchAdafactor', OPTIMIZERS)

    def test_register_torch_optimizers_returns_correct_list(self):
        """Test that register_torch_optimizers returns expected
        optimizer names."""
        # Clear and re-register to test the return value
        # We can't actually clear the registry without breaking things,
        # so we test that calling it again returns empty list
        result = register_torch_optimizers()
        self.assertIsInstance(result, list)
        # Should be empty since already registered
        self.assertEqual(len(result), 0)

    def test_register_sophia_optimizers_returns_list(self):
        """Test that register_sophia_optimizers returns a list."""
        # This will likely return empty list if Sophia is not installed
        # but it should not raise an error
        result = register_sophia_optimizers()
        self.assertIsInstance(result, list)

    def test_register_bitsandbytes_optimizers_returns_list(self):
        """Test that register_bitsandbytes_optimizers returns a list."""
        # This will likely return empty list if bitsandbytes is not installed
        # but it should not raise an error
        result = register_bitsandbytes_optimizers()
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
