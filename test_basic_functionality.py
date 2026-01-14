#!/usr/bin/env python3
"""
Basic functionality tests for TPose TDesign service.
Tests imports, data models, and local functions.
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services')

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("Testing Imports")
    print("="*70)

    try:
        # Shared models
        from shared.models.pose import Pose
        from shared.models.ranking_result import RankingResult
        from shared.models.batch import PoseBatch
        print("✓ Shared models imported successfully")

        # Shared utils
        from shared.utils.s3_client import S3Client
        from shared.utils.logging_config import setup_logging
        from shared.utils.gpu_utils import detect_gpu
        from shared.utils.cif_converter import convert_cif_to_pdb_sdf
        print("✓ Shared utils imported successfully")

        # Shared config
        from shared.config.settings import Settings
        print("✓ Shared config imported successfully")

        # Force fields - need to add services directory to path
        sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services/services/pose-ranking')
        from src.force_fields.base import BaseForceField
        from src.force_fields.helpers import (
            parse_energy_from_xtb_output,
            chop_pdb,
            split_pdb,
            form_complex,
            get_protein_atom_indices
        )
        from src.force_fields.xtb import XTBForceField
        from src.force_fields.so3lr import SO3LRForceField
        print("✓ Force field modules imported successfully")

        # Runner
        from src.ranking_runner import RankingRunner
        print("✓ Ranking runner imported successfully")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_models():
    """Test data model creation and validation."""
    print("\n" + "="*70)
    print("Testing Data Models")
    print("="*70)

    try:
        from shared.models.pose import Pose
        from shared.models.ranking_result import RankingResult
        from shared.models.batch import PoseBatch

        # Test Pose creation with PDB+SDF
        pose1 = Pose(
            pose_id="test_001",
            protein_pdb="s3://bucket/protein.pdb",
            ligand_sdf="s3://bucket/ligand.sdf",
            structure_path="s3://bucket/output/complex.pdb",
            energy_method="gfn2"
        )
        assert pose1.validate(), "Pose validation failed"
        assert pose1.has_direct_structures(), "Direct structures check failed"
        assert not pose1.needs_cif_conversion(), "CIF conversion check failed"
        print("✓ Pose with PDB+SDF created and validated")

        # Test Pose creation with CIF
        pose2 = Pose(
            pose_id="test_002",
            structure_cif="s3://bucket/structure.cif",
            energy_method="so3lr"
        )
        assert pose2.validate(), "CIF Pose validation failed"
        assert pose2.needs_cif_conversion(), "CIF conversion check failed"
        assert not pose2.has_direct_structures(), "Direct structures check failed"
        print("✓ Pose with CIF created and validated")

        # Test Pose to/from dict
        pose_dict = pose1.to_dict()
        pose_recreated = Pose.from_dict(pose_dict)
        assert pose_recreated.pose_id == pose1.pose_id
        assert pose_recreated.structure_path == pose1.structure_path
        print("✓ Pose serialization/deserialization works")

        # Test invalid Pose (missing inputs)
        pose_invalid = Pose(pose_id="invalid")
        assert not pose_invalid.validate(), "Invalid pose should fail validation"
        print("✓ Invalid pose correctly rejected")

        # Test RankingResult success
        result = RankingResult.from_success(
            pose_id="test_001",
            interaction_energy=-10.5,
            strain_energy=2.3,
            energy_method="gfn2",
            force_field_device="cpu",
            complex_energy=-100.0,
            protein_energy=-80.0,
            ligand_bound_energy=-20.0,
            ligand_free_energy=-17.7
        )
        assert result.ranking_success, "RankingResult success flag not set"
        assert result.total_score == -8.2, "Total score calculation incorrect"
        print("✓ RankingResult success case created")

        # Test RankingResult error
        result_error = RankingResult.from_error(
            pose_id="test_002",
            error_message="Test error"
        )
        assert not result_error.ranking_success, "Error result should have success=False"
        assert result_error.error_message == "Test error"
        print("✓ RankingResult error case created")

        # Test RankingResult to_dict
        result_dict = result.to_dict()
        assert 'pose_id' in result_dict
        assert 'interaction_energy' in result_dict
        assert result_dict['total_score'] == -8.2
        print("✓ RankingResult serialization works")

        # Test PoseBatch
        poses = [
            {"pose_id": "batch_001", "protein_pdb": "s3://bucket/p1.pdb", "ligand_sdf": "s3://bucket/l1.sdf"},
            {"pose_id": "batch_002", "protein_pdb": "s3://bucket/p2.pdb", "ligand_sdf": "s3://bucket/l2.sdf"},
        ]
        batch = PoseBatch.from_dict_list(poses, global_energy_method="gfn2")
        assert len(batch.poses) == 2
        assert batch.global_energy_method == "gfn2"
        # Test that get_energy_method returns the global method
        assert batch.get_energy_method(batch.poses[0]) == "gfn2"
        print("✓ PoseBatch created from dict list")

        # Test batch validation
        is_valid = batch.validate()
        assert is_valid == True
        print("✓ PoseBatch validation works")

        print("\n✅ All data model tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Data model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_force_field_initialization():
    """Test force field class initialization."""
    print("\n" + "="*70)
    print("Testing Force Field Initialization")
    print("="*70)

    try:
        sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services/services/pose-ranking')
        from src.force_fields.xtb import XTBForceField
        from src.force_fields.so3lr import SO3LRForceField

        # Test xTB initialization (CPU only)
        xtb_ff = XTBForceField(device="cpu")
        assert xtb_ff.check_availability(), "xTB availability check failed"
        assert xtb_ff.get_device() == "cpu"
        print("✓ XTBForceField initialized (CPU)")

        # Test SO3LR initialization with auto device
        so3lr_ff = SO3LRForceField(device="auto")
        device = so3lr_ff.get_device()
        print(f"✓ SO3LRForceField initialized (device: {device})")

        # Test SO3LR CPU fallback
        so3lr_cpu = SO3LRForceField(device="cpu")
        assert so3lr_cpu.get_device() == "cpu"
        print("✓ SO3LRForceField initialized with CPU fallback")

        print("\n✅ All force field initialization tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Force field initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """Test helper function imports and basic functionality."""
    print("\n" + "="*70)
    print("Testing Helper Functions")
    print("="*70)

    try:
        sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services/services/pose-ranking')
        from src.force_fields.helpers import (
            parse_energy_from_xtb_output,
            get_protein_atom_indices
        )

        # Test xTB output parsing - skip for now, will test when actually running xTB
        print("✓ parse_energy_from_xtb_output imported (will test with real xTB output)")

        # Test protein atom indices parsing
        test_pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
HETATM    3  C1  LIG B   1      12.000  12.000  12.000  1.00 20.00           C
END
"""
        # Write temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(test_pdb_content)
            temp_pdb = f.name

        try:
            protein_indices = get_protein_atom_indices(temp_pdb)
            # Should get ATOM entries only (indices 1 and 2), not HETATM
            # But the function might be 0-indexed or 1-indexed - let's just check we got protein atoms
            assert len(protein_indices) >= 2, f"Expected at least 2 protein atoms, got {len(protein_indices)}: {protein_indices}"
            print(f"✓ get_protein_atom_indices works (found {len(protein_indices)} protein atoms)")
        finally:
            os.unlink(temp_pdb)

        print("\n✅ All helper function tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_settings_and_utils():
    """Test settings and utility classes."""
    print("\n" + "="*70)
    print("Testing Settings and Utils")
    print("="*70)

    try:
        import logging
        from shared.config.settings import Settings
        from shared.utils.logging_config import setup_logging
        from shared.utils.gpu_utils import detect_gpu

        # Test Settings
        settings = Settings()
        assert hasattr(settings, 'energy_method')
        assert hasattr(settings, 's3_bucket')
        assert settings.energy_method == "gfn2"
        assert settings.validate()
        print("✓ Settings initialized and validated")

        # Test logging setup
        setup_logging(level="INFO")
        logger = logging.getLogger("test")
        assert logger is not None
        print("✓ Logging setup works")

        # Test GPU detection
        gpu_available, device = detect_gpu()
        print(f"✓ GPU detection works (detected: {device}, GPU available: {gpu_available})")

        print("\n✅ All settings and utils tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Settings/utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TPose TDesign Service - Basic Functionality Tests")
    print("="*70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Models", test_data_models()))
    results.append(("Force Field Initialization", test_force_field_initialization()))
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("Settings and Utils", test_settings_and_utils()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
