#!/usr/bin/env python3
"""
Integration test for xTB force field ranking using real test data.
Tests actual XTBForceField and RankingRunner code with real structures.
"""

import sys
import os
import tempfile
import shutil

# Add paths
sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services')
sys.path.insert(0, '/home/gavin.bascom/tpose-tdesign-services/services/pose-ranking')

from shared.models.pose import Pose
from shared.models.batch import PoseBatch
from src.ranking_runner import RankingRunner


def test_xtb_single_pose():
    """Test xTB force field ranking on a single real pose."""
    print("\n" + "="*70)
    print("Testing xTB Force Field Ranking - Single Pose")
    print("="*70)

    # Use real test data
    protein_pdb = "/home/gavin.bascom/tpose-tdesign-services/test_data/fixed_prot.pdb"
    ligand_sdf = "/home/gavin.bascom/tpose-tdesign-services/test_data/poses_0.sdf"

    # Check files exist
    assert os.path.exists(protein_pdb), f"Protein PDB not found: {protein_pdb}"
    assert os.path.exists(ligand_sdf), f"Ligand SDF not found: {ligand_sdf}"
    print(f"✓ Test data files found")
    print(f"  Protein: {protein_pdb}")
    print(f"  Ligand: {ligand_sdf}")

    # Create pose
    pose = Pose(
        pose_id="test_pose_001",
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        energy_method="gfn2"
    )

    assert pose.validate(), "Pose validation failed"
    print(f"✓ Pose created and validated")

    # Create runner
    runner = RankingRunner()
    print(f"✓ RankingRunner initialized")

    # Set up parameters for xTB
    params = {
        'energy_method': 'gfn2',
        'device': 'cpu',
        'distance_cutoff': 5.0,
    }

    # Create temporary output location (not using S3 for this test)
    temp_output = tempfile.mkdtemp(prefix="tpose_test_")

    try:
        print(f"\n⏳ Running xTB ranking (this may take 2-5 minutes)...")

        # Run ranking
        result = runner.rank_single_pose(
            pose=pose,
            bucket="test-bucket",  # Not used since files are local
            output_folder=temp_output,
            params=params
        )

        # Check results
        print(f"\n{'='*70}")
        print("Ranking Results")
        print(f"{'='*70}")

        if result.ranking_success:
            print(f"✅ Ranking succeeded!")
            print(f"\nEnergies (kcal/mol):")
            print(f"  Interaction energy: {result.interaction_energy:.2f}")
            print(f"  Strain energy:      {result.strain_energy:.2f}")
            print(f"  Total score:        {result.total_score:.2f}")

            print(f"\nComponent energies (kcal/mol):")
            print(f"  Complex energy:     {result.complex_energy:.2f}")
            print(f"  Protein energy:     {result.protein_energy:.2f}")
            print(f"  Ligand bound:       {result.ligand_bound_energy:.2f}")
            print(f"  Ligand free:        {result.ligand_free_energy:.2f}")

            print(f"\nMetadata:")
            print(f"  Method:             {result.energy_method}")
            print(f"  Device:             {result.force_field_device}")
            print(f"  Time:               {result.computation_time_seconds:.1f}s")

            # Validate energy values are reasonable
            assert result.interaction_energy is not None, "Interaction energy is None"
            assert result.strain_energy is not None, "Strain energy is None"
            assert result.total_score is not None, "Total score is None"
            assert result.computation_time_seconds > 0, "Computation time should be positive"

            print(f"\n✅ All energy values computed successfully")
            return True

        else:
            print(f"❌ Ranking failed!")
            print(f"Error: {result.error_message}")
            return False

    finally:
        # Cleanup
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
            print(f"\n✓ Cleaned up temporary files")


def test_xtb_batch_ranking():
    """Test xTB force field ranking on a batch of poses."""
    print("\n" + "="*70)
    print("Testing xTB Force Field Ranking - Batch")
    print("="*70)

    # Create batch with two poses
    poses_data = [
        {
            "pose_id": "test_pose_001",
            "protein_pdb": "/home/gavin.bascom/tpose-tdesign-services/test_data/fixed_prot.pdb",
            "ligand_sdf": "/home/gavin.bascom/tpose-tdesign-services/test_data/poses_0.sdf",
        },
        {
            "pose_id": "test_pose_002",
            "protein_pdb": "/home/gavin.bascom/tpose-tdesign-services/test_data/fixed_prot.pdb",
            "ligand_sdf": "/home/gavin.bascom/tpose-tdesign-services/test_data/poses_1.sdf",
        },
    ]

    batch = PoseBatch.from_dict_list(poses_data, global_energy_method="gfn2")
    assert batch.validate(), "Batch validation failed"
    print(f"✓ Batch created with {len(batch.poses)} poses")

    # Create runner
    runner = RankingRunner()

    # Set up parameters
    params = {
        'energy_method': 'gfn2',
        'device': 'cpu',
        'distance_cutoff': 5.0,
    }

    temp_output = tempfile.mkdtemp(prefix="tpose_batch_test_")

    try:
        print(f"\n⏳ Running xTB batch ranking (this may take 4-10 minutes)...")

        # Run batch ranking
        results = runner.rank_batch(
            batch=batch,
            bucket="test-bucket",
            output_folder=temp_output,
            params=params
        )

        print(f"\n{'='*70}")
        print("Batch Ranking Results")
        print(f"{'='*70}")

        successful = 0
        failed = 0

        for result in results:
            print(f"\nPose: {result.pose_id}")
            if result.ranking_success:
                successful += 1
                print(f"  ✅ Success")
                print(f"  Interaction: {result.interaction_energy:.2f} kcal/mol")
                print(f"  Strain:      {result.strain_energy:.2f} kcal/mol")
                print(f"  Total:       {result.total_score:.2f} kcal/mol")
                print(f"  Time:        {result.computation_time_seconds:.1f}s")
            else:
                failed += 1
                print(f"  ❌ Failed: {result.error_message}")

        print(f"\n{'='*70}")
        print(f"Summary: {successful} successful, {failed} failed")
        print(f"{'='*70}")

        # Should have at least some successes
        assert successful > 0, "No poses ranked successfully"

        print(f"\n✅ Batch ranking completed with {successful} successes")
        return True

    finally:
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
            print(f"\n✓ Cleaned up temporary files")


def main():
    """Run xTB integration tests."""
    print("\n" + "="*70)
    print("xTB Force Field Integration Tests")
    print("="*70)
    print("Using actual XTBForceField and RankingRunner code")
    print("Testing with real protein-ligand structures")
    print("="*70)

    results = []

    # Test 1: Single pose ranking
    try:
        results.append(("Single Pose Ranking", test_xtb_single_pose()))
    except Exception as e:
        print(f"\n❌ Single pose test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Pose Ranking", False))

    # Test 2: Batch ranking
    try:
        results.append(("Batch Ranking", test_xtb_batch_ranking()))
    except Exception as e:
        print(f"\n❌ Batch test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Batch Ranking", False))

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
