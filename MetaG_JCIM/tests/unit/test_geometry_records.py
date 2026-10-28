"""Ordered geometry identities survive serialization and reject invalid records."""
import json
import numpy as np
import pytest
from metag.energetics.provenance import ordered_geometry_record, content_hash


def test_ordered_geometry_is_a_detached_json_record_with_verifiable_identity():
    positions=np.array([[0.,0.,0.],[0.,0.,1.]])
    record=ordered_geometry_record(['O','H'],positions,-1,1)
    restored=json.loads(json.dumps(record,allow_nan=False))
    digest=restored.pop('sha256')
    assert content_hash(restored)==digest
    np.testing.assert_array_equal(record['positions_angstrom'],positions)
    positions[1,2]=2.
    assert record['positions_angstrom'][1][2]==1.
    assert record['charge']==-1 and record['spin_multiplicity']==1


def test_coordinate_charge_spin_and_order_changes_do_not_reuse_geometry_identity():
    base=ordered_geometry_record(['O','H'],[[0,0,0],[0,0,1]],0,1)
    variants=[ordered_geometry_record(['O','H'],[[0,0,0],[0,0,1.000001]],0,1),
              ordered_geometry_record(['O','H'],[[0,0,0],[0,0,1]],-1,1),
              ordered_geometry_record(['O','H'],[[0,0,0],[0,0,1]],0,2),
              ordered_geometry_record(['H','O'],[[0,0,1],[0,0,0]],0,1)]
    assert all(row['sha256']!=base['sha256'] for row in variants)


@pytest.mark.parametrize('positions', [[[0,0,float('nan')]],[[0,0,float('inf')]],[[0,0]],[]])
def test_nonfinite_or_malformed_geometry_is_not_exported_as_accepted(positions):
    with pytest.raises(ValueError):ordered_geometry_record(['O'],positions,0)


@pytest.mark.parametrize('charge,spin', [(True,1),(.5,1),(float('nan'),1),(0,False),(0,0),(0,1.5)])
def test_invalid_charge_and_spin_are_not_exported(charge,spin):
    with pytest.raises(ValueError):ordered_geometry_record(['O'],[[0,0,0]],charge,spin)
