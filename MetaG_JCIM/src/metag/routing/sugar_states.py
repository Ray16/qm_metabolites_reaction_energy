"""Explicit reducing-sugar state construction; no implicit stereoisomer assignment.

Candidate chemistry for validation before promotion. One reducing centre, both
configurations of each accessible 5/6-membered O-hemiacetal, and the open carbonyl.
Carbonyl hydration belongs to the existing aqueous-state energy treatment. Fixed
nonanomeric centres and glycosidic bonds are retained. This finite subspace omits
septanoses, dimers and chemically different epimers; no universal completeness or
population accuracy is inferred from successful construction.
"""
from dataclasses import dataclass
import hashlib,json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from metag.validation.stereochemistry import inspect_stereochemistry

POLICY='reducing-sugar-explicit-ring-chain-v1'
_HEMI=Chem.MolFromSmarts('[C;X4;R]([O;X2;R])[O;X2;H1]')
_CARBONYL=Chem.MolFromSmarts('[C;X3;!R]=[O;X1]')

def canon(m):
    molecule = Chem.Mol(m)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)
def _formula(m):return rdMolDescriptors.CalcMolFormula(m)
def _centres(m):
 return [h for h in m.GetSubstructMatches(_HEMI)
         if any(len(r) in (5,6) and h[0] in r and h[1] in r for r in m.GetRingInfo().AtomRings())]
def _carbonyls(m):
 out=[]
 for c,o in m.GetSubstructMatches(_CARBONYL):
  atom=m.GetAtomWithIdx(c)
  if atom.GetFormalCharge()!=0:continue
  if all(n.GetAtomicNum() in (1,6) or n.GetIdx()==o for n in atom.GetNeighbors()):out.append((c,o))
 return out

def _open(m,hit):
 c,ring_o,hydroxyl_o=hit;rw=Chem.RWMol(m)
 rw.RemoveBond(c,ring_o);rw.GetBondBetweenAtoms(c,hydroxyl_o).SetBondType(Chem.BondType.DOUBLE)
 rw.GetAtomWithIdx(c).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
 oxygen=rw.GetAtomWithIdx(hydroxyl_o);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(True)
 oxygen=rw.GetAtomWithIdx(ring_o);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(False)
 result=rw.GetMol();Chem.SanitizeMol(result);Chem.AssignStereochemistry(result,cleanIt=True,force=True)
 return result

@dataclass(frozen=True)
class SugarStates:
 applicable:bool
 complete:bool
 reference:str
 open_smiles:str|None=None
 states:tuple=()
 reason:str|None=None
 state_set_sha256:str|None=None

def enumerate_sugar_states(smiles,charge=None,*,max_states=16):
 if isinstance(max_states,bool) or not isinstance(max_states,int) or max_states<1:raise ValueError('positive integer max_states required')
 m=Chem.MolFromSmiles(smiles)
 if m is None:raise ValueError('Invalid sugar SMILES')
 for atom in m.GetAtoms():atom.SetAtomMapNum(0)
 ref=canon(m);q=Chem.GetFormalCharge(m)
 if charge is not None and (isinstance(charge,bool) or charge!=q):raise ValueError('Sugar charge mismatch')
 if len(Chem.GetMolFrags(m))!=1:raise ValueError('Single sugar component required')
 if any(a.GetIsotope() for a in m.GetAtoms()):return SugarStates(True,False,ref,reason='Isotope-specific pools unsupported')
 hits=_centres(m);centres={h[0] for h in hits}
 if len(centres)>1:return SugarStates(True,False,ref,reason='Multiple reducing centres require a joint model')
 inspection=inspect_stereochemistry(smiles)
 if inspection['unsupported'] or inspection['relative_stereo_groups']:return SugarStates(True,False,ref,reason='Unsupported or relative stereochemistry')
 if any(r['kind']!='Atom_Tetrahedral' or r['index'] not in centres for r in inspection['unresolved']):
  return SugarStates(bool(hits or _carbonyls(m)),False,ref,reason='Unresolved nonanomeric stereochemistry')
 opened=_open(m,hits[0]) if hits else Chem.Mol(m)
 carbonyls=_carbonyls(opened)
 if not carbonyls:return SugarStates(False,True,ref)
 if len(carbonyls)!=1:return SugarStates(True,False,ref,reason='Multiple carbonyls require a joint model')
 c,carbonyl_o=carbonyls[0]
 closures=[]
 for oxygen in opened.GetAtoms():
  if oxygen.GetAtomicNum()!=8 or oxygen.GetFormalCharge()!=0 or oxygen.GetDegree()!=1 or oxygen.GetTotalNumHs()!=1:continue
  neighbor=oxygen.GetNeighbors()[0]
  if neighbor.GetAtomicNum()!=6 or neighbor.GetHybridization()!=Chem.HybridizationType.SP3:continue
  path=Chem.GetShortestPath(opened,c,oxygen.GetIdx())
  if len(path) not in (5,6):continue
  if not all(opened.GetAtomWithIdx(i).GetAtomicNum()==6 and opened.GetAtomWithIdx(i).GetHybridization()==Chem.HybridizationType.SP3 for i in path[1:-1]):continue
  closures.append((oxygen.GetIdx(),len(path)))
 if not closures:return SugarStates(False,True,ref)
 open_smi=canon(opened);states={open_smi:{'smiles':open_smi,'kind':'open','ring_size':None}}
 errors=[]
 for ring_o,size in closures:
  for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
   rw=Chem.RWMol(opened);rw.GetBondBetweenAtoms(c,carbonyl_o).SetBondType(Chem.BondType.SINGLE)
   rw.AddBond(c,ring_o,Chem.BondType.SINGLE)
   oxygen=rw.GetAtomWithIdx(ring_o);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(True)
   oxygen=rw.GetAtomWithIdx(carbonyl_o);oxygen.SetNumExplicitHs(1);oxygen.SetNoImplicit(True)
   rw.GetAtomWithIdx(c).SetChiralTag(tag)
   try:
    product=rw.GetMol();Chem.SanitizeMol(product);Chem.AssignStereochemistry(product,cleanIt=True,force=True)
    assert _formula(product)==_formula(m) and Chem.GetFormalCharge(product)==q
    s=canon(product)
    if inspect_stereochemistry(s)['issues']:raise ValueError('Constructed state contains unresolved stereochemistry')
    # Opening each state must recover the identical fully specified carbon skeleton.
    product_hits=_centres(product)
    if len(product_hits)!=1 or canon(_open(product,product_hits[0]))!=open_smi:raise ValueError('State changed the fixed stereochemical skeleton')
    states[s]={'smiles':s,'kind':'ring','ring_size':size}
   except (ValueError,RuntimeError,AssertionError) as exc:errors.append(str(exc))
 rows=tuple(states[k] for k in sorted(states))
 if errors or len(rows)>max_states:return SugarStates(True,False,ref,open_smi,rows,reason=f'{len(rows)}states, {len(errors)}construction failures; limit{max_states}')
 identity=hashlib.sha256(json.dumps({'policy':POLICY,'charge':q,'states':rows},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return SugarStates(True,True,ref,open_smi,rows,state_set_sha256=identity)


def prepare_declared_sugar_inputs(reaction):
    """Choose a reference drawing only for explicitly declared solution pools.

    The reference is a computational gauge. Every state remains required in the
    later physical partition. This function never uses an energy or observation
    to pick an anomer and never mutates the caller's source structures.
    """
    declarations = reaction.get('species_state_pools', {})
    if not isinstance(declarations, dict):
        raise ValueError('species_state_pools must map species names to pool declarations')
    if not declarations:
        return reaction, {}
    if not set(declarations) <= set(reaction['species']):
        raise ValueError('Pool declaration names an absent input species')
    species = {name:list(row) for name,row in reaction['species'].items()}
    plans = {}
    for name, declaration in declarations.items():
        if (not isinstance(declaration, dict) or set(declaration) != {'kind','source'}
                or declaration['kind'] != 'reducing_sugar_equilibrium'
                or not isinstance(declaration['source'], str) or not declaration['source'].strip()):
            raise ValueError('Sugar pools require a reducing_sugar_equilibrium kind and source definition')
        coefficient, charge, smiles = species[name]
        result = enumerate_sugar_states(smiles, charge)
        if not result.applicable or not result.complete:
            raise ValueError(name + ': unsupported declared sugar pool: ' + str(result.reason))
        state_smiles = [row['smiles'] for row in result.states]
        if result.reference in state_smiles:
            reference = result.reference
        else:
            # All states already preserve the declared nonanomeric centres.
            # Match the input connectivity/ring size, then choose a stable gauge.
            topology_mol = Chem.MolFromSmiles(canon(Chem.MolFromSmiles(smiles)))
            topology = Chem.MolToSmiles(topology_mol, isomericSmiles=False)
            matching = [s for s in state_smiles if Chem.MolToSmiles(
                Chem.MolFromSmiles(s), isomericSmiles=False) == topology]
            if not matching:
                raise ValueError(name + ': input drawing absent from the declared state family')
            reference = min(matching)
        species[name] = [coefficient, charge, reference]
        plans[name] = {'policy':POLICY, 'kind':'reducing_sugar_equilibrium',
                       'source':declaration['source'], 'source_input_smiles':smiles,
                       'reference_smiles':reference, 'source_charge':charge,
                       'source_state_set_sha256':result.state_set_sha256,
                       'source_states':list(result.states),
                       'scope':'Complete declared five/six-membered ring/open subspace; '
                               'carbonyl hydration requires the same complete energy treatment.'}
    return dict(reaction, species=species), plans


def pool_aware_stereochemistry(original_species, routed_species, declarations):
    """Retain source ambiguities while distinguishing those explicitly modelled."""
    from metag.validation.stereochemistry import unresolved_species
    source = unresolved_species(original_species)
    covered = {}
    for name, declaration in declarations.items():
        if name not in original_species or original_species[name][2] != declaration['source_input_smiles']:
            raise ValueError('Declared sugar pool does not belong to the source input: ' + name)
        if name in source:
            covered[name] = source.pop(name)
    return {'input':source, 'routed':unresolved_species(routed_species),
            **({'covered_by_declared_sugar_pools':covered} if covered else {})}
