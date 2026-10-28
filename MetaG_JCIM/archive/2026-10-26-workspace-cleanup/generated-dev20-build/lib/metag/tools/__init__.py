"""Maintained tool commands; implementations load only when invoked."""

COMMANDS = {
    'preflight': ('preflight', 'Check input structures, references and routing without QM'),
    'audit-routes': ('audit_routes', 'Check shared representations across a reaction panel'),
    'audit-reference': ('audit_reference', 'Audit experimental-reference conditions'),
    'audit-network': ('audit_network', 'Audit reaction-network consistency'),
    'audit-sampling': ('audit_sampling', 'Audit recorded conformer-search diagnostics'),
    'audit-transfer': ('audit_transfer', 'Audit benchmark and target-domain coverage'),
    'evaluate': ('evaluate', 'Evaluate a declared prediction/reference comparison'),
    'build-reference': ('build_reference', 'Build a source-defined reference ledger'),
    'export-minima': ('export_minima', 'Export verified recorded minima'),
    'refine-ensemble': ('refine_ensemble', 'Reweight a complete recorded solvent ensemble'),
    'compare-states': ('compare_states', 'Compare declared states or a balanced component exchange'),
    'calibrate': ('calibrate', 'Fit historical class-width diagnostics'),
    'fit-uncertainty': ('fit_uq', 'Fit and assess feature-based uncertainty diagnostics'),
}


def command_help():
    return 'Maintained commands:\n' + '\n'.join(
        '  metag ' + name + '  — ' + description for name, (_, description) in COMMANDS.items())


def run_command(name, argv):
    from importlib import import_module
    module, _ = COMMANDS[name]
    return import_module('metag.tools.' + module).main(argv)
