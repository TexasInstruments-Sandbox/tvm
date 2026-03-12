"""Visualize Relax module partitioning as an interactive HTML page.

Generates a graph showing which ops are offloaded to TIDL vs executed
as TVM-generated C code on the C7x scalar pipeline.  Optionally
overlays per-layer cycle profiling data from DSP execution.

Usage::

    from tvm.relax.backend.tidl import TIDLOffloadCompiler
    from tvm.relax.backend.tidl.visualize import (
        visualize_partitioning,
        parse_layer_profile,
    )

    compiler = TIDLOffloadCompiler(config={...})
    prepared = compiler.prepare(mod, params)
    partitioned = compiler.partition(prepared)

    # Without profiling:
    visualize_partitioning(partitioned, "/tmp/graph.html")

    # With profiling (after running on DSP with -profile-layers):
    profile = parse_layer_profile(dsp_stdout)
    visualize_partitioning(partitioned, "/tmp/graph.html",
                           profile_data=profile)

Works with any partitioned Relax module (with or without TIDL offload).
"""

import json
import re
from typing import Dict, List, Optional

from tvm import relax
from tvm.ir import IRModule

from .tidl import _extract_composite_calls


def parse_layer_profile(
    stdout: str, iteration: int = -1
) -> Dict[str, int]:
    """Parse per-layer cycle counts from DSP profile output.

    Extracts layer names and cycle counts from the ``TVMPrintLayerProfile``
    output embedded in DSP stdout.  The expected format per line is::

        [  0] __tvm_ffi_tidl_subgraph_0      170000000 cycles

    When repeat=2 profiling produces multiple profile blocks, the
    ``iteration`` parameter selects which one.

    Parameters
    ----------
    stdout : str
        DSP stdout string (from ``run_dsp_dload`` or ``c7x_compute``).
    iteration : int
        Which profile block to use.  -1 (default) = last block
        (steady-state), 0 = first block (includes init).

    Returns
    -------
    dict
        Mapping of layer name to cycle count,
        e.g. ``{"__tvm_ffi_tidl_subgraph_0": 170000000}``.
    """
    # Split into blocks delimited by the profile header
    blocks = re.split(
        r"={5,}\s*TVM Layer Profile\s*={5,}", stdout
    )
    # Each block after the header contains the layer lines
    pattern = re.compile(r"\[\s*\d+\]\s+(\S+)\s+(\d+)\s+cycles")
    parsed_blocks = []
    for block in blocks:
        profile = {}
        for line in block.split("\n"):
            m = pattern.search(line)
            if m:
                profile[m.group(1)] = int(m.group(2))
        if profile:
            parsed_blocks.append(profile)

    if not parsed_blocks:
        return {}
    idx = min(iteration, len(parsed_blocks) - 1)
    return parsed_blocks[idx]


def _parse_tidl_block(block_text: str) -> List[Dict]:
    """Parse a single TIDL trace block into a list of layer dicts."""
    layers = []
    line_pat = re.compile(
        r"^\s*(\d+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    for line in block_text.split("\n"):
        m = line_pat.match(line)
        if m:
            total = int(m.group(3))
            if total == 0:
                continue
            layers.append({
                "idx": int(m.group(1)),
                "type": m.group(2),
                "total": total,
                "kernel": int(m.group(4)),
                "dma": int(m.group(5)),
            })
    return layers


def parse_tidl_layer_trace(
    stdout: str, iteration: int = -1
) -> List[Dict]:
    """Parse TIDL per-layer cycle trace from DSP profile output.

    Extracts the internal TIDL layer breakdown printed by
    ``tidl_print_layer_perf()`` in ``tidl_api.c``.  The expected
    format is::

        ===== TIDL Per-Layer Cycle Trace =====
         Idx  LayerType   TotalCycles  KernelCycles  DMAPipeup
           1  DataConv        126554       46992      18534
           2  Conv            224952      171227       4668
        ...
        ===== End TIDL Layer Trace =====

    When the output contains multiple trace blocks (from repeat=2
    profiling), the ``iteration`` parameter selects which block to
    use.  Layers with ``total_cycles == 0`` (Data placeholders)
    are skipped.

    Parameters
    ----------
    stdout : str
        DSP output string containing TIDL trace blocks.
    iteration : int
        Which trace block to use.  -1 (default) = last block
        (steady-state), 0 = first block (includes init).

    Returns
    -------
    list of dict
        Each dict has keys: ``idx``, ``type``, ``total``, ``kernel``,
        ``dma``.
    """
    blocks = re.findall(
        r"===== TIDL Per-Layer Cycle Trace =====\n(.*?)"
        r"===== End TIDL Layer Trace =====",
        stdout,
        re.DOTALL,
    )
    if not blocks:
        return []
    idx = min(iteration, len(blocks) - 1)
    return _parse_tidl_block(blocks[idx])


def _get_shape_str(sinfo) -> str:
    if hasattr(sinfo, "shape"):
        try:
            return str(tuple(int(d) for d in sinfo.shape))
        except (TypeError, ValueError):
            pass
    return ""


def _get_dtype_str(sinfo) -> str:
    if hasattr(sinfo, "dtype"):
        return str(sinfo.dtype)
    return ""


def _extract_graph(mod: IRModule, profile_data: Optional[Dict[str, int]] = None) -> dict:
    """Walk the main function and extract nodes + edges."""
    main_fn = mod["main"]
    nodes = []
    edges = []
    nid_counter = 0

    # Var identity -> node id
    var_nodes = []

    def find_var_node(var):
        for v, nid in var_nodes:
            if v.same_as(var):
                return nid
        return None

    def add_node(**kwargs):
        nonlocal nid_counter
        kwargs["id"] = nid_counter
        nid_counter += 1
        nodes.append(kwargs)
        return kwargs["id"]

    # Input params
    for p in main_fn.params:
        nid = add_node(
            label=p.name_hint,
            op="input",
            shape=_get_shape_str(p.struct_info),
            dtype=_get_dtype_str(p.struct_info),
            tidl=False,
            group="input",
            composites=[],
            source="",
            cycles=0,
        )
        var_nodes.append((p, nid))

    # Bindings
    for block in main_fn.body.blocks:
        for b in block.bindings:
            val = b.value
            bvar = b.var
            shape = _get_shape_str(bvar.struct_info)

            if isinstance(val, relax.Call):
                is_tidl = False
                op_name = ""
                composites = []
                source = ""

                # Extract source span (PyTorch module path)
                span = getattr(val, "span", None)
                if span and hasattr(span, "source_name"):
                    source = span.source_name.name

                if isinstance(val.op, relax.GlobalVar):
                    gv_name = val.op.name_hint
                    called_fn = mod.functions.get(val.op)
                    if (
                        called_fn
                        and isinstance(called_fn, relax.Function)
                        and called_fn.attrs
                    ):
                        codegen = str(called_fn.attrs.get("Codegen", ""))
                        if codegen == "tidl":
                            is_tidl = True
                            comps = _extract_composite_calls(called_fn)
                            for cf, orig_call, cv in comps:
                                cname = str(cf.attrs["Composite"])
                                cshape = _get_shape_str(cv.struct_info)
                                # Extract source from inside the composite
                                # function body (the partitioner preserves
                                # spans on inner ops but not the outer call)
                                csource = ""
                                for blk in cf.body.blocks:
                                    for inner_b in blk.bindings:
                                        s = getattr(inner_b.value, "span", None)
                                        if s and hasattr(s, "source_name"):
                                            csource = s.source_name.name
                                composites.append(
                                    {"name": cname, "shape": cshape,
                                     "source": csource}
                                )
                    op_name = gv_name
                elif hasattr(val.op, "name"):
                    op_name = str(val.op.name).replace("relax.", "")
                else:
                    op_name = type(val.op).__name__

                # Match profiling data: profile keys are __tvm_ffi_<name>
                cycles = 0
                if profile_data:
                    ffi_name = f"__tvm_ffi_{op_name}"
                    cycles = profile_data.get(ffi_name, 0)

                nid = add_node(
                    label=bvar.name_hint,
                    op=op_name,
                    shape=shape,
                    dtype=_get_dtype_str(bvar.struct_info),
                    tidl=is_tidl,
                    group="tidl" if is_tidl else "tvm",
                    composites=composites,
                    source=source,
                    cycles=cycles,
                )
                var_nodes.append((bvar, nid))

                for arg in val.args:
                    if isinstance(arg, relax.Var):
                        src = find_var_node(arg)
                        if src is not None:
                            edges.append({"from": src, "to": nid})

            elif isinstance(val, relax.TupleGetItem):
                # Pass through: map this var to the same node as the tuple
                tuple_var = val.tuple_value
                if isinstance(tuple_var, relax.Var):
                    src = find_var_node(tuple_var)
                    if src is not None:
                        var_nodes.append((bvar, src))

            elif isinstance(val, relax.Tuple):
                # Map tuple var to the first element's node for edge continuity
                for field in val.fields:
                    if isinstance(field, relax.Var):
                        src = find_var_node(field)
                        if src is not None:
                            var_nodes.append((bvar, src))
                            break

            elif isinstance(val, relax.Constant):
                nid = add_node(
                    label=bvar.name_hint,
                    op="constant",
                    shape=shape,
                    dtype=_get_dtype_str(bvar.struct_info),
                    tidl=False,
                    group="const",
                    composites=[],
                    source="",
                    cycles=0,
                )
                var_nodes.append((bvar, nid))

    # Add output node connected to the return value
    ret_expr = main_fn.body.body
    if isinstance(ret_expr, relax.Var):
        src = find_var_node(ret_expr)
        # Fallback: match by name if identity lookup fails
        if src is None:
            for v, nid in reversed(var_nodes):
                if v.name_hint == ret_expr.name_hint:
                    src = nid
                    break
        # Last resort: connect to the last node added
        if src is None and nodes:
            src = nodes[-1]["id"]
        if src is not None:
            out_shape = _get_shape_str(ret_expr.struct_info)
            out_nid = add_node(
                label="output",
                op="output",
                shape=out_shape,
                dtype=_get_dtype_str(ret_expr.struct_info),
                tidl=False,
                group="output",
                composites=[],
                source="",
                cycles=0,
            )
            edges.append({"from": src, "to": out_nid})

    # Deduplicate edges (TupleGetItem pass-through can create duplicates)
    seen = set()
    deduped = []
    for e in edges:
        key = (e["from"], e["to"])
        if key not in seen and e["from"] != e["to"]:
            seen.add(key)
            deduped.append(e)

    return {"nodes": nodes, "edges": deduped}


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #fafafa; }
  .tabs { display: flex; background: #333; }
  .tab { padding: 10px 24px; color: #aaa; cursor: pointer; font-size: 14px;
         border-bottom: 3px solid transparent; }
  .tab.active { color: #fff; border-bottom-color: #DE0000; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  /* --- Graph tab --- */
  #tab-graph { display: flex !important; width: 100vw; height: calc(100vh - 40px); }
  #tab-graph:not(.active) { display: none !important; }
  #graph { width: 65vw; height: calc(100vh - 40px); background: #fff; }
  #sidebar { width: 35vw; height: calc(100vh - 40px); overflow-y: auto; padding: 20px; }
  h2 { margin: 0 0 16px 0; color: #333; }
  .legend { margin: 12px 0 20px 0; }
  .legend-item { display: flex; align-items: center; margin: 6px 0; font-size: 14px; }
  .legend-color { width: 16px; height: 16px; margin-right: 10px; border-radius: 3px; }
  .stats { background: #f0f0f0; border-radius: 8px; padding: 14px; margin: 16px 0; }
  .stats table { width: 100%; border-collapse: collapse; }
  .stats td { padding: 4px 0; font-size: 14px; }
  .stats td:last-child { text-align: right; font-weight: 600; }
  #details { background: #fff; border: 1px solid #ddd; border-radius: 8px;
             padding: 16px; margin-top: 16px; display: none; }
  #details h3 { margin: 0 0 10px 0; font-size: 16px; }
  .detail-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }
  .detail-row .label { color: #666; }
  .comp-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
  .comp-table th { text-align: left; padding: 4px 6px; background: #f5f5f5;
                   border-bottom: 1px solid #ddd; }
  .comp-table td { padding: 4px 6px; border-bottom: 1px solid #f0f0f0; font-family: monospace; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }
  .tag-tidl { background: #FDEAEA; color: #DE0000; }
  .tag-tvm { background: #E6F2F4; color: #117788; }
  /* --- Profile tab --- */
  #tab-profile { padding: 24px; width: 100%; height: calc(100vh - 40px);
                 overflow-y: auto; flex-direction: column; }
  .profile-header { display: flex; align-items: flex-start; gap: 32px;
                    flex-wrap: wrap; }
  .profile-header-text { flex: 1; min-width: 300px; }
  .profile-table { width: 100%; max-width: 1100px; border-collapse: collapse;
                   font-size: 13px; margin-top: 16px; }
  .profile-table th { text-align: left; padding: 6px 10px; background: #f5f5f5;
                      border-bottom: 2px solid #ddd; font-size: 12px;
                      color: #666; cursor: pointer; user-select: none;
                      white-space: nowrap; position: relative; }
  .profile-table th:hover { background: #e8e8e8; }
  .profile-table th .sort-arrow { font-size: 10px; margin-left: 4px;
                                  color: #aaa; }
  .profile-table th.sorted .sort-arrow { color: #333; }
  .profile-table th .tip { display: none; position: absolute;
    bottom: 100%; left: 50%; transform: translateX(-50%);
    background: #333; color: #fff; font-size: 11px; font-weight: 400;
    padding: 6px 10px; border-radius: 4px; white-space: nowrap;
    z-index: 100; pointer-events: none; margin-bottom: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
  .profile-table th:hover .tip { display: block; }
  .profile-table td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }
  .profile-table td.name { font-family: monospace; font-size: 12px; }
  .profile-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .bar-cell { width: 30%; }
  .bar { height: 18px; border-radius: 3px; min-width: 2px; }
  .bar-tidl { background: #DE0000; }
  .bar-tvm { background: #117788; }
  .bar-kernel { background: #1565C0; }
  .bar-dma { background: #FFB300; }
  .profile-summary { font-size: 14px; margin-bottom: 8px; }
  .stacked-bar { display: flex; height: 18px; border-radius: 3px;
                 overflow: hidden; }
  .stacked-bar > div { height: 100%; min-width: 1px; }
  .donut-container { width: 220px; flex-shrink: 0; }
  .donut-legend { font-size: 11px; margin-top: 8px; }
  .donut-legend-item { display: flex; align-items: center; margin: 3px 0; }
  .donut-legend-color { width: 12px; height: 12px; margin-right: 6px;
                        border-radius: 2px; }
  .delta-pos { color: #c62828; }
  .delta-neg { color: #2e7d32; }
</style>
</head>
<body>
<div class="tabs">
  <div class="tab active" onclick="switchTab('graph')">Graph</div>
  <div class="tab" onclick="switchTab('profile')" id="profile-tab" style="display:none">Profile</div>
</div>
<div id="tab-graph" class="tab-content active">
  <div id="graph"></div>
  <div id="sidebar">
    <h2>__TITLE__</h2>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-color" style="background:#DE0000"></div>
        TIDL subgraph (MMA accelerator, int8)</div>
      <div class="legend-item">
        <div class="legend-color" style="background:#117788"></div>
        TVM generated C (C7x scalar, float32)</div>
      <div class="legend-item">
        <div class="legend-color" style="background:#BDBDBD"></div>
        Input / Constant</div>
    </div>
    <div class="stats">
      <table>
        <tr><td>TIDL subgraphs</td><td id="s-tidl">-</td></tr>
        <tr><td>TIDL layers (inside subgraphs)</td><td id="s-layers">-</td></tr>
        <tr><td>TVM ops (outside TIDL)</td><td id="s-tvm">-</td></tr>
        __EXTRA_STATS__
      </table>
    </div>
    <p style="font-size:13px;color:#666">Click a node for details.</p>
    <div id="details">
      <h3 id="d-title"></h3>
      <div class="detail-row">
        <span class="label">Op</span><span id="d-op"></span></div>
      <div class="detail-row">
        <span class="label">Output shape</span><span id="d-shape"></span></div>
      <div class="detail-row" id="d-dtype-row" style="display:none">
        <span class="label">dtype</span><span id="d-dtype"></span></div>
      <div class="detail-row">
        <span class="label">Execution</span><span id="d-exec"></span></div>
      <div class="detail-row" id="d-cycles-row" style="display:none">
        <span class="label">Cycles</span>
        <span id="d-cycles" style="font-weight:600"></span></div>
      <div class="detail-row" id="d-source-row" style="display:none">
        <span class="label">PyTorch source</span>
        <span id="d-source" style="font-family:monospace;font-size:12px"></span></div>
      <div id="comp-section" style="display:none;margin-top:12px">
        <b style="font-size:13px">TIDL layers:</b>
        <table class="comp-table">
          <thead><tr><th>#</th><th>Layer</th><th>Output shape</th><th>PyTorch source</th></tr></thead>
          <tbody id="comp-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<div id="tab-profile" class="tab-content">
  <h2>Layer Profile</h2>
  <div class="profile-summary" id="profile-summary"></div>
  <table class="profile-table" id="profile-table">
    <thead id="profile-table-thead"></thead>
    <tbody id="profile-body"></tbody>
  </table>
  <div id="tidl-trace-section" style="display:none;margin-top:24px">
    <div class="profile-header">
      <div class="profile-header-text">
        <h3 style="margin:0 0 4px 0">TIDL Internal Layer Breakdown</h3>
        <div class="profile-summary" id="tidl-trace-summary"></div>
      </div>
      <div class="donut-container" id="donut-container"></div>
    </div>
    <table class="profile-table" id="tidl-trace-table">
      <thead id="tidl-trace-table-thead"></thead>
      <tbody id="tidl-trace-body"></tbody>
    </table>
  </div>
</div>
<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[onclick*="'+name+'"]').classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
  if (name === 'graph' && typeof net !== 'undefined') net.redraw();
}
const G = __GRAPH_DATA__;
const P = __PROFILE_DATA__;
const PI = __PROFILE_DATA_INIT__;
const T = __TIDL_TRACES__;
const TI = __TIDL_TRACES_INIT__;
/* --- Graph tab --- */
const colors = {tidl:'#DE0000', tvm:'#117788', input:'#BDBDBD', 'const':'#BDBDBD', output:'#BDBDBD'};
const visNodes = G.nodes.map(n => ({
  id: n.id,
  label: (function() {
    var l = n.tidl
      ? 'TIDL Backbone\n(' + n.composites.length + ' layers)\nOutput: ' + n.shape
      : n.op + (n.shape ? '\n' + n.shape : '');
    if (n.cycles > 0) l += '\n' + (n.cycles/1e6).toFixed(1) + ' ms';
    return l;
  })(),
  color: {background: colors[n.group]||'#999', border: n.tidl ? '#AA0000' : '#0D5C6A',
          highlight: {background: n.tidl ? '#FF3333' : '#4ABED4'}},
  shape: n.tidl ? 'box' : 'ellipse',
  font: {color: (n.group==='input'||n.group==='const'||n.group==='output') ? '#333' : '#fff',
         size: n.tidl ? 18 : 12, bold: n.tidl},
  borderWidth: n.tidl ? 3 : 1,
  size: n.tidl ? 50 : 20,
  widthConstraint: n.tidl ? {minimum: 280} : undefined,
  margin: n.tidl ? {top:20, bottom:20, left:20, right:20} : undefined,
  _d: n,
}));
const visEdges = G.edges.map(e => ({from:e.from, to:e.to, arrows:'to', color:'#aaa', width:1.5}));
const net = new vis.Network(document.getElementById('graph'), {
  nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges)
}, {
  layout: {hierarchical: {direction:'UD', sortMethod:'directed',
           levelSeparation:140, nodeSpacing:180}},
  physics: false,
  interaction: {hover:true, tooltipDelay:100, zoomView:true,
                keyboard:{enabled:true, bindToWindow:false},
                navigationButtons:true},
});
const tidl = G.nodes.filter(n=>n.tidl), tvmN = G.nodes.filter(n=>n.group==='tvm');
const totalCycles = G.nodes.reduce((s,n)=>s+n.cycles, 0);
document.getElementById('s-tidl').textContent = tidl.length;
document.getElementById('s-layers').textContent = tidl.reduce((s,n)=>s+n.composites.length,0);
document.getElementById('s-tvm').textContent = tvmN.length;
net.on('click', p => {
  if (!p.nodes.length) return;
  const n = visNodes.find(v=>v.id===p.nodes[0]);
  if (!n) return;
  const d = n._d;
  document.getElementById('details').style.display = 'block';
  document.getElementById('d-title').textContent = d.label;
  document.getElementById('d-op').textContent = d.op;
  document.getElementById('d-shape').textContent = d.shape||'N/A';
  var dr = document.getElementById('d-dtype-row');
  if (d.dtype) { dr.style.display = 'flex'; document.getElementById('d-dtype').textContent = d.dtype; }
  else { dr.style.display = 'none'; }
  document.getElementById('d-exec').innerHTML = d.tidl
    ? '<span class="tag tag-tidl">TIDL MMA int8</span>'
    : '<span class="tag tag-tvm">TVM C7x float32</span>';
  const cr = document.getElementById('d-cycles-row');
  if (d.cycles > 0) {
    cr.style.display = 'flex';
    var pct = totalCycles > 0 ? ' (' + (d.cycles/totalCycles*100).toFixed(1) + '%)' : '';
    document.getElementById('d-cycles').textContent =
      d.cycles.toLocaleString() + ' (' + (d.cycles/1e6).toFixed(2) + ' ms)' + pct;
  } else { cr.style.display = 'none'; }
  const sr = document.getElementById('d-source-row');
  if (d.source) {
    sr.style.display = 'flex';
    document.getElementById('d-source').textContent = d.source;
  } else { sr.style.display = 'none'; }
  const cs = document.getElementById('comp-section');
  const cb = document.getElementById('comp-body');
  if (d.composites.length) {
    cs.style.display = 'block';
    cb.innerHTML = d.composites.map((c,i) =>
      '<tr><td>'+i+'</td><td>'+c.name+'</td><td>'+(c.shape||'')+'</td><td style="font-family:monospace;font-size:11px">'+(c.source||'')+'</td></tr>'
    ).join('');
  } else { cs.style.display = 'none'; }
});
/* --- Tooltip helper for table headers --- */
function th(attrs, label, tip) {
  var s = '<th' + (attrs ? ' ' + attrs : '') + '>' + label;
  if (tip) s += '<span class="tip">' + tip + '</span>';
  return s + '</th>';
}
/* --- Sortable table helper --- */
function makeSortable(tableId, data, renderFn) {
  var thead = document.getElementById(tableId+'-thead');
  if (!thead) return;
  var ths = thead.querySelectorAll('th');
  var curCol = -1, curAsc = true;
  ths.forEach(function(th, ci) {
    var key = th.dataset.key;
    if (!key) return;
    th.innerHTML += ' <span class="sort-arrow">\u25B2</span>';
    th.onclick = function() {
      if (curCol === ci) curAsc = !curAsc;
      else { curCol = ci; curAsc = false; }
      ths.forEach(function(h){h.classList.remove('sorted');});
      th.classList.add('sorted');
      th.querySelector('.sort-arrow').textContent = curAsc?'\u25B2':'\u25BC';
      data.sort(function(a,b) {
        var va = a[key], vb = b[key];
        if (typeof va === 'string') return curAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return curAsc ? va - vb : vb - va;
      });
      renderFn();
    };
  });
}
/* --- Donut chart (SVG) --- */
function drawDonut(containerId, segments, size) {
  size = size || 160;
  var r = size/2 - 10, cx = size/2, cy = size/2;
  var total = segments.reduce(function(s,seg){return s+seg.value;},0);
  if (total === 0) return;
  var svg = '<svg width="'+size+'" height="'+size+'">';
  var angle = -Math.PI/2;
  segments.forEach(function(seg) {
    var slice = seg.value/total * 2*Math.PI;
    var x1 = cx+r*Math.cos(angle), y1 = cy+r*Math.sin(angle);
    angle += slice;
    var x2 = cx+r*Math.cos(angle), y2 = cy+r*Math.sin(angle);
    var large = slice > Math.PI ? 1 : 0;
    svg += '<path d="M'+cx+','+cy+' L'+x1+','+y1+' A'+r+','+r+' 0 '+large+',1 '+x2+','+y2+' Z" fill="'+seg.color+'" stroke="#fff" stroke-width="2"/>';
  });
  svg += '</svg>';
  var legend = '<div class="donut-legend">';
  segments.forEach(function(seg) {
    var pct = (seg.value/total*100).toFixed(1);
    legend += '<div class="donut-legend-item"><div class="donut-legend-color" style="background:'+seg.color+'"></div>'+seg.label+' ('+pct+'%)</div>';
  });
  legend += '</div>';
  document.getElementById(containerId).innerHTML = svg + legend;
}
/* --- Profile tab: TVM layers --- */
var hasPI = PI && Object.keys(PI).length > 0;
if (P && Object.keys(P).length > 0) {
  document.getElementById('profile-tab').style.display = '';
  var pEntries = Object.entries(P).map(function(e,i){
    var initC = hasPI ? (PI[e[0]]||0) : 0;
    return {idx:i, name:e[0], cycles:e[1],
            initCycles: initC, delta: initC - e[1]};
  });
  var pTotal = pEntries.reduce(function(s,e){return s+e.cycles;},0);
  var summaryHtml = '<b>Steady-state:</b> ' +
    pTotal.toLocaleString() + ' cycles (' +
    (pTotal/1e6).toFixed(2) + ' ms) &mdash; ' +
    pEntries.length + ' layers';
  if (hasPI) {
    var piTotal = pEntries.reduce(function(s,e){return s+e.initCycles;},0);
    summaryHtml += '<br><b>Init:</b> ' +
      piTotal.toLocaleString() + ' cycles (' +
      (piTotal/1e6).toFixed(2) + ' ms)' +
      ' &mdash; <b>overhead: ' +
      ((piTotal-pTotal)/1e6).toFixed(2) + ' ms</b>';
  }
  document.getElementById('profile-summary').innerHTML = summaryHtml;
  var pHead = document.getElementById('profile-table-thead');
  var phdr = '<tr>' +
    th('data-key="idx"', '#', 'Execution order') +
    th('data-key="name"', 'Layer', 'TVM FFI function name') +
    th('data-key="cycles"', 'Cycles', 'Steady-state DSP cycles (iteration 2)') +
    th('data-key="cycles"', 'ms', 'Time in milliseconds at 1 GHz') +
    th('data-key="cycles"', '%', 'Percentage of total inference time');
  if (hasPI) phdr +=
    th('data-key="initCycles"', 'Init', 'Cycles from iteration 1 (includes one-time init cost)') +
    th('data-key="delta"', '\u0394 Init', 'Init minus steady-state: the one-time init cost per layer');
  phdr += th('class="bar-cell"', 'Distribution', 'Relative cycle cost') + '</tr>';
  pHead.innerHTML = phdr;
  function renderProfile() {
    var maxC = Math.max.apply(null,pEntries.map(function(e){return e.cycles;}));
    var body = document.getElementById('profile-body');
    body.innerHTML = '';
    pEntries.forEach(function(e,i) {
      var pct = (e.cycles/pTotal*100).toFixed(1);
      var barW = maxC>0 ? (e.cycles/maxC*100).toFixed(1) : '0';
      var isTidl = e.name.indexOf('tidl_subgraph') >= 0;
      var cls = isTidl ? 'bar-tidl' : 'bar-tvm';
      var row = '<tr><td class="num">'+e.idx+'</td>' +
        '<td class="name">'+e.name+'</td>' +
        '<td class="num">'+e.cycles.toLocaleString()+'</td>' +
        '<td class="num">'+(e.cycles/1e6).toFixed(2)+'</td>' +
        '<td class="num">'+pct+'%</td>';
      if (hasPI) {
        var dCls = e.delta>0?'delta-pos':(e.delta<0?'delta-neg':'');
        var dSign = e.delta>0?'+':'';
        row += '<td class="num">'+e.initCycles.toLocaleString()+'</td>' +
          '<td class="num '+dCls+'">'+dSign+e.delta.toLocaleString()+'</td>';
      }
      row += '<td class="bar-cell"><div class="bar '+cls+'" style="width:'+barW+'%"></div></td></tr>';
      body.innerHTML += row;
    });
  }
  renderProfile();
  makeSortable('profile-table', pEntries, renderProfile);
}
/* --- TIDL internal layer trace --- */
var hasInit = TI && TI.length > 0;
if (T && T.length > 0) {
  document.getElementById('tidl-trace-section').style.display = '';
  if (!document.getElementById('profile-tab').style.display)
    document.getElementById('profile-tab').style.display = '';
  /* Build init lookup by idx for comparison */
  var initMap = {};
  if (hasInit) TI.forEach(function(l){ initMap[l.idx] = l; });
  /* Augment data with init values */
  var tData = T.map(function(l) {
    var il = initMap[l.idx] || {};
    return {
      idx: l.idx, type: l.type,
      total: l.total, kernel: l.kernel, dma: l.dma,
      initTotal: il.total||0, initKernel: il.kernel||0,
      delta: (il.total||0) - l.total
    };
  });
  var tTotal = tData.reduce(function(s,l){return s+l.total;},0);
  var summaryHtml = '<b>Steady-state:</b> ' +
    tTotal.toLocaleString() + ' cycles (' +
    (tTotal/1e6).toFixed(2) + ' ms) &mdash; ' + tData.length + ' layers';
  if (hasInit) {
    var iTotal = tData.reduce(function(s,l){return s+l.initTotal;},0);
    summaryHtml += '<br><b>Init:</b> ' +
      iTotal.toLocaleString() + ' cycles (' + (iTotal/1e6).toFixed(2) + ' ms)';
  }
  document.getElementById('tidl-trace-summary').innerHTML = summaryHtml;
  /* Donut chart: aggregate by layer type */
  var typeColors = {
    DataConv:'#1565C0', Conv:'#2196F3', Pool:'#00897B',
    ReLU:'#43A047', EltWise:'#7B1FA2', InnerProd:'#E65100',
    SoftMax:'#F9A825', BatchNorm:'#6D4C41', Reshape:'#78909C',
    ConstData:'#BDBDBD', Data:'#E0E0E0', Other:'#9E9E9E'
  };
  var typeAgg = {};
  tData.forEach(function(l) {
    typeAgg[l.type] = (typeAgg[l.type]||0) + l.total;
  });
  var donutSegs = Object.entries(typeAgg)
    .sort(function(a,b){return b[1]-a[1];})
    .map(function(e){
      return {label:e[0], value:e[1],
              color: typeColors[e[0]]||'#9E9E9E'};
    });
  drawDonut('donut-container', donutSegs);
  /* Table header */
  var tHead = document.getElementById('tidl-trace-table-thead');
  var hdrHtml = '<tr>' +
    th('data-key="idx"', '#', 'TIDL layer index') +
    th('data-key="type"', 'Layer Type', 'TIDL layer type (Conv, Pool, EltWise, etc.)') +
    th('data-key="total"', 'Total', 'Total cycles (kernel + DMA + overhead)') +
    th('data-key="kernel"', 'Kernel', 'MMA/DSP compute cycles') +
    th('data-key="dma"', 'DMA', 'DMA pipeline startup cycles') +
    th('data-key="total"', 'ms', 'Time in milliseconds at 1 GHz');
  if (hasInit) hdrHtml +=
    th('data-key="initTotal"', 'Init', 'Total cycles from iteration 1 (cold caches)') +
    th('data-key="delta"', '\u0394 Init', 'Init minus steady-state: cache warmup cost');
  hdrHtml += th('class="bar-cell"', 'Kernel vs DMA', 'Kernel (blue) vs DMA pipeup (amber)') + '</tr>';
  tHead.innerHTML = hdrHtml;
  /* Render rows */
  function renderTidlTrace() {
    var tMax = Math.max.apply(null,tData.map(function(l){return l.total;}));
    var tbody = document.getElementById('tidl-trace-body');
    tbody.innerHTML = '';
    tData.forEach(function(l) {
      var kPct = tMax>0 ? (l.kernel/tMax*100).toFixed(1) : '0';
      var dPct = tMax>0 ? (l.dma/tMax*100).toFixed(1) : '0';
      var row = '<tr>' +
        '<td class="num">'+l.idx+'</td>' +
        '<td class="name">'+l.type+'</td>' +
        '<td class="num">'+l.total.toLocaleString()+'</td>' +
        '<td class="num">'+l.kernel.toLocaleString()+'</td>' +
        '<td class="num">'+l.dma.toLocaleString()+'</td>' +
        '<td class="num">'+(l.total/1e6).toFixed(3)+'</td>';
      if (hasInit) {
        var dCls = l.delta > 0 ? 'delta-pos' : (l.delta < 0 ? 'delta-neg' : '');
        var dSign = l.delta > 0 ? '+' : '';
        row += '<td class="num">'+l.initTotal.toLocaleString()+'</td>' +
          '<td class="num '+dCls+'">'+dSign+l.delta.toLocaleString()+'</td>';
      }
      row += '<td class="bar-cell"><div class="stacked-bar">' +
        '<div class="bar-kernel" style="width:'+kPct+'%"></div>' +
        '<div class="bar-dma" style="width:'+dPct+'%"></div>' +
        '</div></td></tr>';
      tbody.innerHTML += row;
    });
  }
  renderTidlTrace();
  makeSortable('tidl-trace-table', tData, renderTidlTrace);
}
</script>
</body>
</html>"""


def visualize_partitioning(
    mod: IRModule,
    output_path: str,
    title: str = "TIDL Offloading Visualization",
    extra_stats: Optional[Dict[str, str]] = None,
    profile_data: Optional[Dict[str, int]] = None,
    profile_data_init: Optional[Dict[str, int]] = None,
    tidl_traces: Optional[List[Dict]] = None,
    tidl_traces_init: Optional[List[Dict]] = None,
) -> str:
    """Generate an interactive HTML visualization of a partitioned module.

    Parameters
    ----------
    mod : IRModule
        Partitioned Relax module (after ``TIDLOffloadCompiler.partition``).
    output_path : str
        Path to write the HTML file.
    title : str
        Page title.
    extra_stats : dict, optional
        Additional key-value pairs to show in the stats sidebar,
        e.g. ``{"TIDL inference": "170 ms", "Speedup": "46.8x"}``.
    profile_data : dict, optional
        Per-layer cycle counts from ``parse_layer_profile()``.
        Keys are ``__tvm_ffi_<name>`` strings, values are cycle counts.
        Typically the steady-state (last) iteration.
    profile_data_init : dict, optional
        Per-layer cycle counts from init (first) iteration, for
        comparison.  Same format as ``profile_data``.
    tidl_traces : list of dict, optional
        TIDL per-layer cycle trace from ``parse_tidl_layer_trace()``.
        Each dict has keys: idx, type, total, kernel, dma.
        Typically the steady-state (last) iteration.
    tidl_traces_init : list of dict, optional
        TIDL traces from the init (first) iteration, for comparison.
        Same format as ``tidl_traces``.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    graph_data = _extract_graph(mod, profile_data)

    extra_html = ""
    if extra_stats:
        extra_html += (
            '<tr><td colspan="2" style="padding-top:8px;'
            'border-top:1px solid #ccc"></td></tr>'
        )
        for k, v in extra_stats.items():
            bold = "<b>" if "speedup" in k.lower() else ""
            bold_end = "</b>" if bold else ""
            extra_html += (
                f"<tr><td>{bold}{k}{bold_end}</td>"
                f"<td>{bold}{v}{bold_end}</td></tr>"
            )

    html = _HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__PROFILE_DATA__", json.dumps(profile_data or {}))
    html = html.replace(
        "__PROFILE_DATA_INIT__", json.dumps(profile_data_init or {})
    )
    html = html.replace("__TIDL_TRACES__", json.dumps(tidl_traces or []))
    html = html.replace(
        "__TIDL_TRACES_INIT__", json.dumps(tidl_traces_init or [])
    )
    html = html.replace("__EXTRA_STATS__", extra_html)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
