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
from typing import Dict, Optional

from tvm import relax
from tvm.ir import IRModule

from .tidl import _extract_composite_calls


def parse_layer_profile(stdout: str) -> Dict[str, int]:
    """Parse per-layer cycle counts from DSP profile output.

    Extracts layer names and cycle counts from the ``TVMPrintLayerProfile``
    output embedded in DSP stdout.  The expected format per line is::

        [  0] __tvm_ffi_tidl_subgraph_0      170000000 cycles

    Parameters
    ----------
    stdout : str
        DSP stdout string (from ``run_dsp_dload`` or ``c7x_compute``).

    Returns
    -------
    dict
        Mapping of layer name to cycle count,
        e.g. ``{"__tvm_ffi_tidl_subgraph_0": 170000000}``.
    """
    profile = {}
    pattern = re.compile(r"\[\s*\d+\]\s+(\S+)\s+(\d+)\s+cycles")
    for line in stdout.split("\n"):
        m = pattern.search(line)
        if m:
            name = m.group(1)
            cycles = int(m.group(2))
            profile[name] = cycles
    return profile


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
  .profile-table { width: 100%; max-width: 900px; border-collapse: collapse;
                   font-size: 13px; margin-top: 16px; }
  .profile-table th { text-align: left; padding: 6px 10px; background: #f5f5f5;
                      border-bottom: 2px solid #ddd; font-size: 12px; color: #666; }
  .profile-table td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }
  .profile-table td.name { font-family: monospace; font-size: 12px; }
  .bar-cell { width: 40%; }
  .bar { height: 18px; border-radius: 3px; min-width: 2px; }
  .bar-tidl { background: #DE0000; }
  .bar-tvm { background: #117788; }
  .profile-summary { font-size: 14px; margin-bottom: 8px; }
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
    <thead><tr><th>#</th><th>Layer</th><th>Cycles</th><th>Time (ms)</th><th>%</th><th class="bar-cell">Distribution</th></tr></thead>
    <tbody id="profile-body"></tbody>
  </table>
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
/* --- Profile tab --- */
if (P && Object.keys(P).length > 0) {
  document.getElementById('profile-tab').style.display = '';
  var entries = Object.entries(P).sort((a,b) => b[1] - a[1]);
  var pTotal = entries.reduce((s,e) => s + e[1], 0);
  var maxC = entries[0][1];
  document.getElementById('profile-summary').innerHTML =
    '<b>Total:</b> ' + pTotal.toLocaleString() + ' cycles (' +
    (pTotal/1e6).toFixed(2) + ' ms @ 1 GHz) &mdash; ' +
    entries.length + ' layers';
  var body = document.getElementById('profile-body');
  entries.forEach(function(e, i) {
    var name = e[0], cyc = e[1];
    var pct = (cyc/pTotal*100).toFixed(1);
    var barW = (cyc/maxC*100).toFixed(1);
    var isTidl = name.indexOf('tidl_subgraph') >= 0;
    var barCls = isTidl ? 'bar-tidl' : 'bar-tvm';
    body.innerHTML += '<tr>' +
      '<td>' + i + '</td>' +
      '<td class="name">' + name + '</td>' +
      '<td style="text-align:right">' + cyc.toLocaleString() + '</td>' +
      '<td style="text-align:right">' + (cyc/1e6).toFixed(2) + '</td>' +
      '<td style="text-align:right">' + pct + '%</td>' +
      '<td class="bar-cell"><div class="bar ' + barCls + '" style="width:' + barW + '%"></div></td>' +
      '</tr>';
  });
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
    html = html.replace("__EXTRA_STATS__", extra_html)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
