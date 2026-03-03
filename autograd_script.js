/* Backward-mode AD demo
   - Build a small computational graph (nodes with parents and local derivative rules).
   - Forward pass: compute values.
   - Reverse pass: accumulate grads by pushing each node's grad to its parents.
   - Visualize in SVG.
*/

(() => {
  // ---------- Helpers ----------
  const $ = (sel) => document.querySelector(sel);
  const fmt = (x) => (Number.isFinite(x) ? x.toFixed(6) : String(x));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const SUB = { "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉","+":"₊","-":"₋","(":"₍",")":"₎" };
  const SUP = { "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","+":"⁺","-":"⁻","(":"⁽",")":"⁾" };

  function latexishToUnicode(s) {
    if (!s) return s;
    let out = String(s);
    // Convert explicit braced forms first, then simple digit-only suffixes.
    out = out.replace(/_\{([^}]+)\}/g, (_m, grp) => grp.split("").map(ch => SUB[ch] ?? ch).join(""));
    out = out.replace(/_([0-9]+)/g, (_m, grp) => grp.split("").map(ch => SUB[ch] ?? ch).join(""));
    out = out.replace(/\^\{([^}]+)\}/g, (_m, grp) => grp.split("").map(ch => SUP[ch] ?? ch).join(""));
    out = out.replace(/\^([0-9]+)/g, (_m, grp) => grp.split("").map(ch => SUP[ch] ?? ch).join(""));
    return out;
  }

  // Finite difference (central)
  function finiteDiffScalar(f, x, z, h = 1e-5) {
    const fxph = f(x + h, z);
    const fxmh = f(x - h, z);
    const fzph = f(x, z + h);
    const fzmh = f(x, z - h);
    return {
      dx: (fxph - fxmh) / (2 * h),
      dz: (fzph - fzmh) / (2 * h),
    };
  };

  // ---------- AD Node ----------
  class Node {
    constructor(op, parents = [], extra = {}) {
      this.id = Node._nextId++;
      this.op = op;               // e.g. '+', '*', 'sin', 'exp', ...
      this.parents = parents;     // array of Node
      this.value = NaN;
      this.grad = 0;
      this.name = extra.name ?? null;  // for 'var' nodes (i.e. x,z)
      this.constValue = extra.constValue ?? null; // for constants
      this.ui = { x: 0, y: 0, label: "", color: "rgba(255,255,255,0.08)" }; // layout + visuals
      this._localGrads = null;    // function returning local partial derivatives to parents
    }

    // Set the local gradient function, which returns an array of ∂this/∂parent_i values.
    setLocalGradFn(fn) { this._localGrads = fn; return this; }

    // Call the local gradient function with current value and parent values.
    localGrads() {
      if (!this._localGrads) throw new Error(`No localGradFn for op=${this.op}`);
      return this._localGrads(this.value, this.parents.map(p => p.value));
    }
  }
  Node._nextId = 1;

  // ---------- Defining node types ----------
  const Var = (name) => new Node("var", [], { name }); // leaf node with name (e.g. "x" or "z")
  const Const = (c) => new Node("const", [], { constValue: c }); // leaf node with constant value

  function Add(a, b) {
    return new Node("+", [a, b]).setLocalGradFn(() => [1, 1]);
  }
  function Sub(a, b) {
    return new Node("-", [a, b]).setLocalGradFn(() => [1, -1]);
  }
  function Mul(a, b) {
    return new Node("*", [a, b]).setLocalGradFn((_y, [av, bv]) => [bv, av]);
  }
  function Div(a, b) {
    return new Node("/", [a, b]).setLocalGradFn((_y, [av, bv]) => [1 / bv, -av / (bv * bv)]);
  }
  function Sin(a) {
    return new Node("sin", [a]).setLocalGradFn((_y, [av]) => [Math.cos(av)]);
  }
  function Cos(a) {
    return new Node("cos", [a]).setLocalGradFn((_y, [av]) => [-Math.sin(av)]);
  }
  function Exp(a) {
    return new Node("exp", [a]).setLocalGradFn((y) => [y]);
  }
  function Log(a) {
    return new Node("log", [a]).setLocalGradFn((_y, [av]) => [1 / av]);
  }
  function Square(a) {
    return new Node("sq", [a]).setLocalGradFn((_y, [av]) => [2 * av]);
  }
  function ReLU(a) {
    return new Node("relu", [a]).setLocalGradFn((_y, [av]) => [av > 0 ? 1 : 0]);
  }

  // ---------- Graph model ----------
  class GraphModel {
    constructor(svgEl) {
      this.svg = svgEl; // "SVG element"
      this.nodes = [];
      this.edges = []; // {from, to}
      this.inputs = { x: 0, z: 0 };
      this.output = null;
      this.order = [];      // topological order (forward)
      this.revOrder = [];   // reverse topological order (excluding leaves)
      this.revIndex = 0;    // stepping pointer

      // visual state
      this._svgDefsBuilt = false;
      this._nodeEls = new Map();
      this._edgeEls = [];  
      this._nextAlias = 1;  // for naming intermediate nodes t_1, t_2, ...
    }

    build(fnIndex) {
      Node._nextId = 1;
      this.nodes = [];
      this.edges = [];
      this._nodeEls.clear();
      this._edgeEls = [];
      this.order = [];
      this.revOrder = [];
      this.revIndex = 0;

      // Inputs
      const x = Var("x");
      const z = Var("z");

      let y = null;

      // Function set
      if (fnIndex === 0) {
        // y = x^2 + z^2
        const x2 = Square(x);
        const z2 = Square(z);
        y = Add(x2, z2);

        // layout
        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 440],
          [x2, 340, 160], [z2, 340, 400],
          [y, 620, 280],
        ], -60, 1.2);

      } else if (fnIndex === 1) {
        // y = 0.2 ReLU(xz) + sin(x+z)
        const c02 = Const(0.2);
        const xz = Mul(x, z);
        const relu = ReLU(xz);
        const reluScaled = Mul(c02, relu);
        const xpz = Add(x, z);
        const s = Sin(xpz);
        y = Add(reluScaled, s);

        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 440],
          [xz, 280, 340], [relu, 500, 340], [c02, 500, 430], [reluScaled, 700, 340],
          [xpz, 280, 160], [s, 700, 160],
          [y, 900, 250],
        ], -60, 1.4);

      } else if (fnIndex === 2) {
        // y = (x − z)^2 + cos(x+z)
        const xmz = Sub(x, z);
        const sq = Square(xmz);
        const xpz = Add(x, z);
        const c = Cos(xpz);
        y = Add(sq, c);

        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 440],
          [xmz, 310, 320],
          [sq, 520, 320],
          [xpz, 310, 200],
          [c, 520, 200],
          [y, 760, 260],
        ], -60, 1.4);

      } else if (fnIndex === 3) {
        // y = sin(xz) + 0.15(x^2 + z^2)
        const c015 = Const(0.15);
        const xz = Mul(x, z);
        const sxz = Sin(xz);
        const x2 = Square(x);
        const z2 = Square(z);
        const rad = Add(x2, z2);
        const reg = Mul(c015, rad);
        y = Add(sxz, reg);

        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 440],
          [xz, 300, 280], [sxz, 520, 280],
          [x2, 300, 120], [z2, 300, 440],
          [rad, 520, 120], [c015, 520, 440],
          [reg, 700, 300], [y, 860, 250],
        ], -60, 1.4);

      } else if (fnIndex === 4) {
        // y = 0.20(x^2+z^2) + sin(x+z)
        const c02 = Const(0.20);
        const x2 = Square(x);
        const z2 = Square(z);
        const bowlSum = Add(x2, z2);
        const bowl = Mul(c02, bowlSum);
        const xpz = Add(x, z);
        const ripple = Sin(xpz);
        y = Add(bowl, ripple);

        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 420],
          [x2, 320, 120], [z2, 320, 420], [bowlSum, 520, 270], [c02, 520, 130], [bowl, 730, 250],
          [xpz, 520, 430], [ripple, 730, 430],
          [y, 930, 300],
        ], -40, 1.4);

      } else {
        // Fallback to y = x^2 + z^2
        const x2 = Square(x);
        const z2 = Square(z);
        y = Add(x2, z2);
        this._layoutFixedSpread([
          [x, 110, 120], [z, 110, 440],
          [x2, 340, 160], [z2, 340, 400],
          [y, 620, 280],
        ], -60, 1.4);
      }

      // Collect nodes reachable from y
      // Not strictly necessary in our examples, but good practice for larger graphs where some nodes may be disconnected
      const seen = new Set();
      const stack = [y];
      while (stack.length) {
        const n = stack.pop();
        if (seen.has(n.id)) continue;
        seen.add(n.id);
        this.nodes.push(n);
        for (const p of n.parents) stack.push(p);
      }

      // Build edges parent -> child
      const byId = new Map(this.nodes.map(n => [n.id, n]));
      for (const n of this.nodes) {
        for (const p of n.parents) {
          if (byId.has(p.id)) this.edges.push({ from: p, to: n });
        }
      }

      // Store output
      this.output = y;

      // Compute topological order (Kahn)
      this._computeTopologicalOrder();

      // Reverse order for backprop stepping:
      this.revOrder = [...this.order].reverse();
      this.revIndex = 0;
      this._assignAliasesAndExpressions();

      // Build SVG
      this._renderGraph();
      this.resetGradsVisualOnly();
    }

    _layoutFixed(pairs) {
      // pairs: [node, x, y]
      for (const [n, x, y] of pairs) {
        n.ui.x = x; n.ui.y = y;
      }
    }

    _layoutFixedSpread(pairs, x0 = 110, xSpread = 1.25) {
        // pairs are [node, x, y] using original coordinates.
        // We scale x relative to x0 so the whole diagram expands horizontally.
        for (const [n, x, y] of pairs) {
            n.ui.x = x0 + (x - x0) * xSpread;
            n.ui.y = y;
        };
    }

    // Kahn's algorithm for topological sort.
    // See: https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
    _computeTopologicalOrder() {
      const indeg = new Map();
      const children = new Map();
      for (const n of this.nodes) { indeg.set(n.id, 0); children.set(n.id, []); }
      for (const e of this.edges) {
        indeg.set(e.to.id, indeg.get(e.to.id) + 1);
        children.get(e.from.id).push(e.to);
      }

      const q = [];
      for (const n of this.nodes) if (indeg.get(n.id) === 0) q.push(n);

      const order = [];
      while (q.length) {
        q.sort((a, b) => (a.ui.x - b.ui.x) || (a.ui.y - b.ui.y)); // layout-based tie-break for visual consistency
        const n = q.shift();
        order.push(n);
        for (const ch of children.get(n.id)) {
          indeg.set(ch.id, indeg.get(ch.id) - 1);
          if (indeg.get(ch.id) === 0) q.push(ch);
        }
      }
      this.order = order;
    }

    // Assign aliases (e.g. t_1, t_2, ...) and expressions (e.g. t_3 = t_1 + t_2) for display
    _assignAliasesAndExpressions() {
      this._nextAlias = 1;
      let nextConstAlias = 1;
      for (const n of this.order) {
        if (n === this.output) {
          n.ui.alias = "y";
        } else if (n.op === "var") {
          n.ui.alias = n.name;
        } else if (n.op === "const") {
          n.ui.alias = `c_${nextConstAlias++}`;
        } else {
          n.ui.alias = `t_${this._nextAlias++}`;
        }

        const p = n.parents;
        const pa = (idx) => p[idx]?.ui?.alias ?? "?";
        switch (n.op) {
          case "var":
            n.ui.expr = "";
            break;
          case "const":
            n.ui.expr = String(n.constValue);
            break;
          case "+":
            n.ui.expr = `${pa(0)} + ${pa(1)}`;
            break;
          case "-":
            n.ui.expr = `${pa(0)} - ${pa(1)}`;
            break;
          case "*":
            n.ui.expr = `${pa(0)} × ${pa(1)}`;
            break;
          case "/":
            n.ui.expr = `${pa(0)} / ${pa(1)}`;
            break;
          case "sin":
            n.ui.expr = `sin(${pa(0)})`;
            break;
          case "cos":
            n.ui.expr = `cos(${pa(0)})`;
            break;
          case "exp":
            n.ui.expr = `exp(${pa(0)})`;
            break;
          case "log":
            n.ui.expr = `log(${pa(0)})`;
            break;
          case "sq":
            n.ui.expr = `(${pa(0)})^2`;
            break;
          case "relu":
            n.ui.expr = `ReLU(${pa(0)})`;
            break;
          default:
            n.ui.expr = n.op;
        }

        n.ui.display = n.ui.expr ? `${n.ui.alias} = ${n.ui.expr}` : `${n.ui.alias}`;
      }
    }

    // ---------- Forward pass and reverse init ----------
    forward(xVal, zVal) {
      this.inputs.x = xVal;
      this.inputs.z = zVal;

      // reset values/grads
      for (const n of this.nodes) { n.value = NaN; n.grad = 0; }

      // Compute forward pass in topological order
      for (const n of this.order) {
        if (n.op === "var") {
          n.value = (n.name === "x") ? xVal : zVal;
        } else if (n.op === "const") {
          n.value = n.constValue;
        } else {
          const pv = n.parents.map(p => p.value);
          n.value = this._evalOp(n.op, pv);
        }
      }

      // seed gradient at output
      this.output.grad = 1;

      // reset reverse stepping
      this.revIndex = 0;

      // visuals
      this._updateAllNodeLabels();
      this._setForwardComputedStyle();
      this._setSeedStyle(this.output);

      return this.output.value;
    }

    // Backprop step: push current node's grad to its parents
    backpropStep() {
      if (!this.output) return { done: true }; // no graph built

      // Done when we've processed all nodes in reverse order
      if (this.revIndex >= this.revOrder.length) return { done: true };

      // Process the next node in reverse order
      const n = this.revOrder[this.revIndex++];

      // Capture details of this step for display
      const detail = {
        node: n,
        nodeGrad: n.grad,
        nodeValue: n.value,
        updates: []
      };

      // Push n.grad to parents using local derivatives
      if (n.parents.length > 0) {
        const locals = n.localGrads(); // array of partials dn/dparents
        for (let i = 0; i < n.parents.length; i++) {
          const parent = n.parents[i];    // parent node
          const local = locals[i];        // dn/dparent_i
          const contrib = n.grad * local; // contribution to parent.grad
          const before = parent.grad;
          parent.grad += contrib;   
          detail.updates.push({
            parent,
            local,
            contrib,
            before,
            after: parent.grad
          });
        }
      }

      // Update visuals for just-processed node and its parents
      this._markReverseFlow(n, n.parents);
      this._updateAllNodeLabels();

      return { done: this.revIndex >= this.revOrder.length, detail };
    }

    backpropAll() {
      let guard = 0;
      let last = null;
      while (this.revIndex < this.revOrder.length && guard < 10000) {
        last = this.backpropStep();
        guard++;
      }
      return { done: true, last, steps: guard };
    }

    resetGrads() {
      for (const n of this.nodes) n.grad = 0;
      if (this.output) this.output.grad = 1; // keep seed if forward was run
      this.revIndex = 0;
      this.resetGradsVisualOnly();
      this._updateAllNodeLabels();
      if (this.output) this._setSeedStyle(this.output);
    }

    resetGradsVisualOnly() {
      // reset styling
      for (const n of this.nodes) {
        n.ui.color = "rgba(255,255,255,0.08)";
      }
      this._applyNodeStyles();
      this._applyEdgeStyles(null);
    }

    // ---------- Op evaluation ----------
    _evalOp(op, pv) {
      switch (op) {
        case "+": return pv[0] + pv[1];
        case "-": return pv[0] - pv[1];
        case "*": return pv[0] * pv[1];
        case "/": return pv[0] / pv[1];
        case "sin": return Math.sin(pv[0]);
        case "cos": return Math.cos(pv[0]);
        case "exp": return Math.exp(pv[0]);
        case "log": return Math.log(pv[0]);
        case "sq": return pv[0] * pv[0];
        case "relu": return Math.max(0, pv[0]);
        default: throw new Error(`Unknown op: ${op}`);
      }
    }

    // ---------- Visualization (SVG) ----------
    _renderGraph() {
      const svg = this.svg;
      while (svg.firstChild) svg.removeChild(svg.firstChild); // clear old graph

      // Fit the viewBox to actual node extents so nodes never clip at container edges.
      if (this.nodes.length > 0) {
        const NODE_HALF_WIDTH = 103;
        const NODE_HALF_HEIGHT = 36;
        const PAD_X = 24;
        const PAD_Y = 20;

        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const n of this.nodes) {
          if (n.ui.x < minX) minX = n.ui.x;
          if (n.ui.x > maxX) maxX = n.ui.x;
          if (n.ui.y < minY) minY = n.ui.y;
          if (n.ui.y > maxY) maxY = n.ui.y;
        }

        const x0 = minX - NODE_HALF_WIDTH - PAD_X;
        const y0 = minY - NODE_HALF_HEIGHT - PAD_Y;
        const w = (maxX - minX) + 2 * (NODE_HALF_WIDTH + PAD_X);
        const h = (maxY - minY) + 2 * (NODE_HALF_HEIGHT + PAD_Y);
        svg.setAttribute("viewBox", `${x0} ${y0} ${w} ${h}`);
      }

      // defs: arrow markers, subtle glow
      if (!this._svgDefsBuilt) this._svgDefsBuilt = true;
      const defs = el("defs");
      defs.appendChild(marker("arrow", "rgba(230,240,255,0.55)"));
      defs.appendChild(marker("arrowStrong", "rgba(94,234,212,0.9)"));
      svg.appendChild(defs);

      // Draw edges first
      this._edgeEls = [];
      for (const e of this.edges) {
        const p1 = { x: e.from.ui.x, y: e.from.ui.y };
        const p2 = { x: e.to.ui.x, y: e.to.ui.y };
        const path = el("path");
        path.setAttribute("d", curvedPath(p1, p2));
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "rgba(230,240,255,0.35)");
        path.setAttribute("stroke-width", "2");
        path.setAttribute("marker-end", "url(#arrow)");
        path.dataset.from = String(e.from.id);
        path.dataset.to = String(e.to.id);
        svg.appendChild(path);
        this._edgeEls.push(path);
      }

      // Draw nodes
      for (const n of this.nodes) {
        const g = el("g");
        g.setAttribute("transform", `translate(${n.ui.x},${n.ui.y})`);
        g.style.cursor = "default";

        const rect = el("rect");
        rect.setAttribute("x", "-103");
        rect.setAttribute("y", "-36");
        rect.setAttribute("width", "206");
        rect.setAttribute("height", "72");
        rect.setAttribute("rx", "14");
        rect.setAttribute("fill", "rgba(255,255,255,0.05)");
        rect.setAttribute("stroke", "rgba(255,255,255,0.12)");
        rect.setAttribute("stroke-width", "1.5");

        const title = el("text");
        title.setAttribute("x", "0");
        title.setAttribute("y", "-13");
        title.setAttribute("text-anchor", "middle");
        title.setAttribute("font-size", "14");
        title.setAttribute("letter-spacing", "0.4");
        title.setAttribute("fill", "rgb(255, 255, 255)");
        title.textContent = this._nodeTitle(n);

        const subtitle = el("text");
        subtitle.setAttribute("x", "0");
        subtitle.setAttribute("y", "5");
        subtitle.setAttribute("text-anchor", "middle");
        subtitle.setAttribute("font-size", "12");
        subtitle.setAttribute("fill", "rgb(169, 182, 214)");
        subtitle.textContent = ""; // filled by update labels

        const gradline = el("text");
        gradline.setAttribute("x", "0");
        gradline.setAttribute("y", "24");
        gradline.setAttribute("text-anchor", "middle");
        gradline.setAttribute("font-size", "12");
        gradline.setAttribute("fill", "rgb(169, 182, 214)");
        gradline.textContent = "";

        g.appendChild(rect);
        g.appendChild(title);
        g.appendChild(subtitle);
        g.appendChild(gradline);
        svg.appendChild(g);

        this._nodeEls.set(n.id, { g, rect, title, subtitle, gradline });
      }

      this._updateAllNodeLabels();
      this._applyNodeStyles();
    }

    // Determine node title using Unicode characters
    _nodeTitle(n) {
      return latexishToUnicode(n.ui.display ?? n.ui.alias ?? n.op);
    }

    // Update the value and grad labels for all nodes (called after forward or backprop steps)
    _updateAllNodeLabels() {
      for (const n of this.nodes) {
        const ui = this._nodeEls.get(n.id);
        if (!ui) continue;

        const valStr = Number.isFinite(n.value) ? `v = ${fmt(n.value)}` : `v = 0.000000`;
        const gradStr = `g = ${fmt(n.grad)}`;

        ui.subtitle.textContent = valStr;
        ui.gradline.textContent = gradStr;
      }
    }

    // Apply current node.ui.color to all node rectangles
    _applyNodeStyles() {
      for (const n of this.nodes) {
        const ui = this._nodeEls.get(n.id);
        if (!ui) continue;
        ui.rect.setAttribute("fill", n.ui.color);
      }
    }

    // Apply styles to edges based on activeSet
    _applyEdgeStyles(activeSet) {
      for (const path of this._edgeEls) {
        const key = `${path.dataset.from}->${path.dataset.to}`;
        const active = activeSet && activeSet.has(key);

        if (active) {
          path.setAttribute("stroke", "rgba(94,234,212,0.85)");
          path.setAttribute("stroke-width", "4");
          path.setAttribute("marker-end", "url(#arrowStrong)");
        } else {
          path.setAttribute("stroke", "rgba(230,240,255,0.35)");
          path.setAttribute("stroke-width", "2");
          path.setAttribute("marker-end", "url(#arrow)");
        }
      }
    }

    // Tint all nodes slightly to indicate forward values are computed
    _setForwardComputedStyle() {
      // tint all nodes slightly to indicate values computed
      for (const n of this.nodes) n.ui.color = "rgba(122, 162, 255, 0.37)";
      this._applyNodeStyles();
    }

    // Highlight seed node to indicate reverse pass starting point
    _setSeedStyle(node) {
      node.ui.color = "rgba(251,191,36,0.18)";
      this._applyNodeStyles();
    }

    // Highlight the currently processed node and edges to parents
    _markReverseFlow(node, parents) {
      // highlight processed node and edges to parents
      node.ui.color = "rgba(8, 244, 161, 0.29)";
      for (const p of parents) {
        // brighten parents too, but not as much as current node
        if (p === this.output) continue;
        p.ui.color = "rgba(8, 244, 161, 0.13)";
      }

      // find active edges for styling
      const act = new Set();
      for (const p of parents) act.add(`${p.id}->${node.id}`);

      this._applyNodeStyles();
      this._applyEdgeStyles(act);
    }
  }

  // ---------- SVG utilities ----------
  function el(tag) { return document.createElementNS("http://www.w3.org/2000/svg", tag); }

  // Define an arrow marker for edges
  function marker(id, stroke) {
    const m = el("marker");
    m.setAttribute("id", id);
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "8.6");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "7");
    m.setAttribute("markerHeight", "7");
    m.setAttribute("orient", "auto-start-reverse");

    const p = el("path");
    p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill", stroke);
    m.appendChild(p);
    return m;
  }

  // Create a curved path between two points
  function curvedPath(a, b) {
    const NODE_HALF_WIDTH = 85;
    const NODE_HALF_HEIGHT = 34;

    // Vector from a to b
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;

    // Unit vector
    const ux = dx / len;
    const uy = dy / len;

    // Offset start and end so arrows touch rectangle boundary
    const startX = a.x + ux * NODE_HALF_WIDTH;
    const startY = a.y + uy * NODE_HALF_HEIGHT;

    const endX = b.x - ux * NODE_HALF_WIDTH;
    const endY = b.y - uy * NODE_HALF_HEIGHT;

    // Control point for curve
    const mx = (startX + endX) / 2;
    const my = (startY + endY) / 2;

    const bend = clamp(len * 0.15, 12, 55);
    const nx = -uy;
    const ny = ux;

    const cx = mx + nx * bend;
    const cy = my + ny * bend;

    return `M ${startX} ${startY} Q ${cx} ${cy} ${endX} ${endY}`;
  }

  // ---------- App wiring ----------
  const svg = $("#graph");
  const model = new GraphModel(svg);

  const fnSelect = $("#fnSelect");
  const xRange = $("#xRange");
  const zRange = $("#zRange");
  const xValEl = $("#xVal");
  const zValEl = $("#zVal");

  const outYEl = $("#outY");
  const gradXEl = $("#gradX");
  const gradZEl = $("#gradZ");
  const progressEl = $("#progress");
  const chainRuleInfoEl = $("#chainRuleInfo");

  const btnForward = $("#btnForward");
  const btnStep = $("#btnStep");
  const btnRunAll = $("#btnRunAll");
  const btnReset = $("#btnReset");
  const btnFiniteDiff = $("#btnFiniteDiff");
  const btnDescend = $("#btnDescend");
  const btnClearTrace = $("#btnClearTrace");
  const lrRange = $("#lrRange");
  const lrValEl = $("#lrVal");
  const plot3dEl = $("#plot3d");
  let descentTrace = [];
  const functionSurfaceCache = new Map();

  // Initialize
  function currentInputs() {
    return { x: parseFloat(xRange.value), z: parseFloat(zRange.value) };
  }

  // Clamp step size
  function getStepSize() {
    const raw = Number(lrRange?.value);
    if (!Number.isFinite(raw)) return 0.1;
    return clamp(raw, 0.001, 1);
  }

  function updateStepSizeLabel() {
    if (!lrValEl) return;
    lrValEl.textContent = getStepSize().toFixed(3);
  }

  function updateSliderLabels() {
    xValEl.textContent = fmt(parseFloat(xRange.value));
    zValEl.textContent = fmt(parseFloat(zRange.value));
  }

  // Update output, gradients, and progress display after forward or backprop steps
  function setOutputs() {
    if (!model.output) {
      outYEl.textContent = "—";
      gradXEl.textContent = "—";
      gradZEl.textContent = "—";
      progressEl.textContent = "—";
      return;
    }

    outYEl.textContent = fmt(model.output.value);

    const gx = model.nodes.find(n => n.op === "var" && n.name === "x")?.grad ?? NaN;
    const gz = model.nodes.find(n => n.op === "var" && n.name === "z")?.grad ?? NaN;
    gradXEl.textContent = fmt(gx);
    gradZEl.textContent = fmt(gz);

    const done = model.revOrder.length > 0 && model.revIndex >= model.revOrder.length;
    progressEl.textContent = `${model.revIndex}/${model.revOrder.length} nodes${done ? " ✓" : ""}`;
  }

  // Get a display label for a node, using alias if available
  function nodeLabel(n) {
    if (!n) return "node";
    if (n.ui?.alias) return n.ui.alias;
    if (n === model.output) return "y";
    if (n.op === "var") return n.name;
    if (n.op === "const") return String(n.constValue);
    return n.op;
  }

  // Update the chain rule info display with HTML content, and trigger MathJax typesetting if available
  function setChainRuleInfo(html) {
    if (!chainRuleInfoEl) return;
    chainRuleInfoEl.innerHTML = html;
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([chainRuleInfoEl]).catch(() => {});
    }
  }

  // Get a LaTeX-formatted label for a node (e.g. t_12 => t_{12})
  function nodeLatex(n) {
    const label = nodeLabel(n);
    const m = label.match(/^([A-Za-z]+)_(\d+)$/);
    if (m) return `${m[1]}_{${m[2]}}`;
    return label;
  }

  // Get a LaTeX expression for the local derivative of a node with respect to one of its parents
  function localDerivativeLatex(node, parentIndex) {
    const pa = (idx) => nodeLatex(node.parents[idx]);
    switch (node.op) {
      case "+":
        return "1";
      case "-":
        return parentIndex === 0 ? "1" : "-1";
      case "*":
        return parentIndex === 0 ? pa(1) : pa(0);
      case "/":
        return parentIndex === 0 ? `\\frac{1}{${pa(1)}}` : `-\\frac{${pa(0)}}{(${pa(1)})^2}`;
      case "sin":
        return `\\cos(${pa(0)})`;
      case "cos":
        return `-\\sin(${pa(0)})`;
      case "exp":
        return `\\exp(${pa(0)})`;
      case "log":
        return `\\frac{1}{${pa(0)}}`;
      case "sq":
        return `2\\times ${pa(0)}`;
      case "relu":
        return `\\mathbf{1}_{${pa(0)}>0}`;
      default:
        return "?";
    }
  }

  // Render detailed information about a backprop step, including the local derivatives and gradient updates for each parent
  function renderBackpropDetail(stepRes) {
    if (!stepRes?.detail) return;
    const d = stepRes.detail;
    const nLatex = nodeLatex(d.node);
    const lines = [];
    lines.push(`\\(\\mathbf{Processing\\ node}\\;${nLatex}\\):`);
    lines.push(`\\(\\qquad v_{${nLatex}}=${fmt(d.nodeValue)},\\qquad g_{${nLatex}}=${fmt(d.nodeGrad)}.\\)`);
    lines.push(``);

    if (d.updates.length === 0) {
      lines.push(`\\(\\text{Leaf node: no parent gradients updated.}\\)`);
    } else {
      for (let i = 0; i < d.updates.length; i++) {
        const u = d.updates[i];
        const pLatex = nodeLatex(u.parent);
        const localExpr = localDerivativeLatex(d.node, i);
        lines.push(`\\(\\mathbf{Gradient\\ update\\ for\\ }${pLatex}\\):`);
        lines.push(`\\(\\qquad g_{${pLatex}}\\leftarrow g_{${pLatex}} + g_{${nLatex}}\\,\\frac{\\partial ${nLatex}}{\\partial ${pLatex}},\\)`);
        lines.push(`\\(\\qquad \\frac{\\partial ${nLatex}}{\\partial ${pLatex}} = ${localExpr} = ${fmt(u.local)},\\)`);
        lines.push(`\\(\\qquad g_{${pLatex}} = \\ ${fmt(u.before)} + (${fmt(d.nodeGrad)}\\times ${fmt(u.local)}) = ${fmt(u.after)}.\\)`);
        lines.push(``);
      }
    }

    setChainRuleInfo(lines.join("<br>"));
  }

  // "Descend" action only possible when forward and backprop complete
  function canDescendNow() {
    return Number.isFinite(model.output?.value) &&
      model.revOrder.length > 0 &&
      model.revIndex >= model.revOrder.length;
  }

  // Update button highlights based on current stage: forward, backprop, or descend
  function updateActionHighlights() {
    const stage = btnStep.disabled
      ? "forward"
      : (canDescendNow() ? "descend" : "backprop");

    btnForward.classList.toggle("stage-active", stage === "forward");
    btnStep.classList.toggle("stage-active", stage === "backprop");
    btnRunAll.classList.toggle("stage-active", stage === "backprop");
    if (btnDescend) btnDescend.classList.toggle("stage-active", stage === "descend");
  }

  function updateDescendState() {
    if (btnDescend) btnDescend.disabled = !canDescendNow();
    updateActionHighlights();
  }

  function currentInputGrads() {
    const gx = model.nodes.find(n => n.op === "var" && n.name === "x")?.grad ?? NaN;
    const gz = model.nodes.find(n => n.op === "var" && n.name === "z")?.grad ?? NaN;
    return { gx, gz };
  }

  // Clamp a value between min and max
  function syncInputs(x, z) {
    const xMin = parseFloat(xRange.min);
    const xMax = parseFloat(xRange.max);
    const zMin = parseFloat(zRange.min);
    const zMax = parseFloat(zRange.max);
    xRange.value = String(clamp(x, xMin, xMax));
    zRange.value = String(clamp(z, zMin, zMax));
    updateSliderLabels();
  }

  // Compute the next point for gradient descent, ensuring it stays within the specified bounds
  function boundedDescentTarget(x0, z0, gx, gz, step, xMin, xMax, zMin, zMax) {
    const dx = -gx;
    const dz = -gz;
    if (!Number.isFinite(dx) || !Number.isFinite(dz)) {
      return { xNext: x0, zNext: z0, stepUsed: 0 };
    }
    if (Math.abs(dx) < 1e-12 && Math.abs(dz) < 1e-12) {
      return { xNext: x0, zNext: z0, stepUsed: 0 };
    }

    let stepMax = Infinity;

    if (dx > 0) stepMax = Math.min(stepMax, (xMax - x0) / dx);
    else if (dx < 0) stepMax = Math.min(stepMax, (xMin - x0) / dx);

    if (dz > 0) stepMax = Math.min(stepMax, (zMax - z0) / dz);
    else if (dz < 0) stepMax = Math.min(stepMax, (zMin - z0) / dz);

    stepMax = Math.max(0, stepMax);
    const stepUsed = Math.min(step, stepMax);

    return {
      xNext: x0 + stepUsed * dx,
      zNext: z0 + stepUsed * dz,
      stepUsed
    };
  }

  // Build a surface representation of the selected function for plotting
  function buildFunctionSurface(fnIndex, xMin = -3, xMax = 3, zMin = -3, zMax = 3, steps = 100) {
    const cacheKey = `${fnIndex}:${xMin}:${xMax}:${zMin}:${zMax}:${steps}`;
    if (functionSurfaceCache.has(cacheKey)) return functionSurfaceCache.get(cacheKey);

    const f = fScalarFactory(fnIndex);
    const xs = [];
    const zs = [];
    for (let i = 0; i < steps; i++) {
      xs.push(xMin + ((xMax - xMin) * i) / (steps - 1));
      zs.push(zMin + ((zMax - zMin) * i) / (steps - 1));
    }

    const ys = [];
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let zi = 0; zi < zs.length; zi++) {
      const row = [];
      for (let xi = 0; xi < xs.length; xi++) {
        const y = f(xs[xi], zs[zi]);
        row.push(y);
        if (Number.isFinite(y)) {
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }
      }
      ys.push(row);
    }
    const surface = { xs, zs, ys, yMin, yMax, xMin, xMax, zMin, zMax };
    functionSurfaceCache.set(cacheKey, surface);
    return surface;
  }

  // Render the 3D plot of the function surface, current point, descent trace, and gradient direction
  function render3DPlot() {
    if (!plot3dEl || typeof Plotly === "undefined") return;

    const fnIndex = parseInt(fnSelect.value, 10);
    const surface = buildFunctionSurface(fnIndex);
    const traces = [
      {
        type: "surface",
        x: surface.xs,
        y: surface.zs,
        z: surface.ys,
        colorscale: "Viridis",
        opacity: 0.75,
        hovertemplate: "x=%{x:.2f}<br>z=%{y:.2f}<br>y=%{z:.2f}<extra>f(x,z)</extra>",
        showscale: false,
        name: "f(x,z)"
      }
    ];

    // Add forward evaluation point
    const hasForward = !!model.output && Number.isFinite(model.output.value);
    if (hasForward) {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        x: [model.inputs.x],
        y: [model.inputs.z],
        z: [model.output.value],
        marker: { size: 6, color: "#ffffff", line: { color: "#ffffff", width: 1 } },
        name: "Forward eval",
        hovertemplate: "x=%{x:.4f}<br>z=%{y:.4f}<br>y=%{z:.4f}<extra>Forward pass</extra>"
      });
    }

    // Add descent trace
    if (descentTrace.length > 0) {
      traces.push({
        type: "scatter3d",
        mode: "lines+markers",
        x: descentTrace.map(p => p.x),
        y: descentTrace.map(p => p.z),
        z: descentTrace.map(p => p.y),
        marker: { size: 4, color: "#ffffff" },
        line: { width: 4, color: "#ffffff" },
        name: "Gradient descent path",
        hovertemplate: "x=%{x:.4f}<br>z=%{y:.4f}<br>y=%{z:.4f}<extra>Descent trace</extra>"
      });
    }

    // Add gradient descent direction from current point
    const backpropDone = model.revOrder.length > 0 && model.revIndex >= model.revOrder.length;
    if (hasForward && backpropDone) {
      const { gx, gz } = currentInputGrads();
      if (Number.isFinite(gx) && Number.isFinite(gz)) {
        const step = getStepSize();
        const f = fScalarFactory(fnIndex);
        const xCurr = model.inputs.x;
        const zCurr = model.inputs.z;
        const bounded = boundedDescentTarget(
          xCurr, zCurr, gx, gz, step,
          surface.xMin, surface.xMax, surface.zMin, surface.zMax
        );
        const xNext = bounded.xNext;
        const zNext = bounded.zNext;
        const yNext = f(xNext, zNext);
        const ySpan = surface.yMax - surface.yMin || 1;
        const baseY = surface.yMin - 0.06 * ySpan;
        const gradNorm = Math.hypot(gx, gz);

        traces.push({
          type: "scatter3d",
          mode: "lines+markers",
          x: [xCurr, xNext],
          y: [zCurr, zNext],
          z: [baseY, baseY],
          marker: { size: 4, color: "#f59e0b" },
          line: { width: 7, color: "#f59e0b" },
          name: "-∇y in (x,z)",
          showlegend: false,
          hovertemplate: "x=%{x:.4f}<br>z=%{y:.4f}<br><extra>Descent direction</extra>"
        });
        
        traces.push({
          type: "scatter3d",
          mode: "markers",
          x: [xNext],
          y: [zNext],
          z: [yNext],
          marker: { size: 6, color: "#f59e0b", line: { color: "#ffffff", width: 1 } },
          name: "Slider preview",
          showlegend: true,
          hovertemplate:
            `x=%{x:.4f}<br>z=%{y:.4f}<br>y=%{z:.4f}` +
            `<br>step=${bounded.stepUsed.toFixed(3)}<br>||∇y||=${gradNorm.toFixed(4)}<extra>Preview</extra>`
        });
      }
    }

    // Layout with fixed axes ranges and aspect ratio
    const layout = {
      margin: { l: 8, r: 8, t: 24, b: 24 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      showlegend: false,
      uirevision: "autograd-3d",
      scene: {
        bgcolor: "rgba(0,0,0,0)",
        uirevision: "autograd-3d-scene",
        xaxis: {
          title: "x",
          range: [surface.xMin, surface.xMax],
          autorange: false,
          gridcolor: "rgba(255,255,255,0.12)",
          zerolinecolor: "rgba(255,255,255,0.2)"
        },
        yaxis: {
          title: "z",
          range: [surface.zMin, surface.zMax],
          autorange: false,
          gridcolor: "rgba(255,255,255,0.12)",
          zerolinecolor: "rgba(255,255,255,0.2)"
        },
        zaxis: {
          title: "y",
          range: [
            surface.yMin - 0.08 * (surface.yMax - surface.yMin || 1),
            surface.yMax + 0.08 * (surface.yMax - surface.yMin || 1)
          ],
          autorange: false,
          gridcolor: "rgba(255,255,255,0.12)",
          zerolinecolor: "rgba(255,255,255,0.2)"
        },
        aspectmode: "manual",
        aspectratio: { x: 1.5, y: 1.35, z: 0.9 },
        camera: { eye: { x: 1.4, y: 1.25, z: 0.85 } }
      }
    };

    const config = {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso3d", "select2d", "autoScale2d"]
    };

    Plotly.react(plot3dEl, traces, layout, config);
  }

  function descendOneStep() {
    if (!canDescendNow()) {
      setChainRuleInfo(`Complete <b>all backprop steps</b> first, then descend.`);
      updateDescendState();
      return;
    }

    const lr = getStepSize();
    updateStepSizeLabel();

    const { x, z } = currentInputs();
    const { gx, gz } = currentInputGrads();
    if (!Number.isFinite(gx) || !Number.isFinite(gz)) return;

    if (descentTrace.length === 0) {
      descentTrace.push({ x, z, y: model.output.value });
    }

    const bounded = boundedDescentTarget(
      x, z, gx, gz, lr,
      parseFloat(xRange.min), parseFloat(xRange.max),
      parseFloat(zRange.min), parseFloat(zRange.max)
    );
    const xNext = bounded.xNext;
    const zNext = bounded.zNext;
    syncInputs(xNext, zNext);

    model.forward(parseFloat(xRange.value), parseFloat(zRange.value));
    btnStep.disabled = false;
    btnRunAll.disabled = false;
    setChainRuleInfo(
      `Descent step applied. New point loaded. ` +
      `Run <b>Backprop step</b> or <b>Backprop all</b> before descending again.`
    );
    setOutputs();
    updateDescendState();
    descentTrace.push({ x: model.inputs.x, z: model.inputs.z, y: model.output.value });
    render3DPlot();
  }

  function clearDescentTrace() {
    descentTrace = [];
    render3DPlot();
  }

  function rebuildGraph() {
    model.build(parseInt(fnSelect.value, 10));
    descentTrace = [];
    setChainRuleInfo(`Run <b>Forward</b>, then <b>Backprop step</b>.`);
    // disable step/run until forward is run
    btnStep.disabled = true;
    btnRunAll.disabled = true;
    setOutputs();
    updateDescendState();
    render3DPlot();
  }

  function runForward() {
    const { x, z } = currentInputs();
    model.forward(x, z);
    btnStep.disabled = false;
    btnRunAll.disabled = false;
    setChainRuleInfo(
      `Forward complete. Seed set at output: <code>g(y)=1</code>. ` +
      `Click <b>Backprop step</b> to apply chain rule updates.`
    );
    setOutputs();
    updateDescendState();
    render3DPlot();
  }

  function stepBackprop() {
    const res = model.backpropStep();
    renderBackpropDetail(res);
    setOutputs();
    updateDescendState();
    render3DPlot();
  }

  function runAllBackprop() {
    const res = model.backpropAll();
    if (res?.last) renderBackpropDetail(res.last);
    setOutputs();
    updateDescendState();
    render3DPlot();
  }

  function resetGrads() {
    model.resetGrads();
    setChainRuleInfo(`Gradients reset. Current seed: <code>g(y)=1</code>.`);
    setOutputs();
    updateDescendState();
    render3DPlot();
  }

  function fScalarFactory(fnIndex) {
    if (fnIndex === 0) {
      return (x, z) => x * x + z * z;
    } else if (fnIndex === 1) {
      return (x, z) => 0.2 * Math.max(0, x * z) + Math.sin(x + z);
    } else if (fnIndex === 2) {
      return (x, z) => {
        const a = x - z;
        return a * a + Math.cos(x + z);
      };
    } else if (fnIndex === 3) {
      return (x, z) => Math.sin(x * z) + 0.15 * (x * x + z * z);
    } else if (fnIndex === 4) {
      return (x, z) => 0.20 * (x * x + z * z) + Math.sin(x + z);
    }
    return (x, z) => x * x + z * z;
  }

  function finiteDiffCheck() {
    // Ensure we have a forward run and full backprop so gradients are available
    runForward();
    runAllBackprop();

    const { x, z } = currentInputs();
    const fnIndex = parseInt(fnSelect.value, 10);
    const f = fScalarFactory(fnIndex);
    const fd = finiteDiffScalar(f, x, z, 1e-5);

    const gx = model.nodes.find(n => n.op === "var" && n.name === "x")?.grad ?? NaN;
    const gz = model.nodes.find(n => n.op === "var" && n.name === "z")?.grad ?? NaN;

    const msg =
      `Finite-difference check (central, h=1e-5)\n\n` +
      `∂y/∂x: reverse-mode autodiff = ${fmt(gx)}\n` +
      `∂y/∂x: finite-difference approximation = ${fmt(fd.dx)}\n` +
      `abs error          = ${fmt(Math.abs(gx - fd.dx))}\n\n` +
      `∂y/∂z: reverse-mode autodiff = ${fmt(gz)}\n` +
      `∂y/∂z: finite-difference approximation = ${fmt(fd.dz)}\n` +
      `abs error          = ${fmt(Math.abs(gz - fd.dz))}`;
    alert(msg);
  }

  // Events
  fnSelect.addEventListener("change", () => { rebuildGraph(); });
  xRange.addEventListener("input", () => {
    updateSliderLabels();
    descentTrace = [];
    if (Number.isFinite(model.output?.value)) runForward();
    else render3DPlot();
  });
  zRange.addEventListener("input", () => {
    updateSliderLabels();
    descentTrace = [];
    if (Number.isFinite(model.output?.value)) runForward();
    else render3DPlot();
  });
  lrRange?.addEventListener("input", () => {
    updateStepSizeLabel();
    render3DPlot();
  });

  btnForward.addEventListener("click", runForward);
  btnStep.addEventListener("click", stepBackprop);
  btnRunAll.addEventListener("click", runAllBackprop);
  btnReset.addEventListener("click", resetGrads);
  btnFiniteDiff.addEventListener("click", finiteDiffCheck);
  btnDescend?.addEventListener("click", descendOneStep);
  btnClearTrace?.addEventListener("click", clearDescentTrace);

  // Init
  updateSliderLabels();
  updateStepSizeLabel();
  rebuildGraph();
  updateDescendState();
})();
