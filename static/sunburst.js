/**
 * Sunburst chart for mood centroid selection.
 * Shared between similarity and alchemy pages.
 *
 * Usage:
 *   const tree = SunburstChart.buildTree(centroidsData);
 *   SunburstChart.init(containerEl, tree, onSelect);
 *
 * onSelect(node, best) is called when a terminal node is clicked.
 *   node  — the terminal tree node
 *   best  — node.bestCentroid {centroidIndex, nSongs, moodScore, clusterId, tags}
 */
const SunburstChart = (() => {
    const MOOD_COLORS = {
        happy:      {h:48,  s:90, l:55},
        sad:        {h:220, s:70, l:55},
        aggressive: {h:0,   s:80, l:50},
        party:      {h:300, s:75, l:55},
        relaxed:    {h:160, s:60, l:45},
        danceable:  {h:270, s:70, l:55}
    };

    function buildTree(data) {
        const TAG_DEPTH = 3;
        const root = {name: 'root', children: [], depth: 0};
        for (const mood of Object.keys(data).sort()) {
            const moodNode = {name: mood, children: [], depth: 1, mood: mood};
            data[mood].forEach(c => {
                let parent = moodNode;
                const tags = c.top_tags || [];
                for (let lvl = 0; lvl < TAG_DEPTH && lvl < tags.length; lvl++) {
                    const tag = tags[lvl];
                    const isTerminal = (lvl === TAG_DEPTH - 1);
                    let child = parent.children.find(ch => ch.name === tag);
                    if (!child) {
                        child = {
                            name: tag, children: [], depth: parent.depth + 1,
                            mood: mood, isTerminal: isTerminal,
                            candidates: isTerminal ? [] : undefined
                        };
                        parent.children.push(child);
                    }
                    if (isTerminal) {
                        if (!child.candidates) child.candidates = [];
                        child.candidates.push({
                            centroidIndex: c.index,
                            nSongs: c.n_songs,
                            moodScore: c.mood_score || 0,
                            clusterId: c.cluster_id != null ? c.cluster_id : c.index,
                            tags: tags
                        });
                    }
                    parent = child;
                }
            });
            root.children.push(moodNode);
        }
        (function resolveBest(node) {
            if (node.isTerminal && node.candidates && node.candidates.length > 0) {
                node.candidates.sort((a, b) => b.moodScore !== a.moodScore ? b.moodScore - a.moodScore : a.clusterId - b.clusterId);
                node.bestCentroid = node.candidates[0];
                node.size = 1;
                return;
            }
            if (node.children) {
                node.children.forEach(resolveBest);
                node.size = node.children.reduce((s, ch) => s + (ch.size || 1), 0);
                // Propagate best centroid upward: pick child with highest moodScore
                const best = node.children.reduce((a, b) => {
                    if (!a || !a.bestCentroid) return b;
                    if (!b || !b.bestCentroid) return a;
                    return b.bestCentroid.moodScore > a.bestCentroid.moodScore ? b : a;
                }, null);
                if (best && best.bestCentroid) node.bestCentroid = best.bestCentroid;
            }
        })(root);
        return root;
    }

    function arcColor(node, alpha) {
        const mc = MOOD_COLORS[node.mood] || {h:0, s:0, l:60};
        const l = Math.min(80, mc.l + (node.depth - 1) * 8);
        return `hsla(${mc.h}, ${mc.s}%, ${l}%, ${alpha !== undefined ? alpha : 1})`;
    }

    function layoutArcs(focusNode) {
        const arcs = [], maxR = 230, centerR = maxR * 0.32;
        const n = focusNode.children.length;
        if (n === 0) return arcs;
        const sweep = (Math.PI * 2) / n;
        let offset = 0;
        for (const child of focusNode.children) {
            arcs.push({node: child, startAngle: offset, endAngle: offset + sweep, innerR: centerR + 4, outerR: maxR});
            offset += sweep;
        }
        return arcs;
    }

    function findParent(root, target) {
        if (!root.children) return null;
        for (const ch of root.children) {
            if (ch === target) return root;
            const p = findParent(ch, target);
            if (p) return p;
        }
        return null;
    }

    function draw(st) {
        const {canvas, ctx, tree} = st;
        if (!tree) return;
        const f = st.focus || tree;
        const cx = 240, cy = 240, maxR = 230, centerR = maxR * 0.32;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = 480 * dpr;
        canvas.height = 480 * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, 480, 480);
        const arcs = layoutArcs(f);
        canvas._arcs = arcs;
        canvas._focus = f;
        const isDark = document.body.classList.contains('dark-mode');

        for (const arc of arcs) {
            const isHovered = st.hover === arc;
            const isSelected = st.selected && arc.node.isTerminal && st.selected.isTerminal && arc.node === st.selected;
            ctx.beginPath();
            ctx.arc(cx, cy, arc.outerR, arc.startAngle - Math.PI/2, arc.endAngle - Math.PI/2);
            ctx.arc(cx, cy, arc.innerR, arc.endAngle - Math.PI/2, arc.startAngle - Math.PI/2, true);
            ctx.closePath();
            ctx.fillStyle = arcColor(arc.node, isHovered || isSelected ? 1 : 0.78);
            ctx.fill();
            if (isSelected) { ctx.strokeStyle = '#000'; ctx.lineWidth = 4; }
            else { ctx.strokeStyle = isDark ? '#1F2937' : '#fff'; ctx.lineWidth = 2; }
            ctx.stroke();

            const midAngle = (arc.startAngle + arc.endAngle) / 2 - Math.PI/2;
            const midR = (arc.innerR + arc.outerR) / 2;
            const arcLen = midR * (arc.endAngle - arc.startAngle);
            ctx.save();
            ctx.translate(cx + Math.cos(midAngle) * midR, cy + Math.sin(midAngle) * midR);
            let rot = midAngle;
            if (rot > Math.PI/2 && rot < Math.PI*1.5) rot += Math.PI;
            ctx.rotate(rot);
            ctx.fillStyle = isDark ? '#fff' : '#1a1a2e';
            let label = arc.node.name.charAt(0).toUpperCase() + arc.node.name.slice(1);
            const fontSize = Math.min(20, Math.max(12, arcLen / (label.length * 0.55)));
            ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const maxChars = Math.floor(arcLen / (fontSize * 0.52));
            if (label.length > maxChars && maxChars > 2) label = label.slice(0, maxChars - 1) + '\u2026';
            ctx.fillText(label, 0, 0);
            ctx.restore();
        }

        // Center circle
        ctx.beginPath();
        ctx.arc(cx, cy, centerR, 0, Math.PI * 2);
        if (f !== tree) {
            const mc = MOOD_COLORS[f.mood] || {h:0, s:0, l:60};
            ctx.fillStyle = `hsl(${mc.h}, ${mc.s}%, ${isDark ? mc.l - 10 : mc.l + 20}%)`;
        } else {
            ctx.fillStyle = isDark ? '#374151' : '#f3f4f6';
        }
        ctx.fill();
        ctx.strokeStyle = isDark ? '#6B7280' : '#d1d5db';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Center text
        ctx.fillStyle = isDark ? '#e5e7eb' : '#374151';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (f !== tree) {
            ctx.font = 'bold 18px system-ui, sans-serif';
            ctx.fillText('\u2190 Back', cx, cy - 11);
            ctx.font = '18px system-ui, sans-serif';
            ctx.fillText(f.name.charAt(0).toUpperCase() + f.name.slice(1), cx, cy + 11);
        } else {
            ctx.font = 'bold 18px system-ui, sans-serif';
            ctx.fillText('Mood', cx, cy);
        }

        // Breadcrumb
        if (st.breadcrumbEl) {
            if (!f || f === tree) {
                st.breadcrumbEl.textContent = '';
            } else {
                const path = [];
                let nd = f;
                while (nd && nd !== tree) {
                    path.unshift(nd.name.charAt(0).toUpperCase() + nd.name.slice(1));
                    nd = findParent(tree, nd);
                }
                st.breadcrumbEl.textContent = path.length ? '\u2B05 ' + path.join(' \u203A ') + ' (click to go back)' : '';
            }
        }
    }

    function hitTest(x, y, canvas) {
        const cx = 240, cy = 240, dx = x - cx, dy = y - cy;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const maxR = 230, centerR = maxR * 0.32;
        if (dist <= centerR) return {type: 'center'};
        let angle = Math.atan2(dy, dx) + Math.PI/2;
        if (angle < 0) angle += Math.PI * 2;
        for (const arc of (canvas._arcs || [])) {
            if (dist >= arc.innerR && dist <= arc.outerR && angle >= arc.startAngle && angle <= arc.endAngle)
                return {type: 'arc', arc};
        }
        return null;
    }

    /**
     * Initialise a sunburst on a container element.
     *
     * @param {HTMLElement} container — must contain:
     *   canvas.sunburst-canvas, .sunburst-selection, .sunburst-breadcrumb
     * @param {Object} tree — from buildTree()
     * @param {Function} onSelect(node, best) — called on terminal selection
     * @param {Function} [onClear] — called when selection is cleared (back/navigate)
     * @returns {Object} state object (st) with reset() method
     */
    function init(container, tree, onSelect, onClear) {
        const canvas = container.querySelector('.sunburst-canvas');
        const ctx = canvas.getContext('2d');
        const selDiv = container.querySelector('.sunburst-selection');
        const bcDiv = container.querySelector('.sunburst-breadcrumb');

        const st = {canvas, ctx, tree, focus: null, hover: null, selected: null, breadcrumbEl: bcDiv};
        const redraw = () => draw(st);

        function doSelect(node) {
            st.selected = node;
            const best = node.bestCentroid;
            // Build label from what the user actually clicked, not the auto-resolved path
            const path = [];
            let nd = node;
            while (nd && nd !== tree) {
                path.unshift(nd.name.charAt(0).toUpperCase() + nd.name.slice(1));
                nd = findParent(tree, nd);
            }
            selDiv.innerHTML = '<strong>\u2713 Selected:</strong> ' + path.join(' \u2014 ');
            if (onSelect) onSelect(node, best);
        }

        function doClear() {
            st.selected = null;
            selDiv.textContent = '';
            if (onClear) onClear();
        }

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (480 / rect.width);
            const y = (e.clientY - rect.top) * (480 / rect.height);
            const hit = hitTest(x, y, canvas);
            const prev = st.hover;
            st.hover = hit && hit.type === 'arc' ? hit.arc : null;
            if (st.hover !== prev) redraw();
            if (hit && hit.type === 'arc') {
                canvas.style.cursor = 'pointer';
                if (!st.selected) selDiv.textContent = hit.arc.node.name.charAt(0).toUpperCase() + hit.arc.node.name.slice(1);
            } else if (hit && hit.type === 'center') {
                canvas.style.cursor = st.focus && st.focus !== st.tree ? 'pointer' : 'default';
                if (!st.selected) selDiv.textContent = '';
            } else {
                canvas.style.cursor = 'default';
                if (!st.selected) selDiv.textContent = '';
            }
        });

        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (480 / rect.width);
            const y = (e.clientY - rect.top) * (480 / rect.height);
            const hit = hitTest(x, y, canvas);
            if (!hit) return;
            if (hit.type === 'center') {
                if (st.focus && st.focus !== st.tree) {
                    const par = findParent(st.tree, st.focus);
                    st.focus = par && par !== st.tree ? par : st.tree;
                    // When going back to root, clear; otherwise auto-select parent's best
                    if (st.focus === st.tree) doClear();
                    else if (st.focus.bestCentroid) doSelect(st.focus);
                    else doClear();
                    redraw();
                }
                return;
            }
            if (hit.type === 'arc') {
                const node = hit.arc.node;
                if (node.isTerminal && node.bestCentroid) {
                    doSelect(node);
                    redraw();
                } else if (!node.isTerminal && node.children && node.children.length > 0) {
                    let target = node;
                    while (!target.isTerminal && target.children && target.children.length === 1) target = target.children[0];
                    if (target.isTerminal && target.bestCentroid) {
                        doSelect(target);
                        const p = findParent(st.tree, target);
                        if (p) st.focus = p;
                    } else {
                        st.focus = target;
                        // Auto-select the best centroid from this subtree
                        if (target.bestCentroid) doSelect(target);
                        else doClear();
                    }
                    redraw();
                }
            }
        });

        if (bcDiv) {
            bcDiv.addEventListener('click', () => {
                if (st.focus && st.focus !== st.tree) {
                    const par = findParent(st.tree, st.focus);
                    st.focus = par && par !== st.tree ? par : st.tree;
                    if (st.focus === st.tree) doClear();
                    else if (st.focus.bestCentroid) doSelect(st.focus);
                    else doClear();
                    redraw();
                }
            });
        }

        st.reset = () => {
            st.focus = null;
            st.selected = null;
            st.hover = null;
            selDiv.textContent = '';
            redraw();
        };

        st.redraw = redraw;

        // Initial draw
        redraw();
        canvas._sbState = st;
        return st;
    }

    return {buildTree, init, MOOD_COLORS};
})();
