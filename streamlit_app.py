
# app.py
# Streamlit Snowflake SQL Builder (POC, hardcoded metadata)

import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import streamlit as st

from hardcoded_metadata import TABLES_HARDCODED  # baked-in metadata

# ------------------------------
# Config
# ------------------------------

APP_TITLE = "🚚 Truckload Analytics — No‑Code Snowflake SQL Builder (POC)"
APP_CAPTION = "Pick tables by theme, confirm joins, choose columns, and copy the ready-to-run Snowflake SQL."

FRIENDLY_SCHEMA_NAMES = {
    "TRUCKLOAD_ETA_ANALYTICS": "ETA Analytics",
    "TL_ANALYTICS": "Analytics (Core TL)",
    "TL_ANALYTICS_INTERNAL": "Analytics (Internal)",
    "TRUCKLOADTRACKING": "Tracking Events (Raw/Operational)",
}

GENERIC_JOIN_BLACKLIST_REGEX = [
    r"^ID$", r"^CREATED_BY(_ID)?$", r"^UPDATED_BY(_ID)?$", r"^CREATED_DATE$",
    r"^UPDATED_DATE$", r"^DELETED_DATE$", r"^IS_ACTIVE$", r"^IS_DELETED$",
    r"^_.*", r"^LOAD$", r"^NAME$", r"^TYPE$"
]

GOOD_KEY_PATTERNS = [
    r"P44_", r"SHIPMENT", r"LEG", r"STOP", r"ORDER", r"LOAD",
    r"MASTER_SHIPMENT", r"ENTITY_KEY", r"UUID", r"CARRIER_?ID$", r"TENANT_ID$"
]

def quote_ident(name: str) -> str:
    """Return a Snowflake-safe identifier: unquoted if simple UPPER_CASE_WITH_UNDERSCORES,
    else double-quoted with internal quotes escaped."""
    if name is None:
        return '""'
    n = str(name)
    if re.match(r"^[A-Z_][A-Z0-9_]*$", n):
        return n
    return '"' + n.replace('"', '""') + '"'

@dataclass
class TableMeta:
    database: str
    schema: str
    table: str
    description: str
    columns: List[str] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        return f"{quote_ident(self.database)}.{quote_ident(self.schema)}.{quote_ident(self.table)}"

    @property
    def id(self) -> str:
        # id should remain case-insensitive for internal uses
        return f"{self.database}.{self.schema}.{self.table}".upper()

def _is_generic(col: str) -> bool:
    u = col.strip().upper()
    for pat in GENERIC_JOIN_BLACKLIST_REGEX:
        if re.match(pat, u):
            return True
    return False

def _key_score(col: str) -> float:
    s = 0.0
    u = col.strip().upper()
    for p in GOOD_KEY_PATTERNS:
        if re.search(p, u):
            s += 1.0
    if u == "TENANT_ID":
        s -= 0.5
    return s

def _rank_keys(keys: List[str]) -> List[str]:
    return sorted(keys, key=lambda k: (_key_score(k), len(k)), reverse=True)

@st.cache_data(show_spinner=False)
def load_metadata_hardcoded() -> Dict[str, TableMeta]:
    tables: Dict[str, TableMeta] = {}
    for t in TABLES_HARDCODED:
        meta = TableMeta(
            database=str(t["database"]).strip(),
            schema=str(t["schema"]).strip(),
            table=str(t["table"]).strip(),
            description=str(t.get("description", "")).strip(),
            columns=[str(c).strip() for c in t.get("columns", []) if str(c).strip() != ""],
        )
        tables.setdefault(meta.id, meta)
    return tables

def common_columns(a: TableMeta, b: TableMeta, ignore_generic=True) -> List[str]:
    ca = set(c.strip().upper() for c in a.columns)
    cb = set(c.strip().upper() for c in b.columns)
    inter = ca.intersection(cb)
    if ignore_generic:
        inter = {c for c in inter if not _is_generic(c)}
    return sorted(inter)

def overlap_count(a: TableMeta, b: TableMeta) -> int:
    return len(common_columns(a, b, ignore_generic=True))

def build_greedy_join_plan(selected: List[TableMeta]):
    if not selected:
        return None, [], []
    overlaps = {t.id: 0 for t in selected}
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = selected[i], selected[j]
            ov = overlap_count(a, b)
            overlaps[a.id] += ov
            overlaps[b.id] += ov
    base = max(selected, key=lambda t: overlaps.get(t.id, 0))
    plan = []
    used: Set[str] = {base.id}
    remaining = [t for t in selected if t.id != base.id]
    while remaining:
        best = None
        best_score = -1
        best_keys: List[str] = []
        for t in remaining:
            for u_id in list(used):
                u = next(x for x in selected if x.id == u_id)
                keys = common_columns(t, u, ignore_generic=True)
                score = len(keys)
                if score > best_score:
                    best = (u, t)
                    best_score = score
                    best_keys = keys
        if best_score <= 0:
            break
        left, right = best
        plan.append((left, right, _rank_keys(best_keys)))
        used.add(right.id)
        remaining = [x for x in remaining if x.id != right.id]
    unjoinable = remaining
    return base, plan, unjoinable

def suggest_bridges(a: TableMeta, b: TableMeta, universe: Dict[str, TableMeta], exclude: Set[str], top_k: int = 5):
    results = []
    for t_id, t in universe.items():
        if t_id in exclude:
            continue
        inter_a = set(common_columns(a, t))
        inter_b = set(common_columns(b, t))
        score = min(len(inter_a), len(inter_b))
        if score > 0:
            results.append((score, t, sorted(inter_a), sorted(inter_b)))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def build_sql(base: TableMeta,
              join_edges: List[Tuple[TableMeta, TableMeta, List[str]]],
              selected_columns: Dict[str, List[str]],
              join_key_overrides: Dict[Tuple[str, str], List[str]],
              join_type: str = "LEFT JOIN") -> str:
    aliases: Dict[str, str] = {}
    tables_in_order: List[TableMeta] = [base]
    for (l, r, _) in join_edges:
        if l.id not in [t.id for t in tables_in_order]:
            tables_in_order.append(l)
        if r.id not in [t.id for t in tables_in_order]:
            tables_in_order.append(r)
    for idx, t in enumerate(tables_in_order, start=1):
        aliases[t.id] = f"t{idx}"
    select_parts: List[str] = []
    for t in tables_in_order:
        alias = aliases[t.id]
        cols = selected_columns.get(alias, [])
        if not cols:
            continue
        for c in cols:
            # Keep columns unquoted to lean on Snowflake's case-insensitive behavior for typical uppercase schemas
            select_parts.append(f"    {alias}.{c} AS {alias}_{c}")
    if not select_parts:
        select_parts = [f"    {aliases[base.id]}.*"]
    select_clause = "SELECT\n" + ",\n".join(select_parts)
    from_clause = f"FROM {base.fqn} {aliases[base.id]}"
    join_clauses: List[str] = []
    for (left, right, suggested) in join_edges:
        l_alias = aliases[left.id]
        r_alias = aliases[right.id]
        keys = join_key_overrides.get((left.id, right.id), suggested)
        keys = [k for k in keys if k]
        if not keys:
            continue
        using_list = ", ".join(keys)
        join_clauses.append(f"{join_type} {right.fqn} {r_alias} USING ({using_list})")
    sql = textwrap.dedent(f"""
    {select_clause}
    {from_clause}
    """).strip()
    for jc in join_clauses:
        sql += "\n" + jc
    return sql + "\n"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_CAPTION)

with st.sidebar:
    st.header("⚙️ Options")
    default_join_type = st.selectbox("Default JOIN type", ["LEFT JOIN", "INNER JOIN", "RIGHT JOIN", "FULL OUTER JOIN"], index=0)
    show_graph = st.checkbox("Show join graph", value=True)
    show_matrix = st.checkbox("Show joinability matrix", value=False)
    global_select_all = st.checkbox("Select ALL columns by default", value=False)

tables_dict = load_metadata_hardcoded()
if not tables_dict:
    st.error("No baked-in metadata found. Ensure hardcoded_metadata.py contains TABLES_HARDCODED.")
    st.stop()

schemas = sorted({t.schema for t in tables_dict.values()})
tables_by_schema: Dict[str, List[TableMeta]] = {s: [] for s in schemas}
for tm in tables_dict.values():
    tables_by_schema[tm.schema].append(tm)
for s in schemas:
    tables_by_schema[s].sort(key=lambda x: x.table)

st.text_input("🔎 Filter tables (by name/description)", key="table_filter")
flt = st.session_state.get("table_filter", "").strip().lower()

st.header("1) Pick your tables")
chosen_ids: List[str] = []
cols = st.columns(2)
col_idx = 0
for schema in schemas:
    friendly = FRIENDLY_SCHEMA_NAMES.get(schema.upper(), schema)
    with cols[col_idx].expander(f"📂 {friendly}", expanded=False):
        st.caption(f"Schema: `{schema}`")
        for tm in tables_by_schema[schema]:
            label = f"{tm.table}"
            desc = tm.description or "(no description)"
            if flt and (flt not in tm.table.lower() and flt not in desc.lower() and flt not in schema.lower()):
                continue
            key = f"chk_{tm.id}"
            if st.checkbox(label, key=key, help=desc):
                chosen_ids.append(tm.id)
    col_idx = 1 - col_idx

if not chosen_ids:
    st.info("Select one or more tables to continue.")
    st.stop()

selected_tables = [tables_dict[_id] for _id in chosen_ids]
st.header("2) Join planner & compatibility")
if len(selected_tables) == 1:
    st.success("Only one table selected — no JOINs needed. Proceed to **Step 3** to pick columns and generate SQL.")
    base = selected_tables[0]
    plan_edges = []
    unjoinable = []
else:
    base, plan_edges, unjoinable = build_greedy_join_plan(selected_tables)
    if show_matrix:
        ids = [t.id for t in selected_tables]
        header = [""] + [tables_dict[_id].table for _id in ids]
        data = []
        for a in ids:
            row = [tables_dict[a].table]
            for b in ids:
                row.append("—" if a == b else str(overlap_count(tables_dict[a], tables_dict[b])))
            data.append(row)
        st.subheader("Joinability matrix (non-generic column overlaps)")
        st.dataframe(pd.DataFrame(data, columns=header), use_container_width=True, height=250)
    if show_graph:
        st.subheader("Join graph (edges labeled with up to 3 suggested keys)")
        nodes = {t.id: f"{t.table}\\n({t.schema})" for t in selected_tables}
        edges = []
        for i in range(len(selected_tables)):
            for j in range(i + 1, len(selected_tables)):
                a, b = selected_tables[i], selected_tables[j]
                keys = _rank_keys(common_columns(a, b))
                if keys:
                    edges.append((a.id, b.id, ", ".join(keys[:3])))
        dot = ["graph G {", "  rankdir=LR;", '  node [shape=box];']
        for nid, label in nodes.items():
            dot.append(f'  "{nid}" [label="{label}"];')
        for u, v, lbl in edges:
            dot.append(f'  "{u}" -- "{v}" [label="{lbl}"];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
    if unjoinable:
        st.error("Some selected tables could not be connected by shared columns. See suggestions below.")
        st.write("**Unjoinable tables:** " + ", ".join(f"`{t.table}`" for t in unjoinable))
    else:
        st.success(f"Joinable! Proposed base: `{base.table}`. Review keys below.")

join_key_overrides: Dict[Tuple[str, str], List[str]] = {}
if len(selected_tables) > 1 and (('unjoinable' in locals() and not unjoinable) or 'unjoinable' not in locals()):
    st.subheader("Refine join keys")
    st.caption("Pick the exact keys to use for each join edge. The defaults are ranked by likelihood.")
    for (left, right, suggested) in plan_edges:
        key = (left.id, right.id)
        options = _rank_keys(common_columns(left, right))
        default_sel = suggested[:2] if suggested else []
        chosen = st.multiselect(
            f"Join `{right.table}` onto `{left.table}` USING ...",
            options=options,
            default=default_sel,
            help=f"From left `{left.table}` and right `{right.table}` shared columns"
        )
        join_key_overrides[key] = chosen

st.header("3) Choose columns to SELECT")
tables_in_order = [selected_tables[0]] if len(selected_tables) == 1 else []
if len(selected_tables) > 1 and 'base' in locals() and base:
    ordered = [base]
    for (l, r, _) in plan_edges:
        if l.id not in [t.id for t in ordered]:
            ordered.append(l)
        if r.id not in [t.id for t in ordered]:
            ordered.append(r)
    tables_in_order = ordered

selected_columns: Dict[str, List[str]] = {}
if tables_in_order:
    for idx, t in enumerate(tables_in_order, start=1):
        alias = f"t{idx}"
        with st.expander(f"🧱 Columns from `{t.table}` (alias `{alias}`)", expanded=(idx == 1)):
            cols_sorted = sorted(set(c.strip().upper() for c in t.columns))
            if st.checkbox(f"Select ALL columns from `{t.table}`", key=f"all_{t.id}"):
                selected_columns[alias] = cols_sorted
                st.caption(f"Selected {len(cols_sorted)} columns.")
            else:
                chosen = st.multiselect(
                    f"Pick columns from `{t.table}`",
                    options=cols_sorted,
                    default=[],
                    key=f"cols_{t.id}"
                )
                selected_columns[alias] = chosen

st.header("4) Generate SQL")
can_generate = True
problems = []
if len(selected_tables) > 1 and 'unjoinable' in locals() and unjoinable:
    can_generate = False
    problems.append("Selected tables are not fully joinable. Add bridging tables or adjust selection.")
if len(selected_tables) > 1 and 'plan_edges' in locals():
    for (left, right, suggested) in plan_edges:
        keys = join_key_overrides.get((left.id, right.id), suggested)
        if not keys:
            can_generate = False
            problems.append(f"No join keys selected for `{right.table}` onto `{left.table}`.")
if not any(selected_columns.values()):
    st.info("No columns selected — the query will default to `SELECT *` from the base table.")
if not can_generate:
    st.error("Cannot generate SQL yet:")
    for p in problems:
        st.write(f"- {p}")
else:
    sql = build_sql(
        base=base if len(selected_tables) > 1 else selected_tables[0],
        join_edges=plan_edges if len(selected_tables) > 1 else [],
        selected_columns=selected_columns,
        join_key_overrides=join_key_overrides,
        join_type=default_join_type
    )
    st.code(sql, language="sql")
    st.download_button("⬇️ Download SQL", sql.encode("utf-8"), file_name="query.sql", mime="text/sql")

st.markdown("---")
with st.expander("ℹ️ Notes"):
    st.write("""
- Shared columns are matched by **name** (case-insensitive) and we ignore generic fields (e.g., `ID`, `CREATED_DATE`, ...).
- Suggested join keys are **heuristics**. Always review and refine.
- SQL uses `JOIN ... USING (col1, col2, ...)` (same column names on both sides).
- Database/Schema/Table names are **safely quoted** if they contain spaces/mixed case.
""")
