# -*- coding: utf-8 -*-
"""
SF Tagger – bulk tag editor for Campaign Members

Highlights:
• Immediate tagging: Click a tag checkbox to instantly apply/remove it for all selected rows.
• Smart Next: The "Next Untagged" button automatically finds the next untagged household, looping from the top.
• Save/Load Session: Save your work-in-progress to a file and resume later.
• Column order: First Name | Last Name | Company | Member | Final Tags | Email | Phone | SFID.
• Final Tags column is lightly highlighted for visibility.
• Tri-state checkboxes show the status of tags across multiple selected rows.
• Auto-select household (by Company) on row click; toggle to disable. Ctrl/⌘ or Shift bypasses.
"""

from __future__ import annotations
import os
import sys
from typing import List, Optional
from datetime import datetime

import pandas as pd

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QSize,
    QRegularExpression, QItemSelectionModel
)
from PySide6.QtGui import QAction, QFont, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTableView, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QLineEdit,
    QSplitter, QStyleFactory, QCheckBox, QFrame, QToolBar,
    QHeaderView, QToolButton
)

# ====== Tags used in SF ======
TAGS: List[str] = ["Buyer", "Seller", "Renter", "Landlord", "Investor", "Luxury", "Settled", "Sphere"]

# Display columns (Final Tags is computed)
DISPLAY_COLS = ["First Name", "Last Name", "Company", "Member", "Final Tags", "Email", "Phone", "SFID"]

# Column-name candidates to map from export
CANDIDATES = {
    "member":  ["Member Type", "Member", "MemberType"],
    "id":      ["Related Record ID", "RelatedRecordId", "SFID", "Id", "Record ID", "Related Record Id"],
    "first":   ["First Name", "FirstName"],
    "last":    ["Last Name", "LastName"],
    "email":   ["Email", "Email Address"],
    "phone":   ["Phone", "Business Phone", "Phone Number", "Mobile Phone", "Mobile"],
    "company": ["Company", "Account Name", "Account"],
    "owner":   ["Related Record Owner", "Record Owner", "Owner"] # Added for filename
}
PREEXISTING_TAGS_COLS = ["Tags__c", "Tags", "FinalTags"]

# ---------- helpers ----------

def pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        c = lower.get(n.lower())
        if c:
            return c
    return None

def infer_member_from_id(sfid: str) -> str:
    if not isinstance(sfid, str) or len(sfid) < 3:
        return ""
    p = sfid[:3]
    if p == "003":
        return "Contact"
    if p == "00Q":
        return "Lead"
    return ""

def coerce_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().upper()
    return s in {"1", "TRUE", "Y", "YES", "✓", "CHECKED"}

def ensure_bool_columns(df: pd.DataFrame, tags: List[str], pre_col: Optional[str]) -> None:
    for t in tags:
        if t not in df.columns:
            df[t] = False
        else:
            df[t] = df[t].apply(coerce_bool)
    if pre_col and pre_col in df.columns:
        def to_set(v):
            if not isinstance(v, str):
                return set()
            return {x.strip() for x in v.split(";") if x.strip()}
        parsed = df[pre_col].fillna("").apply(to_set)
        for t in tags:
            df[t] = df[t] | parsed.apply(lambda s: t in s)

def compose_tags_row(row: pd.Series, tags: List[str]) -> str:
    return ";".join([t for t in tags if bool(row.get(t, False))])

# ---------- models ----------

class DataFrameModel(QAbstractTableModel):
    """Read-only model; 'Final Tags' is computed."""
    def __init__(self, df: pd.DataFrame, columns: List[str], parent=None):
        super().__init__(parent)
        self._df = df
        self._cols = columns

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._cols)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col = self._cols[index.column()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == "Final Tags":
                row = self._df.iloc[index.row()]
                return compose_tags_row(row, TAGS)
            v = self._df.iloc[index.row()].get(col, "")
            return "" if pd.isna(v) else str(v)
        if role == Qt.TextAlignmentRole:
            if col in ("Member", "Phone"):
                return Qt.AlignCenter
            return Qt.AlignVCenter | Qt.AlignLeft
        if role == Qt.BackgroundRole and col == "Final Tags":
            return QBrush(QColor(255, 252, 229))
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._cols[section]
        return section + 1

class MultiColFilter(QSortFilterProxyModel):
    """Filter across multiple columns using a QRegularExpression."""
    def __init__(self, cols: List[str], parent=None):
        super().__init__(parent)
        self.cols = cols
    def filterAcceptsRow(self, source_row, source_parent):
        rx: QRegularExpression = self.filterRegularExpression()
        if not rx.pattern():
            return True
        mdl: DataFrameModel = self.sourceModel()
        for col in self.cols:
            try:
                ix = mdl._cols.index(col)
                idx = mdl.index(source_row, ix)
                val = mdl.data(idx, Qt.DisplayRole) or ""
                if rx.match(str(val)).hasMatch():
                    return True
            except Exception:
                pass
        return False

# ---------- Tagging panel ----------

class TagPanel(QWidget):
    """Immediate-apply tagging driven by clicked() to avoid tri-state signal quirks."""
    def __init__(self, tags: List[str], parent=None):
        super().__init__(parent)
        self.tags = tags
        self._df: Optional[pd.DataFrame] = None
        self.selected_rows: List[int] = []
        
        self.refresh_table_cb = None
        self.sync_panel_cb = None
        self.move_prev_cb = None
        self.move_next_cb = None

        self.setMinimumWidth(350)
        self.setMaximumWidth(430)

        # UI Styling
        self.setStyleSheet("""
            QPushButton#nextBtn {
                background-color: #ffb300;
                color: black;
                font-weight: 600;
                border: 1px solid #c68400;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton#nextBtn:disabled {
                background-color: #ffe082;
                border-color: #ffca28;
            }
        """)

        self.title = QLabel("Tag selected rows")
        f = QFont(); f.setPointSize(12); f.setBold(True)
        self.title.setFont(f)
        self.subtitle = QLabel("Open a report to begin.")
        self.subtitle.setStyleSheet("color:#666;"); self.subtitle.setWordWrap(True)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)

        self.infoBtn = QToolButton(); self.infoBtn.setText("ℹ︎"); self.infoBtn.setCheckable(True)
        self.hint = QLabel("Select rows. Click a tag to instantly apply/remove it for the entire selection.");
        self.hint.setStyleSheet("color:#777; font-size:11px;"); self.hint.setWordWrap(True); self.hint.setVisible(False)
        self.infoBtn.toggled.connect(self.hint.setVisible)
        infoRow = QHBoxLayout(); infoRow.addWidget(self.infoBtn, 0, Qt.AlignLeft); infoRow.addStretch(1)

        self.chkHousehold = QCheckBox("Auto-select household"); self.chkHousehold.setChecked(True)

        self.tagChecks: List[QCheckBox] = []
        tags_grid = QGridLayout(); tags_grid.setHorizontalSpacing(18); tags_grid.setVerticalSpacing(6)
        for i, t in enumerate(self.tags):
            c = QCheckBox(t)
            c.setTristate(True)
            c.clicked.connect(lambda checked, tag=t: self._on_tag_clicked(tag))
            self.tagChecks.append(c)
            tags_grid.addWidget(c, i // 2, i % 2)

        self.btnPickNone = QPushButton("Clear All Tags")
        self.btnPickAll = QPushButton("Select All Tags")
        self.btnPrev = QPushButton("◀ Prev Household")
        self.btnNext = QPushButton("Next Untagged ▶")
        self.btnNext.setObjectName("nextBtn")

        self.btnPickNone.clicked.connect(lambda: self._batch_set_all(False))
        self.btnPickAll.clicked.connect(lambda: self._batch_set_all(True))
        self.btnPrev.clicked.connect(lambda: self.move_prev_cb() if self.move_prev_cb else None)
        self.btnNext.clicked.connect(lambda: self.move_next_cb() if self.move_next_cb else None)
        
        button_grid = QGridLayout()
        button_grid.addWidget(self.btnPickNone, 0, 0)
        button_grid.addWidget(self.btnPickAll, 0, 1)
        button_grid.addWidget(self.btnPrev, 1, 0)
        button_grid.addWidget(self.btnNext, 1, 1)

        lay = QVBoxLayout(self); lay.setContentsMargins(12, 12, 12, 12)
        lay.addWidget(self.title); lay.addWidget(self.subtitle); lay.addWidget(line)
        lay.addLayout(infoRow); lay.addWidget(self.hint)
        lay.addWidget(self.chkHousehold); lay.addSpacing(4)
        lay.addLayout(tags_grid)
        lay.addSpacing(10)
        lay.addLayout(button_grid)
        lay.addStretch(1)
        
        self._set_panel_enabled(False)

    def bind(self, df: pd.DataFrame, refresh_table_cb, sync_panel_cb, move_prev_cb, move_next_cb):
        self._df = df
        self.refresh_table_cb = refresh_table_cb
        self.sync_panel_cb = sync_panel_cb
        self.move_prev_cb = move_prev_cb
        self.move_next_cb = move_next_cb
        self._set_panel_enabled(True)
        self.subtitle.setText("No rows selected")

    def _set_panel_enabled(self, on: bool):
        widgets = [self.btnPickAll, self.btnPickNone, self.chkHousehold, self.btnPrev, self.btnNext] + self.tagChecks
        for w in widgets:
            w.setEnabled(on)

    def set_selection(self, rows: List[int]):
        self.selected_rows = rows
        if self._df is None or not rows:
            self.subtitle.setText("No rows selected")
            self._set_panel_enabled(bool(self._df is not None))
            self.btnPrev.setEnabled(False)
            self.btnNext.setEnabled(bool(self._df is not None))
        else:
            self._set_panel_enabled(True)
            rec = self._df.iloc[rows[0]]
            name = f"{rec.get('First Name', '')} {rec.get('Last Name', '')}".strip() or "(No name)"
            if len(rows) == 1:
                self.subtitle.setText(f"{name} • {rec.get('Company', '')}")
            else:
                self.subtitle.setText(f"{name} + {len(rows)-1} more selected")
            self.btnPrev.setEnabled(True)
            self.btnNext.setEnabled(True)
        
        if not self._df is None:
            for cb in self.tagChecks: cb.blockSignals(True)
            for i, tag in enumerate(self.tags):
                state = Qt.Unchecked
                if rows:
                    vals = [bool(self._df.at[r, tag]) for r in rows]
                    if all(vals):
                        state = Qt.Checked
                    elif any(vals):
                        state = Qt.PartiallyChecked
                self.tagChecks[i].setCheckState(state)
            for cb in self.tagChecks: cb.blockSignals(False)

    def _on_tag_clicked(self, tag: str):
        if self._df is None or not self.selected_rows:
            return
        
        rows = self.selected_rows
        vals = [bool(self._df.at[r, tag]) for r in rows]
        make_true = not all(vals)
        
        self._df.loc[rows, tag] = make_true
        
        if self.refresh_table_cb:
            self.refresh_table_cb()
        if self.sync_panel_cb:
            self.sync_panel_cb()

    def _batch_set_all(self, make_true: bool):
        if self._df is None or not self.selected_rows:
            return
        for tag in self.tags:
            self._df.loc[self.selected_rows, tag] = make_true
        
        if self.refresh_table_cb:
            self.refresh_table_cb()
        if self.sync_panel_cb:
            self.sync_panel_cb()

# ---------- Main window ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Northrop Salesforce Relationship Tagger")
        self.resize(1350, 800)
        self.setStyleSheet("""
            QPushButton#exportBtn { 
                background-color: #d32f2f; 
                color: white; 
                font-weight: 600; 
                padding: 6px 12px; 
                border-radius: 6px;
                border: 1px solid #c62828;
            }
            QPushButton#exportBtn:disabled { 
                background-color: #ef9a9a; 
                border-color: #e57373;
            }
            QTableView { 
                selection-background-color: #2f6de1; 
            }
            QTableView::item:selected { 
                color: white; 
                background-color: #2f6de1; 
            }
        """)
        self._df: Optional[pd.DataFrame] = None
        self._record_owner = "Unknown_Owner"

        tb = QToolBar("Main"); tb.setIconSize(QSize(16, 16)); self.addToolBar(tb)
        actOpen = QAction("Open Report/Session…", self); actOpen.triggered.connect(self.open_report); tb.addAction(actOpen)
        actSave = QAction("Save Session…", self); actSave.triggered.connect(self.save_session); tb.addAction(actSave)
        actHelp = QAction("Help", self); actHelp.triggered.connect(self.show_help); tb.addAction(actHelp)
        tb.addSeparator()
        self.filterEdit = QLineEdit(); self.filterEdit.setPlaceholderText("Quick filter…"); tb.addWidget(self.filterEdit)
        tb.addSeparator()
        self.exportBtn = QPushButton("Export for Salesforce"); self.exportBtn.setObjectName("exportBtn"); tb.addWidget(self.exportBtn)
        self.exportBtn.clicked.connect(self.export_files)

        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setStyle(QStyleFactory.create("Fusion"))
        self.table.setEditTriggers(QTableView.NoEditTriggers)

        self.tagPanel = TagPanel(TAGS, parent=self)
        split = QSplitter(); split.addWidget(self.table); split.addWidget(self.tagPanel)
        split.setStretchFactor(0, 7); split.setStretchFactor(1, 2)
        split.setSizes([1000, 350])
        container = QWidget(); lay = QVBoxLayout(container); lay.addWidget(split); lay.setContentsMargins(6, 4, 6, 6)
        self.setCentralWidget(container)

        self.proxy = MultiColFilter(DISPLAY_COLS, self)
        self.filterEdit.textChanged.connect(self._on_filter_text)
        self.table.clicked.connect(self._on_table_clicked)

    def _on_filter_text(self, s: str):
        escaped_text = QRegularExpression.escape(s)
        rx = QRegularExpression(escaped_text, QRegularExpression.CaseInsensitiveOption)
        self.proxy.setFilterRegularExpression(rx)
        self.sync_panel()

    def show_help(self):
        help_text = """
        <p><b>1. Open Report/Session</b><br>
        Choose a CSV/XLSX report or a previously saved session file (.pkl) to begin.</p>
        
        <p><b>2. Select Rows</b><br>
        Click a row to select it. Hold Ctrl/⌘ to select multiple individual rows, or hold Shift to select a range of rows.</p>
        
        <p><b>3. Auto-Select Household</b><br>
        With 'Auto-select household' enabled, clicking any person will automatically select all other records that share the same 'Company' value.</p>
        
        <p><b>4. Tag Records</b><br>
        Click the tag checkboxes on the right panel. Changes are applied instantly to all selected rows, and the 'Final Tags' column will update immediately.</p>
        
        <p><b>5. Navigate</b><br>
        Use the <b>◀ Prev Household</b> and <b>Next Untagged ▶</b> buttons to efficiently move through the records.</p>
        
        <p><b>6. Save & Export</b><br>
        Use <b>Save Session</b> to save your progress as a .pkl file. When you are finished, click <b>Export for Salesforce</b> to generate your final CSV files.</p>
        """
        QMessageBox.information(self, "How to use SF Tagger", help_text)

    def get_selected_source_rows(self) -> List[int]:
        sm = self.table.selectionModel()
        if not sm: return []
        proxy_rows = sorted({ix.row() for ix in sm.selectedIndexes()})
        src_rows = [self.proxy.mapToSource(self.proxy.index(r, 0)).row() for r in proxy_rows]
        return sorted([r for r in src_rows if r >= 0])

    def save_session(self):
        if self._df is None:
            QMessageBox.warning(self, "Nothing to Save", "Please open a report first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "tagger_session.pkl", "Tagger Session Files (*.pkl)")
        if not path: return
        try:
            self._df.to_pickle(path)
            QMessageBox.information(self, "Session Saved", f"Session saved successfully to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save the session file:\n{e}")

    def _normalize_dataframe(self, raw: pd.DataFrame) -> pd.DataFrame:
        col_id = pick_col(raw, CANDIDATES["id"])
        if not col_id:
            raise ValueError("Couldn’t find an Id column (e.g., 'Related Record ID' or 'SFID').")
        
        out = pd.DataFrame({
            "Member": raw.get(pick_col(raw, CANDIDATES["member"]), ""), "SFID": raw[col_id],
            "First Name": raw.get(pick_col(raw, CANDIDATES["first"]), ""), "Last Name": raw.get(pick_col(raw, CANDIDATES["last"]), ""),
            "Email": raw.get(pick_col(raw, CANDIDATES["email"]), ""), "Phone": raw.get(pick_col(raw, CANDIDATES["phone"]), ""),
            "Company": raw.get(pick_col(raw, CANDIDATES["company"]), "")
        })
        out["Member"] = out.apply(lambda r: r["Member"] if str(r["Member"]).strip() else infer_member_from_id(str(r["SFID"])), axis=1)
        ensure_bool_columns(out, TAGS, pick_col(raw, PREEXISTING_TAGS_COLS))
        for c in DISPLAY_COLS + TAGS:
            if c in out.columns:
                out[c] = out[c].fillna("")
        return out

    def open_report(self):
        file_filter = "All Supported Files (*.csv *.xlsx *.xls *.pkl);;Reports (*.csv *.xlsx *.xls);;Tagger Sessions (*.pkl)"
        path, _ = QFileDialog.getOpenFileName(self, "Open Report or Session", "", file_filter)
        if not path: return
        try:
            if path.lower().endswith(".pkl"):
                df = pd.read_pickle(path)
                self._record_owner = "Unknown_Owner" # Owner info isn't saved in pkl
            else:
                raw = pd.read_csv(path, dtype=str, keep_default_na=False) if path.lower().endswith(".csv") else pd.read_excel(path, dtype=str, keep_default_na=False)
                
                # Find and store the record owner for filenames
                owner_col = pick_col(raw, CANDIDATES["owner"])
                if owner_col and not raw.empty:
                    owner_name = str(raw.iloc[0][owner_col]).replace(" ", "_")
                    self._record_owner = owner_name if owner_name else "Unknown_Owner"
                else:
                    self._record_owner = "Unknown_Owner"

                df = self._normalize_dataframe(raw)
        except Exception as e:
            QMessageBox.critical(self, "Error Opening File", f"Could not open or process the file:\n{e}")
            return

        self._df = df
        model = DataFrameModel(df, DISPLAY_COLS, self)
        self.proxy.setSourceModel(model)
        self.table.setModel(self.proxy)

        widths = {"First Name": 120, "Last Name": 140, "Company": 250, "Member": 90, "Final Tags": 190, "Email": 230, "Phone": 120, "SFID": 150}
        for i, name in enumerate(DISPLAY_COLS):
            self.table.setColumnWidth(i, widths.get(name, 120))
        self.table.horizontalHeader().setStretchLastSection(True)

        if hasattr(self, '_selection_handler'):
             self.table.selectionModel().selectionChanged.disconnect(self._selection_handler)
        self._selection_handler = self._on_selection_changed
        self.table.selectionModel().selectionChanged.connect(self._selection_handler)
        
        self.tagPanel.bind(df, self.refresh_table_view, self.sync_panel, self._move_prev_household, self._move_next)
        self.sync_panel()

    def _on_table_clicked(self, proxy_index: QModelIndex):
        if self._df is None: return
        mods = QApplication.keyboardModifiers()
        bypass = bool(mods & (Qt.ControlModifier | Qt.ShiftModifier | Qt.MetaModifier))
        if self.tagPanel.chkHousehold.isChecked() and not bypass:
            src_row = self.proxy.mapToSource(proxy_index).row()
            if src_row >= 0:
                self._select_household_by_source_row(src_row)
        else:
            self.sync_panel()

    def _select_household_by_source_row(self, src_row: int):
        if self._df is None or src_row < 0: return
        company = str(self._df.iloc[src_row].get("Company", "")).strip()
        if not company:
            self._select_single_source_row(src_row)
            return

        sm = self.table.selectionModel()
        sm.clearSelection()
        src_model: DataFrameModel = self.proxy.sourceModel()
        first_proxy_row = -1
        for r_idx, r_company in self._df["Company"].items():
            if str(r_company).strip() == company:
                pr = self.proxy.mapFromSource(src_model.index(r_idx, 0)).row()
                if pr != -1:
                    if first_proxy_row == -1:
                        first_proxy_row = pr
                    sm.select(self.proxy.index(pr, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        
        if first_proxy_row != -1:
            self.table.scrollTo(self.proxy.index(first_proxy_row, 0), QTableView.ScrollHint.PositionAtCenter)

    def _on_selection_changed(self, *_):
        self.sync_panel()

    def sync_panel(self):
        """Syncs the right-hand panel to reflect the current table selection."""
        rows = self.get_selected_source_rows()
        self.tagPanel.set_selection(rows)
        
    def _select_single_source_row(self, srow: int):
        """Selects a single row in the table given its source index and scrolls to it."""
        if srow < 0 or self.proxy.sourceModel() is None: return
        proxy_row = self.proxy.mapFromSource(self.proxy.sourceModel().index(srow, 0)).row()
        if proxy_row < 0: return

        sm = self.table.selectionModel()
        sm.clearSelection()
        sm.select(self.proxy.index(proxy_row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.table.scrollTo(self.proxy.index(proxy_row, 0), QTableView.ScrollHint.PositionAtCenter)

    def _get_current_focus_proxy_row(self) -> Optional[int]:
        proxy_rows = sorted({ix.row() for ix in self.table.selectionModel().selectedIndexes()})
        return proxy_rows[0] if proxy_rows else None

    def _move_next(self):
        """Finds the next untagged row, looping from the top if necessary, and selects its household."""
        start_row = self._get_current_focus_proxy_row()
        if start_row is None:
            start_row = -1

        try:
            final_tags_col_idx = DISPLAY_COLS.index("Final Tags")
        except ValueError:
            return

        num_rows = self.proxy.rowCount()
        if num_rows == 0: return
        
        search_order = [(start_row + 1 + i) % num_rows for i in range(num_rows)]

        for i in search_order:
            proxy_index = self.proxy.index(i, final_tags_col_idx)
            final_tags_value = self.proxy.data(proxy_index)
            if not final_tags_value:
                src_row_to_select = self.proxy.mapToSource(proxy_index).row()
                if self.tagPanel.chkHousehold.isChecked():
                    self._select_household_by_source_row(src_row_to_select)
                else:
                    self._select_single_source_row(src_row_to_select)
                return
        
        QMessageBox.information(self, "Search Complete", "No untagged records found in the entire list.")

    def _move_prev_household(self):
        """Moves the selection to the previous household."""
        if self._df is None: return
        
        visible_companies = []
        seen = set()
        company_col = DISPLAY_COLS.index("Company")
        for prow in range(self.proxy.rowCount()):
            company = self.proxy.data(self.proxy.index(prow, company_col))
            if company and company not in seen:
                visible_companies.append(company)
                seen.add(company)

        if not visible_companies: return

        current_rows = self.get_selected_source_rows()
        current_company = self._df.iloc[current_rows[0]].get("Company") if current_rows else None

        try:
            current_idx = visible_companies.index(current_company) if current_company else 0
        except ValueError:
            current_idx = 0
            
        target_idx = (current_idx - 1 + len(visible_companies)) % len(visible_companies)
        target_company = visible_companies[target_idx]

        for idx, company in self._df["Company"].items():
            if company == target_company:
                self._select_household_by_source_row(idx)
                break

    def refresh_table_view(self):
        """Forces the table to repaint its data, recalculating computed columns."""
        if self.proxy.rowCount() > 0 and self.table.model() is not None:
            self.table.model().layoutChanged.emit()

    def export_files(self):
        if self._df is None or self._df.empty:
            QMessageBox.information(self, "Nothing to export", "Open a report first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir: return

        df = self._df.copy()
        df["Tags__c"] = df.apply(lambda r: compose_tags_row(r, TAGS), axis=1)
        df["Id"] = df["SFID"].astype(str)

        leads = df[(df["Member"] == "Lead") & (df["Tags__c"] != "")]
        contacts = df[(df["Member"] == "Contact") & (df["Tags__c"] != "")]

        date_str = datetime.now().strftime("%m_%d_%y")
        contacts_filename = f"{self._record_owner}_Contacts_Tagged_For_Upload_{date_str}.csv"
        leads_filename = f"{self._record_owner}_Leads_Tagged_For_Upload_{date_str}.csv"
        
        export_cols = ["Id", "First Name", "Last Name", "Tags__c"]

        try:
            contacts[export_cols].to_csv(os.path.join(out_dir, contacts_filename), index=False)
            leads[export_cols].to_csv(os.path.join(out_dir, leads_filename), index=False)
            QMessageBox.information(
                self, "Export complete",
                f"Saved:\n• {contacts_filename}\n• {leads_filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Write Error", f"Could not write files:\n{e}")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()