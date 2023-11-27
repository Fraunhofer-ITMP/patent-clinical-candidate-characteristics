# coding: utf-8

import os
import logging
import multiprocessing as mp
import json
import sqlite3
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import (
    MACCSkeys,
    rdReducedGraphs,
    RDKFingerprint,
    MolFromSmiles,
    Descriptors,
    Lipinski,
)
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from itertools import chain

RAW_DIR = "../data/raw"
MAPPING_DIR = "../data/mappings"
PROCESSED_DIR = "../data/processed"
FIG_DIR = "../data/figures"

os.makedirs(PROCESSED_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

tqdm.pandas()


def find_multipatent_compounds(data_df: pd.DataFrame) -> pd.DataFrame:
    """Find compounds that have been patented in previous years."""

    if os.path.exists(f"{PROCESSED_DIR}/repeated_compound.tsv"):
        return pd.read_csv(f"{PROCESSED_DIR}/repeated_compound.tsv", sep="\t")
    
    if os.path.exists(f"{PROCESSED_DIR}/surechembl_unique_inchikey_dump.pq.gzip"):
        df = pd.read_parquet(f"{PROCESSED_DIR}/surechembl_unique_inchikey_dump.pq.gzip")
    else:
        df = data_df.drop_duplicates(subset=["InChIKey", "year"], keep="first")
        df.to_parquet(f"{PROCESSED_DIR}/surechembl_unique_inchikey_dump.pq.gzip", compression="gzip")
    
    year_range = df["year"].unique().tolist()
    year_range.sort()

    # 2015 as the base year
    base_compounds = set(df[df["year"] == "2015"]["InChIKey"].unique().tolist())

    count = []

    for last_year, current_year in tqdm(
        zip(year_range, year_range[1:]), total=len(year_range) - 1
    ):
        previous_year_compound = set(
            df[(df["year"] == str(last_year))]["InChIKey"].unique().tolist()
        )
        current_year_compound = set(
            df[(df["year"] == str(current_year))]["InChIKey"].unique().tolist()
        )

        base_compounds = base_compounds.union(previous_year_compound)

        unique_cmps = current_year_compound.difference(base_compounds)
        common_cmps = current_year_compound.intersection(base_compounds)

        count.append(
            {
                "year": current_year,
                "patented_drug_count": len(common_cmps),
                "unique_compounds": len(unique_cmps),
            }
        )

    c = pd.DataFrame(count)
    c.to_csv(f"{PROCESSED_DIR}/repeated_compound.tsv", sep="\t", index=False)
    return c


def cross_references(from_pubchem: bool = False, from_chembl: bool = False):
    """Get external references (pubchem and chembl ids) for surechembl compounds."""

    surechem_df = pd.read_parquet(f"{RAW_DIR}/surechembl_dump.pq")
    inchikeys_of_interest = surechem_df["InChIKey"].unique().tolist()

    if from_pubchem:
        pubchem_df = pd.read_csv(
            f"{MAPPING_DIR}/CID-InChI-Key.gz",
            sep="\t",
            compression="gzip",
            names=["cid", "inchi", "inchikey"],
        )
        pubchem_df = pubchem_df[
            pubchem_df["inchikey"].isin(inchikeys_of_interest)
        ]  # 10129899 compounds
        pubchem_df.to_parquet(
            f"{PROCESSED_DIR}/surechembl_pubchem_map.pq.gzip", compression="gzip"
        )

    """ChEMBL ID to InChIKey mapping"""
    if from_chembl:
        conn = sqlite3.connect(f"{RAW_DIR}/chembl_32.db")

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        try:
            assert len(cursor.fetchall()) > 1
        except AssertionError:
            print("Incorrect database. Please download the database again.")

        _sql = """
        SELECT
            MOLECULE_DICTIONARY.chembl_id as chembl_id,
            MOLECULE_DICTIONARY.max_phase as clinical_phase,
            COMPOUND_STRUCTURES.STANDARD_INCHI_KEY as inchikey
        FROM MOLECULE_DICTIONARY
        JOIN COMPOUND_STRUCTURES ON MOLECULE_DICTIONARY.molregno = COMPOUND_STRUCTURES.molregno
        """

        chembl_df = pd.read_sql(_sql, con=conn)
        chembl_df.to_parquet(f"{PROCESSED_DIR}/chembl.pq.gzip", compression="gzip")


"""Parallelization of compound fingerprint calucation"""


def _calculate_fingerprints(smiles: str):
    """Get the different fingerprints for compounds."""

    try:
        molecule = Chem.MolFromSmiles(smiles)
    except:
        return smiles, {"maccs": None, "rdkit": None, "erg": None}

    if molecule is None:
        return smiles, {"maccs": None, "rdkit": None, "erg": None}

    maccs = MACCSkeys.GenMACCSKeys(molecule).ToBitString()
    rdkit = RDKFingerprint(molecule).ToBitString()
    erg = rdReducedGraphs.GetErGFingerprint(molecule).ToBitString()

    return smiles, {"maccs": maccs, "rdkit": rdkit, "erg": erg}


def get_fingerprints():
    pool = mp.Pool(10)  # We have 24 cores on our linux machine

    surechembl_df = pd.read_parquet(f"{RAW_DIR}/surechembl_dump.pq")
    logger.warning(f"Loading data")
    surechembl_df["year"] = surechembl_df["PUBLICATION_DATE"].progress_apply(
        lambda x: x.split("-")[0]
    )
    logger.warning(f"Loading completed")

    # Drop duplicate entries
    surechembl_df = surechembl_df.drop_duplicates(
        subset=["InChIKey", "year"], keep="first"
    )

    global smiles_to_fingerprint_dict

    smiles_to_fingerprint_dict = defaultdict(dict)

    unique_compounds = surechembl_df.SMILES.unique().tolist()

    results = []

    # Parallelize the process
    for row in tqdm(
        unique_compounds,
        total=len(unique_compounds),
        desc="Extracting molecular properties",
    ):
        results.append(pool.apply_async(_calculate_fingerprints, args=(row,)))

    pool.close()
    pool.join()

    for res in results:
        result = res.get()
        smiles_to_fingerprint_dict[result[0]] = result[1]

    with open(f"{MAPPING_DIR}/fingerprint.json", "w") as f:
        json.dump(smiles_to_fingerprint_dict, f, ensure_ascii=False, indent=2)


"""Parallelization of compound properties calucation"""


def _calcualte_props(smiles: str):
    """Get molecular properties from RDKit"""

    try:
        molecule = MolFromSmiles(smiles)
    except:
        return smiles, {
            "mw": None,
            "logp": None,
            "n_hba": None,
            "n_hbd": None,
            "rot_bonds": None,
            "tpsa": None,
            "fsp3": None,
            "n_chiral": None,
        }

    try:
        molecular_weight = Descriptors.ExactMolWt(molecule)
    except:
        return smiles, {
            "mw": None,
            "logp": None,
            "n_hba": None,
            "n_hbd": None,
            "rot_bonds": None,
            "tpsa": None,
            "fsp3": None,
            "n_chiral": None,
        }

    logp = Descriptors.MolLogP(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
    tpsa = Descriptors.TPSA(molecule)
    fsp3 = Lipinski.FractionCSP3(molecule)
    chiral_count = len(tuple(EnumerateStereoisomers(molecule)))

    return smiles, {
        "mw": molecular_weight,
        "logp": logp,
        "n_hba": n_hba,
        "n_hbd": n_hbd,
        "rot_bonds": rotatable_bonds,
        "tpsa": tpsa,
        "fsp3": fsp3,
        "n_chiral": chiral_count,
    }


def get_properties():
    pool = mp.Pool(10)  # We have 24 cores on our linux machine

    surechembl_df = pd.read_parquet(f"{RAW_DIR}/surechembl_dump.pq")
    logger.warning(f"Loading data")
    surechembl_df["year"] = surechembl_df["PUBLICATION_DATE"].progress_apply(
        lambda x: x.split("-")[0]
    )
    logger.warning(f"Loading completed")

    # Drop duplicate entries
    surechembl_df = surechembl_df.drop_duplicates(
        subset=["InChIKey", "year"], keep="first"
    )

    global smiles_to_property_dict

    smiles_to_property_dict = defaultdict(dict)

    unique_compounds = surechembl_df.SMILES.unique().tolist()

    results = []

    # Parallelize the process
    for row in tqdm(
        unique_compounds,
        total=len(unique_compounds),
        desc="Extracting molecular properties",
    ):
        results.append(pool.apply_async(_calcualte_props, args=(row,)))

    pool.close()
    pool.join()

    for res in results:
        result = res.get()
        smiles_to_property_dict[result[0]] = result[1]

    with open(f"{MAPPING_DIR}/properties.json", "w") as f:
        json.dump(smiles_to_property_dict, f, ensure_ascii=False, indent=2)


"""Venn diagram 
Code adapted from - https://github.com/tctianchi/pyvenn/tree/master
"""


def get_labels(data, fill=["number"]):
    """
    get a dict of labels for groups in data
    """

    N = len(data)

    sets_data = [set(data[i]) for i in range(N)]  # sets for separate groups
    s_all = set(chain(*data))  # union of all sets

    # bin(3) --> '0b11', so bin(3).split('0b')[-1] will remove "0b"
    set_collections = {}
    for n in range(1, 2**N):
        key = bin(n).split("0b")[-1].zfill(N)
        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(N) if key[i] == "1"]
        sets_for_difference = [sets_data[i] for i in range(N) if key[i] == "0"]
        for s in sets_for_intersection:
            value = value & s
        for s in sets_for_difference:
            value = value - s
        set_collections[key] = value

    labels = {k: "" for k in set_collections}
    if "logic" in fill:
        for k in set_collections:
            labels[k] = k + ": "
    if "number" in fill:
        for k in set_collections:
            labels[k] += str(len(set_collections[k]))
    if "percent" in fill:
        data_size = len(s_all)
        for k in set_collections:
            labels[k] += "(%.1f%%)" % (100.0 * len(set_collections[k]) / data_size)

    return labels


def draw_ellipse(fig, ax, x, y, w, h, a, fillcolor):
    e = patches.Ellipse(xy=(x, y), width=w, height=h, angle=a, color=fillcolor)
    ax.add_patch(e)


def draw_text(
    fig, ax, x, y, text, color=[0, 0, 0, 1], fontsize=14, ha="center", va="center"
):
    ax.text(
        x,
        y,
        text,
        horizontalalignment=ha,
        verticalalignment=va,
        fontsize=fontsize,
        color="black",
    )


def venn4(data, names=["A", "B", "C", "D"], **options):
    """
    plots a 4-set Venn diagram

    input
      data: a list of sets
      names:  group names
      more:   colors, figsize, dpi, fontsize
    """

    sns.set(style="white")
    default_colors = [
        # r, g, b, a
        [92, 192, 98, 0.5],
        [90, 155, 212, 0.5],
        [246, 236, 86, 0.6],
        [241, 90, 96, 0.4],
        [255, 117, 0, 0.3],
        [82, 82, 190, 0.2],
    ]
    default_colors = [
        [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]]
        for i in default_colors
    ]

    if (data is None) or len(data) != 4:
        raise Exception("length of data should be 4!")
    if (names is None) or (len(names) != 4):
        names = ("set 1", "set 2", "set 3", "set 4")

    colors = options.get("colors", default_colors)
    figsize = options.get("figsize", (12, 12))
    dpi = options.get("dpi", 96)
    fontsize = options.get("fontsize", 14)

    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)

    fill_typ = options.get("fill", 'number')
    labels = get_labels(data, fill=fill_typ)

    # body
    draw_ellipse(fig, ax, 0.350, 0.400, 0.72, 0.45, 140.0, colors[0])
    draw_ellipse(fig, ax, 0.450, 0.500, 0.72, 0.45, 140.0, colors[1])
    draw_ellipse(fig, ax, 0.544, 0.500, 0.72, 0.45, 40.0, colors[2])
    draw_ellipse(fig, ax, 0.644, 0.400, 0.72, 0.45, 40.0, colors[3])
    draw_text(fig, ax, 0.85, 0.42, labels.get("0001", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.68, 0.72, labels.get("0010", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.77, 0.59, labels.get("0011", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.32, 0.72, labels.get("0100", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.71, 0.30, labels.get("0101", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.66, labels.get("0110", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.65, 0.50, labels.get("0111", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.14, 0.42, labels.get("1000", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.17, labels.get("1001", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.29, 0.30, labels.get("1010", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.39, 0.24, labels.get("1011", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.23, 0.59, labels.get("1100", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.61, 0.24, labels.get("1101", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.35, 0.50, labels.get("1110", ""), fontsize=fontsize)
    draw_text(fig, ax, 0.50, 0.38, labels.get("1111", ""), fontsize=fontsize)

    # legend
    draw_text(fig, ax, 0.13, 0.18, names[0], colors[0], fontsize=fontsize, ha="right")
    draw_text(
        fig,
        ax,
        0.18,
        0.83,
        names[1],
        colors[1],
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )
    draw_text(
        fig,
        ax,
        0.82,
        0.83,
        names[2],
        colors[2],
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )
    draw_text(
        fig, ax, 0.87, 0.18, names[3], colors[3], fontsize=fontsize, ha="left", va="top"
    )
    leg = ax.legend(names, loc="center left", bbox_to_anchor=(1.0, 0.5), fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/figure_2.png", dpi=400)
    plt.show()
