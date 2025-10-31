"""Jarvis-dgl data loaders and DGLGraph utilities."""

import random
from pathlib import Path
from typing import Optional

# from typing import Dict, List, Optional, Set, Tuple

import os
import torch
import dgl
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

# from sklearn.pipeline import Pipeline
import pickle as pk

# from sklearn.decomposition import PCA  # ,KernelPCA
from sklearn.preprocessing import StandardScaler

# use pandas progress_apply
tqdm.pandas()


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    # d = d.replace("na", np.nan)
    return d


# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_torch_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # print("df", df)
    vals = df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    graphs = load_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )

    data = StructureDataset(
        df,
        graphs,
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )
    return data


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    train_data=None,
    test_data=None,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    cv = 0, # cross validation fold
):
    # print ('output_dir data',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (
        train_data is not None
        and test_data is not None
    ):
        # If train_data and test_data are already lists (passed from outside)
        if isinstance(train_data, list) and isinstance(test_data, list):
            d_train = train_data
            d_test = test_data
        # Otherwise load from jarvis database
        else:
            d_train = jdata(train_data)
            d_test = jdata(test_data)

        dat_train = []
        all_targets_train = []
        for i in d_train:
            if isinstance(i[target], list):  # multioutput target
                all_targets_train.append(torch.tensor(i[target]))
                dat_train.append(i)

            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                dat_train.append(i)
                all_targets_train.append(i[target])

        # id_test = ids[-test_size:]
        # if standardize:
        #    data.setup_standardizer(id_train)
        dat_test = []
        all_targets_test = []
        for i in d_test:
            if isinstance(i[target], list):  # multioutput target
                all_targets_test.append(torch.tensor(i[target]))
                dat_test.append(i)

            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                dat_test.append(i)
                all_targets_test.append(i[target])
        id_train = list(range(len(dat_train)))
        id_val = list(range(len(dat_test)))
        id_test = list(range(len(dat_test)))

        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat_train[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat_test[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat_test[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )     
        dataset_train = [dat_train[x] for x in id_train]
        dataset_val = [dat_test[x] for x in id_val]
        dataset_test = [dat_test[x] for x in id_test]

        if standard_scalar_and_pca:
            y_data = [i[target] for i in dataset_train]
            # pipe = Pipeline([('scale', StandardScaler())])
            if not isinstance(y_data[0], list):
                print("Running StandardScalar")
                y_data = np.array(y_data).reshape(-1, 1)
            sc = StandardScaler()

            sc.fit(y_data)
            print("Mean", sc.mean_)
            print("Variance", sc.var_)
            try:
                print("New max", max(y_data))
                print("New min", min(y_data))
            except Exception as exp:
                print(exp)
                pass
            # pc = PCA(n_components=output_features)
            # pipe = Pipeline(
            #    [
            #        ("scale", StandardScaler()),
            #        ("reduce_dims", PCA(n_components=output_features)),
            #    ]
            # )
            pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))
            # pc = PCA(n_components=10)
            # pc.fit(y_data)
            # pk.dump(pc, open("pca.pkl", "wb"))

        if classification_threshold is None:
            try:
                from sklearn.metrics import mean_absolute_error

                all_targets = all_targets_train + all_targets_test
                print("MAX val:", max(all_targets))
                print("MIN val:", min(all_targets))
                print("MAD:", mean_absolute_deviation(all_targets))
                try:
                    f = open(os.path.join(output_dir, "mad"), "w")
                    line = "MAX val:" + str(max(all_targets)) + "\n"
                    line += "MIN val:" + str(min(all_targets)) + "\n"
                    line += (
                        "MAD val:"
                        + str(mean_absolute_deviation(all_targets))
                        + "\n"
                    )
                    f.write(line)
                    f.close()
                except Exception as exp:
                    print("Cannot write mad", exp)
                    pass
                # Random model precited value
                x_bar = np.mean(np.array([i[target] for i in dataset_train]))
                baseline_mae = mean_absolute_error(
                    np.array([i[target] for i in dataset_test]),
                    np.array([x_bar for i in dataset_test]),
                )
                print("Baseline MAE:", baseline_mae)
            except Exception as exp:
                print("Data error", exp)
                pass
        tmp_name = f"train_data"
        train_data = get_torch_dataset(
            dataset=dataset_train,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name=tmp_name,
        )
        tmp_name = "val_data"
        val_data = (
            get_torch_dataset(
                dataset=dataset_val,
                id_tag=id_tag,
                atom_features=atom_features,
                target=target,
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                name=dataset,
                line_graph=line_graph,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                classification=classification_threshold is not None,
                output_dir=output_dir,
                tmp_name=tmp_name,
            )
            if len(dataset_val) > 0
            else None
        )
        tmp_name = f"test_data"
        test_data = (
            get_torch_dataset(
                dataset=dataset_test,
                id_tag=id_tag,
                atom_features=atom_features,
                target=target,
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                name=dataset,
                line_graph=line_graph,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                classification=classification_threshold is not None,
                output_dir=output_dir,
                tmp_name="test_data",
            )
            if len(dataset_test) > 0
            else None
        )

        collate_fn = train_data.collate
        if line_graph:
            collate_fn = train_data.collate_line_graph

        # use a regular pytorch dataloader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        ) if val_data is not None else None

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        ) if test_data is not None else None
        if save_dataloader:
            torch.save(train_loader, train_sample)
            if val_loader is not None:
                torch.save(val_loader, val_sample)
            if test_loader is not None:
                torch.save(test_loader, test_sample)
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset) if val_loader is not None and val_loader.dataset is not None else 0)
    print("n_test:", len(test_loader.dataset) if test_loader is not None and test_loader.dataset is not None else 0)
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
    )