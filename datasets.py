import datetime
import pandas as pd
import pathlib
import copy
import glob
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import tqdm


from sklearn import \
    utils as sk_utils, \
    model_selection as sk_model_selection, \
    metrics as sk_metrics, \
    preprocessing as sk_preprocessing
from pathlib import Path

# Root directory holding the raw cohorts (IXI/, FCDBONN/, IDEAS/).
# Override with the SURFPREP_DATA_ROOT environment variable.
DATA_ROOT = os.environ.get("SURFPREP_DATA_ROOT", "../data")

# Änderung: 13. Feb
# USED IN 3D_to_2D, vqvae3D, split

class SuperDataset():

    def __init__(self):
        super(SuperDataset, self).__init__()

    def get_template_dict(self):
        template_dict = {
            "t1": [],
            "t1ks": [],
            "t2": [],
            "swi": [],
            "flair": [],
            "ct": [],
            "xray": [],
            "etc": [],
            "seg": [],
            "not_healthy": [],
            "class_label": [],
            "cdr": [],
            "mask": [],
            "keyword": [],
        }
        return template_dict


    def check_if_dict_empty(self, dd):
        dd = copy.deepcopy(dd)
        empty = False

        try:
            del dd["seg"]
        except:
            pass
        try:
            del dd["not_healthy"]
        except:
            pass

        try:
            del dd["class_label"]
        except:
            pass

        try:
            del dd["mask"]
        except:
            pass

        try:
            del dd["cdr"]
        except:
            pass

        try:
            del dd["keyword"]
        except:
            pass

        lst_values = []
        for k, v in dd.items():
            lst_values.extend(v)

        if len(lst_values) == 0:
            empty = True
        return empty


    def filter_seq_types(self, data_dict, seq_types):
        data_dict = copy.deepcopy(data_dict)
        if seq_types is not None:
            for seq_type in self.get_template_dict().keys():
                if seq_type not in seq_types:
                    del data_dict[seq_type]
        if self.check_if_dict_empty(data_dict):
            data_dict = None
        return data_dict


    def filter_max_1(self, data_dict, seq_types):
        data_dict = copy.deepcopy(data_dict)
        if seq_types is not None:
            for seq_type in self.get_template_dict().keys():
                if seq_type in seq_types and len(data_dict[seq_type]) > 1:
                    data_dict[seq_type] = [data_dict[seq_type][0]]

        return data_dict


    def unpack_data(self, data_dict):
        data_dict = copy.deepcopy(data_dict)
        lst_data_dict = []
        if len(data_dict["seg"]) == 0:
            seg = []
        else:
            seg = [data_dict["seg"][0]]
        del data_dict["seg"]

        if len(data_dict["class_label"]) == 0:
            class_label = []
        else:
            class_label = [data_dict["class_label"][0]]
        del data_dict["class_label"]

        if len(data_dict["mask"]) == 0:
            mask = []
        else:
            mask = [data_dict["mask"][0]]
        del data_dict["mask"]

        not_healthy = data_dict["not_healthy"]
        del data_dict["not_healthy"]

        # cdr = None
        # if 'cdr' in data_dict:
        #     cdr = data_dict["cdr"]
        #     del data_dict["cdr"]

        keyword = data_dict["keyword"]
        del data_dict["keyword"]

        cdr = None
        if 'cdr' in data_dict:
            cdr = data_dict["cdr"]
            del data_dict["cdr"]

        for label, (k, value_lst) in enumerate(data_dict.items()):
            for value in value_lst:
                template_dict = copy.deepcopy(self.get_template_dict())
                template_dict[k].append(value)
                template_dict["seg"] = seg
                template_dict["not_healthy"] = not_healthy
                if cdr is not None:
                    template_dict["cdr"] = cdr
                template_dict["class_label"] = class_label
                template_dict["mask"] = mask
                template_dict["keyword"] = keyword
                lst_data_dict.append(template_dict)
        return lst_data_dict


    def unpack_helper(self, data_dict):
        lst_data_dict = []
        #for data_dict in lst_data_dicts:
        lst_data_dict_unpacked = self.unpack_data(data_dict)
        lst_data_dict.extend(lst_data_dict_unpacked)
        return lst_data_dict


    def get_files(
            self, lst_template_dict, seq_type,
            req_seq=[], unpack=False, train_test=0.1, test_val=0.5
    ):
        lst_data_dict = []
        for data_dict in lst_template_dict:
            save_seq = True
            for seq in req_seq:
                if len(data_dict[seq]) == 0:
                    save_seq = False
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None and save_seq:
                lst_data_dict.append(data_dict)

        lst_data_split_dict = {}
        for data_dict in lst_data_dict:
            keyword = data_dict['keyword'][0]
            if keyword not in lst_data_split_dict:
                lst_data_split_dict[keyword] = []
            lst_data_split_dict[keyword].append(data_dict)

        lst_train = []
        lst_val = []
        lst_test = []
        for keyword, lst_data_dict in lst_data_split_dict.items():
            _train, _test = sk_model_selection.train_test_split(
                lst_data_dict, test_size=train_test, random_state=42
            )
            _test, _val = sk_model_selection.train_test_split(
                _test, test_size=test_val, random_state=42
            )
            lst_train.extend(_train)
            lst_val.extend(_val)
            lst_test.extend(_test)

        if unpack:
            _train_unpack = []
            for data_dict in lst_train:
                data_dict = self.unpack_helper(data_dict)
                _train_unpack.extend(data_dict)

            _val_unpack = []
            for data_dict in lst_val:
                data_dict = self.unpack_helper(data_dict)
                _val_unpack.extend(data_dict)

            _test_unpack = []
            for data_dict in lst_test:
                data_dict = self.unpack_helper(data_dict)
                _test_unpack.extend(data_dict)
            train_final = _train_unpack
            val_final = _val_unpack
            test_final = _test_unpack
        else:
            train_final = lst_train
            val_final = lst_val
            test_final = lst_test

        return train_final, val_final, test_final


    def split_data_minmax_1(self, lst_template_dict, seq_type, req_seq=["t1", "t2", "swi", "flair"]):
        lst_data_dict = []
        for data_dict in lst_template_dict:
            save_seq = True
            for seq in req_seq:
                if len(data_dict[seq]) == 0:
                    save_seq = False
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None and save_seq:
                data_dict = self.filter_max_1(data_dict, seq_type)
                lst_data_dict.append(data_dict)

        _train, _test = sk_model_selection.train_test_split(
            lst_data_dict, test_size=0.1, random_state=42
        )
        _test, _val = sk_model_selection.train_test_split(
            _test, test_size=0.5, random_state=42
        )

        _train_unpack = []
        for data_dict in _train:
            data_dict = self.unpack_helper(data_dict)
            _train_unpack.extend(data_dict)

        train = []
        for data_dict in _train_unpack:
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None:
                train.append(data_dict)

        _val_unpack = []
        for data_dict in _val:
            data_dict = self.unpack_helper(data_dict)
            _val_unpack.extend(data_dict)

        val = []
        for data_dict in _val_unpack:
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None:
                val.append(data_dict)

        _test_unpack = []
        for data_dict in _test:
            data_dict = self.unpack_helper(data_dict)
            _test_unpack.extend(data_dict)

        test = []
        for data_dict in _test_unpack:
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None:
                test.append(data_dict)

        return train, val, test


    def split_data_seq(self, lst_template_dict, seq_type):
        _lst_data_dict = []
        for data_dict in lst_template_dict:
            lst_data_dict = self.unpack_helper(data_dict)
            _lst_data_dict.extend(lst_data_dict)

        _lst_template_dict = []
        for data_dict in _lst_data_dict:
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None:
                _lst_template_dict.append(data_dict)

        lst_y = self.get_seq_label(_lst_template_dict)
        assert len(lst_y) == len(_lst_template_dict)

        _train, _val, _test = [], [], []
        if len(_lst_template_dict) != 0:
            _train, _test, _, _test_y = sk_model_selection.train_test_split(
                _lst_template_dict, lst_y, test_size=0.1, random_state=42, stratify=lst_y
            )
            _test, _val, _, _ = sk_model_selection.train_test_split(
                _test, _test_y, test_size=0.5, random_state=42, stratify=_test_y,
            )
        return _train, _val, _test


    def split_data_class(self, lst_template_dict, seq_type):
        _lst_data_dict = []
        for data_dict in lst_template_dict:
            lst_data_dict = self.unpack_helper(data_dict)
            _lst_data_dict.extend(lst_data_dict)

        _lst_template_dict = []
        for data_dict in _lst_data_dict:
            data_dict = self.filter_seq_types(data_dict, seq_type)
            if data_dict is not None:
                _lst_template_dict.append(data_dict)

        lst_y = self.get_class_label(_lst_template_dict)
        assert len(lst_y) == len(_lst_template_dict)

        _train, _test, _, _test_y = sk_model_selection.train_test_split(
            _lst_template_dict, lst_y, test_size=0.1, random_state=42, stratify=lst_y
        )
        _test, _val, _, _ = sk_model_selection.train_test_split(
            _test, _test_y, test_size=0.5, random_state=42, stratify=_test_y,
        )
        return _train, _val, _test


    def get_class_label(self, lst_data_dicts):
        y = []
        for data_dict in lst_data_dicts:
            y.append(data_dict["class_label"][0])
        return y


    def get_seq_label(self, lst_data_dicts):
        y = []
        for data_dict in lst_data_dicts:
            data_dict = copy.deepcopy(data_dict)
            try:
                del data_dict["seg"]
            except:
                pass
            try:
                del data_dict["not_healthy"]
            except:
                pass

            try:
                del data_dict["class_label"]
            except:
                pass

            try:
                del data_dict["mask"]
            except:
                pass
            for label, (k, value_lst) in enumerate(data_dict.items()):
                if (len(value_lst) != 0):
                    y.append(label)
        return y


# -------------------------------- IXI --------------------------------
        # healthy
class DatasetIXI(SuperDataset):

    def __init__(self):
        super(DatasetIXI, self).__init__()
        self.file_cat = "IXI"

    def read(self):
        template_dict = super().get_template_dict()
        lst_template_dict = []

        path_to_t1, path_to_t2 = f"{DATA_ROOT}/IXI/IXI-T1/*.nii.gz", f"{DATA_ROOT}/IXI/IXI-T2/"
        lst_files_t1 = glob.glob(path_to_t1)

        for t1 in lst_files_t1:
            name_t1 = os.path.basename(os.path.normpath(t1))[:-10]
            name_t2 = "{}/{}-T2.nii.gz".format(path_to_t2, name_t1)

            data_dict = copy.deepcopy(template_dict)
            data_dict["t1"] = [t1]
            data_dict["t2"] = []
            data_dict["not_healthy"] = [False]
            data_dict["annotations"] = []

            if os.path.isfile(name_t2):
                data_dict["t2"] = [name_t2]

            data_dict["keyword"] = ["ixi"]
            lst_template_dict.append(data_dict)
        return lst_template_dict



# -------------------------------- OPENMS --------------------------------
# not healthy
# -------------------------------- YALELOW --------------------------------
# healthy
# -------------------------------- YALEHIGH --------------------------------
# healthy
# -------------------------------- ATLAS1 --------------------------------
# not healthy
# -------------------------------- ATLAS2 --------------------------------
# not healthy
# -------------------------------- EPISURG --------------------------------
# not healthy
# -------------------------------- OASIS3 --------------------------------
# not healthy
# -------------------------------- FCDBONN --------------------------------
# not healthy
class DatasetFCDBONN(SuperDataset):

    def __init__(self):
        super(DatasetFCDBONN, self).__init__()
        self.file_cat = "FCDBONN"

    def get_healthy_not_healthy(self, lst_template_dict):
        lst_healthy = []
        lst_nhealthy = []
        for data_dict in lst_template_dict:
            if data_dict["not_healthy"][0] == True:
                lst_nhealthy.append(data_dict)
            elif data_dict["not_healthy"][0] == False:
                lst_healthy.append(data_dict)
        return lst_healthy, lst_nhealthy

    def read(self):
        template_dict = super().get_template_dict()
        lst_template_dict = []

        path_to_t1, path_to_flair = \
            f"{DATA_ROOT}/FCDBONN/*/*/*_T1w.nii.gz", \
            f"{DATA_ROOT}/FCDBONN/*/*/*_FLAIR.nii.gz"

        lst_files_t1 = glob.glob(path_to_t1)
        lst_files_flair = glob.glob(path_to_flair)

        count = 0
        for t1, flair in zip(lst_files_t1, lst_files_flair):
            data_dict = copy.deepcopy(template_dict)
            data_dict["t1"] = [t1]
            data_dict["flair"] = [flair]
            lst_files_seg = glob.glob(str(pathlib.Path(t1).parent / "*_roi.nii.gz"))
            if len(lst_files_seg) > 0:
                data_dict["seg"] = lst_files_seg
                data_dict["not_healthy"] = [True]
                count +=1
            else:
                data_dict["seg"] = []
                data_dict["not_healthy"] = [False]

            data_dict["keyword"] = ["fcdbonn"]
            lst_template_dict.append(data_dict)

        return lst_template_dict


# -------------------------------- MSLESSEG --------------------------------
# not healthy
# -------------------------------- IDEAS --------------------------------
# not healthy
# -------------------------------- IDEAS --------------------------------
class DatasetIdeas(SuperDataset):
    def __init__(self):
        super(DatasetIdeas, self).__init__()
        self.file_cat = "IDEAS"

    def read(self):
        template_dict = super().get_template_dict()
        lst_template_dict = []

        base = f"{DATA_ROOT}/IDEAS/ds005602"
        mask_base = f"{DATA_ROOT}/IDEAS/ds005602_masks"

        path_to_t1 = f"{base}/*/anat/*_T1w.nii.gz"
        lst_files_t1 = sorted(glob.glob(path_to_t1))

        for t1 in lst_files_t1:
            # sub-1, sub-2, ... → extract number
            sub_dir = pathlib.Path(t1).parent.parent.name  # "sub-1"
            sub_num = sub_dir.replace("sub-", "")           # "1"

            data_dict = copy.deepcopy(template_dict)
            data_dict["t1"] = [t1]

            # FLAIR (not all subjects have it)
            flair_path = os.path.join(base, sub_dir, "anat", f"{sub_dir}_FLAIR.nii.gz")
            if os.path.exists(flair_path):
                data_dict["flair"] = [flair_path]

            # Resection mask in orig space
            mask_path = os.path.join(mask_base, sub_num, f"{sub_num}_MaskInOrig.nii.gz")
            if os.path.exists(mask_path):
                data_dict["seg"] = [mask_path]
                data_dict["not_healthy"] = [True]
            else:
                data_dict["seg"] = []
                data_dict["not_healthy"] = [False]

            data_dict["keyword"] = ["ideas"]
            lst_template_dict.append(data_dict)

        return lst_template_dict



if __name__ == "__main__":
    datasets = {
        "IXI": DatasetIXI,
        "FCDBONN": DatasetFCDBONN,
        "IDEAS": DatasetIdeas,
    }

    for name, cls in datasets.items():
        print(f"\n{'='*60}")
        print(f"Loading: {name}")
        print(f"{'='*60}")
        try:
            ds = cls()
            data = ds.read()
            print(f"  Subjects loaded: {len(data)}")
            if len(data) > 0:
                d = data[0]
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, cls in datasets.items():
        try:
            ds = cls()
            data = ds.read()
            n = len(data)
            n_healthy = sum(1 for d in data if d.get("not_healthy", [None])[0] == False)
            n_sick = sum(1 for d in data if d.get("not_healthy", [None])[0] == True)
            n_seg = sum(1 for d in data if len(d.get("seg", [])) > 0)
            print(f"  {name:12s} | total: {n:5d} | healthy: {n_healthy:5d} | not_healthy: {n_sick:5d} | with seg: {n_seg:5d}")
        except Exception as e:
            print(f"  {name:12s} | FAILED: {e}")