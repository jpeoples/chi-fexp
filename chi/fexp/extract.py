import radiomics, radiomics.featureextractor
import SimpleITK as sitk
from tqdm import tqdm
import pandas
import os, os.path

import argparse

def make_2d_extractor(fname):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(fname)

    extractor.enableFeatureClassByName('shape2D', enabled=False)
    extractor.enableFeatureClassByName('shape', enabled=False)
    return extractor

def make_3d_extractor(fname):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(fname)
    extractor.enableFeatureClassByName('shape2D', enabled=False)
    return extractor

class Processor:
    def __init__(self, extractors, dataset_root=None, image_column="Image", mask_column="Mask", label_column="MaskLabel", default_label=1, dump_preprocessed=False, dump_dir=None):
        self.extractors = extractors
        self.dataset_root = dataset_root

        self.image_column=image_column
        self.mask_column=mask_column
        self.label_column=label_column
        self.default_label=default_label
        self.dump_preprocessed = dump_preprocessed
        self.dump_dir = dump_dir

    def get_path(self, p):
        if self.dataset_root is None:
            return p
        else:
            return os.path.join(self.dataset_root, p)

    def dump_process(self, row):
        index, row = row
        print(index, row[self.mask_column])
        im = row[self.image_column]
        msk = row[self.mask_column]
        impath = im
        mskpath = msk

        label = row.get(self.label_column, self.default_label)


        im = sitk.ReadImage(self.get_path(im))
        msk = sitk.ReadImage(self.get_path(msk)) == label

        try:
            ress = {k: extractor.loadImage(im, msk, generalInfo=None, **extractor.settings.copy()) for k, extractor in self.extractors.items()}
        except:
            print("Error was in row", index, row)
            print("Image:", row[self.image_column])
            print("Mask:", row[self.mask_column])
            raise

        result_rows = {}
        for conf, (lim, lmsk) in ress.items():
            conf_name = os.path.basename(conf).replace(".yaml", "")
            label_num = f"{label:02d}"
            root = os.path.join(self.dump_dir, conf_name, label_num)
            imoutpath = os.path.join(root, impath)
            mskoutpath = os.path.join(root, mskpath)
            os.makedirs(os.path.dirname(imoutpath), exist_ok=True)
            os.makedirs(os.path.dirname(mskoutpath), exist_ok=True)
            sitk.WriteImage(lim, imoutpath)
            sitk.WriteImage(lmsk, mskoutpath)

            out = row.to_dict()
            out[self.image_column] = os.path.relpath(imoutpath, self.dump_dir)
            out[self.mask_column] = os.path.relpath(mskoutpath, self.dump_dir)
            out[self.label_column] = 1
            out["OriginalMaskLabel"] = label

            result_rows[conf] = out

        return index, result_rows

    def process_row(self, row):
        if self.dump_preprocessed:
            return self.dump_process(row)
        index, row = row
        print(index, row[self.mask_column])
        im = row[self.image_column]
        msk = row[self.mask_column]

        label = row.get(self.label_column, self.default_label)


        im = sitk.ReadImage(self.get_path(im))
        msk = sitk.ReadImage(self.get_path(msk)) == label

        try:
            ress = {k: extractor.execute(im, msk) for k, extractor in self.extractors.items()}
        except:
            print("Error was in row", index, row)
            print("Image:", row[self.image_column])
            print("Mask:", row[self.mask_column])
            raise

        def add_to_row(x):
            dct = row.to_dict()
            dct.update(x)
            return dct

        ress = {k: add_to_row(r) for k, r in ress.items()}

        return index, ress

    @staticmethod
    def tabulate_results(results):
        output = {}
        for index, ress in results:
            for extractor, features in ress.items():
                output.setdefault(extractor, {})[index] = features

        output = {k: pandas.DataFrame.from_dict(v, orient='index') for k, v in output.items()}

        return output

def parse_confs(conf):
    cases = {}
    for c in conf:
        cases[c] = make_3d_extractor(c)
    return cases

# TODO load conf files from resource files!
import time
class TicToc:
    def __init__(self, scaler=1e9):
        self.last = None
        self.scaler = scaler
        self.tic()
    
    def tic(self):
        self.last = time.perf_counter_ns()
    
    def toc(self):
        diff = time.perf_counter_ns() - self.last
        return diff / self.scaler

def do_execute(args, ix, row):
    extractors = parse_confs(args.conf)
    processor = Processor(extractors, dataset_root=args.dataset_root, image_column=args.image_column, mask_column=args.mask_column, label_column=args.label_column, default_label=args.use_label, dump_preprocessed=args.dump_preprocessed, dump_dir=args.dump_dir)
    return processor.process_row((ix, row))


from joblib import Parallel, delayed
def main():
    tt = TicToc()
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, nargs='+')
    parser.add_argument('--output', required=True, nargs='+')
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_root", required=False, default=None)
    parser.add_argument("--jobs", default=1, type=int)
    parser.add_argument("--image_column", default="Image")
    parser.add_argument("--mask_column", default="Mask")
    parser.add_argument("--label_column", default="MaskLabel")
    parser.add_argument("--dump_preprocessed", action='store_true')
    parser.add_argument("--dump_dir")
    parser.add_argument("--use_label", default=1, type=int)
    parser.add_argument("--start", default=-1, type=int)
    parser.add_argument("--count", default=-1, type=int)



    args = parser.parse_args()

    table = pandas.read_csv(args.dataset)

    if args.start > -1:
        assert args.count > -1
        istart = args.start
        iend = args.start + args.count
        iend = min(iend, table.shape[0])

        print("Batch mode, computing rows", istart, "to", iend)

        table = table.iloc[istart:iend]


    extractors = parse_confs(args.conf)
    outputs = dict(zip(args.conf, args.output))
    #processor = Processor(extractors, dataset_root=args.dataset_root, image_column=args.image_column, mask_column=args.mask_column, label_column=args.label_column, default_label=args.use_label)


    print("Time to load up:", tt.toc())

    

    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(do_execute)(args, ix, row) for ix, row in table.iterrows())
    print("Merging results")
    results = Processor.tabulate_results(results)

    #if args.jobs == 1:
    #    results = processor.tabulate_results(map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))
    #else:
    #    import multiprocessing as mp
    #    with mp.Pool(processes=args.jobs) as pool:
    #        results = processor.tabulate_results(pool.map(processor.process_row, table.iterrows()))

    print("Saving results")
    for name, result in results.items():
        output = outputs[name]
        result.to_csv(output, index=False)

    print("Total run time", tt.toc())

if __name__=="__main__": main()
    






