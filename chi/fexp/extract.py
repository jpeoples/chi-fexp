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
    def __init__(self, extractors, dataset_root=None):
        self.extractors = extractors
        self.dataset_root = dataset_root

    def get_path(self, p):
        if self.dataset_root is None:
            return p
        else:
            return os.path.join(self.dataset_root, p)

    def process_row(self, row):
        index, row = row
        im = row['Image']
        msk = row['Mask']

        im = sitk.ReadImage(self.get_path(im))
        msk = sitk.ReadImage(self.get_path(msk))

        ress = {k: extractor.execute(im, msk) for k, extractor in self.extractors.items()}

        def add_to_row(x):
            dct = row.to_dict()
            dct.update(x)
            return dct

        ress = {k: add_to_row(r) for k, r in ress.items()}

        return index, ress

    def tabulate_results(self, results):
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

def main():
    tt = TicToc()
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True, nargs='+')
    parser.add_argument('--output', required=True, nargs='+')
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_root", required=False, default=None)
    parser.add_argument("--jobs", default=1, type=int)

    args = parser.parse_args()

    table = pandas.read_csv(args.dataset, index_col=0)

    extractors = parse_confs(args.conf)
    outputs = dict(zip(args.conf, args.output))
    processor = Processor(extractors, dataset_root=args.dataset_root)

    print("Time to load up:", tt.toc())


    if args.jobs == 1:
        results = processor.tabulate_results(map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))
    else:
        import multiprocessing as mp
        with mp.Pool(processes=args.jobs) as pool:
            results = processor.tabulate_results(pool.map(processor.process_row, tqdm(table.iterrows(), total=table.shape[0])))

    for name, result in results.items():
        output = outputs[name]
        result.to_csv(output)

    print("Total run time", tt.toc())

if __name__=="__main__": main()
    






